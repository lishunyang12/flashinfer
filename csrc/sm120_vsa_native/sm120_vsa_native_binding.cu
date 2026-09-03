/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace flashinfer::sm120_vsa_native {

namespace {

constexpr int kTile = 64;
constexpr int kHeadDim = 128;
constexpr int kMathWarps = 4;
constexpr int kIoWarp = 4;
constexpr int kThreads = 160;
constexpr int kVecBf16 = 8;
constexpr int kVecsPerRow = kHeadDim / kVecBf16;
constexpr int kVecsPerTile = kTile * kVecsPerRow;

using bf16 = __nv_bfloat16;

struct alignas(128) SharedStorage {
  bf16 qv[kTile][kHeadDim];
  bf16 k[2][kTile][kHeadDim];
};

struct MmaResult {
  float x0, x1, x2, x3;
};

__device__ __forceinline__ int swizzled_col(int row, int col) {
  return col ^ ((row & 7) << 3);
}

__device__ __forceinline__ MmaResult mma_bf16_m16n8k16(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
  MmaResult out;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(out.x0), "=f"(out.x1), "=f"(out.x2), "=f"(out.x3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
        "f"(c1), "f"(c2), "f"(c3));
  return out;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

__device__ __forceinline__ uint32_t load_bf16x2(const bf16* ptr) {
  return *reinterpret_cast<const uint32_t*>(ptr);
}

__device__ __forceinline__ uint32_t load_strided_bf16x2(const bf16* lo,
                                                         const bf16* hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(*lo)) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(*hi)) << 16);
}

__device__ __forceinline__ void update_mma(MmaResult result, float (&dst)[4]) {
  dst[0] = result.x0;
  dst[1] = result.x1;
  dst[2] = result.x2;
  dst[3] = result.x3;
}

__device__ __forceinline__ float quad_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1, 4));
  return fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2, 4));
}

__device__ __forceinline__ float quad_sum(float value) {
  value += __shfl_xor_sync(0xffffffff, value, 1, 4);
  return value + __shfl_xor_sync(0xffffffff, value, 2, 4);
}

__device__ __forceinline__ int64_t qkv_offset(int batch_idx, int token_idx,
                                               int head_idx, int seqlen,
                                               int num_heads) {
  return ((static_cast<int64_t>(batch_idx) * seqlen + token_idx) * num_heads +
          head_idx) *
         kHeadDim;
}

__device__ __forceinline__ void load_tile(bf16 (&dst)[kTile][kHeadDim],
                                          const bf16* src, int batch_idx,
                                          int token_base, int head_idx,
                                          int seqlen, int num_heads,
                                          int lane) {
#pragma unroll 1
  for (int vec = lane; vec < kVecsPerTile; vec += 32) {
    const int row = vec / kVecsPerRow;
    const int col = (vec % kVecsPerRow) * kVecBf16;
    const int dst_col = swizzled_col(row, col);
    const int64_t src_idx =
        qkv_offset(batch_idx, token_base + row, head_idx, seqlen, num_heads) + col;
    *reinterpret_cast<uint4*>(&dst[row][dst_col]) =
        *reinterpret_cast<const uint4*>(src + src_idx);
  }
}

__device__ __forceinline__ int sparse_index(
    const int32_t* indices, int logical_idx, int qo_tile_idx, int head_idx,
    int batch_idx, int64_t stride_k, int64_t stride_q, int64_t stride_h,
    int64_t stride_b) {
  const int64_t offset = static_cast<int64_t>(logical_idx) * stride_k +
                         static_cast<int64_t>(qo_tile_idx) * stride_q +
                         static_cast<int64_t>(head_idx) * stride_h +
                         static_cast<int64_t>(batch_idx) * stride_b;
  return indices[offset];
}

__device__ __forceinline__ uint32_t load_q_a(const bf16* q, int warp,
                                              int k_step, int lane,
                                              int fragment) {
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int row = warp * 16 + group + ((fragment == 1 || fragment == 3) ? 8 : 0);
  const int col = k_step * 16 + thread_in_group * 2 + (fragment >= 2 ? 8 : 0);
  return load_bf16x2(q + row * kHeadDim + swizzled_col(row, col));
}

__device__ __forceinline__ uint32_t load_k_b(const bf16* k, int n_step,
                                              int k_step, int lane,
                                              int fragment) {
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int row = n_step * 8 + group;
  const int col = k_step * 16 + thread_in_group * 2 + (fragment ? 8 : 0);
  return load_bf16x2(k + row * kHeadDim + swizzled_col(row, col));
}

__device__ __forceinline__ uint32_t load_v_b(const bf16* v, int n_step,
                                              int k_step, int lane,
                                              int fragment) {
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int row0 = k_step * 16 + thread_in_group * 2 + (fragment ? 8 : 0);
  const int col = n_step * 8 + group;
  return load_strided_bf16x2(
      v + row0 * kHeadDim + swizzled_col(row0, col),
      v + (row0 + 1) * kHeadDim + swizzled_col(row0 + 1, col));
}

__device__ __forceinline__ void zero_output(bf16* output, int batch_idx,
                                             int qo_tile_idx, int head_idx,
                                             int seqlen, int num_heads,
                                             int math_warp, int lane) {
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int row0 = qo_tile_idx * kTile + math_warp * 16 + group;
  const int row1 = row0 + 8;
#pragma unroll
  for (int n_step = 0; n_step < 16; ++n_step) {
    const int col = n_step * 8 + thread_in_group * 2;
    const int64_t off0 = qkv_offset(batch_idx, row0, head_idx, seqlen, num_heads) + col;
    const int64_t off1 = qkv_offset(batch_idx, row1, head_idx, seqlen, num_heads) + col;
    *reinterpret_cast<uint32_t*>(output + off0) = 0;
    *reinterpret_cast<uint32_t*>(output + off1) = 0;
  }
}

__global__ __launch_bounds__(kThreads, 2) void vsa_bf16_h3_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ v, bf16* __restrict__ output,
    const int32_t* __restrict__ indices, const int32_t* __restrict__ block_nums,
    int batch, int seqlen_q, int seqlen_k, int num_heads, int num_q_tiles,
    int64_t idx_stride_k, int64_t idx_stride_q, int64_t idx_stride_h,
    int64_t idx_stride_b, int64_t num_stride_q, int64_t num_stride_h,
    int64_t num_stride_b, float softmax_scale_log2e) {
  const int qo_tile_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int batch_idx = blockIdx.z;
  if (qo_tile_idx >= num_q_tiles || head_idx >= num_heads || batch_idx >= batch) {
    return;
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool is_math = warp < kMathWarps;
  const bool is_io = warp == kIoWarp;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int64_t num_offset = static_cast<int64_t>(qo_tile_idx) * num_stride_q +
                             static_cast<int64_t>(head_idx) * num_stride_h +
                             static_cast<int64_t>(batch_idx) * num_stride_b;
  const int num_tiles = block_nums[num_offset];

  if (is_io) {
    load_tile(smem.qv, q, batch_idx, qo_tile_idx * kTile, head_idx, seqlen_q,
              num_heads, lane);
    if (num_tiles > 0) {
      const int physical = sparse_index(indices, 0, qo_tile_idx, head_idx,
                                        batch_idx, idx_stride_k, idx_stride_q,
                                        idx_stride_h, idx_stride_b);
      load_tile(smem.k[0], k, batch_idx, physical * kTile, head_idx, seqlen_k,
                num_heads, lane);
    }
  }
  __syncthreads();

  uint32_t q_frag[8][4];
  if (is_math) {
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        q_frag[ks][f] = load_q_a(&smem.qv[0][0], warp, ks, lane, f);
      }
    }
  }
  __syncthreads();

  if (!is_math && !is_io) return;

  if (is_math) {
    float out_acc[16][4];
#pragma unroll
    for (int ns = 0; ns < 16; ++ns) {
#pragma unroll
      for (int f = 0; f < 4; ++f) out_acc[ns][f] = 0.0f;
    }
    float row_max0 = -CUDART_INF_F;
    float row_max1 = -CUDART_INF_F;
    float row_sum0 = 0.0f;
    float row_sum1 = 0.0f;

#pragma unroll 1
    for (int tile = 0; tile < num_tiles; ++tile) {
      float score[8][4];
#pragma unroll
      for (int ns = 0; ns < 8; ++ns) {
#pragma unroll
        for (int f = 0; f < 4; ++f) score[ns][f] = 0.0f;
      }

#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
        for (int ns = 0; ns < 8; ++ns) {
          const uint32_t b0 = load_k_b(&smem.k[tile & 1][0][0], ns, ks, lane, 0);
          const uint32_t b1 = load_k_b(&smem.k[tile & 1][0][0], ns, ks, lane, 1);
          update_mma(mma_bf16_m16n8k16(
                         q_frag[ks][0], q_frag[ks][1], q_frag[ks][2],
                         q_frag[ks][3], b0, b1, score[ns][0], score[ns][1],
                         score[ns][2], score[ns][3]),
                     score[ns]);
        }
      }

      __syncthreads();

      float local_max0 = -CUDART_INF_F;
      float local_max1 = -CUDART_INF_F;
#pragma unroll
      for (int ns = 0; ns < 8; ++ns) {
        local_max0 = fmaxf(local_max0, fmaxf(score[ns][0], score[ns][1]));
        local_max1 = fmaxf(local_max1, fmaxf(score[ns][2], score[ns][3]));
      }
      const float next_max0 = fmaxf(row_max0, quad_max(local_max0));
      const float next_max1 = fmaxf(row_max1, quad_max(local_max1));
      const float scale0 = row_max0 == next_max0
                               ? 1.0f
                               : (isinf(row_max0)
                                      ? 0.0f
                                      : exp2f((row_max0 - next_max0) *
                                              softmax_scale_log2e));
      const float scale1 = row_max1 == next_max1
                               ? 1.0f
                               : (isinf(row_max1)
                                      ? 0.0f
                                      : exp2f((row_max1 - next_max1) *
                                              softmax_scale_log2e));

      float tile_sum0 = 0.0f;
      float tile_sum1 = 0.0f;
#pragma unroll
      for (int ns = 0; ns < 8; ++ns) {
        score[ns][0] = exp2f((score[ns][0] - next_max0) * softmax_scale_log2e);
        score[ns][1] = exp2f((score[ns][1] - next_max0) * softmax_scale_log2e);
        score[ns][2] = exp2f((score[ns][2] - next_max1) * softmax_scale_log2e);
        score[ns][3] = exp2f((score[ns][3] - next_max1) * softmax_scale_log2e);
        tile_sum0 += score[ns][0] + score[ns][1];
        tile_sum1 += score[ns][2] + score[ns][3];
      }
      tile_sum0 = quad_sum(tile_sum0);
      tile_sum1 = quad_sum(tile_sum1);
      row_sum0 = row_sum0 * scale0 + tile_sum0;
      row_sum1 = row_sum1 * scale1 + tile_sum1;
      row_max0 = next_max0;
      row_max1 = next_max1;

      const bool unit_scale = __all_sync(0xffffffff, scale0 == 1.0f && scale1 == 1.0f);
      if (!unit_scale) {
#pragma unroll
        for (int ns = 0; ns < 16; ++ns) {
          out_acc[ns][0] *= scale0;
          out_acc[ns][1] *= scale0;
          out_acc[ns][2] *= scale1;
          out_acc[ns][3] *= scale1;
        }
      }

#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t a0 = pack_bf16(score[ks * 2][0], score[ks * 2][1]);
        const uint32_t a1 = pack_bf16(score[ks * 2][2], score[ks * 2][3]);
        const uint32_t a2 = pack_bf16(score[ks * 2 + 1][0], score[ks * 2 + 1][1]);
        const uint32_t a3 = pack_bf16(score[ks * 2 + 1][2], score[ks * 2 + 1][3]);
#pragma unroll
        for (int ns = 0; ns < 16; ++ns) {
          const uint32_t b0 = load_v_b(&smem.qv[0][0], ns, ks, lane, 0);
          const uint32_t b1 = load_v_b(&smem.qv[0][0], ns, ks, lane, 1);
          update_mma(mma_bf16_m16n8k16(
                         a0, a1, a2, a3, b0, b1, out_acc[ns][0],
                         out_acc[ns][1], out_acc[ns][2], out_acc[ns][3]),
                     out_acc[ns]);
        }
      }

      __syncthreads();
    }

    if (num_tiles == 0) {
      zero_output(output, batch_idx, qo_tile_idx, head_idx, seqlen_q,
                  num_heads, warp, lane);
      return;
    }

    const float inv_sum0 = 1.0f / row_sum0;
    const float inv_sum1 = 1.0f / row_sum1;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int row0 = qo_tile_idx * kTile + warp * 16 + group;
    const int row1 = row0 + 8;
#pragma unroll
    for (int ns = 0; ns < 16; ++ns) {
      const int col = ns * 8 + thread_in_group * 2;
      const int64_t off0 = qkv_offset(batch_idx, row0, head_idx, seqlen_q,
                                      num_heads) +
                           col;
      const int64_t off1 = qkv_offset(batch_idx, row1, head_idx, seqlen_q,
                                      num_heads) +
                           col;
      *reinterpret_cast<uint32_t*>(output + off0) =
          pack_bf16(out_acc[ns][0] * inv_sum0, out_acc[ns][1] * inv_sum0);
      *reinterpret_cast<uint32_t*>(output + off1) =
          pack_bf16(out_acc[ns][2] * inv_sum1, out_acc[ns][3] * inv_sum1);
    }
  } else {
#pragma unroll 1
    for (int tile = 0; tile < num_tiles; ++tile) {
      const int physical = sparse_index(indices, tile, qo_tile_idx, head_idx,
                                        batch_idx, idx_stride_k, idx_stride_q,
                                        idx_stride_h, idx_stride_b);
      load_tile(smem.qv, v, batch_idx, physical * kTile, head_idx, seqlen_k,
                num_heads, lane);
      if (tile + 1 < num_tiles) {
        const int next_physical = sparse_index(
            indices, tile + 1, qo_tile_idx, head_idx, batch_idx, idx_stride_k,
            idx_stride_q, idx_stride_h, idx_stride_b);
        load_tile(smem.k[(tile + 1) & 1], k, batch_idx,
                  next_physical * kTile, head_idx, seqlen_k, num_heads, lane);
      }
      __syncthreads();
      __syncthreads();
    }
  }
}

}  // namespace

void run(TensorView q, TensorView k, TensorView v, TensorView output,
         TensorView indices, TensorView block_nums, double softmax_scale) {
  CHECK_DIM(4, q);
  CHECK_DIM(4, k);
  CHECK_DIM(4, v);
  CHECK_DIM(4, output);
  CHECK_DIM(4, indices);
  CHECK_DIM(3, block_nums);
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_CUDA(output);
  CHECK_CUDA(indices);
  CHECK_CUDA(block_nums);
  CHECK_DEVICE(k, q);
  CHECK_DEVICE(v, q);
  CHECK_DEVICE(output, q);
  CHECK_DEVICE(indices, q);
  CHECK_DEVICE(block_nums, q);
  TVM_FFI_ICHECK(q.dtype().code == kDLBfloat && q.dtype().bits == 16)
      << "q must be bfloat16";
  TVM_FFI_ICHECK(k.dtype() == q.dtype() && v.dtype() == q.dtype() &&
                 output.dtype() == q.dtype())
      << "q, k, v, and output must all be bfloat16";
  TVM_FFI_ICHECK(indices.dtype().code == kDLInt && indices.dtype().bits == 32)
      << "indices must be int32";
  TVM_FFI_ICHECK(block_nums.dtype().code == kDLInt && block_nums.dtype().bits == 32)
      << "block_nums must be int32";
  TVM_FFI_ICHECK(q.IsContiguous() && k.IsContiguous() && v.IsContiguous() &&
                 output.IsContiguous())
      << "q, k, v, and output must be contiguous BSHD tensors";

  const int batch = static_cast<int>(q.size(0));
  const int seqlen_q = static_cast<int>(q.size(1));
  const int num_heads = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(3));
  const int seqlen_k = static_cast<int>(k.size(1));
  TVM_FFI_ICHECK(head_dim == kHeadDim && k.size(3) == kHeadDim &&
                 v.size(3) == kHeadDim)
      << "native SM120 VSA requires head_dim=128";
  TVM_FFI_ICHECK(seqlen_q % kTile == 0 && seqlen_k % kTile == 0)
      << "native SM120 VSA requires sequence lengths divisible by 64";
  TVM_FFI_ICHECK(k.size(0) == batch && k.size(2) == num_heads &&
                 v.size(0) == batch && v.size(1) == seqlen_k &&
                 v.size(2) == num_heads && output.size(0) == batch &&
                 output.size(1) == seqlen_q && output.size(2) == num_heads &&
                 output.size(3) == kHeadDim)
      << "native SM120 VSA requires matching BSHD shapes and MHA heads";

  const int num_q_tiles = seqlen_q / kTile;
  TVM_FFI_ICHECK(indices.size(1) == num_q_tiles &&
                 indices.size(2) == num_heads && indices.size(3) == batch &&
                 block_nums.size(0) == num_q_tiles &&
                 block_nums.size(1) == num_heads && block_nums.size(2) == batch)
      << "native SM120 VSA sparse metadata shape mismatch";
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  dim3 grid(num_q_tiles, num_heads, batch);
  dim3 block(kThreads);
  const size_t smem_bytes = sizeof(SharedStorage);
  const auto kernel = vsa_bf16_h3_kernel;
  auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(smem_bytes));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);
  status = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                cudaSharedmemCarveoutMaxShared);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);
  const cudaStream_t stream = get_stream(q.device());
  kernel<<<grid, block, smem_bytes, stream>>>(
      static_cast<const bf16*>(q.data_ptr()),
      static_cast<const bf16*>(k.data_ptr()),
      static_cast<const bf16*>(v.data_ptr()), static_cast<bf16*>(output.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()),
      static_cast<const int32_t*>(block_nums.data_ptr()), batch, seqlen_q,
      seqlen_k, num_heads, num_q_tiles, indices.stride(0), indices.stride(1),
      indices.stride(2), indices.stride(3), block_nums.stride(0),
      block_nums.stride(1), block_nums.stride(2),
      static_cast<float>(softmax_scale * 1.44269504088896340736));
  status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "native SM120 VSA launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::sm120_vsa_native

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm120_vsa_native_run,
                              flashinfer::sm120_vsa_native::run);
