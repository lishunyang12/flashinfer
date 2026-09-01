# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""Probe whether the generic BF16 CuTe-DSL FMHA can run on SM120."""

import argparse
import math
import traceback

import torch
import torch.nn.functional as F

from flashinfer.attention.cute_dsl.fmha import cute_dsl_fmha_ragged_prefill
from flashinfer.cute_dsl.attention.fmha.compile import (
    compile_cute_dsl_fmha_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--skip-scale-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return float((difference / denominator).item())


def compile_kernel(args: argparse.Namespace, *, enable_skip: bool):
    return compile_cute_dsl_fmha_kernel(
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        args.num_heads,
        args.num_heads,
        args.head_dim,
        args.head_dim,
        False,
        False,
        False,
        enable_skip,
        False,
        torch.device("cuda"),
    )


def run_kernel(
    kernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    indptr: torch.Tensor,
    *,
    skip_scale_factor: float | None,
) -> None:
    cute_dsl_fmha_ragged_prefill(
        q=q,
        k=k,
        v=v,
        o=out,
        qo_indptr=indptr,
        kv_indptr=indptr,
        is_causal=False,
        sm_scale=1.0 / math.sqrt(q.size(-1)),
        max_qo_len=q.size(0),
        max_kv_len=k.size(0),
        kernel_fn=kernel,
        skip_softmax_threshold_scale_factor=skip_scale_factor,
        enable_pdl=False,
    )


def main() -> None:
    args = parse_args()
    if args.seq_len < 256 or args.seq_len % 128:
        raise ValueError("seq-len must be a multiple of 128 and at least 256")
    if args.num_heads <= 0:
        raise ValueError("num-heads must be positive")
    if args.head_dim not in (32, 64, 128):
        raise ValueError("head-dim must be 32, 64, or 128")
    if not math.isfinite(args.skip_scale_factor) or args.skip_scale_factor <= 0:
        raise ValueError("skip-scale-factor must be finite and positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe requires an SM120 GPU")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    shape = (args.seq_len, args.num_heads, args.head_dim)

    # KV tiles are visited right-to-left. The rightmost tile establishes a
    # dominant row max, making earlier tiles eligible for skip-softmax.
    q = torch.ones(shape, device=device, dtype=dtype)
    k = torch.zeros(shape, device=device, dtype=dtype)
    k[-128:] = 1
    v = torch.randint(-2, 3, shape, device=device, dtype=torch.int32).to(dtype)
    indptr = torch.tensor([0, args.seq_len], device=device, dtype=torch.int32)
    dense = torch.empty_like(q)
    skipped = torch.empty_like(q)

    print("SM120_BF16_SKIP_SOFTMAX_PROBE")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(
        f"gpu={torch.cuda.get_device_name()} "
        f"capability={torch.cuda.get_device_capability()}"
    )
    print(
        f"seq_len={args.seq_len} heads={args.num_heads} "
        f"head_dim={args.head_dim} dtype={dtype} "
        f"skip_scale_factor={args.skip_scale_factor}"
    )

    stage = "compile_dense"
    try:
        dense_kernel = compile_kernel(args, enable_skip=False)
        stage = "run_dense"
        run_kernel(dense_kernel, q, k, v, dense, indptr, skip_scale_factor=None)
        torch.cuda.synchronize()

        stage = "reference"
        reference = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        dense_max_abs = float((dense.float() - reference.float()).abs().max().item())
        dense_rel_l2 = relative_l2(dense, reference)
        print(
            f"dense_max_abs_vs_reference={dense_max_abs:.6g} "
            f"dense_relative_l2_vs_reference={dense_rel_l2:.6g}"
        )
        if dense_max_abs > 0.1 or dense_rel_l2 > 0.03:
            raise RuntimeError("dense BF16 CuTe-DSL output failed reference check")

        stage = "compile_skip"
        skip_kernel = compile_kernel(args, enable_skip=True)
        stage = "run_skip"
        run_kernel(
            skip_kernel,
            q,
            k,
            v,
            skipped,
            indptr,
            skip_scale_factor=args.skip_scale_factor,
        )
        torch.cuda.synchronize()

        skip_max_abs = float((skipped.float() - dense.float()).abs().max().item())
        skip_rel_l2 = relative_l2(skipped, dense)
        print(
            f"skip_max_abs_vs_dense={skip_max_abs:.6g} "
            f"skip_relative_l2_vs_dense={skip_rel_l2:.6g}"
        )
        if skip_max_abs > 0.1 or skip_rel_l2 > 0.03:
            raise RuntimeError("skip BF16 CuTe-DSL output failed dense comparison")
    except Exception:
        print(f"SM120_BF16_SKIP_SOFTMAX_PROBE: FAIL stage={stage}", flush=True)
        traceback.print_exc()
        raise

    print("SM120_BF16_SKIP_SOFTMAX_PROBE: PASS")


if __name__ == "__main__":
    main()
