# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""Compare lossless internal K/V tile sizes for SM120 blk64 VSA.

The sparse mask always uses 64-token logical blocks.  Only the CuTeDSL
compute tile is varied, so this benchmark does not change model sparsity or
FastH3's trained block-selection geometry.
"""

import argparse
import statistics

import torch

from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.testing import bench_gpu_time


LOGICAL_BLOCK_SIZE = 64
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=109632)
    parser.add_argument("--num-heads", type=int, default=7)
    parser.add_argument("--prefix-blocks", type=int, default=8)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--kv-tiles", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_h3_style_bsr(
    num_blocks: int, prefix_blocks: int, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build prefix-dense and video-top-k rows in 64-token logical blocks."""
    video_blocks = num_blocks - prefix_blocks
    keep_video = min(topk, video_blocks)
    prefix = torch.arange(prefix_blocks, dtype=torch.int32)
    all_blocks = torch.arange(num_blocks, dtype=torch.int32)
    rows = []
    for q_block in range(num_blocks):
        if q_block < prefix_blocks:
            rows.append(all_blocks)
            continue
        start = (q_block - prefix_blocks) % video_blocks
        selected = (torch.arange(keep_video, dtype=torch.int32) + start) % video_blocks
        rows.append(torch.cat((prefix, selected + prefix_blocks)))

    counts = torch.tensor([row.numel() for row in rows], dtype=torch.int32)
    indptr = torch.empty(num_blocks + 1, dtype=torch.int32)
    indptr[0] = 0
    indptr[1:] = counts.cumsum(0)
    return indptr, torch.cat(rows)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm()
    denom = expected.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return float((diff / denom).item())


def main() -> None:
    args = parse_args()
    if args.seq_len < LOGICAL_BLOCK_SIZE or args.seq_len % LOGICAL_BLOCK_SIZE:
        raise ValueError("seq-len must be a positive multiple of 64")
    if args.num_heads <= 0 or args.warmup <= 0 or args.repeat <= 0:
        raise ValueError("num-heads, warmup, and repeat must be positive")
    if any(tile not in (16, 32, 64) for tile in args.kv_tiles):
        raise ValueError("kv-tiles must contain only 16, 32, and 64")
    if 64 not in args.kv_tiles:
        raise ValueError("kv-tiles must include the 64-token baseline")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark requires an SM120 GPU")

    num_blocks = args.seq_len // LOGICAL_BLOCK_SIZE
    if not 0 <= args.prefix_blocks < num_blocks:
        raise ValueError("prefix-blocks must be in [0, num_blocks)")
    if args.topk <= 0:
        raise ValueError("topk must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    shape = (args.seq_len, args.num_heads, HEAD_DIM)
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    output = torch.empty_like(q)
    reference = torch.empty_like(q)
    indptr, indices = make_h3_style_bsr(num_blocks, args.prefix_blocks, args.topk)
    indptr = indptr.to(device)
    indices = indices.to(device)

    workspace = torch.empty(64 << 20, dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="vsa_sm120_blk64")
    wrapper.plan(
        indptr,
        indices,
        args.seq_len,
        args.seq_len,
        LOGICAL_BLOCK_SIZE,
        LOGICAL_BLOCK_SIZE,
        args.num_heads,
        args.num_heads,
        HEAD_DIM,
        q_data_type=dtype,
        o_data_type=dtype,
    )

    print("SM120_VSA_KV_TILE_SWEEP")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(
        f"gpu={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}"
    )
    print(
        f"seq_len={args.seq_len} heads={args.num_heads} head_dim={HEAD_DIM} "
        f"logical_block=64 blocks={num_blocks} prefix_blocks={args.prefix_blocks} "
        f"topk={args.topk} logical_nnz={indices.numel()}"
    )

    tile_sizes = [64, *dict.fromkeys(tile for tile in args.kv_tiles if tile != 64)]
    timings: dict[int, float] = {}
    for tile in tile_sizes:
        wrapper.run(q, k, v, out=output, sm120_kv_tile_size=tile)
        torch.cuda.synchronize()
        if tile == 64:
            reference.copy_(output)
        else:
            max_abs = float((output.float() - reference.float()).abs().max().item())
            rel_l2 = relative_l2(output, reference)
            print(
                f"correctness kv_tile={tile} max_abs_vs_64={max_abs:.6g} "
                f"relative_l2_vs_64={rel_l2:.6g}"
            )
            if max_abs > 0.04 or rel_l2 > 0.01:
                raise RuntimeError(f"kv_tile={tile} failed the kv_tile=64 comparison")

        samples = bench_gpu_time(
            lambda current=tile: wrapper.run(
                q, k, v, out=output, sm120_kv_tile_size=current
            ),
            dry_run_iters=args.warmup,
            repeat_iters=args.repeat,
            enable_cupti=True,
            cold_l2_cache=False,
        )
        timings[tile] = statistics.median(samples)

    baseline_ms = timings[64]
    for tile in tile_sizes:
        median_ms = timings[tile]
        print(
            f"kv_tile={tile} median_ms={median_ms:.6f} "
            f"speedup_vs_64={baseline_ms / median_ms:.4f}x"
        )
    print("SM120_VSA_KV_TILE_SWEEP: PASS")


if __name__ == "__main__":
    main()
