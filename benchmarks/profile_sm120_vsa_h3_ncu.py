# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""Capture one steady-state SM120 VSA launch at the MiniMax-H3 shape."""

import argparse

import torch

from flashinfer.sparse import BlockSparseAttentionWrapper


BLOCK_SIZE = 64
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=109632)
    parser.add_argument("--num-heads", type=int, default=7)
    parser.add_argument("--prefix-blocks", type=int, default=8)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
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


def main() -> None:
    args = parse_args()
    if args.seq_len < BLOCK_SIZE or args.seq_len % BLOCK_SIZE:
        raise ValueError("seq-len must be a positive multiple of 64")
    if args.num_heads <= 0 or args.warmup <= 0:
        raise ValueError("num-heads and warmup must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This profile requires an SM120 GPU")

    num_blocks = args.seq_len // BLOCK_SIZE
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
    out = torch.empty_like(q)
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
        BLOCK_SIZE,
        BLOCK_SIZE,
        args.num_heads,
        args.num_heads,
        HEAD_DIM,
        q_data_type=dtype,
        o_data_type=dtype,
    )

    for _ in range(args.warmup):
        wrapper.run(q, k, v, out=out)
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        raise RuntimeError("warmup output contains non-finite values")

    print(
        "PROFILE_CONFIG "
        f"seq_len={args.seq_len} heads={args.num_heads} head_dim={HEAD_DIM} "
        f"logical_block={BLOCK_SIZE} blocks={num_blocks} "
        f"prefix_blocks={args.prefix_blocks} topk={args.topk} "
        f"logical_nnz={indices.numel()}",
        flush=True,
    )

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("sm120_vsa_h3_blk64")
    wrapper.run(q, k, v, out=out)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print("PROFILE_CAPTURED sm120_vsa_h3_blk64", flush=True)


if __name__ == "__main__":
    main()
