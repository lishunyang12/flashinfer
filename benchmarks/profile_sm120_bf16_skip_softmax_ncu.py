# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""Capture one native SM120 BF16 VSA launch with Nsight Compute."""

import argparse

import torch

from flashinfer.sparse import BlockSparseAttentionWrapper


BLOCK_SIZE = 64
MODES = {
    "dense": None,
    "skip_no_tiles": 1e-30,
    "skip_active": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--num-heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return float((difference / denominator).item())


def main() -> None:
    args = parse_args()
    if args.seq_len < 2 * BLOCK_SIZE or args.seq_len % BLOCK_SIZE:
        raise ValueError("seq-len must be a multiple of 64 and at least 128")
    if args.num_heads <= 0 or args.head_dim != 128:
        raise ValueError("num-heads must be positive and head-dim must be 128")
    if args.warmup <= 0:
        raise ValueError("warmup must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This profile requires an SM120 GPU")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    shape = (args.seq_len, args.num_heads, args.head_dim)
    num_blocks = args.seq_len // BLOCK_SIZE

    q = torch.ones(shape, device=device, dtype=dtype)
    k = torch.zeros(shape, device=device, dtype=dtype)
    k[-BLOCK_SIZE:].fill_(1)
    v = torch.randint(-2, 3, shape, device=device, dtype=torch.int32).to(dtype)
    block_mask = torch.ones(
        args.num_heads,
        num_blocks,
        num_blocks,
        device=device,
        dtype=torch.bool,
    )
    workspace = torch.empty(64 << 20, device=device, dtype=torch.uint8)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="vsa_sm120_blk64")
    wrapper.plan(
        None,
        None,
        args.seq_len,
        args.seq_len,
        BLOCK_SIZE,
        BLOCK_SIZE,
        args.num_heads,
        args.num_heads,
        args.head_dim,
        q_data_type=dtype,
        o_data_type=dtype,
        block_mask=block_mask,
    )

    dense_out = torch.empty_like(q)
    profile_out = torch.empty_like(q)

    def run(out: torch.Tensor, threshold: float | None) -> None:
        wrapper.run(
            q,
            k,
            v,
            out=out,
            skip_softmax_threshold_scale_factor=threshold,
        )

    run(dense_out, None)
    threshold = MODES[args.mode]
    for _ in range(args.warmup):
        run(profile_out, threshold)
    torch.cuda.synchronize()

    max_abs = float((profile_out.float() - dense_out.float()).abs().max().item())
    rel_l2 = relative_l2(profile_out, dense_out)
    if args.mode == "skip_no_tiles" and (max_abs != 0 or rel_l2 != 0):
        raise RuntimeError(
            "skip_no_tiles diverged from dense: "
            f"max_abs={max_abs:g} relative_l2={rel_l2:g}"
        )
    if args.mode == "skip_active" and (max_abs > 0.1 or rel_l2 > 0.03):
        raise RuntimeError(
            f"skip_active failed accuracy: max_abs={max_abs:g} relative_l2={rel_l2:g}"
        )

    expected_skip_fraction = (
        (num_blocks - 1) / num_blocks if args.mode == "skip_active" else 0.0
    )
    print(
        "PROFILE_CONFIG "
        f"mode={args.mode} seq_len={args.seq_len} heads={args.num_heads} "
        f"head_dim={args.head_dim} threshold={threshold} "
        f"expected_skip_fraction={expected_skip_fraction:.6f} "
        f"max_abs_vs_dense={max_abs:.6g} relative_l2_vs_dense={rel_l2:.6g}",
        flush=True,
    )

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"sm120_bf16_skip_softmax::{args.mode}")
    run(profile_out, threshold)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"PROFILE_CAPTURED mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
