# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""Capture one steady-state SM120 PRIMS attention launch with Nsight Compute."""

import argparse
import math

import torch

import flashinfer


MODES = {
    "dense": None,
    "skip_no_tiles": 1e-30,
    "skip_active": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--q-len", type=int, default=63744)
    parser.add_argument("--kv-len", type=int, default=63744)
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
    if args.q_len <= 0 or args.kv_len < 256:
        raise ValueError("q-len must be positive and kv-len must be at least 256")
    if args.num_heads <= 0 or args.head_dim not in (32, 64, 128, 256):
        raise ValueError("num-heads must be positive and head-dim must be supported")
    if args.warmup <= 0:
        raise ValueError("warmup must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This profile requires an SM120 GPU")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    fp8 = torch.float8_e4m3fn
    q = torch.ones(
        args.q_len,
        args.num_heads,
        args.head_dim,
        device=device,
        dtype=fp8,
    )
    k = torch.zeros(
        args.kv_len,
        args.num_heads,
        args.head_dim,
        device=device,
        dtype=fp8,
    )
    k[-128:] = 1
    v = torch.randint(
        -2,
        3,
        (args.kv_len, args.num_heads, args.head_dim),
        device=device,
        dtype=torch.float32,
    ).to(fp8)
    indptr_q = torch.tensor([0, args.q_len], device=device, dtype=torch.int32)
    indptr_kv = torch.tensor([0, args.kv_len], device=device, dtype=torch.int32)
    workspace = torch.empty(64 << 20, device=device, dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        backend="cute-dsl-prims",
    )
    wrapper.plan(
        indptr_q,
        indptr_kv,
        args.num_heads,
        args.num_heads,
        args.head_dim,
        causal=False,
        q_data_type=fp8,
        kv_data_type=fp8,
        o_data_type=torch.bfloat16,
    )

    dense_out = torch.empty_like(q, dtype=torch.bfloat16)
    profile_out = torch.empty_like(q, dtype=torch.bfloat16)

    def run(out: torch.Tensor, threshold: float | None) -> None:
        wrapper.run(
            q,
            k,
            v,
            out=out,
            enable_pdl=True,
            skip_softmax_threshold_scale_factor=threshold,
        )

    run(dense_out, None)
    threshold = MODES[args.mode]
    for _ in range(args.warmup):
        run(profile_out, threshold)
    torch.cuda.synchronize()

    max_abs = float((profile_out.float() - dense_out.float()).abs().max().item())
    rel_l2 = relative_l2(profile_out, dense_out)
    if args.mode == "skip_no_tiles" and (max_abs > 0.2 or rel_l2 > 0.05):
        raise RuntimeError(
            "skip_no_tiles diverged from dense: "
            f"max_abs={max_abs:g} relative_l2={rel_l2:g}"
        )

    num_kv_tiles = math.ceil(args.kv_len / 128)
    dominant_tiles = 1 if args.kv_len % 128 == 0 else 2
    expected_skip_fraction = (
        max(num_kv_tiles - dominant_tiles, 0) / num_kv_tiles
        if args.mode == "skip_active"
        else 0.0
    )
    print(
        "PROFILE_CONFIG "
        f"mode={args.mode} q_len={args.q_len} kv_len={args.kv_len} "
        f"heads={args.num_heads} head_dim={args.head_dim} threshold={threshold} "
        f"expected_skip_fraction={expected_skip_fraction:.6f} "
        f"max_abs_vs_dense={max_abs:.6g} relative_l2_vs_dense={rel_l2:.6g}",
        flush=True,
    )

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"sm120_skip_softmax::{args.mode}")
    run(profile_out, threshold)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"PROFILE_CAPTURED mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
