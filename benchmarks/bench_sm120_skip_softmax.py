# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""Measure SM120 PRIMS skip-softmax overhead and skip-friendly speedup."""

import argparse
import math
import statistics

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=256)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--skip-scale-factor", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "q_len", "kv_len", "num_q_heads", "num_kv_heads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.num_q_heads % args.num_kv_heads:
        raise ValueError("num-q-heads must be divisible by num-kv-heads")
    if args.head_dim not in (32, 64, 128, 256):
        raise ValueError("head-dim must be one of 32, 64, 128, 256")
    if args.kv_len < 256 or args.kv_len % 128:
        raise ValueError("kv-len must be at least 256 and divisible by 128")
    if args.skip_scale_factor <= 0 or not math.isfinite(args.skip_scale_factor):
        raise ValueError("skip-scale-factor must be finite and positive")
    if args.warmup <= 0 or args.repeat <= 0:
        raise ValueError("warmup and repeat must be positive")


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return float((difference / denominator).item())


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark requires an SM120 GPU")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    fp8 = torch.float8_e4m3fn
    total_q = args.batch_size * args.q_len
    total_kv = args.batch_size * args.kv_len

    # The SM120 kernel traverses KV from right to left. The final tile receives
    # dominant logits; earlier tiles are negligible and can be skipped with the
    # default factor while retaining a close dense-attention result.
    q = torch.ones(total_q, args.num_q_heads, args.head_dim, device=device, dtype=fp8)
    k = torch.zeros(
        total_kv, args.num_kv_heads, args.head_dim, device=device, dtype=fp8
    )
    for batch_idx in range(args.batch_size):
        end = (batch_idx + 1) * args.kv_len
        k[end - 128 : end] = 1
    v = torch.randint(
        -2,
        3,
        (total_kv, args.num_kv_heads, args.head_dim),
        device=device,
        dtype=torch.float32,
    ).to(fp8)
    qo_indptr = torch.arange(
        args.batch_size + 1, device=device, dtype=torch.int32
    ).mul_(args.q_len)
    kv_indptr = torch.arange(
        args.batch_size + 1, device=device, dtype=torch.int32
    ).mul_(args.kv_len)
    workspace = torch.empty(64 << 20, device=device, dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cute-dsl-prims"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim,
        causal=False,
        q_data_type=fp8,
        kv_data_type=fp8,
        o_data_type=torch.bfloat16,
    )

    outputs = {
        name: torch.empty(
            total_q,
            args.num_q_heads,
            args.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        for name in ("dense", "skip_no_tiles", "skip_active")
    }
    thresholds = {
        "dense": None,
        "skip_no_tiles": 1e-30,
        "skip_active": args.skip_scale_factor,
    }

    def run(name: str) -> None:
        wrapper.run(
            q,
            k,
            v,
            out=outputs[name],
            enable_pdl=True,
            skip_softmax_threshold_scale_factor=thresholds[name],
        )

    # Compile and warm every specialization before correctness or timing.
    for name in thresholds:
        run(name)
    torch.cuda.synchronize()

    dense = outputs["dense"]
    no_tiles = outputs["skip_no_tiles"]
    active = outputs["skip_active"]
    no_tiles_max_abs = float((no_tiles.float() - dense.float()).abs().max().item())
    active_max_abs = float((active.float() - dense.float()).abs().max().item())
    no_tiles_rel_l2 = relative_l2(no_tiles, dense)
    active_rel_l2 = relative_l2(active, dense)
    if no_tiles_max_abs > 0.2 or no_tiles_rel_l2 > 0.05:
        raise RuntimeError(
            "Tiny-threshold skip specialization diverged from dense attention: "
            f"max_abs={no_tiles_max_abs:g}, relative_l2={no_tiles_rel_l2:g}"
        )

    timings: dict[str, list[float]] = {}
    for name in thresholds:
        timings[name] = bench_gpu_time(
            fn=lambda current=name: run(current),
            dry_run_iters=args.warmup,
            repeat_iters=args.repeat,
            cold_l2_cache=False,
        )

    medians = {name: statistics.median(values) for name, values in timings.items()}
    print("SM120_SKIP_SOFTMAX_BENCHMARK")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(
        f"gpu={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}"
    )
    print(
        "config="
        f"batch={args.batch_size},q_len={args.q_len},kv_len={args.kv_len},"
        f"hq={args.num_q_heads},hkv={args.num_kv_heads},d={args.head_dim},"
        f"warmup={args.warmup},repeat={args.repeat},seed={args.seed}"
    )
    print(
        "correctness="
        f"skip_no_tiles_max_abs={no_tiles_max_abs:.6g},"
        f"skip_no_tiles_rel_l2={no_tiles_rel_l2:.6g},"
        f"skip_active_max_abs={active_max_abs:.6g},"
        f"skip_active_rel_l2={active_rel_l2:.6g}"
    )
    dense_ms = medians["dense"]
    for name in thresholds:
        values = timings[name]
        median_ms = medians[name]
        print(
            f"mode={name} threshold={thresholds[name]} median_ms={median_ms:.6f} "
            f"min_ms={min(values):.6f} max_ms={max(values):.6f} "
            f"speedup_vs_dense={dense_ms / median_ms:.4f}x"
        )


if __name__ == "__main__":
    main()
