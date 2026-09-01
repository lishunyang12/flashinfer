# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""Measure SM120 PRIMS skip-softmax across representative video DiT shapes."""

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time


@dataclass(frozen=True)
class BenchCase:
    name: str
    q_len: int
    kv_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int = 128
    batch_size: int = 1


# These are attention-layer shapes, not end-to-end model benchmarks. The exact
# model points follow the public model layouts:
#
# * MiniMax H3: 56 heads x 128, 768x1344 output, [1,2,2] DiT patches.
#   The packed lengths use the 537-token prompt from the vLLM-Omni SM120
#   benchmark and include its 40 Hz stereo audio rows and 64-row padding.
# * Wan2.2 A14B: 40 heads x 128, [1,2,2] DiT patches after the 4x8x8
#   temporal/spatial VAE compression. The listed lengths are the exact latent
#   token counts for the named resolution and frame count.
# The TP2+Ulysses2 and Ulysses4 entries additionally cover the per-rank head
# shapes used by the corresponding four- and eight-GPU vLLM-Omni recipes.
MODEL_CASES = (
    BenchCase("minimax_h3_sweep_4k", 4096, 4096, 56, 56),
    BenchCase("minimax_h3_sweep_16k", 16384, 16384, 56, 56),
    BenchCase("minimax_h3_768p_4s_107f", 33152, 33152, 56, 56),
    BenchCase("minimax_h3_768p_5s_124f", 38272, 38272, 56, 56),
    BenchCase("minimax_h3_tp2_usp2_5s_124f", 38272, 38272, 14, 14),
    BenchCase("minimax_h3_768p_8p7s_209f", 63744, 63744, 56, 56),
    BenchCase("minimax_h3_tp2_usp2_8p7s_209f", 63744, 63744, 14, 14),
    BenchCase("minimax_h3_768p_15s_362f", 109632, 109632, 56, 56),
    BenchCase("wan22_sweep_4k", 4096, 4096, 40, 40),
    BenchCase("wan22_sweep_16k", 16384, 16384, 40, 40),
    BenchCase("wan22_480p_81f", 32760, 32760, 40, 40),
    BenchCase("wan22_720p_81f", 75600, 75600, 40, 40),
    BenchCase("wan22_usp4_720p_81f", 75600, 75600, 10, 10),
    BenchCase("wan22_720p_121f", 111600, 111600, 40, 40),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("models", "minimax-h3", "wan2.2", "custom"),
        default="models",
        help="Shape matrix to run; models covers MiniMax H3 and Wan2.2.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=256)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-tile", type=int, choices=(64, 128), default=128)
    parser.add_argument("--kv-tile", type=int, choices=(64, 128), default=128)
    parser.add_argument("--skip-scale-factor", type=float, default=1.0)
    parser.add_argument(
        "--active-skip-fraction",
        type=float,
        default=None,
        help=(
            "Target fraction of complete 128-token KV tiles made negligible in "
            "skip_active mode. The default preserves the legacy upper-bound case."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def selected_cases(args: argparse.Namespace) -> tuple[BenchCase, ...]:
    if args.preset == "models":
        return MODEL_CASES
    if args.preset == "minimax-h3":
        return tuple(case for case in MODEL_CASES if case.name.startswith("minimax_h3"))
    if args.preset == "wan2.2":
        return tuple(case for case in MODEL_CASES if case.name.startswith("wan22"))
    return (
        BenchCase(
            "custom",
            args.q_len,
            args.kv_len,
            args.num_q_heads,
            args.num_kv_heads,
            args.head_dim,
            args.batch_size,
        ),
    )


def validate_args(args: argparse.Namespace, cases: tuple[BenchCase, ...]) -> None:
    if args.skip_scale_factor <= 0 or not math.isfinite(args.skip_scale_factor):
        raise ValueError("skip-scale-factor must be finite and positive")
    if args.active_skip_fraction is not None and not (
        math.isfinite(args.active_skip_fraction) and 0 <= args.active_skip_fraction < 1
    ):
        raise ValueError("active-skip-fraction must be finite and in [0, 1)")
    if args.warmup <= 0 or args.repeat < 2:
        raise ValueError("warmup must be positive and repeat must be at least 2")
    for case in cases:
        for name in (
            "batch_size",
            "q_len",
            "kv_len",
            "num_q_heads",
            "num_kv_heads",
        ):
            if getattr(case, name) <= 0:
                raise ValueError(
                    f"{case.name}: {name.replace('_', '-')} must be positive"
                )
        if case.num_q_heads % case.num_kv_heads:
            raise ValueError(
                f"{case.name}: num-q-heads must be divisible by num-kv-heads"
            )
        if case.head_dim not in (32, 64, 128, 256):
            raise ValueError(f"{case.name}: unsupported head-dim {case.head_dim}")
        if case.kv_len < 256:
            raise ValueError(f"{case.name}: kv-len must be at least 256")


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return float((difference / denominator).item())


def benchmark_case(
    case: BenchCase,
    *,
    skip_scale_factor: float,
    active_skip_fraction: float | None,
    q_tile: int,
    kv_tile: int,
    warmup: int,
    repeat: int,
    seed: int,
) -> list[dict[str, object]]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    fp8 = torch.float8_e4m3fn
    total_q = case.batch_size * case.q_len
    total_kv = case.batch_size * case.kv_len

    # The SM120 kernel traverses KV from right to left. Give the trailing tiles
    # dominant logits and make a controlled fraction of earlier complete tiles
    # negligible. This isolates speedup as a function of tiles skipped.
    q = torch.ones(total_q, case.num_q_heads, case.head_dim, device=device, dtype=fp8)
    k = torch.zeros(
        total_kv, case.num_kv_heads, case.head_dim, device=device, dtype=fp8
    )
    complete_kv_tiles = case.kv_len // kv_tile
    if active_skip_fraction is None:
        negligible_tiles = max(complete_kv_tiles - 1, 0)
    else:
        negligible_tiles = min(
            int(complete_kv_tiles * active_skip_fraction),
            max(complete_kv_tiles - 1, 0),
        )
    negligible_tokens = negligible_tiles * kv_tile
    dominant_tokens = case.kv_len - negligible_tokens
    realized_skip_fraction = (
        negligible_tiles / complete_kv_tiles if complete_kv_tiles else 0.0
    )
    for batch_idx in range(case.batch_size):
        end = (batch_idx + 1) * case.kv_len
        k[end - dominant_tokens : end] = 1
    v = torch.randint(
        -2,
        3,
        (total_kv, case.num_kv_heads, case.head_dim),
        device=device,
        dtype=torch.float32,
    ).to(fp8)
    qo_indptr = torch.arange(
        case.batch_size + 1, device=device, dtype=torch.int32
    ).mul_(case.q_len)
    kv_indptr = torch.arange(
        case.batch_size + 1, device=device, dtype=torch.int32
    ).mul_(case.kv_len)
    from flashinfer.attention.cute_dsl.sm120_fmha import (
        sm120_fmha_fp8_ragged_prefill,
    )

    outputs = {
        name: torch.empty(
            total_q,
            case.num_q_heads,
            case.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        for name in ("dense", "skip_no_tiles", "skip_active")
    }
    thresholds = {
        "dense": None,
        "skip_no_tiles": 1e-30,
        "skip_active": skip_scale_factor,
    }

    def run(name: str) -> None:
        sm120_fmha_fp8_ragged_prefill(
            q,
            k,
            v,
            outputs[name],
            qo_indptr,
            kv_indptr,
            max_seqlen_q=case.q_len,
            max_seqlen_k=case.kv_len,
            sm_scale=1.0 / math.sqrt(case.head_dim),
            q_tile=q_tile,
            kv_tile=kv_tile,
            enable_pdl=True,
            skip_softmax_threshold_scale_factor=thresholds[name],
        )

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
            f"{case.name}: tiny-threshold specialization diverged from dense: "
            f"max_abs={no_tiles_max_abs:g}, relative_l2={no_tiles_rel_l2:g}"
        )

    timings = {
        name: bench_gpu_time(
            fn=lambda current=name: run(current),
            dry_run_iters=warmup,
            repeat_iters=repeat,
            enable_cupti=True,
            cold_l2_cache=False,
        )
        for name in thresholds
    }
    medians = {name: statistics.median(values) for name, values in timings.items()}
    dense_ms = medians["dense"]
    dense_equivalent_flops = (
        4
        * case.batch_size
        * case.q_len
        * case.kv_len
        * case.num_q_heads
        * case.head_dim
    )
    rows = []
    for name, threshold in thresholds.items():
        values = timings[name]
        median_ms = medians[name]
        rows.append(
            {
                "case": case.name,
                "batch_size": case.batch_size,
                "q_len": case.q_len,
                "kv_len": case.kv_len,
                "num_q_heads": case.num_q_heads,
                "num_kv_heads": case.num_kv_heads,
                "head_dim": case.head_dim,
                "q_tile": q_tile,
                "kv_tile": kv_tile,
                "target_skip_fraction": active_skip_fraction,
                "realized_skip_fraction": realized_skip_fraction,
                "mode": name,
                "threshold_scale_factor": threshold,
                "median_ms": median_ms,
                "std_ms": statistics.stdev(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "cv": statistics.stdev(values) / statistics.fmean(values),
                "speedup_vs_dense": dense_ms / median_ms,
                "dense_equivalent_tflops": dense_equivalent_flops / median_ms / 1e9,
                "max_abs_vs_dense": {
                    "dense": 0.0,
                    "skip_no_tiles": no_tiles_max_abs,
                    "skip_active": active_max_abs,
                }[name],
                "relative_l2_vs_dense": {
                    "dense": 0.0,
                    "skip_no_tiles": no_tiles_rel_l2,
                    "skip_active": active_rel_l2,
                }[name],
            }
        )

    print(
        "case="
        f"{case.name} batch={case.batch_size} q_len={case.q_len} kv_len={case.kv_len} "
        f"hq={case.num_q_heads} hkv={case.num_kv_heads} d={case.head_dim} "
        f"q_tile={q_tile} kv_tile={kv_tile} "
        f"negligible_tiles={negligible_tiles}/{complete_kv_tiles} "
        f"realized_skip_fraction={realized_skip_fraction:.6f}"
    )
    for row in rows:
        print(
            f"mode={row['mode']} threshold={row['threshold_scale_factor']} "
            f"median_ms={row['median_ms']:.6f} std_ms={row['std_ms']:.6f} "
            f"cv={row['cv']:.4%} min_ms={row['min_ms']:.6f} max_ms={row['max_ms']:.6f} "
            f"speedup_vs_dense={row['speedup_vs_dense']:.4f}x "
            f"dense_equivalent_tflops={row['dense_equivalent_tflops']:.3f} "
            f"max_abs_vs_dense={row['max_abs_vs_dense']:.6g} "
            f"relative_l2_vs_dense={row['relative_l2_vs_dense']:.6g}"
        )
    return rows


def main() -> None:
    args = parse_args()
    cases = selected_cases(args)
    validate_args(args, cases)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark requires an SM120 GPU")

    print("SM120_SKIP_SOFTMAX_BENCHMARK")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(
        f"gpu={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}"
    )
    print(
        f"preset={args.preset} cases={len(cases)} warmup={args.warmup} "
        f"repeat={args.repeat} seed={args.seed} active_skip_factor={args.skip_scale_factor}"
    )
    print(f"q_tile={args.q_tile} kv_tile={args.kv_tile}")
    print(f"active_skip_fraction={args.active_skip_fraction}")
    print("active_mode_data=synthetic_controlled_skip_fraction")
    print("timing=CUPTI_preferred_CUDA_events_fallback warm_L2")

    rows = []
    for index, case in enumerate(cases):
        print(f"CASE_START {index + 1}/{len(cases)} {case.name}", flush=True)
        rows.extend(
            benchmark_case(
                case,
                skip_scale_factor=args.skip_scale_factor,
                active_skip_fraction=args.active_skip_fraction,
                q_tile=args.q_tile,
                kv_tile=args.kv_tile,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed,
            )
        )
        torch.cuda.empty_cache()

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV={args.csv}")
    print(f"SM120_SKIP_SOFTMAX_BENCHMARK: PASS ({len(cases)} cases)")


if __name__ == "__main__":
    main()
