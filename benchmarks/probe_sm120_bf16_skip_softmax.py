# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate and time the native SM120 BF16 warp-MMA skip-softmax path."""

import argparse
import math
import statistics
import traceback

import torch
import torch.nn.functional as F

from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.testing import bench_gpu_time


BLOCK_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--skip-scale-factor", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
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
    if args.num_heads <= 0:
        raise ValueError("num-heads must be positive")
    if args.head_dim != 128:
        raise ValueError("the native sm120_blk64 kernel requires head-dim=128")
    if not math.isfinite(args.skip_scale_factor) or args.skip_scale_factor <= 0:
        raise ValueError("skip-scale-factor must be finite and positive")
    if args.warmup < 1 or args.repeat < 1:
        raise ValueError("warmup and repeat must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe requires an SM120 GPU")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    shape = (args.seq_len, args.num_heads, args.head_dim)
    num_blocks = args.seq_len // BLOCK_SIZE

    # Blocks are visited right-to-left. The last block establishes a dominant
    # row maximum, so all earlier blocks are eligible for skip-softmax.
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

    modes = {
        "dense": None,
        "skip_no_tiles": 1e-30,
        "skip_active": args.skip_scale_factor,
    }
    outputs = {name: torch.empty_like(q) for name in modes}

    def run(name: str) -> None:
        wrapper.run(
            q,
            k,
            v,
            out=outputs[name],
            skip_softmax_threshold_scale_factor=modes[name],
        )

    print("SM120_BF16_SKIP_SOFTMAX_PROBE")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(
        f"gpu={torch.cuda.get_device_name()} "
        f"capability={torch.cuda.get_device_capability()}"
    )
    print(
        f"seq_len={args.seq_len} heads={args.num_heads} "
        f"head_dim={args.head_dim} dtype={dtype} "
        f"skip_scale_factor={args.skip_scale_factor} "
        f"expected_skip_fraction={(num_blocks - 1) / num_blocks:.6f}"
    )

    stage = "run_dense"
    try:
        for name in modes:
            stage = f"run_{name}"
            run(name)
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
        dense_max_abs = float(
            (outputs["dense"].float() - reference.float()).abs().max().item()
        )
        dense_rel_l2 = relative_l2(outputs["dense"], reference)
        no_skip_max_abs = float(
            (outputs["skip_no_tiles"].float() - outputs["dense"].float())
            .abs()
            .max()
            .item()
        )
        no_skip_rel_l2 = relative_l2(outputs["skip_no_tiles"], outputs["dense"])
        skip_max_abs = float(
            (outputs["skip_active"].float() - outputs["dense"].float())
            .abs()
            .max()
            .item()
        )
        skip_rel_l2 = relative_l2(outputs["skip_active"], outputs["dense"])
        print(
            f"dense_max_abs_vs_reference={dense_max_abs:.6g} "
            f"dense_relative_l2_vs_reference={dense_rel_l2:.6g}"
        )
        print(
            f"no_skip_max_abs_vs_dense={no_skip_max_abs:.6g} "
            f"no_skip_relative_l2_vs_dense={no_skip_rel_l2:.6g}"
        )
        print(
            f"skip_max_abs_vs_dense={skip_max_abs:.6g} "
            f"skip_relative_l2_vs_dense={skip_rel_l2:.6g}"
        )
        if dense_max_abs > 0.1 or dense_rel_l2 > 0.03:
            raise RuntimeError("dense SM120 BF16 output failed reference check")
        if no_skip_max_abs != 0 or no_skip_rel_l2 != 0:
            raise RuntimeError("tiny-threshold specialization diverged from dense")
        if skip_max_abs > 0.1 or skip_rel_l2 > 0.03:
            raise RuntimeError("skip SM120 BF16 output failed dense comparison")

        stage = "benchmark"
        timings = {
            name: bench_gpu_time(
                fn=lambda current=name: run(current),
                dry_run_iters=args.warmup,
                repeat_iters=args.repeat,
                enable_cupti=True,
                cold_l2_cache=False,
            )
            for name in modes
        }
        dense_ms = statistics.median(timings["dense"])
        for name in modes:
            median_ms = statistics.median(timings[name])
            std_ms = statistics.pstdev(timings[name])
            print(
                f"mode={name} median_ms={median_ms:.6f} std_ms={std_ms:.6f} "
                f"speedup_vs_dense={dense_ms / median_ms:.4f}x"
            )
    except Exception:
        print(f"SM120_BF16_SKIP_SOFTMAX_PROBE: FAIL stage={stage}", flush=True)
        traceback.print_exc()
        raise

    print("SM120_BF16_SKIP_SOFTMAX_PROBE: PASS")


if __name__ == "__main__":
    main()
