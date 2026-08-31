# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side tests for the SM120 PRIMS FMHA API and JIT specialization."""

import inspect
import math

import pytest
import torch

from flashinfer.attention.cute_dsl.sm120_fmha import (
    _skip_softmax_threshold_log2,
)


def test_skip_softmax_threshold_conversion_and_validation():
    assert _skip_softmax_threshold_log2(None, 256) is None
    assert _skip_softmax_threshold_log2(0, 256) is None
    assert _skip_softmax_threshold_log2(2.0, 256).value == pytest.approx(
        math.log2(2.0 / 256)
    )
    with pytest.raises(ValueError, match="finite and >= 0"):
        _skip_softmax_threshold_log2(-1.0, 256)
    with pytest.raises(ValueError, match="finite and >= 0"):
        _skip_softmax_threshold_log2(float("nan"), 256)


def test_ragged_wrapper_exposes_skip_softmax_parameter():
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    parameters = inspect.signature(BatchPrefillWithRaggedKVCacheWrapper.run).parameters
    assert "skip_softmax_threshold_scale_factor" in parameters


def test_skip_softmax_has_a_distinct_jit_specialization(monkeypatch):
    pytest.importorskip("cutlass.experimental")

    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_ragged_kernel,
    )
    from flashinfer.jit import cute_dsl_core

    observed_names = []

    def fake_build_and_load(module_name, kernel_name, compile_fn, **kwargs):
        observed_names.append((module_name, kernel_name))
        return kernel_name

    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    monkeypatch.setattr(
        cute_dsl_core, "build_and_load_cute_dsl_kernel", fake_build_and_load
    )

    common = dict(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=128,
        is_causal=False,
        kv_tile=128,
        q_tile=128,
        device=torch.device("cuda"),
    )
    dense = compile_sm120_fmha_fp8_ragged_kernel(**common)
    sparse = compile_sm120_fmha_fp8_ragged_kernel(**common, enable_skip_softmax=True)

    assert dense != sparse
    assert observed_names[0][1].endswith("_skip0")
    assert observed_names[1][1].endswith("_skip1")
    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()
