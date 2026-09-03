"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from pathlib import Path

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm120a_nvcc_flags


_MODULE_NAME = "sm120_vsa_native_pipeline_v3"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_tree() -> tuple[Path, Path]:
    installed_source = (
        jit_env.FLASHINFER_CSRC_DIR
        / "sm120_vsa_native"
        / "sm120_vsa_native_binding.cu"
    )
    if installed_source.exists():
        return installed_source, jit_env.FLASHINFER_CSRC_DIR

    root = _repo_root()
    checkout_source = (
        root / "csrc" / "sm120_vsa_native" / "sm120_vsa_native_binding.cu"
    )
    if checkout_source.exists():
        return checkout_source, root / "csrc"
    raise FileNotFoundError(
        "native SM120 VSA source not found in the installed package or checkout"
    )


@functools.cache
def gen_sm120_vsa_native_module() -> JitSpec:
    root = _repo_root()
    source, csrc_dir = _source_tree()
    include_dir = (
        jit_env.FLASHINFER_INCLUDE_DIR
        if jit_env.FLASHINFER_INCLUDE_DIR.exists()
        else root / "include"
    )
    return gen_jit_spec(
        _MODULE_NAME,
        [source],
        extra_cuda_cflags=[
            *sm120a_nvcc_flags,
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
        ],
        extra_include_paths=[csrc_dir, include_dir],
    )


@functools.cache
def load_sm120_vsa_native_module():
    return gen_sm120_vsa_native_module().build_and_load()


__all__ = ["gen_sm120_vsa_native_module", "load_sm120_vsa_native_module"]
