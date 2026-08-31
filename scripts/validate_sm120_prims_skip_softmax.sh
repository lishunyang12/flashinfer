#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlashInfer contributors.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
required_code_commit="f764a6c868543fcf581da43b4ecef4de9b5aaa6a"
gpu_id="${GPU_ID:-0}"
base_python="${PYTHON_BIN:-python3}"
validation_venv="${VALIDATION_VENV:-${repo_root}/.venv-sm120-validation}"
result_base="${RESULT_BASE:-$(dirname "${repo_root}")/sm120-skip-results}"
result_root="${result_base}/$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${result_root}"
exec > >(tee "${result_root}/validation.log") 2>&1

echo "RESULT_ROOT=${result_root}"
echo "REPO_ROOT=${repo_root}"
echo "GPU_ID=${gpu_id}"
hostname
date -u

git -C "${repo_root}" merge-base --is-ancestor "${required_code_commit}" HEAD || {
  echo "ERROR: checkout does not contain required code commit ${required_code_commit}" >&2
  exit 2
}
echo "FLASHINFER_COMMIT=$(git -C "${repo_root}" rev-parse HEAD)"
git -C "${repo_root}" status --short

command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi not found" >&2
  exit 2
}
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

gpu_uuid="$(nvidia-smi -i "${gpu_id}" --query-gpu=uuid --format=csv,noheader)"
busy_count="$(
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
    | grep -c "${gpu_uuid}" || true
)"
if ((busy_count != 0)); then
  echo "REFUSED: GPU ${gpu_id} has ${busy_count} active compute process(es)" >&2
  exit 3
fi

command -v "${base_python}" >/dev/null || {
  echo "ERROR: PYTHON_BIN=${base_python} is not executable" >&2
  exit 2
}
CUDA_VISIBLE_DEVICES="${gpu_id}" "${base_python}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: selected Python cannot access CUDA")
capability = torch.cuda.get_device_capability(0)
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("gpu=", torch.cuda.get_device_name(0))
print("compute_capability=", capability)
if capability != (12, 0):
    raise SystemExit(f"ERROR: expected SM120, got compute capability {capability}")
PY

if [[ ! -x "${validation_venv}/bin/python" ]]; then
  "${base_python}" -m venv --system-site-packages "${validation_venv}"
fi
# shellcheck disable=SC1091
source "${validation_venv}/bin/activate"
python -m pip install --upgrade pytest

export BUILD_NVEP=0
export CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.7.1}"
# setup_test_env pins the inherited CUDA stack, installs only missing runtime
# requirements, and exposes install_flashinfer_editable.
# shellcheck disable=SC1091
source "${repo_root}/scripts/setup_test_env.sh"
install_flashinfer_editable "${repo_root}"

CUDA_VISIBLE_DEVICES="${gpu_id}" python - <<'PY'
from importlib.metadata import version

import torch

print("flashinfer-python=", version("flashinfer-python"))
print("nvidia-cutlass-dsl=", version("nvidia-cutlass-dsl"))
print("runtime_compute_capability=", torch.cuda.get_device_capability(0))
PY

CUDA_VISIBLE_DEVICES="${gpu_id}" python -m pytest -q \
  "${repo_root}/tests/attention/test_sm120_fmha_api.py"
CUDA_VISIBLE_DEVICES="${gpu_id}" python -m pytest -q -s \
  "${repo_root}/tests/attention/test_sm120_fmha.py" \
  -k "sm120_ragged_skip_softmax_omits_negligible_tiles"
CUDA_VISIBLE_DEVICES="${gpu_id}" python -m pytest -q -s \
  "${repo_root}/tests/attention/test_sm120_prims_prefill_backend.py" \
  -k "ragged_public_wrapper_skip_softmax"

echo "SM120_SKIP_SOFTMAX_VALIDATION: PASS"
