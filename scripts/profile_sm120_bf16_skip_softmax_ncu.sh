#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlashInfer contributors.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
repo_parent="$(dirname "${repo_root}")"
gpu_id="${GPU_ID:-0}"
base_python="${PYTHON_BIN:-${repo_parent}/minimax-h3-native/.venv/bin/python}"
validation_overlay="${VALIDATION_OVERLAY:-${repo_parent}/.flashinfer-sm120-validation}"
export FLASHINFER_WORKSPACE_BASE="${JIT_CACHE_BASE:-${repo_parent}/.flashinfer-sm120-bf16-ncu-jit}"
result_base="${RESULT_BASE:-${repo_parent}/sm120-bf16-skip-ncu-results}"
result_root="${result_base}/$(date -u +%Y%m%dT%H%M%SZ)"
telemetry_pid=""

cleanup() {
  if [[ -n "${telemetry_pid}" ]] && kill -0 "${telemetry_pid}" 2>/dev/null; then
    kill "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

mkdir -p "${result_root}" "${FLASHINFER_WORKSPACE_BASE}"
exec > >(tee "${result_root}/profile.log") 2>&1

ncu_bin="${NCU_BIN:-$(command -v ncu || true)}"
if [[ -z "${ncu_bin}" && -x /usr/local/cuda/bin/ncu ]]; then
  ncu_bin=/usr/local/cuda/bin/ncu
fi

echo "RESULT_ROOT=${result_root}"
echo "REPO_ROOT=${repo_root}"
echo "GPU_ID=${gpu_id}"
echo "PYTHON_BIN=${base_python}"
echo "NCU_BIN=${ncu_bin}"
echo "FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}"
echo "FLASHINFER_COMMIT=$(git -C "${repo_root}" rev-parse HEAD)"
hostname
date -u

if [[ ! -x "${base_python}" ]]; then
  echo "ERROR: Python not found at ${base_python}; set PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "${validation_overlay}" ]]; then
  echo "ERROR: validation overlay not found; run validation first" >&2
  exit 2
fi
if [[ -z "${ncu_bin}" || ! -x "${ncu_bin}" ]]; then
  echo "ERROR: Nsight Compute CLI (ncu) was not found; set NCU_BIN" >&2
  exit 2
fi

"${ncu_bin}" --version
bash "${repo_root}/scripts/pro5000_preflight.sh" "${gpu_id}"
nvidia-smi -i "${gpu_id}" \
  --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,clocks.sm,power.draw,memory.used \
  --format=csv,noheader,nounits --loop-ms=1000 >"${result_root}/gpu_telemetry.csv" &
telemetry_pid="$!"

validation_pythonpath="${repo_root}:${validation_overlay}${PYTHONPATH:+:${PYTHONPATH}}"
sections=(
  SpeedOfLight
  ComputeWorkloadAnalysis
  MemoryWorkloadAnalysis
  SchedulerStats
  WarpStateStats
  InstructionStats
  LaunchStats
  Occupancy
)
section_args=()
for section in "${sections[@]}"; do
  section_args+=(--section "${section}")
done
metrics=(
  gpu__time_duration.sum
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  dram__bytes_op_read.sum
  dram__bytes_op_write.sum
  lts__throughput.avg.pct_of_peak_sustained_elapsed
  lts__t_bytes.sum
  smsp__inst_executed_pipe_tensor.sum
  smsp__inst_executed_pipe_fma.sum
  smsp__inst_executed_pipe_alu.sum
  smsp__inst_executed_pipe_lsu.sum
  smsp__issue_active.avg.pct_of_peak_sustained_active
  smsp__warps_active.avg.per_cycle_active
  smsp__warps_eligible.avg.per_cycle_active
  smsp__warp_issue_stalled_barrier_per_warp_active.pct
  smsp__warp_issue_stalled_membar_per_warp_active.pct
  smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
  smsp__warp_issue_stalled_wait_per_warp_active.pct
  smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
  smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
  smsp__warp_issue_stalled_not_selected_per_warp_active.pct
)
metrics_csv="$(IFS=,; echo "${metrics[*]}")"

for mode in dense skip_no_tiles skip_active; do
  report_base="${result_root}/${mode}"
  echo "NCU_PROFILE_START mode=${mode}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${validation_pythonpath}" \
    "${ncu_bin}" \
      --target-processes all \
      --replay-mode kernel \
      --profile-from-start off \
      --launch-count 1 \
      --cache-control none \
      --clock-control none \
      --print-summary per-kernel \
      --force-overwrite \
      --export "${report_base}" \
      --log-file "${report_base}.ncu.log" \
      "${section_args[@]}" \
      --metrics "${metrics_csv}" \
      "${base_python}" "${repo_root}/benchmarks/profile_sm120_bf16_skip_softmax_ncu.py" \
      --mode "${mode}" "$@"
  report="${report_base}.ncu-rep"
  if [[ ! -f "${report}" ]]; then
    echo "ERROR: Nsight Compute did not create ${report}" >&2
    exit 4
  fi
  "${ncu_bin}" --import "${report}" --page raw --csv --print-units base \
    --log-file "${report_base}.raw.csv"
  echo "NCU_PROFILE_DONE mode=${mode} report=${report}"
done

"${base_python}" "${repo_root}/scripts/summarize_sm120_skip_softmax_ncu.py" \
  --result-root "${result_root}" | tee "${result_root}/summary.txt"
echo "SM120_BF16_SKIP_SOFTMAX_NCU: PASS"
