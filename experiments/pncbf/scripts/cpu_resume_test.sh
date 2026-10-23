#!/usr/bin/env bash
# Kill-and-restart test of runner.resume on CPU (16 envs).
# 1) start a run with resume=True and a checkpoint every 3 learning iterations,
# 2) SIGKILL it as soon as the first resume checkpoint exists,
# 3) rerun the identical command and check that it restores that checkpoint and continues from its step.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-$(mktemp -d)}"
RUN_NAME="${RUN_NAME:-cpu_resume_test}"
RUN_DIR="${REPO_DIR}/experiments/runs/pncbf_cpu_test/resume/${RUN_NAME}"
rm -rf "${RUN_DIR}"
export RUN_NAME TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-16384}" EVAL_FREQ="${EVAL_FREQ:-1024}"
ARGS=(--runner.resume=True --runner.save_interval_iterations=3 --runner.save_interval_minutes=0 --runner.save_model=True)

"${SCRIPT_DIR}/cpu_tiny_run.sh" "${ARGS[@]}" > "${LOG_DIR}/attempt1.log" 2>&1 &
PID=$!
while kill -0 "${PID}" 2>/dev/null && [[ ! -f "${RUN_DIR}/resume/LATEST" ]]; do sleep 0.2; done
kill -9 "${PID}" 2>/dev/null; wait "${PID}" 2>/dev/null
SAVED="$(cat "${RUN_DIR}/resume/LATEST")"
echo "attempt 1 killed; LATEST=${SAVED}; last logged step: $(grep -o 'Step: [0-9]*' "${LOG_DIR}/attempt1.log" | tail -1)"

"${SCRIPT_DIR}/cpu_tiny_run.sh" "${ARGS[@]}" > "${LOG_DIR}/attempt2.log" 2>&1
echo "attempt 2 exit code: $?"
grep -E "Resume|Step: " "${LOG_DIR}/attempt2.log" | sed 's/^\[[^]]*\] //' | head -8
echo "..."
grep -E "Step: " "${LOG_DIR}/attempt2.log" | tail -1
echo "final LATEST=$(cat "${RUN_DIR}/resume/LATEST"); logs in ${LOG_DIR}"
