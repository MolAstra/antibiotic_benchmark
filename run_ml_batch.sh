#!/usr/bin/env bash
set -euo pipefail

# Batch run:
#   data:   datasets/dpnet_db/<name>/processed/<name>/
#   models: svm, rf, lightgbm, xgb
#   cuts:   2,4,6,8,16,32,64,128
#
# Output:
#   resutls/<data_name>/<model>_<cut_off>.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_ROOT="${DB_ROOT:-${ROOT_DIR}/datasets/dpnet_db}"
MODELS="${MODELS:-svm,rf,lightgbm,xgb}"
CUTOFFS="${CUTOFFS:-2,4,6,8,16,32,64,128}"
PYTHON_BIN="${PYTHON_BIN:-python}"

IFS=',' read -r -a MODEL_ARR <<< "${MODELS}"
IFS=',' read -r -a CUTOFF_ARR <<< "${CUTOFFS}"

shopt -s nullglob
TASK_DIRS=( "${DB_ROOT}"/*/processed/* )
if [[ "${#TASK_DIRS[@]}" -eq 0 ]]; then
  echo "ERROR: no task dirs found under: ${DB_ROOT}/*/processed/*" >&2
  exit 1
fi

echo "Task dirs: ${#TASK_DIRS[@]}"
echo "Models: ${MODELS}"
echo "Cutoffs: ${CUTOFFS}"

FAIL_N=0
for task_dir in "${TASK_DIRS[@]}"; do
  if [[ ! -f "${task_dir}/train.csv" || ! -f "${task_dir}/test.csv" ]]; then
    echo "WARN: skip invalid task dir (missing train/test): ${task_dir}" >&2
    continue
  fi
  for model in "${MODEL_ARR[@]}"; do
    for cutoff in "${CUTOFF_ARR[@]}"; do
      echo ">>> task_dir=${task_dir} model=${model} cutoff=${cutoff}"
      if ! "${PYTHON_BIN}" "${ROOT_DIR}/run_ml_single.py" \
        --task-dir "${task_dir}" \
        --model "${model}" \
        --cutoff "${cutoff}" \
        --out-root "${ROOT_DIR}/resutls"; then
        echo "WARN: failed -> task_dir=${task_dir} model=${model} cutoff=${cutoff}" >&2
        FAIL_N=$((FAIL_N + 1))
      fi
    done
  done
done

echo "Batch done. fail_count=${FAIL_N}"
