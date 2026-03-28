#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_ml_single.sh <data_name> <cutoff> [model]
#
# Example:
#   bash run_ml_single.sh Acinetobacter 32 svm
#   bash run_ml_single.sh Acinetobacter 32 rf

if [[ "${#}" -lt 2 ]]; then
  echo "Usage: bash run_ml_single.sh <data_name> <cutoff> [model]" >&2
  exit 1
fi

DATA_NAME="${1}"
CUTOFF="${2}"
MODEL="${3:-svm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_ROOT="${DB_ROOT:-${ROOT_DIR}/datasets/dpnet_db}"
TASK_DIR="${DB_ROOT}/${DATA_NAME}/processed/${DATA_NAME}"

"${PYTHON_BIN}" run_ml_single.py \
  --task-dir "${TASK_DIR}" \
  --cutoff "${CUTOFF}" \
  --model "${MODEL}" \
  --out-root "resutls"
