#!/usr/bin/env bash
set -euo pipefail

# Build DPNet tasks from genus_top15 without thresholding labels.
# No external task_meta template is required.
#
# After processing, summarize ratio(value >= threshold) for
# thresholds: 2,4,6,8,16,32,64,128 in train/val/test,
# and write to datasets/data_pos_ratio.csv.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${ROOT_DIR}/datasets/genus_top15"
DB_DIR="${ROOT_DIR}/datasets/dpnet_db"
RATIO_CSV="${ROOT_DIR}/datasets/data_pos_ratio.csv"
PYTHON_BIN="${PYTHON_BIN:-python}"
THRESHOLDS="${THRESHOLDS:-2,4,6,8,16,32,64,128}"
SEED="${SEED:-42}"
STRICT_TEST="${STRICT_TEST:-true}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: Input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi

if ! command -v dpnet >/dev/null 2>&1; then
  echo "ERROR: dpnet command not found. Install first:" >&2
  echo "  pip install \"git+https://github.com/zhaisilong/dpnet.git\"" >&2
  exit 1
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "DB_DIR=${DB_DIR}"
echo "THRESHOLDS=${THRESHOLDS}"
echo "SEED=${SEED}"
echo "STRICT_TEST=${STRICT_TEST}"

shopt -s nullglob
csv_files=("${INPUT_DIR}"/*.csv)
if [[ "${#csv_files[@]}" -eq 0 ]]; then
  echo "ERROR: no CSV files found in ${INPUT_DIR}" >&2
  exit 1
fi

for src_csv in "${csv_files[@]}"; do
  genus="$(basename "${src_csv}" .csv)"
  task_root="${DB_DIR}/${genus}"
  raw_dir="${task_root}/raw"
  raw_csv="${raw_dir}/${genus}.csv"
  task_meta="${task_root}/task_meta.json"
  processed_dir="${task_root}/processed/${genus}"

  mkdir -p "${raw_dir}"

  # Keep original value (no thresholding). Ensure cid/smiles/value exist.
  "${PYTHON_BIN}" - "${src_csv}" "${raw_csv}" <<'PY'
import sys
import pandas as pd

src_csv, dst_csv = sys.argv[1:]
df = pd.read_csv(src_csv).copy()
required = {"smiles", "value"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in {src_csv}: {sorted(missing)}")

df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["smiles", "value"]).reset_index(drop=True)
if df.empty:
    raise SystemExit(f"No valid rows after filtering for {src_csv}")

if "cid" not in df.columns:
    df["cid"] = [f"cid_{i:08d}" for i in range(len(df))]
else:
    cid = df["cid"].astype(str).str.strip()
    missing_mask = cid.eq("") | cid.eq("nan")
    if missing_mask.any():
        fill = [f"cid_{i:08d}" for i in range(missing_mask.sum())]
        df.loc[missing_mask, "cid"] = fill

df = df[["cid", "smiles", "value"]]
df.to_csv(dst_csv, index=False)
print(f"Prepared raw: {dst_csv} | n={len(df)}")
PY

  # Build task_meta from scratch (regression label, no template dependency).
  "${PYTHON_BIN}" - "${task_meta}" "${genus}" "${SEED}" "${STRICT_TEST}" <<'PY'
import json
import sys
from pathlib import Path

out_meta, genus, seed, strict_test = sys.argv[1:]
strict_bool = strict_test.lower() in {"1", "true", "yes", "y"}
meta = {
    "name": genus,
    "seed": int(seed),
    "id_col": "cid",
    "smiles_col": "smiles",
    "labels": [
        {
            "id": "mic",
            "label_col": "value",
            "problem_type": "regression",
            "num_classes": None,
        }
    ],
    "strict_test": strict_bool,
    "processed_dir": "processed",
    "extra_cols": None,
    "dialect": "dpnet",
    "version": 1,
}
Path(out_meta).write_text(json.dumps(meta, indent=4) + "\n", encoding="utf-8")
print(f"Wrote meta: {out_meta}")
PY

  if [[ -d "${processed_dir}" ]]; then
    echo "Skip ${genus}: processed exists -> ${processed_dir}"
    continue
  fi

  echo "Run dpnet for ${genus}"
  dpnet process "${genus}" task_meta --root_dir "${task_root}"
done

# Summarize threshold ratios from processed splits.
"${PYTHON_BIN}" - "${DB_DIR}" "${RATIO_CSV}" "${THRESHOLDS}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

db_dir = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
thresholds = [float(x) for x in sys.argv[3].split(",") if x.strip()]

rows = []
for task_root in sorted([p for p in db_dir.iterdir() if p.is_dir()]):
    genus = task_root.name
    proc_dir = task_root / "processed" / genus
    if not proc_dir.is_dir():
        continue

    for split_in, split_out in [("train", "train"), ("valid", "val"), ("test", "test")]:
        fp = proc_dir / f"{split_in}.csv"
        if not fp.exists():
            continue

        df = pd.read_csv(fp)
        if "mic" in df.columns:
            values = pd.to_numeric(df["mic"], errors="coerce")
        elif "value" in df.columns:
            values = pd.to_numeric(df["value"], errors="coerce")
        else:
            continue

        values = values.dropna()
        n = len(values)
        if n == 0:
            continue

        for thr in thresholds:
            ratio = float((values >= thr).mean())
            rows.append(
                {
                    "genus": genus,
                    "split": split_out,
                    "n_samples": int(n),
                    "threshold": thr,
                    "pos_ratio_ge": ratio,
                }
            )

out_csv.parent.mkdir(parents=True, exist_ok=True)
out_df = pd.DataFrame(rows)
if out_df.empty:
    out_df = pd.DataFrame(columns=["genus", "split", "n_samples", "threshold", "pos_ratio_ge"])
else:
    out_df = out_df.sort_values(["genus", "split", "threshold"]).reset_index(drop=True)
out_df.to_csv(out_csv, index=False)
print(f"Wrote ratio summary: {out_csv} | rows={len(out_df)}")
PY

echo "Done. Ratio summary: ${RATIO_CSV}"
