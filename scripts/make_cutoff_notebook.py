#!/usr/bin/env python3
"""Generate notebooks/cutoff.ipynb for data_pos_ratio.csv visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text,
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code,
    }


def build_notebook(csv_path: Path, notebook_path: Path) -> dict:
    title = (
        "# Cutoff Ratio Dashboard\n\n"
        f"- Notebook: `{notebook_path}`\n"
        f"- Data: `{csv_path}`\n\n"
        "This notebook visualizes ratio(`value >= threshold`) across "
        "`train / val / test` splits."
    )

    load_code = f"""from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path(r\"{csv_path}\")
df = pd.read_csv(csv_path)
df = df.sort_values([\"genus\", \"split\", \"threshold\"]).reset_index(drop=True)

print(\"csv:\", csv_path)
print(\"shape:\", df.shape)
print(\"columns:\", list(df.columns))
display(df.head(10))
"""

    overview_code = """summary = (
    df.groupby([\"split\", \"threshold\"], as_index=False)[\"pos_ratio_ge\"]
      .mean()
      .sort_values([\"split\", \"threshold\"])
)
display(summary.head(20))

plt.figure(figsize=(8, 5))
for split_name in [\"train\", \"val\", \"test\"]:
    sub = summary[summary[\"split\"] == split_name]
    if len(sub) == 0:
        continue
    plt.plot(sub[\"threshold\"], sub[\"pos_ratio_ge\"], marker=\"o\", label=split_name)
plt.xscale(\"log\", base=2)
plt.xlabel(\"Threshold\")
plt.ylabel(\"Mean pos_ratio_ge\")
plt.title(\"Mean ratio(value >= threshold) by split\")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
"""

    heatmap_code = """split_name = \"test\"  # change to train / val / test
sub = df[df[\"split\"] == split_name].copy()
pivot = sub.pivot(index=\"genus\", columns=\"threshold\", values=\"pos_ratio_ge\").sort_index()

plt.figure(figsize=(10, max(6, 0.35 * len(pivot))))
im = plt.imshow(pivot.values, aspect=\"auto\", vmin=0, vmax=1)
plt.colorbar(im, label=\"pos_ratio_ge\")
plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title(f\"{split_name} ratio(value >= threshold) heatmap\")
plt.tight_layout()
plt.show()
"""

    compare_code = """# Train-test shift per genus at each threshold
train_df = df[df[\"split\"] == \"train\"][ [\"genus\", \"threshold\", \"pos_ratio_ge\"] ].rename(columns={\"pos_ratio_ge\": \"train_ratio\"})
test_df = df[df[\"split\"] == \"test\"][ [\"genus\", \"threshold\", \"pos_ratio_ge\"] ].rename(columns={\"pos_ratio_ge\": \"test_ratio\"})

shift = train_df.merge(test_df, on=[\"genus\", \"threshold\"], how=\"inner\")
shift[\"abs_shift\"] = (shift[\"train_ratio\"] - shift[\"test_ratio\"]).abs()
display(shift.sort_values(\"abs_shift\", ascending=False).head(20))
"""

    nb = {
        "cells": [
            md_cell(title),
            code_cell(load_code),
            md_cell("## Split-Level Overview"),
            code_cell(overview_code),
            md_cell("## Genus Heatmap"),
            code_cell(heatmap_code),
            md_cell("## Train-Test Shift"),
            code_cell(compare_code),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate cutoff notebook from ratio CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("datasets/data_pos_ratio.csv"),
        help="Input ratio CSV path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("notebooks/cutoff.ipynb"),
        help="Output notebook path.",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    out_path = args.out.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook(csv_path=csv_path, notebook_path=out_path)
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote notebook: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
