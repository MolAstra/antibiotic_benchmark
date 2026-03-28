#!/usr/bin/env python3
"""Train one ML model from DPNet processed splits.

Reads:
  <task_dir>/train.csv
  <task_dir>/valid.csv (optional)
  <task_dir>/test.csv

Output:
  resutls/<data_name>/<model>_<cut_off>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one model for one DPNet processed task/cutoff.")
    parser.add_argument(
        "--task-dir",
        type=Path,
        required=True,
        help="DPNet processed task dir, e.g. datasets/dpnet_db/Acinetobacter/processed/Acinetobacter",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="svm",
        choices=["svm", "rf", "lightgbm", "xgb"],
        help="Model name.",
    )
    parser.add_argument("--cutoff", type=float, required=True, help="MIC cutoff.")
    parser.add_argument(
        "--positive-rule",
        type=str,
        default="resistant",
        choices=["resistant", "sensitive"],
        help="resistant: value>=cutoff is positive; sensitive: value<cutoff is positive.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument(
        "--rebuild-ecfp",
        action="store_true",
        help="Force rebuild ECFP cache even if cached arrays exist.",
    )
    parser.add_argument("--out-root", type=Path, default=Path("resutls"))
    return parser.parse_args()


def featurize_smiles(smiles: pd.Series, radius: int, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    idxs: list[int] = []
    for i, smi in enumerate(smiles.astype(str).tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        xs.append(arr)
        idxs.append(i)
    if not xs:
        return np.empty((0, n_bits), dtype=np.int8), np.array([], dtype=int)
    return np.vstack(xs), np.array(idxs, dtype=int)


def build_model(model_name: str, seed: int):
    if model_name == "svm":
        return SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300, random_state=seed, class_weight="balanced_subsample", n_jobs=-1
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("lightgbm is not installed") from exc
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        )
    if model_name == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("xgboost is not installed") from exc
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def get_value_series(df: pd.DataFrame) -> pd.Series:
    if "mic" in df.columns:
        return pd.to_numeric(df["mic"], errors="coerce")
    if "value" in df.columns:
        return pd.to_numeric(df["value"], errors="coerce")
    raise ValueError("Neither 'mic' nor 'value' column found")


def format_cutoff_tag(cutoff: float) -> str:
    # 16.0 -> 16 ; keep decimals for non-integers
    if float(cutoff).is_integer():
        return str(int(cutoff))
    return f"{cutoff:g}"


def load_clean_split_df(split_fp: Path) -> pd.DataFrame:
    df = pd.read_csv(split_fp)
    if "smiles" not in df.columns:
        raise ValueError(f"Missing 'smiles' in {split_fp}")
    values = get_value_series(df)
    clean = (
        pd.DataFrame({"smiles": df["smiles"], "value": values})
        .dropna(subset=["smiles", "value"])
        .reset_index(drop=True)
    )
    if clean.empty:
        raise ValueError(f"No valid rows after cleaning: {split_fp}")
    return clean


def load_or_build_ecfp(
    split_fp: Path,
    cache_dir: Path,
    radius: int,
    n_bits: int,
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_fp = cache_dir / f"{split_fp.stem}_r{radius}_b{n_bits}.npz"

    if cache_fp.exists() and not rebuild:
        cache = np.load(cache_fp)
        return cache["x"], cache["values"]

    clean = load_clean_split_df(split_fp)
    x, valid_idx = featurize_smiles(clean["smiles"], radius=radius, n_bits=n_bits)
    if valid_idx.size == 0:
        raise ValueError(f"No valid SMILES after RDKit featurization: {split_fp}")
    clean = clean.iloc[valid_idx].reset_index(drop=True)
    values = clean["value"].to_numpy(dtype=float)
    np.savez_compressed(cache_fp, x=x, values=values)
    return x, values


def to_binary(values: np.ndarray, cutoff: float, positive_rule: str) -> np.ndarray:
    if positive_rule == "resistant":
        return (values >= cutoff).astype(int)
    return (values < cutoff).astype(int)


def main() -> int:
    args = parse_args()
    task_dir = args.task_dir
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task dir not found: {task_dir}")

    data_name = task_dir.name
    cutoff_tag = format_cutoff_tag(args.cutoff)
    out_dir = args.out_root / data_name
    out_csv = out_dir / f"{args.model}_{cutoff_tag}.csv"
    if out_csv.exists():
        print(f"WARN: result exists, skip -> {out_csv}")
        return 0

    train_fp = task_dir / "train.csv"
    valid_fp = task_dir / "valid.csv"
    test_fp = task_dir / "test.csv"

    if not train_fp.exists() or not test_fp.exists():
        raise FileNotFoundError(f"Missing train/test in {task_dir}")

    cache_dir = task_dir / ".ecfp_cache"
    x_train, train_values = load_or_build_ecfp(
        train_fp, cache_dir, args.radius, args.n_bits, args.rebuild_ecfp
    )
    x_test, test_values = load_or_build_ecfp(
        test_fp, cache_dir, args.radius, args.n_bits, args.rebuild_ecfp
    )
    y_train = to_binary(train_values, args.cutoff, args.positive_rule)
    y_test = to_binary(test_values, args.cutoff, args.positive_rule)
    n_train = int(len(y_train))
    n_test = int(len(y_test))
    n_valid = 0
    valid_pos_ratio = float("nan")
    if valid_fp.exists():
        _, valid_values = load_or_build_ecfp(
            valid_fp, cache_dir, args.radius, args.n_bits, args.rebuild_ecfp
        )
        y_valid = to_binary(valid_values, args.cutoff, args.positive_rule)
        n_valid = int(len(y_valid))
        valid_pos_ratio = float(y_valid.mean())

    if np.unique(y_train).size < 2:
        raise ValueError(f"Train split has one class after cutoff={args.cutoff}")
    if np.unique(y_test).size < 2:
        print("WARN: test split has one class; some metrics may be NaN")

    model = build_model(args.model, seed=args.seed)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x_test)
    else:
        y_score = y_pred

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        auc = float(roc_auc_score(y_test, y_score))
    except ValueError:
        auc = float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(
        [
            {
                "data_name": data_name,
                "model": args.model,
                "cutoff": args.cutoff,
                "positive_rule": args.positive_rule,
                "n_train": int(n_train),
                "n_valid": int(n_valid),
                "n_test": int(n_test),
                "train_pos_ratio": float(y_train.mean()),
                "valid_pos_ratio": valid_pos_ratio,
                "test_pos_ratio": float(y_test.mean()),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_test, y_pred)),
                "roc_auc": auc,
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "task_dir": str(task_dir),
                "output_csv": str(out_csv),
            }
        ]
    )
    result.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
