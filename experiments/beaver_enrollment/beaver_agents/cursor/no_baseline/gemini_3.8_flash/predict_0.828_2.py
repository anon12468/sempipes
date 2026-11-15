#!/usr/bin/env python3
"""
Prediction and evaluation script for the high-enrollment prediction task.

Usage:
    # 1. Run predictions & evaluation on gold_enrollment_test.csv (default):
    python predict.py

    # 2. Score arbitrary keys file:
    python predict.py --keys path/to/keys.csv

    # 3. Score a held-out test folder once available:
    python predict.py --test-dir /path/to/test --gold-test /path/to/test/gold_enrollment_test.csv

Prediction keys: TERM_CODE, SUBJECT_ID_SORT
Target column:   HIGH_ENROLLMENT ('Y' or 'N')
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from feature_engineering import (
    _read_csv,
    add_features,
    clean_cat,
    feature_columns,
    load_offering_frame,
    prepare_matrix,
)

DEFAULT_TRAIN_DIR = Path("table_splits/train")
DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")
FALLBACK_MODEL_PATH = Path("table_splits/lgbm_model.pkl")
DEFAULT_OUTPUT_PATH = Path("table_splits/predictions.csv")


def load_saved_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Load model bundle from joblib or pickle."""
    if model_path.exists():
        return joblib.load(model_path)
    if FALLBACK_MODEL_PATH.exists():
        import pickle
        with open(FALLBACK_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"Model file not found at {model_path} or {FALLBACK_MODEL_PATH}. Run train_model.py first.")


def build_test_features(
    train_dir: Path,
    test_dir: Optional[Path],
    gold_keys_df: pd.DataFrame,
    bundle: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build features for test offerings without data leakage.

    If test_dir is provided (future terms), train offerings and test offerings
    are combined to maintain historical lookback, with test labels masked.
    If test_dir is None, historical offerings from train_dir are used.
    """
    keys = gold_keys_df[["TERM_CODE", "SUBJECT_ID_SORT"]].drop_duplicates().copy()
    keys["TERM_CODE"] = keys["TERM_CODE"].astype(str).str.strip()
    keys["SUBJECT_ID_SORT"] = keys["SUBJECT_ID_SORT"].astype(str).str.strip()

    if test_dir is not None and test_dir.exists():
        print(f"Loading historical context from {train_dir} and test offerings from {test_dir} …")
        train_raw = load_offering_frame(train_dir, train_dir / "gold_enrollment_train.csv", split="train")
        # Mask any test offerings in train_raw to prevent duplicate keys
        test_raw = load_offering_frame(test_dir, None, split="test")
        test_keys_set = set(zip(keys["TERM_CODE"], keys["SUBJECT_ID_SORT"]))
        train_raw = train_raw[
            ~train_raw.set_index(["TERM_CODE", "SUBJECT_ID_SORT"]).index.isin(test_keys_set)
        ].copy()

        # Restrict test offerings to the requested keys
        test_raw = keys.merge(test_raw.drop(columns=["y"], errors="ignore"), on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left")
        test_raw["y"] = np.nan
        test_raw["split"] = "test"

        combined = pd.concat([train_raw, test_raw], ignore_index=True)
        featured = add_features(combined)
        test_featured = featured[featured["split"] == "test"].copy()
    else:
        # Test keys are within the training directory terms (e.g. table_splits/train/gold_enrollment_test.csv)
        cache_path = Path("artifacts/train_features.parquet")
        if cache_path.exists():
            print(f"Using feature cache {cache_path} …")
            featured = pd.read_parquet(cache_path)
        else:
            print(f"Building features from {train_dir} …")
            raw = load_offering_frame(train_dir, train_dir / "gold_enrollment_train.csv", split="train")
            featured = add_features(raw)

        test_featured = keys.merge(
            featured,
            on=["TERM_CODE", "SUBJECT_ID_SORT"],
            how="left",
        )

    return test_featured, keys


def predict(
    keys_df: pd.DataFrame,
    train_dir: Path = DEFAULT_TRAIN_DIR,
    test_dir: Optional[Path] = None,
    bundle: Optional[dict] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """
    Generate predictions for each row in keys_df.
    Maintains exact row-for-row alignment and order with keys_df.
    """
    if bundle is None:
        bundle = load_saved_bundle(model_path)

    test_featured, uniq_keys = build_test_features(train_dir, test_dir, keys_df, bundle)

    features = bundle["features"]
    cat_maps = bundle["cat_maps"]
    threshold = bundle.get("threshold", 0.40)
    positive_rate = bundle.get("positive_rate", 0.08)

    # Prepare matrix
    x, _ = prepare_matrix(test_featured, features, cat_maps=cat_maps, fit=False)
    scores = bundle["classifier"].predict_proba(x)[:, 1]

    # Predict using calibrated threshold
    pred_labels = np.where(scores >= threshold, "Y", "N")

    # Unique offering lookup
    lookup = pd.DataFrame(
        {
            "TERM_CODE": test_featured["TERM_CODE"].to_numpy(),
            "SUBJECT_ID_SORT": test_featured["SUBJECT_ID_SORT"].to_numpy(),
            "pred_proba": scores,
            "HIGH_ENROLLMENT_PRED": pred_labels,
        }
    ).drop_duplicates(["TERM_CODE", "SUBJECT_ID_SORT"])

    # Align with input keys preserving original order and row count
    input_keys = keys_df.copy()
    input_keys["TERM_CODE"] = input_keys["TERM_CODE"].astype(str).str.strip()
    input_keys["SUBJECT_ID_SORT"] = input_keys["SUBJECT_ID_SORT"].astype(str).str.strip()
    input_keys["_orig_idx"] = np.arange(len(input_keys))

    merged = input_keys.merge(lookup, on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left")
    merged = merged.sort_values("_orig_idx").reset_index(drop=True)

    # Fill any unpredicted keys with 'N'
    if merged["HIGH_ENROLLMENT_PRED"].isna().any():
        n_missing = int(merged["HIGH_ENROLLMENT_PRED"].isna().sum())
        print(f"Warning: {n_missing} keys were not found in offering summary; predicted N.")
        merged["HIGH_ENROLLMENT_PRED"] = merged["HIGH_ENROLLMENT_PRED"].fillna("N")
        merged["pred_proba"] = merged["pred_proba"].fillna(0.0)

    out = pd.DataFrame(
        {
            "TERM_CODE": merged["TERM_CODE"],
            "SUBJECT_ID_SORT": merged["SUBJECT_ID_SORT"],
            "HIGH_ENROLLMENT": merged["HIGH_ENROLLMENT_PRED"],
            "pred_proba": merged["pred_proba"],
        }
    )

    if "HIGH_ENROLLMENT" in keys_df.columns:
        out["HIGH_ENROLLMENT_TRUE"] = keys_df["HIGH_ENROLLMENT"].values

    return out


def evaluate_predictions(pred_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Evaluate predictions against ground truth if present."""
    truth_col = None
    if "HIGH_ENROLLMENT_TRUE" in pred_df.columns:
        truth_col = "HIGH_ENROLLMENT_TRUE"
    elif "HIGH_ENROLLMENT" in pred_df.columns and "pred_proba" in pred_df.columns:
        # Check if caller passed a df with gold column
        pass

    if truth_col is None:
        print("No ground truth labels found in input keys — skipping evaluation.")
        return None

    raw_true = pred_df[truth_col].astype(str).str.strip().str.upper()
    valid_mask = raw_true.isin(["Y", "1", "TRUE", "N", "0", "FALSE"])
    if not valid_mask.any():
        print("No valid Y/N or 1/0 ground truth labels found.")
        return None

    y_true = np.where(raw_true[valid_mask].isin(["Y", "1", "TRUE"]), "Y", "N")
    y_pred = pred_df.loc[valid_mask, "HIGH_ENROLLMENT"].astype(str).str.strip().str.upper().to_numpy()

    macro = float(f1_score(y_true, y_pred, average="macro", labels=["N", "Y"], zero_division=0))
    scores = f1_score(y_true, y_pred, average=None, labels=["N", "Y"], zero_division=0)
    y_true_bin = (y_true == "Y").astype(int)
    y_pred_bin = (y_pred == "Y").astype(int)
    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))

    metrics = {
        "macro_f1": macro,
        "f1_n": float(scores[0]),
        "f1_y": float(scores[1]),
        "precision": prec,
        "recall": rec,
        "rows_total": int(len(pred_df)),
        "rows_scored": int(valid_mask.sum()),
        "positive_rate_true": float((y_true == "Y").mean()),
        "positive_rate_pred": float((y_pred == "Y").mean()),
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Rows Scored:    {metrics['rows_scored']:,}")
    print(f"Macro F1 Score:       {macro:.4f}")
    print(f"Class N (Low) F1:     {metrics['f1_n']:.4f}")
    print(f"Class Y (High) F1:    {metrics['f1_y']:.4f}")
    print(f"Precision (Y):        {prec:.4f}")
    print(f"Recall (Y):           {rec:.4f}")
    print(f"Ground Truth Pos Rate: {metrics['positive_rate_true']:.3%}")
    print(f"Predicted Pos Rate:    {metrics['positive_rate_pred']:.3%}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=["N", "Y"], zero_division=0))
    print("=" * 50 + "\n")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict and evaluate HIGH_ENROLLMENT.")
    parser.add_argument(
        "--keys",
        type=Path,
        default=None,
        help="Path to CSV containing TERM_CODE, SUBJECT_ID_SORT (and optional HIGH_ENROLLMENT).",
    )
    parser.add_argument(
        "--gold-test",
        type=Path,
        default=Path("/Users/USER/BEAVER/test/gold_enrollment_test.csv"),
        help="Alias for --keys: path to gold_enrollment_test.csv.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Path to training tables directory (default: table_splits/train).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("/Users/USER/BEAVER/test"),
        help="Path to held-out test tables directory (default: None).",
    )
    parser.add_argument(
        "--output",
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save output predictions CSV (default: table_splits/predictions.csv).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to saved model artifact (default: artifacts/model.joblib).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine test dir and keys path with smart auto-detection
    test_dir = args.test_dir
    if test_dir is None:
        if Path("table_splits/test").exists():
            test_dir = Path("table_splits/test")
        elif Path("../BEAVER/test").exists():
            test_dir = Path("../BEAVER/test")

    keys_path = args.keys or args.gold_test
    if keys_path is None:
        if test_dir is not None and (test_dir / "gold_enrollment_test.csv").exists():
            keys_path = test_dir / "gold_enrollment_test.csv"
        elif Path("../BEAVER/eval/gold_enrollment_test.csv").exists():
            keys_path = Path("../BEAVER/eval/gold_enrollment_test.csv")
        elif (args.train_dir / "gold_enrollment_test.csv").exists():
            keys_path = args.train_dir / "gold_enrollment_test.csv"
        elif (Path("eval") / "gold_enrollment_test.csv").exists():
            keys_path = Path("eval") / "gold_enrollment_test.csv"
        else:
            raise FileNotFoundError(
                "No keys or test gold file specified, and default gold_enrollment_test.csv not found. "
                "Specify with --keys or --gold-test."
            )

    print(f"Loading keys from {keys_path} …")
    keys_df = _read_csv(keys_path)
    print(f"Loaded {len(keys_df):,} rows.")

    # Run predictions
    print(f"Generating predictions using model {args.model} …")
    pred_df = predict(
        keys_df=keys_df,
        train_dir=args.train_dir,
        test_dir=test_dir,
        model_path=args.model,
    )

    # Save submission file with required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT
    submission_df = pred_df[["TERM_CODE", "SUBJECT_ID_SORT", "HIGH_ENROLLMENT"]].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")

    # Also write a copy to root predictions.csv for convenience
    root_pred = Path("predictions.csv")
    submission_df.to_csv(root_pred, index=False)
    print(f"Predictions also mirrored to {root_pred}")

    # Evaluate if ground truth is present
    evaluate_predictions(pred_df)


if __name__ == "__main__":
    main()
