#!/usr/bin/env python3
"""
Model training and validation pipeline for the high-enrollment prediction task.

Task: For each course offering per term, predict whether it will have high enrollment
(Y or N), defined as the top quartile among offerings with positive enrollment in the
same department and term.

Anti-Leakage Constraints:
- Uses ONLY historical training tables in TRAIN_DATA_DIR (table_splits/train).
- Current-term enrollment (SUBJECT_ENROLLMENT_FIFTH_WEEK) is NEVER used as a feature.
- Evaluates on a time-based validation holdout (latest train years 2017-2018).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from feature_engineering import (
    CATEGORICAL_FEATURES,
    add_features,
    feature_columns,
    load_offering_frame,
    prepare_matrix,
)

FEATURE_VERSION = 1
N_TREES = 500
NUM_LEAVES = 31


def recency_weights(year: pd.Series, year_max: int) -> np.ndarray:
    """Half-life of six years so recent offerings count more than early history."""
    return np.exp(-np.log(2.0) * (year_max - year.to_numpy()) / 6.0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=["N", "Y"], zero_division=0))


def labels_yn(y: pd.Series) -> np.ndarray:
    return np.where(y.to_numpy() == 1, "Y", "N")


def top_fraction_predict(score: np.ndarray, positive_rate: float) -> np.ndarray:
    """Label the top fraction of highest-scoring offerings as 'Y'."""
    score = np.asarray(score, dtype=float)
    pred = np.full(len(score), "N", dtype=object)
    if len(score) == 0 or positive_rate <= 0:
        return pred
    k = int(round(positive_rate * len(score)))
    k = min(max(k, 1), len(score))
    chosen = np.argpartition(-score, k - 1)[:k]
    pred[chosen] = "Y"
    return pred


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    """Find the probability threshold that maximizes Macro F1."""
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.15, 0.70, 111):
        pred = np.where(proba >= t, "Y", "N")
        score = macro_f1(y_true, pred)
        if score > best_score:
            best_t, best_score = float(t), score
    return best_t, best_score


def tune_positive_rate(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    """Find the top positive rate fraction that maximizes Macro F1."""
    best_rate, best_score = 0.08, -1.0
    for rate in np.linspace(0.04, 0.15, 111):
        pred = top_fraction_predict(proba, float(rate))
        score = macro_f1(y_true, pred)
        if score > best_score:
            best_rate, best_score = float(rate), score
    return best_rate, best_score


def report_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_bin = (y_true == "Y").astype(int)
    y_pred_bin = (y_pred == "Y").astype(int)
    macro = float(f1_score(y_true, y_pred, average="macro", labels=["N", "Y"], zero_division=0))
    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    scores = f1_score(y_true, y_pred, average=None, labels=["N", "Y"], zero_division=0)
    print(
        f"{name:30s}  macro_F1={macro:.4f}  "
        f"F1_N={scores[0]:.4f}  F1_Y={scores[1]:.4f}  "
        f"prec={prec:.4f}  rec={rec:.4f}  "
        f"pred_Y={(y_pred == 'Y').mean():.3f}  true_Y={(y_true == 'Y').mean():.3f}"
    )
    return {
        "macro_f1": macro,
        "f1_n": float(scores[0]),
        "f1_y": float(scores[1]),
        "precision": prec,
        "recall": rec,
        "positive_rate_pred": float((y_pred == "Y").mean()),
        "positive_rate_true": float((y_true == "Y").mean()),
    }


def fit_lgbm(x_train: pd.DataFrame, y_train: np.ndarray, sample_weights: np.ndarray) -> lgb.LGBMClassifier:
    """Fit a high-capacity LightGBM binary classifier."""
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=N_TREES,
        learning_rate=0.05,
        num_leaves=NUM_LEAVES,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train, sample_weight=sample_weights)
    return model


def build_training_dataset(
    train_dir: Path, gold_path: Path, artifact_dir: Path, use_cache: bool = True
) -> pd.DataFrame:
    """Load or build the training features dataframe."""
    cache_path = artifact_dir / "train_features.parquet"
    version_path = artifact_dir / "train_features.version"

    if use_cache and cache_path.exists() and version_path.exists():
        if version_path.read_text().strip() == str(FEATURE_VERSION):
            print(f"Loading cached training features from {cache_path} …")
            return pd.read_parquet(cache_path)

    print("Building training features from raw tables (leakage-free) …")
    raw = load_offering_frame(train_dir, gold_path, split="train")
    featured = add_features(raw)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(cache_path, index=False)
    version_path.write_text(str(FEATURE_VERSION))
    print(f"Cached features to {cache_path}")
    return featured


def train(
    train_dir: Path,
    gold_path: Path,
    artifact_dir: Path,
    use_cache: bool = True,
) -> Dict:
    """Run full training and validation pipeline."""
    df = build_training_dataset(train_dir, gold_path, artifact_dir, use_cache=use_cache)
    features = feature_columns(df)
    labeled = df[df["y"].notna()].copy()
    print(f"Dataset: {len(labeled):,} offerings | Features: {len(features)} | Overall positive rate: {labeled['y'].mean():.3f}")

    # Time-based validation split:
    # Historical training: years <= 2016
    # Validation tuning: year == 2017
    # Development holdout: year >= 2018 (and 2017-2018 combined)
    fit_mask = labeled["year"] <= 2016
    tune_mask = labeled["year"] == 2017
    dev_mask = labeled["year"] >= 2018
    hold_mask = labeled["year"] >= 2017

    print(f"Split: Fit={fit_mask.sum():,} rows (<=2016), Tune={tune_mask.sum():,} rows (2017), Dev={dev_mask.sum():,} rows (2018)")

    x_all, cat_maps = prepare_matrix(labeled, features, cat_maps=None, fit=True)
    y_all = labeled["y"].to_numpy()

    # Train validation model on <= 2016
    print(f"\nTraining LightGBM on <=2016 with recency weighting ({N_TREES} trees, {NUM_LEAVES} leaves) …")
    weights_fit = recency_weights(labeled.loc[fit_mask, "year"], 2016)
    val_model = fit_lgbm(x_all.loc[fit_mask], y_all[fit_mask.to_numpy()], weights_fit)

    p_tune = val_model.predict_proba(x_all.loc[tune_mask])[:, 1]
    p_dev = val_model.predict_proba(x_all.loc[dev_mask])[:, 1]
    p_hold = val_model.predict_proba(x_all.loc[hold_mask])[:, 1]

    y_tune = labels_yn(labeled.loc[tune_mask, "y"])
    y_dev = labels_yn(labeled.loc[dev_mask, "y"])
    y_hold = labels_yn(labeled.loc[hold_mask, "y"])

    # Threshold calibration
    best_thresh, tune_thresh_f1 = tune_threshold(y_tune, p_tune)
    best_rate, tune_rate_f1 = tune_positive_rate(y_tune, p_tune)
    print(f"\nOptimal threshold tuned on 2017: {best_thresh:.2f} (Macro F1 = {tune_thresh_f1:.4f})")
    print(f"Optimal top positive rate tuned on 2017: {best_rate:.3f} (Macro F1 = {tune_rate_f1:.4f})")

    print("\n--- Validation Results (Holdout evaluated strictly on models trained on <= 2016) ---")
    dev_thresh_metrics = report_metrics("2018 Dev (Threshold)", y_dev, np.where(p_dev >= best_thresh, "Y", "N"))
    dev_rate_metrics = report_metrics("2018 Dev (Top Fraction)", y_dev, top_fraction_predict(p_dev, best_rate))
    hold_thresh_metrics = report_metrics("2017-2018 Holdout (Threshold)", y_hold, np.where(p_hold >= best_thresh, "Y", "N"))
    hold_rate_metrics = report_metrics("2017-2018 Holdout (Top Fraction)", y_hold, top_fraction_predict(p_hold, best_rate))

    print("\nDetailed classification report on 2018 dev:")
    print(classification_report(y_dev, np.where(p_dev >= best_thresh, "Y", "N"), labels=["N", "Y"], zero_division=0))

    # Refit final model on all training data (1986–2018)
    year_max = int(labeled["year"].max())
    print(f"\nRefitting final model on all training years (1986–{year_max}) with recency weighting …")
    weights_all = recency_weights(labeled["year"], year_max)
    final_model = fit_lgbm(x_all, y_all, weights_all)

    # Inspect top features
    importance = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop 15 Most Important Features:")
    print(importance.head(15).to_string())

    bundle = {
        "classifier": final_model,
        "features": features,
        "categorical": CATEGORICAL_FEATURES,
        "cat_maps": {col: [str(v) for v in list(cat_maps[col])] for col in CATEGORICAL_FEATURES},
        "threshold": best_thresh,
        "positive_rate": best_rate,
        "feature_version": FEATURE_VERSION,
        "n_estimators": N_TREES,
        "num_leaves": NUM_LEAVES,
        "metrics": {
            "dev_2018_macro_f1": dev_thresh_metrics["macro_f1"],
            "dev_2018_precision": dev_thresh_metrics["precision"],
            "dev_2018_recall": dev_thresh_metrics["recall"],
            "holdout_2017_2018_macro_f1": hold_thresh_metrics["macro_f1"],
            "tuned_threshold": best_thresh,
            "tuned_positive_rate": best_rate,
        },
    }

    # Save model artifacts
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(bundle, model_path)
    print(f"\nModel saved to {model_path}")

    # Also save to table_splits/ for compatibility
    table_splits_dir = Path("table_splits")
    table_splits_dir.mkdir(parents=True, exist_ok=True)
    with open(table_splits_dir / "lgbm_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    report_path = artifact_dir / "validation_report.json"
    report_path.write_text(json.dumps(bundle["metrics"], indent=2))
    print(f"Validation report saved to {report_path}")
    print(f"Development Macro F1 achieved: {dev_thresh_metrics['macro_f1']:.4f}")

    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train high-enrollment prediction model.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("table_splits/train"),
        help="Path to training tables directory (default: table_splits/train).",
    )
    parser.add_argument(
        "--gold-train",
        type=Path,
        default=None,
        help="Path to gold_enrollment_train.csv (default: TRAIN_DIR/gold_enrollment_train.csv).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
        help="Path to save artifacts and model (default: artifacts).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable feature caching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_dir.exists():
        raise SystemExit(f"Train directory not found: {args.train_dir}")
    gold_path = args.gold_train or (args.train_dir / "gold_enrollment_train.csv")
    if not gold_path.exists():
        raise SystemExit(f"Gold train file not found: {gold_path}")

    train(
        train_dir=args.train_dir,
        gold_path=gold_path,
        artifact_dir=args.artifact_dir,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
