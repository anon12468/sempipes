#!/usr/bin/env python3
"""Forecast parking-violation counts from the previous year's aggregates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


SCRIPT_DIR = Path(__file__).resolve().parent
KEY_COLUMNS = ["Street Name", "Violation Description"]
TARGET_COLUMN = "violation_count"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on 2022 aggregates and optionally predict a later file."
    )
    parser.add_argument(
        "--train-path",
        default="violations_per_street_2022.csv",
        help="Labeled current-year CSV (default: violations_per_street_2022.csv).",
    )
    parser.add_argument(
        "--history-path",
        default="corpus/violations_per_street_2021.csv",
        help="Previous-year CSV used to construct lag features.",
    )
    parser.add_argument(
        "--features-path",
        default="corpus/corpus_features.parquet",
        help="Optional street-level augmentation parquet.",
    )
    parser.add_argument(
        "--test-path",
        default=None,
        help="Optional keys-only or labeled evaluation CSV.",
    )
    parser.add_argument(
        "--output-path",
        default="submission.csv",
        help="Prediction output path (default: submission.csv).",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_input(path_text: str) -> Path:
    """Resolve a CLI path, with the script directory as a convenient fallback."""
    path = Path(path_text).expanduser()
    if path.exists() or path.is_absolute():
        return path
    local_path = SCRIPT_DIR / path
    return local_path if local_path.exists() else path


def load_core(path: Path, require_target: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if require_target and TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{path} is missing required column: {TARGET_COLUMN}")
    if require_target:
        frame[TARGET_COLUMN] = pd.to_numeric(
            frame[TARGET_COLUMN], errors="coerce"
        )
        if frame[TARGET_COLUMN].isna().any():
            count = int(frame[TARGET_COLUMN].isna().sum())
            raise ValueError(f"{path} has {count} non-numeric target values")
        if (frame[TARGET_COLUMN] < 0).any():
            raise ValueError(f"{path} contains negative target values")
    return frame


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize case and whitespace without merging distinct street spellings."""
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def make_history_tables(history: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    base = pd.DataFrame(
        {
            "norm_street": normalize_text(history["Street Name"]),
            "violation_type": normalize_text(history["Violation Description"]),
            "history_count": history[TARGET_COLUMN].astype(float),
        }
    )
    pair = (
        base.groupby(["norm_street", "violation_type"], as_index=False)[
            "history_count"
        ]
        .sum()
        .rename(columns={"history_count": "lag_pair"})
    )
    street = (
        base.groupby("norm_street")["history_count"]
        .agg(["sum", "mean", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": "lag_street_total",
                "mean": "lag_street_mean",
                "max": "lag_street_max",
                "count": "lag_street_types",
            }
        )
    )
    violation = (
        base.groupby("violation_type")["history_count"]
        .agg(["sum", "mean", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": "lag_violation_total",
                "mean": "lag_violation_mean",
                "max": "lag_violation_max",
                "count": "lag_violation_streets",
            }
        )
    )
    return pair, street, violation


def load_augmentations(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        if path is not None:
            print(
                f"Warning: augmentation file not found at {path}; continuing without it.",
                file=sys.stderr,
            )
        return None
    frame = pd.read_parquet(path)
    if "norm_street" not in frame.columns:
        print(
            f"Warning: {path} has no norm_street column; continuing without it.",
            file=sys.stderr,
        )
        return None
    frame = frame.copy()
    frame["norm_street"] = normalize_text(frame["norm_street"])
    # The supplied feature table is street-level. This also prevents an accidental
    # many-to-many merge if a replacement table contains repeated street names.
    return frame.drop_duplicates("norm_street", keep="first")


def build_features(
    keys: pd.DataFrame,
    history: pd.DataFrame,
    augmentations: pd.DataFrame | None,
) -> pd.DataFrame:
    raw_street = keys["Street Name"].fillna("").astype(str).str.strip()
    features = pd.DataFrame(
        {
            "norm_street": normalize_text(keys["Street Name"]),
            "violation_type": normalize_text(keys["Violation Description"]),
            "street_is_upper": (raw_street == raw_street.str.upper()).astype(
                "int8"
            ),
            "street_is_title": (raw_street == raw_street.str.title()).astype(
                "int8"
            ),
            "street_length": raw_street.str.len().astype(float),
            "street_words": (raw_street.str.count(r"\s+") + 1).astype(float),
            "street_digits": raw_street.str.count(r"\d").astype(float),
            "street_letters": raw_street.str.count(r"[A-Za-z]").astype(float),
            "street_suffix": normalize_text(keys["Street Name"])
            .str.extract(r"([A-Z]+)$", expand=False)
            .fillna("NONE"),
        }
    )

    pair, street, violation = make_history_tables(history)
    features = (
        features.merge(pair, on=["norm_street", "violation_type"], how="left")
        .merge(street, on="norm_street", how="left")
        .merge(violation, on="violation_type", how="left")
    )
    lag_columns = [column for column in features if column.startswith("lag_")]
    features[lag_columns] = features[lag_columns].fillna(0.0)
    features["lag_pair_share_street"] = features["lag_pair"] / (
        features["lag_street_total"] + 1.0
    )
    features["pair_seen"] = (features["lag_pair"] > 0).astype("int8")
    features["street_seen"] = (features["lag_street_total"] > 0).astype("int8")

    if augmentations is not None:
        features = features.merge(augmentations, on="norm_street", how="left")
    return features


def harmonize_categories(
    train_features: pd.DataFrame, test_features: pd.DataFrame | None
) -> list[str]:
    categorical = [
        column
        for column in ["norm_street", "violation_type", "street_suffix", "pave_borough"]
        if column in train_features.columns
    ]
    for column in categorical:
        values = [train_features[column].fillna("MISSING").astype(str)]
        if test_features is not None:
            values.append(test_features[column].fillna("MISSING").astype(str))
        categories = pd.Index(pd.concat(values, ignore_index=True).unique())
        train_features[column] = pd.Categorical(
            values[0], categories=categories
        )
        if test_features is not None:
            test_features[column] = pd.Categorical(
                values[1], categories=categories
            )
    return categorical


def model_parameters(seed: int, estimators: int = 2_000) -> dict[str, object]:
    return {
        "objective": "regression_l2",
        "metric": "rmse",
        "n_estimators": estimators,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 30,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": seed,
    }


def fit_trend_slopes(
    features: pd.DataFrame, target: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    slopes: dict[str, float] = {}
    violation = features["violation_type"].astype(str).to_numpy()
    lag = features["lag_pair"].to_numpy(dtype=float)
    for category in np.unique(violation[indices]):
        selected = indices[violation[indices] == category]
        x = lag[selected]
        denominator = float(np.dot(x, x))
        slopes[category] = (
            float(np.dot(x, target[selected]) / denominator)
            if denominator > 0
            else 0.0
        )
    return slopes


def trend_predictions(
    features: pd.DataFrame, indices: np.ndarray, slopes: dict[str, float]
) -> np.ndarray:
    violation = features["violation_type"].astype(str).to_numpy()[indices]
    lag = features["lag_pair"].to_numpy(dtype=float)[indices]
    scale = np.fromiter(
        (slopes.get(category, 0.0) for category in violation),
        dtype=float,
        count=len(indices),
    )
    return lag * scale


def select_blend_weight(
    target: np.ndarray,
    model_prediction: np.ndarray,
    trend_prediction: np.ndarray,
    seen: np.ndarray,
) -> float:
    """Choose the least-squares model weight for seen pairs only."""
    if not seen.any():
        return 1.0
    delta = model_prediction[seen] - trend_prediction[seen]
    denominator = float(np.dot(delta, delta))
    if denominator == 0:
        return 1.0
    weight = np.dot(target[seen] - trend_prediction[seen], delta) / denominator
    return float(np.clip(weight, 0.0, 1.0))


def blend_predictions(
    model_prediction: np.ndarray,
    trend_prediction: np.ndarray,
    seen: np.ndarray,
    model_weight: float,
) -> np.ndarray:
    prediction = np.maximum(model_prediction, 0.0)
    prediction[seen] = (
        model_weight * prediction[seen]
        + (1.0 - model_weight) * trend_prediction[seen]
    )
    return np.maximum(prediction, 0.0)


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.validation_fraction <= 0.5:
        raise ValueError("--validation-fraction must be between 0.05 and 0.5")

    train_path = resolve_input(args.train_path)
    history_path = resolve_input(args.history_path)
    features_path = resolve_input(args.features_path) if args.features_path else None
    if not history_path.exists():
        raise FileNotFoundError(
            f"History file not found: {history_path}. "
            "A prior year is needed for leakage-free temporal features."
        )

    train = load_core(train_path, require_target=True)
    history = load_core(history_path, require_target=True)
    augmentations = load_augmentations(features_path)
    train_features = build_features(train[KEY_COLUMNS], history, augmentations)
    target = train[TARGET_COLUMN].to_numpy(dtype=float)

    test = None
    test_features = None
    if args.test_path:
        test = load_core(resolve_input(args.test_path), require_target=False)
        # Deliberately pass only keys: a supplied test target cannot enter features.
        test_features = build_features(test[KEY_COLUMNS], train, augmentations)

    categorical = harmonize_categories(train_features, test_features)
    groups = train_features["norm_street"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.validation_fraction,
        random_state=args.seed,
    )
    fit_indices, validation_indices = next(
        splitter.split(train_features, target, groups=groups)
    )

    validation_model = lgb.LGBMRegressor(**model_parameters(args.seed))
    validation_model.fit(
        train_features.iloc[fit_indices],
        target[fit_indices],
        categorical_feature=categorical,
        eval_set=[
            (train_features.iloc[validation_indices], target[validation_indices])
        ],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    model_validation_prediction = validation_model.predict(
        train_features.iloc[validation_indices]
    )
    fit_slopes = fit_trend_slopes(train_features, target, fit_indices)
    trend_validation_prediction = trend_predictions(
        train_features, validation_indices, fit_slopes
    )
    validation_seen = (
        train_features.iloc[validation_indices]["pair_seen"].to_numpy() == 1
    )
    blend_weight = select_blend_weight(
        target[validation_indices],
        model_validation_prediction,
        trend_validation_prediction,
        validation_seen,
    )
    validation_prediction = blend_predictions(
        model_validation_prediction,
        trend_validation_prediction,
        validation_seen,
        blend_weight,
    )
    validation_rmse = mean_squared_error(
        target[validation_indices], validation_prediction
    ) ** 0.5
    validation_unseen = int((~validation_seen).sum())
    print(
        f"Development grouped-holdout RMSE: {validation_rmse:.6f} "
        f"({len(validation_indices):,} rows)"
    )
    print(
        "Validation pairs unseen in 2021 after normalization: "
        f"{validation_unseen:,}/{len(validation_indices):,}"
    )
    print(
        f"Selected boosting rounds: {validation_model.best_iteration_}; "
        f"seen-pair model blend weight: {blend_weight:.3f}"
    )

    if test is None or test_features is None:
        print("No --test-path supplied; validation completed without a submission.")
        return

    rounds = max(int(validation_model.best_iteration_ or 1), 50)
    final_model = lgb.LGBMRegressor(
        **model_parameters(args.seed, estimators=rounds)
    )
    final_model.fit(
        train_features,
        target,
        categorical_feature=categorical,
    )
    model_test_prediction = final_model.predict(test_features)
    all_indices = np.arange(len(train_features))
    test_indices = np.arange(len(test_features))
    final_slopes = fit_trend_slopes(train_features, target, all_indices)
    trend_test_prediction = trend_predictions(
        test_features, test_indices, final_slopes
    )
    test_seen = test_features["pair_seen"].to_numpy() == 1
    prediction = blend_predictions(
        model_test_prediction,
        trend_test_prediction,
        test_seen,
        blend_weight,
    )

    submission = pd.DataFrame(
        {
            "street_name": test["Street Name"],
            "violation_type": test["Violation Description"],
            "predicted_count": prediction,
        }
    )
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    unseen_count = int((~test_seen).sum())
    print(f"Wrote {len(submission):,} predictions to {output_path}")
    print(
        "Test pairs unseen in training after normalization: "
        f"{unseen_count:,}/{len(test):,}"
    )
    print(
        "Unseen pairs use violation, street, lexical, and corpus context; "
        "seen pairs additionally use the prior count and learned trend."
    )

    if TARGET_COLUMN in test.columns:
        truth = pd.to_numeric(test[TARGET_COLUMN], errors="coerce")
        if truth.isna().any():
            raise ValueError("The test target contains non-numeric values")
        test_rmse = mean_squared_error(truth.to_numpy(), prediction) ** 0.5
        print(f"Supplied test/eval RMSE: {test_rmse:.6f}")


if __name__ == "__main__":
    main()
