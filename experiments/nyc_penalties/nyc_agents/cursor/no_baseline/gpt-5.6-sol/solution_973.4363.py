#!/usr/bin/env python3
"""Train and run the NYC parking-violation count forecast."""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


KEY_COLUMNS = ["Street Name", "Violation Description"]
TARGET_COLUMN = "violation_count"
CATEGORICAL_COLUMNS = [
    "street_raw",
    "violation_raw",
    "street_norm",
    "violation_norm",
    "violation_code",
    "street_suffix",
    "pave_borough",
]
STREET_REPLACEMENTS = {
    "AV": "AVENUE",
    "AVE": "AVENUE",
    "BLVD": "BOULEVARD",
    "CT": "COURT",
    "DR": "DRIVE",
    "EXPY": "EXPRESSWAY",
    "HWY": "HIGHWAY",
    "LN": "LANE",
    "PKWY": "PARKWAY",
    "PL": "PLACE",
    "RD": "ROAD",
    "ST": "STREET",
    "STR": "STREET",
    "TER": "TERRACE",
    "TPKE": "TURNPIKE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast parking violations by street and violation type."
    )
    parser.add_argument(
        "--train-path",
        default="violations_per_street_2022.csv",
        help="Training CSV (default: violations_per_street_2022.csv).",
    )
    parser.add_argument(
        "--test-path",
        default=None,
        help="Optional keys-only or labeled evaluation CSV.",
    )
    parser.add_argument(
        "--history-path",
        default="corpus/violations_per_street_2021.csv",
        help="Previous-year counts used as lag features.",
    )
    parser.add_argument(
        "--features-path",
        default="corpus/corpus_features.parquet",
        help="Optional street-level augmentation parquet.",
    )
    parser.add_argument(
        "--output-path",
        default="submission.csv",
        help="Prediction output path (default: submission.csv).",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help="Fraction of 2022 rows used for development validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_input(path_text: str) -> Path:
    """Resolve a CLI path from CWD, then relative to this script."""
    path = Path(path_text).expanduser()
    if path.exists() or path.is_absolute():
        return path
    local_path = Path(__file__).resolve().parent / path
    return local_path if local_path.exists() else path


def read_core_csv(path: Path, require_target: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    frame = pd.read_csv(path)
    missing = set(KEY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if require_target and TARGET_COLUMN not in frame:
        raise ValueError(f"{path} is missing required column: {TARGET_COLUMN}")
    if frame[KEY_COLUMNS].isna().any().any():
        warnings.warn(f"{path} contains missing key values; they will become empty strings.")
    if require_target:
        frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="raise")
        if frame[TARGET_COLUMN].isna().any():
            raise ValueError(f"{path} contains missing training targets.")
    return frame


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def canonicalize_street(value: str) -> str:
    value = re.sub(r"[^A-Z0-9@ ]+", " ", value)
    return " ".join(STREET_REPLACEMENTS.get(token, token) for token in value.split())


def prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    prepared = history.copy()
    prepared["street_norm"] = normalize_text(prepared["Street Name"])
    prepared["violation_norm"] = normalize_text(prepared["Violation Description"])
    # Historical source rows can repeat after harmless whitespace/case normalization.
    return (
        prepared.groupby(["street_norm", "violation_norm"], as_index=False, sort=False)[
            TARGET_COLUMN
        ]
        .sum()
        .rename(columns={TARGET_COLUMN: "previous_count"})
    )


def load_augmentation(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"Augmentation file not found; continuing without it: {path}")
        return None
    augmentation = pd.read_parquet(path)
    if "norm_street" not in augmentation:
        warnings.warn(
            f"Augmentation file has no norm_street column; ignoring it: {path}"
        )
        return None
    return augmentation.drop_duplicates("norm_street")


def make_features(
    rows: pd.DataFrame,
    history: pd.DataFrame,
    augmentation: pd.DataFrame | None,
) -> pd.DataFrame:
    features = pd.DataFrame(index=rows.index)
    features["street_raw"] = rows["Street Name"].fillna("").astype(str).str.strip()
    features["violation_raw"] = (
        rows["Violation Description"].fillna("").astype(str).str.strip()
    )
    features["street_norm"] = normalize_text(rows["Street Name"])
    features["violation_norm"] = normalize_text(rows["Violation Description"])

    features = features.merge(
        history,
        on=["street_norm", "violation_norm"],
        how="left",
        sort=False,
    )
    features["previous_present"] = features["previous_count"].notna().astype("int8")
    features["previous_count"] = features["previous_count"].fillna(0.0)

    street_history = history.groupby("street_norm")["previous_count"].agg(
        ["sum", "mean", "max", "count"]
    )
    street_history.columns = [f"previous_street_{name}" for name in street_history]
    violation_history = history.groupby("violation_norm")["previous_count"].agg(
        ["sum", "mean", "max", "count"]
    )
    violation_history.columns = [
        f"previous_violation_{name}" for name in violation_history
    ]
    features = features.join(street_history, on="street_norm")
    features = features.join(violation_history, on="violation_norm")

    features["canonical_street"] = features["street_norm"].map(canonicalize_street)
    if augmentation is not None:
        features = features.merge(
            augmentation,
            left_on="canonical_street",
            right_on="norm_street",
            how="left",
            sort=False,
        )

    features["street_is_upper"] = (
        features["street_raw"] == features["street_raw"].str.upper()
    ).astype("int8")
    features["violation_is_upper"] = (
        features["violation_raw"] == features["violation_raw"].str.upper()
    ).astype("int8")
    features["street_length"] = features["street_norm"].str.len()
    features["street_token_count"] = features["street_norm"].str.count(" ") + 1
    features["street_has_at_sign"] = (
        features["street_norm"].str.contains("@", regex=False).astype("int8")
    )
    features["street_starts_with_number"] = (
        features["street_norm"].str.match(r"^\d").astype("int8")
    )
    features["violation_code"] = (
        features["violation_norm"]
        .str.extract(r"^(\d+[A-Z]?)", expand=False)
        .fillna("OTHER")
    )
    features["street_suffix"] = (
        features["canonical_street"].str.rsplit(n=1).str[-1]
    )
    for column in CATEGORICAL_COLUMNS:
        if column not in features:
            features[column] = "UNKNOWN"

    return features.drop(
        columns=["canonical_street", "norm_street"], errors="ignore"
    )


def align_categories(*frames: pd.DataFrame) -> None:
    """Give train/test identical category dictionaries for LightGBM."""
    for column in CATEGORICAL_COLUMNS:
        values = pd.concat(
            [frame[column].fillna("UNKNOWN").astype(str) for frame in frames],
            ignore_index=True,
        )
        categories = pd.Index(values.unique())
        for frame in frames:
            frame[column] = pd.Categorical(
                frame[column].fillna("UNKNOWN").astype(str), categories=categories
            )


def model_parameters(seed: int, n_estimators: int = 1800) -> dict:
    return {
        "objective": "regression",
        "n_estimators": n_estimators,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 20,
        "colsample_bytree": 0.90,
        "reg_lambda": 10.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def validation_split(y: pd.Series, size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < size < 1.0:
        raise ValueError("--validation-size must be strictly between 0 and 1.")
    # Stratifying log-count bins keeps the heavy target tail represented.
    bins = pd.qcut(np.log1p(y), q=10, duplicates="drop").cat.codes
    indices = np.arange(len(y))
    return train_test_split(
        indices, test_size=size, random_state=seed, stratify=bins
    )


def fit_validation_model(
    features: pd.DataFrame, y: pd.Series, validation_size: float, seed: int
) -> tuple[float, int]:
    train_indices, validation_indices = validation_split(y, validation_size, seed)
    model = LGBMRegressor(**model_parameters(seed))
    model.fit(
        features.iloc[train_indices],
        y.iloc[train_indices],
        categorical_feature=CATEGORICAL_COLUMNS,
        eval_set=[(features.iloc[validation_indices], y.iloc[validation_indices])],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    prediction = np.clip(model.predict(features.iloc[validation_indices]), 0.0, None)
    rmse = mean_squared_error(y.iloc[validation_indices], prediction) ** 0.5
    return float(rmse), int(model.best_iteration_)


def exact_key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame[KEY_COLUMNS].fillna("").astype(str).apply(lambda column: column.str.strip())
    )


def main() -> None:
    args = parse_args()
    train_path = resolve_input(args.train_path)
    history_path = resolve_input(args.history_path)
    features_path = resolve_input(args.features_path)

    train = read_core_csv(train_path, require_target=True)
    history_raw = read_core_csv(history_path, require_target=True)
    history = prepare_history(history_raw)
    augmentation = load_augmentation(features_path)
    train_features = make_features(train, history, augmentation)
    y_train = train[TARGET_COLUMN].astype(float)

    test = None
    test_features = None
    if args.test_path:
        test = read_core_csv(resolve_input(args.test_path), require_target=False)
        # violation_count, if present, is deliberately not passed to make_features.
        test_features = make_features(test[KEY_COLUMNS], history, augmentation)
        align_categories(train_features, test_features)
    else:
        align_categories(train_features)

    validation_rmse, best_iteration = fit_validation_model(
        train_features, y_train, args.validation_size, args.seed
    )
    print(
        f"Development validation: RMSE={validation_rmse:.4f} "
        f"({args.validation_size:.0%} stratified 2022 holdout, "
        f"best_iteration={best_iteration})"
    )

    if test is None or test_features is None:
        print("No --test-path supplied; validation completed without writing predictions.")
        return

    final_model = LGBMRegressor(
        **model_parameters(args.seed, n_estimators=max(best_iteration, 50))
    )
    final_model.fit(
        train_features,
        y_train,
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[log_evaluation(0)],
    )
    prediction = np.clip(final_model.predict(test_features), 0.0, None)

    output = pd.DataFrame(
        {
            "street_name": test["Street Name"],
            "violation_type": test["Violation Description"],
            "predicted_count": prediction,
        }
    )
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    train_keys = exact_key_index(train)
    test_keys = exact_key_index(test)
    unseen_count = int((~test_keys.isin(train_keys)).sum())
    print(
        f"Wrote {len(output):,} predictions to {output_path}. "
        f"Unseen exact key pairs: {unseen_count:,}/{len(test):,}; "
        "handled using lag aggregates, street enrichment, and categorical fallback."
    )

    if TARGET_COLUMN in test:
        truth = pd.to_numeric(test[TARGET_COLUMN], errors="coerce")
        labeled = truth.notna()
        if labeled.any():
            eval_rmse = mean_squared_error(truth[labeled], prediction[labeled]) ** 0.5
            print(
                f"Supplied evaluation labels: RMSE={eval_rmse:.4f} "
                f"over {int(labeled.sum()):,} labeled rows."
            )


if __name__ == "__main__":
    main()
