#!/usr/bin/env python3
"""Leakage-safe high-enrollment training, prediction, and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

KEY_COLUMNS = ["TERM_CODE", "SUBJECT_ID_SORT"]
TARGET_COLUMN = "HIGH_ENROLLMENT"
HISTORY_COLUMNS = [
    "lag1_enrollment",
    "lag2_enrollment",
    "roll3_enrollment_mean",
]
OFFERED_COLUMNS = [
    "COURSE_NUMBER",
    "MASTER_SUBJECT_ID_SORT",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
    "SUBJECT_TITLE",
]
FEATURE_COLUMNS = [
    "SUBJECT_ID_SORT",
    "COURSE_NUMBER",
    "MASTER_SUBJECT_ID_SORT",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
    *HISTORY_COLUMNS,
    "YEAR",
    "SEASON",
    "SUBJECT_PREFIX",
    "SUBJECT_LEVEL",
    "SUBJECT_NUMERIC",
    "TITLE_LEN",
]
CATEGORICAL_COLUMNS = [
    "SUBJECT_ID_SORT",
    "COURSE_NUMBER",
    "MASTER_SUBJECT_ID_SORT",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "SEASON",
    "SUBJECT_PREFIX",
    "SUBJECT_LEVEL",
]


def _read_csv(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a CSV and produce a useful error for missing columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Required table not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if columns is not None:
        columns = list(columns)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        frame = frame[columns]
    return frame


def _validate_keys(frame: pd.DataFrame, name: str) -> None:
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{name} contains null prediction keys")
    duplicated = frame.duplicated(KEY_COLUMNS)
    if duplicated.any():
        examples = frame.loc[duplicated, KEY_COLUMNS].head().to_dict("records")
        raise ValueError(f"{name} contains duplicate keys, for example: {examples}")


def load_gold(path: Path) -> pd.DataFrame:
    gold = _read_csv(path, [*KEY_COLUMNS, TARGET_COLUMN])
    _validate_keys(gold, path.name)
    labels = set(gold[TARGET_COLUMN].dropna().unique())
    if not labels <= {"Y", "N"} or gold[TARGET_COLUMN].isna().any():
        raise ValueError(f"{path.name} labels must contain only Y or N")
    return gold


def load_prediction_keys(test_dir: Path) -> pd.DataFrame:
    """Use the offered summary as the prediction universe."""
    keys = _read_csv(test_dir / "SUBJECT_OFFERED_SUMMARY.csv", KEY_COLUMNS)
    _validate_keys(keys, "test SUBJECT_OFFERED_SUMMARY.csv")
    return keys


def load_raw_features(data_dir: Path, keys: pd.DataFrame) -> pd.DataFrame:
    """Join only features that are available before the target enrollment."""
    offered = _read_csv(
        data_dir / "SUBJECT_OFFERED_SUMMARY.csv",
        [*KEY_COLUMNS, *OFFERED_COLUMNS],
    )
    history_path = data_dir / "ENROLLMENT_HISTORY_FEATURES.csv"
    if history_path.is_file():
        history = _read_csv(history_path, [*KEY_COLUMNS, *HISTORY_COLUMNS])
    else:
        history = keys.copy()
        for column in HISTORY_COLUMNS:
            history[column] = np.nan

    _validate_keys(offered, f"{data_dir.name} offered features")
    _validate_keys(history, f"{data_dir.name} history features")
    result = keys.merge(offered, on=KEY_COLUMNS, how="left", validate="one_to_one")
    result = result.merge(history, on=KEY_COLUMNS, how="left", validate="one_to_one")
    return add_derived_features(result)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    term = result["TERM_CODE"].astype("string")
    result["YEAR"] = pd.to_numeric(term.str.extract(r"^(\d{4})", expand=False))
    result["SEASON"] = term.str.extract(r"^\d{4}(.+)$", expand=False)

    subject = result["SUBJECT_ID_SORT"].astype("string")
    result["SUBJECT_PREFIX"] = subject.str.split(".", n=1).str[0]
    suffix = subject.str.split(".", n=1).str[1].fillna("")
    result["SUBJECT_LEVEL"] = suffix.str.extract(r"(\d)", expand=False).fillna("X")
    result["SUBJECT_NUMERIC"] = pd.to_numeric(
        suffix.str.extract(r"(\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )
    result["TITLE_LEN"] = result["SUBJECT_TITLE"].fillna("").astype(str).str.len()
    return result


def make_feature_matrices(
    train_raw: pd.DataFrame,
    prediction_raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Create aligned pandas categoricals for LightGBM."""
    frames = [train_raw] if prediction_raw is None else [train_raw, prediction_raw]
    lengths = [len(frame) for frame in frames]
    combined = pd.concat(
        [frame[FEATURE_COLUMNS] for frame in frames],
        ignore_index=True,
    )

    for column in CATEGORICAL_COLUMNS:
        combined[column] = (
            combined[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype("category")
        )
    for column in set(FEATURE_COLUMNS) - set(CATEGORICAL_COLUMNS):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    train_features = combined.iloc[: lengths[0]].copy()
    train_features.index = train_raw.index
    if prediction_raw is None:
        return train_features, None

    prediction_features = combined.iloc[lengths[0] :].copy()
    prediction_features.index = prediction_raw.index
    return train_features, prediction_features


def build_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )


def validation_mask(raw: pd.DataFrame) -> pd.Series:
    years = raw["YEAR"].dropna().astype(int)
    if years.nunique() < 3:
        raise ValueError("At least three distinct years are needed for time validation")
    first_validation_year = int(years.max()) - 1
    mask = raw["YEAR"] >= first_validation_year
    if mask.all() or not mask.any():
        raise ValueError("Could not construct a non-empty time validation split")
    return mask


def tune_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.20, 0.65, 181)
    scored = [
        (float(f1_score(labels, probabilities >= threshold, average="macro")), threshold)
        for threshold in candidates
    ]
    score, threshold = max(scored)
    return float(threshold), score


def run_validation(
    train_dir: Path,
    *,
    quiet: bool = False,
) -> tuple[float, dict[str, object]]:
    gold = load_gold(train_dir / "gold_enrollment_train.csv")
    raw = load_raw_features(train_dir, gold[KEY_COLUMNS])
    features, _ = make_feature_matrices(raw)
    labels = gold[TARGET_COLUMN].eq("Y").to_numpy(dtype=np.int8)
    holdout = validation_mask(raw).to_numpy()

    model = build_model()
    model.fit(
        features.loc[~holdout],
        labels[~holdout],
        categorical_feature=CATEGORICAL_COLUMNS,
    )
    probabilities = model.predict_proba(features.loc[holdout])[:, 1]
    threshold, macro_f1 = tune_threshold(labels[holdout], probabilities)
    predicted = probabilities >= threshold

    report: dict[str, object] = {
        "training_rows": int((~holdout).sum()),
        "validation_rows": int(holdout.sum()),
        "validation_years": sorted(raw.loc[holdout, "YEAR"].astype(int).unique().tolist()),
        "threshold": threshold,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_matrix(labels[holdout], predicted).tolist(),
        "classification_report": classification_report(
            labels[holdout],
            predicted,
            target_names=["N", "Y"],
            output_dict=True,
            zero_division=0,
        ),
    }
    if not quiet:
        print(json.dumps(report, indent=2))
    return threshold, report


def evaluate_predictions(gold_path: Path, predictions_path: Path) -> dict[str, object]:
    gold = load_gold(gold_path)
    predictions = _read_csv(
        predictions_path,
        [*KEY_COLUMNS, TARGET_COLUMN],
    )
    _validate_keys(predictions, predictions_path.name)
    labels = set(predictions[TARGET_COLUMN].dropna().unique())
    if not labels <= {"Y", "N"} or predictions[TARGET_COLUMN].isna().any():
        raise ValueError("Predictions must contain only Y or N")

    joined = gold.merge(
        predictions,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_GOLD", "_PREDICTED"),
        indicator=True,
        validate="one_to_one",
    )
    key_counts = joined["_merge"].value_counts()
    if key_counts.get("left_only", 0) or key_counts.get("right_only", 0):
        raise ValueError(
            "Gold and prediction key sets differ: "
            f"{key_counts.to_dict()}"
        )

    actual = joined[f"{TARGET_COLUMN}_GOLD"]
    predicted = joined[f"{TARGET_COLUMN}_PREDICTED"]
    result: dict[str, object] = {
        "rows": len(joined),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=["N", "Y"]).tolist(),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=["N", "Y"],
            output_dict=True,
            zero_division=0,
        ),
    }
    print(json.dumps(result, indent=2))
    return result


def train_and_predict(
    train_dir: Path,
    test_dir: Path,
    output_path: Path,
    test_gold_path: Path | None = None,
) -> None:
    # This is the only threshold-selection step. It reads training data only.
    threshold, validation_report = run_validation(train_dir, quiet=True)
    print(
        "Validation macro F1: "
        f"{validation_report['macro_f1']:.6f}; threshold: {threshold:.4f}"
    )

    train_gold = load_gold(train_dir / "gold_enrollment_train.csv")
    test_keys = load_prediction_keys(test_dir)
    train_raw = load_raw_features(train_dir, train_gold[KEY_COLUMNS])
    test_raw = load_raw_features(test_dir, test_keys)
    train_features, test_features = make_feature_matrices(train_raw, test_raw)
    assert test_features is not None

    labels = train_gold[TARGET_COLUMN].eq("Y").to_numpy(dtype=np.int8)
    model = build_model()
    model.fit(
        train_features,
        labels,
        categorical_feature=CATEGORICAL_COLUMNS,
    )
    probabilities = model.predict_proba(test_features)[:, 1]

    output = test_keys.copy()
    output[TARGET_COLUMN] = np.where(probabilities >= threshold, "Y", "N")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote {len(output):,} predictions to {output_path}")

    # Test gold is optional and is opened only after training and prediction.
    if test_gold_path is not None:
        evaluate_predictions(test_gold_path, output_path)


def parse_args() -> argparse.Namespace:
    default_train = Path("table_splits/train")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="run time-based validation")
    validate_parser.add_argument("--train-dir", type=Path, default=default_train)

    predict_parser = subparsers.add_parser("predict", help="train and predict a future split")
    predict_parser.add_argument("--train-dir", type=Path, default=default_train)
    predict_parser.add_argument("--test-dir", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    predict_parser.add_argument(
        "--test-gold",
        type=Path,
        default=None,
        help="optional gold_enrollment_test.csv; evaluated only after prediction",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="evaluate an existing prediction file against test gold",
    )
    evaluate_parser.add_argument("--gold-test", type=Path, required=True)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        run_validation(args.train_dir)
    elif args.command == "predict":
        train_and_predict(
            args.train_dir,
            args.test_dir,
            args.output,
            args.test_gold,
        )
    elif args.command == "evaluate":
        evaluate_predictions(args.gold_test, args.predictions)


if __name__ == "__main__":
    main()
