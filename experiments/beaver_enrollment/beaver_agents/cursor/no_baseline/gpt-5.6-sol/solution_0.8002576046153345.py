#!/usr/bin/env python3
"""Leakage-aware high-enrollment training, prediction, and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score


KEY_COLUMNS = ["TERM_CODE", "SUBJECT_ID_SORT"]
TARGET_COLUMN = "HIGH_ENROLLMENT"
HISTORY_FILE = "ENROLLMENT_HISTORY_FEATURES.csv"
OFFERING_FILE = "SUBJECT_OFFERED_SUMMARY.csv"
TRAIN_GOLD_FILE = "gold_enrollment_train.csv"
TEST_GOLD_FILE = "gold_enrollment_test.csv"

HISTORY_COLUMNS = [
    *KEY_COLUMNS,
    "lag1_enrollment",
    "lag2_enrollment",
    "roll3_enrollment_mean",
]
OFFERING_COLUMNS = [
    *KEY_COLUMNS,
    "COURSE_NUMBER",
    "MASTER_SUBJECT_ID_SORT",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "HGN_CODE",
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
    "SUBJECT_TITLE",
]
CATEGORICAL_FEATURES = [
    "SUBJECT_ID_SORT",
    "COURSE_NUMBER",
    "MASTER_SUBJECT_ID_SORT",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "HGN_CODE",
    "season",
    "subject_prefix",
]
TITLE_WORDS = [
    "thesis",
    "seminar",
    "introduction",
    "introductory",
    "special",
    "research",
    "project",
    "laboratory",
    "independent",
]
NUMERIC_FEATURES = [
    "lag1_enrollment",
    "lag2_enrollment",
    "roll3_enrollment_mean",
    "log_lag1_enrollment",
    "log_lag2_enrollment",
    "log_roll3_enrollment_mean",
    "roll_rank_pct",
    "roll_positive_share",
    "roll_vs_dept_mean",
    "dept_term_size",
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
    "subject_number",
    "title_len",
    *[f"has_{word}" for word in TITLE_WORDS],
]
MODEL_FEATURES = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(
        path,
        usecols=columns,
        dtype={"TERM_CODE": str, "SUBJECT_ID_SORT": str},
        low_memory=False,
    )


def _validate_keys(frame: pd.DataFrame, name: str) -> None:
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{name} contains null prediction keys")
    duplicates = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicates:
        raise ValueError(f"{name} contains {duplicates} duplicate prediction keys")


def read_gold(path: Path, require_target: bool) -> pd.DataFrame:
    columns = [*KEY_COLUMNS, TARGET_COLUMN] if require_target else KEY_COLUMNS
    gold = _read_csv(path, columns)
    _validate_keys(gold, path.name)
    if require_target:
        invalid = sorted(set(gold[TARGET_COLUMN].dropna()) - {"Y", "N"})
        if gold[TARGET_COLUMN].isna().any() or invalid:
            raise ValueError(f"{path.name} has invalid labels: {invalid}")
    return gold


def build_feature_frame(data_dir: Path, keys: pd.DataFrame) -> pd.DataFrame:
    """Build forecast-time features without loading same-term enrollment outcomes."""
    history = _read_csv(data_dir / HISTORY_FILE, HISTORY_COLUMNS)
    offering = _read_csv(data_dir / OFFERING_FILE, OFFERING_COLUMNS)
    _validate_keys(history, HISTORY_FILE)
    _validate_keys(offering, OFFERING_FILE)

    frame = keys[KEY_COLUMNS].merge(
        history, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    frame = frame.merge(
        offering, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    missing_history = int(frame[HISTORY_COLUMNS[2:]].isna().all(axis=1).sum())
    missing_offering = int(frame["OFFER_DEPT_CODE"].isna().sum())
    if missing_history or missing_offering:
        raise ValueError(
            "Feature tables do not cover all gold keys: "
            f"{missing_history} missing history rows, "
            f"{missing_offering} missing offering rows"
        )

    year = pd.to_numeric(frame["TERM_CODE"].str[:4], errors="coerce")
    if year.isna().any():
        bad = frame.loc[year.isna(), "TERM_CODE"].drop_duplicates().tolist()
        raise ValueError(f"Unrecognized TERM_CODE values: {bad[:10]}")
    frame["year"] = year.astype("int16")
    frame["season"] = frame["TERM_CODE"].str[4:].replace("", "__MISSING__")
    frame["subject_prefix"] = (
        frame["SUBJECT_ID_SORT"].str.split(".", n=1).str[0]
    )
    frame["subject_number"] = pd.to_numeric(
        frame["SUBJECT_ID_SORT"].str.extract(r"\.(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )

    title = frame["SUBJECT_TITLE"].fillna("").astype(str)
    frame["title_len"] = title.str.len()
    lower_title = title.str.lower()
    for word in TITLE_WORDS:
        frame[f"has_{word}"] = lower_title.str.contains(
            word, regex=False
        ).astype("int8")

    history_features = [
        "lag1_enrollment",
        "lag2_enrollment",
        "roll3_enrollment_mean",
    ]
    for column in history_features:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        frame[column] = values
        frame[f"log_{column}"] = np.log1p(values.clip(lower=0.0))

    group = frame.groupby(
        ["TERM_CODE", "OFFER_DEPT_CODE"], dropna=False, observed=True
    )
    frame["dept_term_size"] = group["SUBJECT_ID_SORT"].transform("size")
    frame["roll_rank_pct"] = group["roll3_enrollment_mean"].rank(
        method="average", pct=True
    )
    frame["roll_positive_share"] = group["roll3_enrollment_mean"].transform(
        lambda values: values.gt(0).mean()
    )
    group_mean = group["roll3_enrollment_mean"].transform("mean")
    frame["roll_vs_dept_mean"] = frame["roll3_enrollment_mean"] / (
        group_mean + 1.0
    )

    return frame


def harmonize_categories(*frames: pd.DataFrame) -> None:
    """Give train and inference rows identical, deterministic category codes."""
    for column in CATEGORICAL_FEATURES:
        values = pd.concat(
            [frame[column].fillna("__MISSING__").astype(str) for frame in frames],
            ignore_index=True,
        )
        categories = pd.Index(values.unique())
        dtype = pd.CategoricalDtype(categories=categories)
        for frame in frames:
            frame[column] = (
                frame[column].fillna("__MISSING__").astype(str).astype(dtype)
            )


def make_model(n_estimators: int = 1500) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )


def fit_with_early_stopping(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    validation_mask: pd.Series,
) -> tuple[lgb.LGBMClassifier, np.ndarray]:
    model = make_model()
    model.fit(
        frame.loc[train_mask, MODEL_FEATURES],
        frame.loc[train_mask, "target"],
        categorical_feature=CATEGORICAL_FEATURES,
        eval_set=[
            (
                frame.loc[validation_mask, MODEL_FEATURES],
                frame.loc[validation_mask, "target"],
            )
        ],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    probabilities = model.predict_proba(
        frame.loc[validation_mask, MODEL_FEATURES]
    )[:, 1]
    return model, probabilities


def optimize_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.02, 0.80, 391)
    scores = [
        f1_score(y_true, probabilities >= threshold, average="macro")
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def metric_report(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    details = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["N", "Y"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "rows": int(len(y_true)),
        "true_positive_rows": int(y_true.sum()),
        "predicted_positive_rows": int(predictions.sum()),
        "threshold": threshold,
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "class_f1": {
            "N": float(details["N"]["f1-score"]),
            "Y": float(details["Y"]["f1-score"]),
        },
    }


def run_validation(train_dir: Path) -> dict[str, Any]:
    gold = read_gold(train_dir / TRAIN_GOLD_FILE, require_target=True)
    frame = build_feature_frame(train_dir, gold)
    frame["target"] = gold[TARGET_COLUMN].eq("Y").astype("int8")
    harmonize_categories(frame)

    years = sorted(frame["year"].unique())
    if len(years) < 3:
        raise ValueError("At least three distinct training years are required")
    tuning_year, evaluation_year = years[-2], years[-1]
    train_mask = frame["year"] < tuning_year
    tuning_mask = frame["year"] == tuning_year
    evaluation_mask = frame["year"] == evaluation_year

    model, tuning_probabilities = fit_with_early_stopping(
        frame, train_mask, tuning_mask
    )
    threshold = optimize_threshold(
        frame.loc[tuning_mask, "target"], tuning_probabilities
    )
    evaluation_probabilities = model.predict_proba(
        frame.loc[evaluation_mask, MODEL_FEATURES]
    )[:, 1]

    return {
        "design": {
            "training_years": [
                int(frame.loc[train_mask, "year"].min()),
                int(frame.loc[train_mask, "year"].max()),
            ],
            "threshold_tuning_year": int(tuning_year),
            "untouched_evaluation_year": int(evaluation_year),
            "same_term_enrollment_columns_used": False,
        },
        "best_iteration": int(model.best_iteration_),
        "tuning": metric_report(
            frame.loc[tuning_mask, "target"],
            tuning_probabilities,
            threshold,
        ),
        "evaluation": metric_report(
            frame.loc[evaluation_mask, "target"],
            evaluation_probabilities,
            threshold,
        ),
    }


def run_prediction(
    train_dir: Path,
    test_dir: Path,
    gold_test_path: Path,
    output_path: Path,
    evaluate: bool,
) -> dict[str, Any]:
    if train_dir.resolve() == test_dir.resolve():
        raise ValueError("TRAIN_DATA_DIR and TEST_DATA_DIR must be different")

    train_gold = read_gold(train_dir / TRAIN_GOLD_FILE, require_target=True)
    test_keys = read_gold(gold_test_path, require_target=False)
    train_frame = build_feature_frame(train_dir, train_gold)
    test_frame = build_feature_frame(test_dir, test_keys)
    train_frame["target"] = train_gold[TARGET_COLUMN].eq("Y").astype("int8")
    harmonize_categories(train_frame, test_frame)

    latest_year = int(train_frame["year"].max())
    calibration_train = train_frame["year"] < latest_year
    calibration_holdout = train_frame["year"] == latest_year
    calibration_model, calibration_probabilities = fit_with_early_stopping(
        train_frame, calibration_train, calibration_holdout
    )
    threshold = optimize_threshold(
        train_frame.loc[calibration_holdout, "target"],
        calibration_probabilities,
    )
    n_estimators = max(1, int(calibration_model.best_iteration_))

    final_model = make_model(n_estimators=n_estimators)
    final_model.fit(
        train_frame[MODEL_FEATURES],
        train_frame["target"],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    probabilities = final_model.predict_proba(test_frame[MODEL_FEATURES])[:, 1]
    predicted_labels = np.where(probabilities >= threshold, "Y", "N")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = test_keys[KEY_COLUMNS].copy()
    predictions[TARGET_COLUMN] = predicted_labels
    predictions.to_csv(output_path, index=False)

    result: dict[str, Any] = {
        "output": str(output_path),
        "rows": int(len(predictions)),
        "predicted_positive_rows": int((predicted_labels == "Y").sum()),
        "threshold": threshold,
        "n_estimators": n_estimators,
    }
    if evaluate:
        labeled_test = read_gold(gold_test_path, require_target=True)
        if not labeled_test[KEY_COLUMNS].equals(predictions[KEY_COLUMNS]):
            raise ValueError("Gold-test key order changed during prediction")
        y_true = labeled_test[TARGET_COLUMN].eq("Y").astype("int8")
        result["evaluation"] = metric_report(
            y_true, probabilities, threshold
        )
    return result


def write_report(report: dict[str, Any], path: Path | None) -> None:
    text = json.dumps(report, indent=2)
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Run time-based validation using training data only"
    )
    validate.add_argument("--train-dir", type=Path, required=True)
    validate.add_argument(
        "--report", type=Path, default=Path("validation_report.json")
    )

    predict = subparsers.add_parser(
        "predict", help="Train on all training rows and predict a test split"
    )
    predict.add_argument("--train-dir", type=Path, required=True)
    predict.add_argument("--test-dir", type=Path, required=True)
    predict.add_argument(
        "--gold-test",
        type=Path,
        help="Defaults to TEST_DATA_DIR/gold_enrollment_test.csv",
    )
    predict.add_argument(
        "--output", type=Path, default=Path("predictions.csv")
    )
    predict.add_argument("--report", type=Path, default=Path("test_report.json"))
    predict.add_argument(
        "--evaluate",
        action="store_true",
        help="Score predictions if gold_enrollment_test.csv contains labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        report = run_validation(args.train_dir)
    else:
        gold_test = args.gold_test or args.test_dir / TEST_GOLD_FILE
        report = run_prediction(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            gold_test_path=gold_test,
            output_path=args.output,
            evaluate=args.evaluate,
        )
    write_report(report, args.report)


if __name__ == "__main__":
    main()
