#!/usr/bin/env python3
"""Leakage-safe high-enrollment training, prediction, and evaluation.

The model uses SUBJECT_OFFERED_SUMMARY and precomputed historical (lagged)
enrollment features.  It intentionally never uses current-term enrollment
columns from SUBJECT_SUMMARY.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


KEYS = ["TERM_CODE", "SUBJECT_ID_SORT"]
TARGET = "HIGH_ENROLLMENT"
HISTORY_COLUMNS = [
    "lag1_enrollment",
    "lag2_enrollment",
    "roll3_enrollment_mean",
]
OFFERED_COLUMNS = [
    *KEYS,
    "COURSE_NUMBER",
    "SUBJECT_TITLE",
    "MASTER_SUBJECT_ID_SORT",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
]
CATEGORICAL_COLUMNS = [
    "term_season",
    "subject_prefix",
    "course_band",
    "course_suffix",
    "COURSE_NUMBER",
    "MASTER_SUBJECT_ID_SORT",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "OFFER_DEPT_CODE",
    "DEPARTMENT_CODE",
    "OFFER_SCHOOL_NAME",
]


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required table not found: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def _assert_unique_keys(frame: pd.DataFrame, name: str) -> None:
    missing = [column for column in KEYS if column not in frame]
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")
    duplicate_count = int(frame.duplicated(KEYS).sum())
    if duplicate_count:
        raise ValueError(f"{name} has {duplicate_count:,} duplicate prediction keys")


def load_features(data_dir: Path) -> pd.DataFrame:
    """Load one feature row per subject-offering summary row."""
    offered_path = data_dir / "SUBJECT_OFFERED_SUMMARY.csv"
    offered_header = _read_csv(offered_path, nrows=0).columns
    available = [column for column in OFFERED_COLUMNS if column in offered_header]
    offered = _read_csv(offered_path, usecols=available, dtype={key: str for key in KEYS})
    _assert_unique_keys(offered, offered_path.name)

    history_path = data_dir / "ENROLLMENT_HISTORY_FEATURES.csv"
    history = _read_csv(
        history_path,
        usecols=[*KEYS, *HISTORY_COLUMNS],
        dtype={key: str for key in KEYS},
    )
    _assert_unique_keys(history, history_path.name)

    features = offered.merge(history, on=KEYS, how="left", validate="one_to_one")
    subject_path = data_dir / "SUBJECT_SUMMARY.csv"
    if subject_path.is_file():
        # Department is safe catalog metadata.  Do not load any current-term
        # enrollment columns from this table.
        subject = _read_csv(
            subject_path,
            usecols=[*KEYS, "DEPARTMENT_CODE"],
            dtype={key: str for key in KEYS},
        )
        _assert_unique_keys(subject, subject_path.name)
        features = features.merge(subject, on=KEYS, how="left", validate="one_to_one")
    else:
        features["DEPARTMENT_CODE"] = np.nan
    if len(features) != len(offered):
        raise AssertionError("Feature join changed the prediction row count")
    return features


def load_training_data(train_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load training features and labels; never inspect any test gold file."""
    gold_path = train_dir / "gold_enrollment_train.csv"
    gold = _read_csv(gold_path, dtype={key: str for key in KEYS})
    expected = [*KEYS, TARGET]
    if list(gold.columns) != expected:
        raise ValueError(f"{gold_path.name} must have columns {expected}")
    _assert_unique_keys(gold, gold_path.name)
    invalid = sorted(set(gold[TARGET].dropna()) - {"Y", "N"})
    if invalid:
        raise ValueError(f"Unexpected labels in {gold_path.name}: {invalid}")

    features = load_features(train_dir)
    joined = gold.merge(features, on=KEYS, how="left", validate="one_to_one")
    feature_columns = [column for column in features if column not in KEYS]
    unmatched = joined[feature_columns].isna().all(axis=1)
    if unmatched.any():
        raise ValueError(
            f"{int(unmatched.sum()):,} training labels have no subject summary row"
        )
    return joined.drop(columns=TARGET), gold[TARGET].eq("Y").astype(np.int8)


def _subject_components(subject_ids: pd.Series) -> pd.DataFrame:
    values = subject_ids.fillna("").astype(str).str.upper()
    prefix = values.str.extract(r"^([^.\d]+|\d+[A-Z]*)", expand=False).fillna("UNKNOWN")
    after_dot = values.str.split(".", n=1).str[-1]
    number_text = after_dot.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    number = pd.to_numeric(number_text, errors="coerce")
    suffix = after_dot.str.extract(r"([A-Z]+)$", expand=False).fillna("NONE")
    band = np.floor(number.fillna(-100) / 100).astype(int).astype(str)
    return pd.DataFrame(
        {
            "subject_prefix": prefix,
            "subject_number": number,
            "course_band": band,
            "course_suffix": suffix,
            "subject_has_letters": after_dot.str.contains("[A-Z]", regex=True).astype(np.int8),
        },
        index=subject_ids.index,
    )


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create model features using only information available before outcome."""
    frame = raw.copy()
    term = frame["TERM_CODE"].fillna("").astype(str).str.upper()
    frame["term_year"] = pd.to_numeric(term.str[:4], errors="coerce")
    frame["term_season"] = term.str[4:].replace("", "UNKNOWN")
    frame = pd.concat([frame, _subject_components(frame["SUBJECT_ID_SORT"])], axis=1)

    title = frame.get("SUBJECT_TITLE", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["title_length"] = title.str.len()
    frame["title_words"] = title.str.count(r"\S+")
    frame["title_has_intro"] = title.str.contains(
        r"\b(?:intro|introduction|principles|fundamentals)\b", case=False, regex=True
    ).astype(np.int8)
    frame["title_has_thesis"] = title.str.contains(
        r"\b(?:thesis|research|seminar|special topics?)\b", case=False, regex=True
    ).astype(np.int8)

    numeric_columns = [
        *HISTORY_COLUMNS,
        "TOTAL_UNITS",
        "LECTURE_UNITS",
        "LAB_UNITS",
        "PREPARATION_UNITS",
    ]
    for column in numeric_columns:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in HISTORY_COLUMNS:
        values = frame[column].fillna(0).clip(lower=0)
        frame[f"log1p_{column}"] = np.log1p(values)
        frame[f"{column}_positive"] = values.gt(0).astype(np.int8)

    lag1 = frame["lag1_enrollment"].fillna(0).clip(lower=0)
    lag2 = frame["lag2_enrollment"].fillna(0).clip(lower=0)
    rolling = frame["roll3_enrollment_mean"].fillna(0).clip(lower=0)
    frame["lag1_minus_lag2"] = lag1 - lag2
    frame["lag1_over_lag2"] = (lag1 + 1.0) / (lag2 + 1.0)
    frame["lag1_over_roll3"] = (lag1 + 1.0) / (rolling + 1.0)
    frame["history_max"] = pd.concat([lag1, lag2, rolling], axis=1).max(axis=1)
    frame["history_nonzero_count"] = pd.concat(
        [lag1.gt(0), lag2.gt(0), rolling.gt(0)], axis=1
    ).sum(axis=1)

    frame["model_department"] = frame.get(
        "DEPARTMENT_CODE", pd.Series(np.nan, index=frame.index)
    ).fillna(frame.get("OFFER_DEPT_CODE", "UNKNOWN"))
    group_columns = ["TERM_CODE", "model_department"]
    if all(column in frame for column in group_columns):
        grouped = frame.groupby(group_columns, dropna=False, observed=True)
        for column in ["lag1_enrollment", "roll3_enrollment_mean"]:
            values = frame[column].fillna(0).clip(lower=0)
            group_mean = grouped[column].transform("mean").fillna(0)
            group_median = grouped[column].transform("median").fillna(0)
            group_max = grouped[column].transform("max").fillna(0)
            positive_values = values.where(values.gt(0))
            positive_groups = positive_values.groupby(
                [frame[group] for group in group_columns],
                dropna=False,
                observed=True,
            )
            frame[f"{column}_dept_mean"] = group_mean
            frame[f"{column}_dept_median"] = group_median
            frame[f"{column}_dept_max"] = group_max
            frame[f"{column}_dept_positive_q75"] = positive_groups.transform(
                lambda values: values.quantile(0.75)
            ).fillna(0)
            frame[f"{column}_dept_positive_count"] = positive_groups.transform(
                "count"
            ).fillna(0)
            frame[f"{column}_vs_dept_mean"] = values / (group_mean + 1.0)
            frame[f"{column}_vs_dept_median"] = values / (group_median + 1.0)
            frame[f"{column}_dept_pct_rank"] = grouped[column].rank(
                pct=True, method="average"
            )
            frame[f"{column}_dept_positive_pct_rank"] = (
                positive_groups.rank(pct=True, method="average").fillna(0)
            )
        frame["dept_term_offerings"] = grouped["SUBJECT_ID_SORT"].transform("size")

    drop_columns = [*KEYS, "SUBJECT_TITLE", "model_department"]
    frame = frame.drop(columns=[column for column in drop_columns if column in frame])
    return frame


@dataclass
class FeatureEncoder:
    """Stable category maps and model column order."""

    category_maps: dict[str, dict[str, int]]
    columns: list[str]
    categorical_columns: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "FeatureEncoder":
        engineered = engineer_features(frame)
        category_maps: dict[str, dict[str, int]] = {}
        categorical = [column for column in CATEGORICAL_COLUMNS if column in engineered]
        for column in categorical:
            values = engineered[column].fillna("__MISSING__").astype(str)
            uniques = pd.Index(values.unique())
            category_maps[column] = {value: index for index, value in enumerate(uniques)}
        return cls(category_maps, list(engineered.columns), categorical)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        engineered = engineer_features(frame)
        engineered = engineered.reindex(columns=self.columns)
        for column, mapping in self.category_maps.items():
            values = engineered[column].fillna("__MISSING__").astype(str)
            engineered[column] = values.map(mapping).fillna(-1).astype(np.int32)
        for column in engineered:
            if column not in self.category_maps:
                engineered[column] = pd.to_numeric(engineered[column], errors="coerce")
        return engineered


def make_model(seed: int = 2026) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """Tune for the stated macro-F1 metric without assuming 0.5."""
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.015, 0.35, 100),
                np.quantile(probabilities, np.linspace(0.65, 0.98, 80)),
            ]
        )
    )
    scores = np.array(
        [f1_score(y_true, probabilities >= value, average="macro") for value in candidates]
    )
    index = int(np.argmax(scores))
    return float(candidates[index]), float(scores[index])


def validation_split(
    features: pd.DataFrame, validation_years: int
) -> tuple[np.ndarray, np.ndarray, int]:
    years = pd.to_numeric(features["TERM_CODE"].str[:4], errors="coerce")
    unique_years = sorted(years.dropna().astype(int).unique())
    if len(unique_years) <= validation_years:
        raise ValueError("Not enough distinct years for the requested time validation")
    cutoff = unique_years[-validation_years]
    validation = years.ge(cutoff).to_numpy()
    return ~validation, validation, int(cutoff)


def validate(
    train_dir: Path, validation_years: int = 2, seed: int = 2026
) -> dict[str, Any]:
    raw, target = load_training_data(train_dir)
    train_mask, validation_mask, cutoff = validation_split(raw, validation_years)
    encoder = FeatureEncoder.fit(raw.loc[train_mask])
    x_train = encoder.transform(raw.loc[train_mask])
    x_validation = encoder.transform(raw.loc[validation_mask])
    model = make_model(seed)
    model.fit(
        x_train,
        target.loc[train_mask],
        categorical_feature=encoder.categorical_columns,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    probabilities = model.predict_proba(x_validation)[:, 1]
    threshold, macro_f1 = best_threshold(target.loc[validation_mask], probabilities)
    predictions = probabilities >= threshold
    report = classification_report(
        target.loc[validation_mask],
        predictions,
        target_names=["N", "Y"],
        output_dict=True,
        zero_division=0,
    )
    result = {
        "validation_start_year": cutoff,
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "positive_rate": float(target.loc[validation_mask].mean()),
        "threshold": threshold,
        "macro_f1": macro_f1,
        "f1_N": float(report["N"]["f1-score"]),
        "f1_Y": float(report["Y"]["f1-score"]),
        "confusion_matrix": confusion_matrix(
            target.loc[validation_mask], predictions
        ).tolist(),
    }
    print(json.dumps(result, indent=2))
    return result


def train(
    train_dir: Path,
    model_path: Path,
    validation_years: int = 2,
    seed: int = 2026,
) -> dict[str, Any]:
    raw, target = load_training_data(train_dir)
    train_mask, validation_mask, cutoff = validation_split(raw, validation_years)

    tuning_encoder = FeatureEncoder.fit(raw.loc[train_mask])
    x_train = tuning_encoder.transform(raw.loc[train_mask])
    x_validation = tuning_encoder.transform(raw.loc[validation_mask])
    tuning_model = make_model(seed)
    tuning_model.fit(
        x_train,
        target.loc[train_mask],
        categorical_feature=tuning_encoder.categorical_columns,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    validation_probabilities = tuning_model.predict_proba(x_validation)[:, 1]
    threshold, validation_score = best_threshold(
        target.loc[validation_mask], validation_probabilities
    )

    final_encoder = FeatureEncoder.fit(raw)
    x_all = final_encoder.transform(raw)
    final_model = make_model(seed)
    final_model.fit(
        x_all,
        target,
        categorical_feature=final_encoder.categorical_columns,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    artifact = {
        "model": final_model,
        "encoder": final_encoder,
        "threshold": threshold,
        "metadata": {
            "train_dir": str(train_dir),
            "rows": len(raw),
            "validation_start_year": cutoff,
            "validation_macro_f1": validation_score,
            "threshold": threshold,
            "seed": seed,
            "leakage_exclusions": [
                "SUBJECT_SUMMARY current-term enrollment columns",
                "gold_enrollment_test.csv",
            ],
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    print(json.dumps(artifact["metadata"], indent=2))
    print(f"Saved model to {model_path}")
    return artifact["metadata"]


def predict(model_path: Path, test_dir: Path, output_path: Path) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    raw = load_features(test_dir)
    matrix = artifact["encoder"].transform(raw)
    probabilities = artifact["model"].predict_proba(matrix)[:, 1]
    labels = np.where(probabilities >= artifact["threshold"], "Y", "N")
    predictions = raw[KEYS].copy()
    predictions[TARGET] = labels
    _assert_unique_keys(predictions, "predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(
        f"Wrote {len(predictions):,} predictions to {output_path} "
        f"({int((labels == 'Y').sum()):,} positive)"
    )
    return predictions


def evaluate(prediction_path: Path, gold_path: Path) -> dict[str, Any]:
    """Evaluate an already-created prediction file against held-out gold."""
    predictions = _read_csv(prediction_path, dtype={key: str for key in KEYS})
    gold = _read_csv(gold_path, dtype={key: str for key in KEYS})
    _assert_unique_keys(predictions, prediction_path.name)
    _assert_unique_keys(gold, gold_path.name)
    required = [*KEYS, TARGET]
    for frame, name in [(predictions, prediction_path.name), (gold, gold_path.name)]:
        if any(column not in frame for column in required):
            raise ValueError(f"{name} must contain {required}")

    compared = gold.merge(
        predictions,
        on=KEYS,
        how="outer",
        suffixes=("_gold", "_pred"),
        indicator=True,
        validate="one_to_one",
    )
    if not compared["_merge"].eq("both").all():
        counts = compared["_merge"].value_counts().to_dict()
        raise ValueError(f"Prediction and gold keys do not match: {counts}")
    y_true = compared[f"{TARGET}_gold"]
    y_pred = compared[f"{TARGET}_pred"]
    invalid = sorted((set(y_true) | set(y_pred)) - {"Y", "N"})
    if invalid:
        raise ValueError(f"Unexpected labels: {invalid}")
    result = {
        "rows": len(compared),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_labels": ["N", "Y"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["N", "Y"]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=["N", "Y"], output_dict=True, zero_division=0
        ),
    }
    print(json.dumps(result, indent=2))
    return result


def run_pipeline(
    train_dir: Path,
    model_path: Path,
    test_dir: Path | None = None,
    output_path: Path = Path("predictions.csv"),
    gold_test: Path | None = None,
    validation_years: int = 2,
    seed: int = 2026,
) -> None:
    """Train now; optionally predict and evaluate when test data arrives."""
    train(train_dir, model_path, validation_years, seed)
    if test_dir is None:
        print("No test directory supplied; training is complete.")
        return
    predict(model_path, test_dir, output_path)
    if gold_test is not None:
        evaluate(output_path, gold_test)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Run time-based validation")
    validate_parser.add_argument("--train-dir", type=Path, required=True)
    validate_parser.add_argument("--validation-years", type=int, default=2)
    validate_parser.add_argument("--seed", type=int, default=2026)

    train_parser = subparsers.add_parser("train", help="Tune and train the final model")
    train_parser.add_argument("--train-dir", type=Path, required=True)
    train_parser.add_argument("--model-path", type=Path, required=True)
    train_parser.add_argument("--validation-years", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=2026)

    predict_parser = subparsers.add_parser("predict", help="Predict a future table split")
    predict_parser.add_argument("--model-path", type=Path, required=True)
    predict_parser.add_argument("--test-dir", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate predictions after held-out gold is available"
    )
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--gold-test", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run", help="Train, then optionally predict and evaluate"
    )
    run_parser.add_argument("--train-dir", type=Path, required=True)
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--test-dir", type=Path, default=None)
    run_parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    run_parser.add_argument("--gold-test", type=Path, default=None)
    run_parser.add_argument("--validation-years", type=int, default=2)
    run_parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        validate(args.train_dir, args.validation_years, args.seed)
    elif args.command == "train":
        train(
            args.train_dir,
            args.model_path,
            args.validation_years,
            args.seed,
        )
    elif args.command == "predict":
        predict(args.model_path, args.test_dir, args.output)
    elif args.command == "evaluate":
        evaluate(args.predictions, args.gold_test)
    elif args.command == "run":
        run_pipeline(
            train_dir=args.train_dir,
            model_path=args.model_path,
            test_dir=args.test_dir,
            output_path=args.output,
            gold_test=args.gold_test,
            validation_years=args.validation_years,
            seed=args.seed,
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
