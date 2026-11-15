#!/usr/bin/env python3
"""Leakage-safe high-enrollment training, prediction, and evaluation.

The model uses only course-offering metadata, the supplied historical
enrollment features, and labels from earlier training terms. In particular,
current-term enrollment columns from SUBJECT_SUMMARY.csv are never loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


KEYS = ["TERM_CODE", "SUBJECT_ID_SORT"]
TARGET = "HIGH_ENROLLMENT"
MISSING = "__MISSING__"

OFFERING_COLUMNS = [
    *KEYS,
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
HISTORY_COLUMNS = [
    *KEYS,
    "lag1_enrollment",
    "lag2_enrollment",
    "roll3_enrollment_mean",
]
NUMERIC_BASE_FEATURES = [
    "lag1_enrollment",
    "lag2_enrollment",
    "roll3_enrollment_mean",
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
    "year",
    "subject_number",
    "subject_level",
]
CATEGORICAL_FEATURES = [
    "season",
    "SUBJECT_ID_SORT",
    "MASTER_SUBJECT_ID_SORT",
    "title_normalized",
    "title_prefix",
    "subject_prefix",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
]
HISTORICAL_LABEL_FEATURES = [
    "hist_count_subject",
    "hist_rate_subject",
    "hist_last_subject",
    "hist_count_subject_season",
    "hist_rate_subject_season",
    "hist_last_subject_season",
]
FEATURES = NUMERIC_BASE_FEATURES + CATEGORICAL_FEATURES + HISTORICAL_LABEL_FEATURES
SEASON_ORDER = {"FA": 0, "JA": 1, "SP": 2, "SU": 3, "10": 4}


def _check_unique(frame: pd.DataFrame, source: Path) -> None:
    duplicate_count = int(frame.duplicated(KEYS).sum())
    if duplicate_count:
        raise ValueError(f"{source} has {duplicate_count} duplicate prediction keys")


def _read_available_columns(path: Path, requested: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required table not found: {path}")
    available = set(pd.read_csv(path, nrows=0).columns)
    missing_keys = set(KEYS) - available
    if missing_keys:
        raise ValueError(f"{path} is missing key columns: {sorted(missing_keys)}")
    selected = [column for column in requested if column in available]
    frame = pd.read_csv(path, usecols=selected, low_memory=False)
    for column in requested:
        if column not in frame:
            frame[column] = np.nan
    return frame[list(requested)]


def load_feature_rows(data_dir: Path) -> pd.DataFrame:
    """Load one row per offering without touching any gold file."""
    offering_path = data_dir / "SUBJECT_OFFERED_SUMMARY.csv"
    history_path = data_dir / "ENROLLMENT_HISTORY_FEATURES.csv"
    offerings = _read_available_columns(offering_path, OFFERING_COLUMNS)
    history = _read_available_columns(history_path, HISTORY_COLUMNS)
    _check_unique(offerings, offering_path)
    _check_unique(history, history_path)

    rows = offerings.merge(history, on=KEYS, how="left", validate="one_to_one")
    for column in HISTORY_COLUMNS[2:]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    return rows


def load_training_rows(train_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Join training labels to the offering rows and verify exact key coverage."""
    rows = load_feature_rows(train_dir)
    gold_path = train_dir / "gold_enrollment_train.csv"
    if not gold_path.is_file():
        raise FileNotFoundError(f"Training labels not found: {gold_path}")
    gold = pd.read_csv(gold_path, dtype={TARGET: "string"})
    required = set(KEYS + [TARGET])
    if not required.issubset(gold.columns):
        raise ValueError(f"{gold_path} must contain {sorted(required)}")
    gold = gold[KEYS + [TARGET]]
    _check_unique(gold, gold_path)

    joined = rows.merge(gold, on=KEYS, how="left", validate="one_to_one")
    if len(joined) != len(gold) or joined[TARGET].isna().any():
        offering_keys = pd.MultiIndex.from_frame(rows[KEYS])
        gold_keys = pd.MultiIndex.from_frame(gold[KEYS])
        missing = len(gold_keys.difference(offering_keys))
        extra = len(offering_keys.difference(gold_keys))
        raise ValueError(
            "Training gold and offering summary keys differ "
            f"(gold-only={missing}, summary-only={extra})"
        )
    invalid = sorted(set(joined[TARGET].dropna()) - {"Y", "N"})
    if invalid:
        raise ValueError(f"Unexpected {TARGET} values: {invalid}")
    target = joined.pop(TARGET).eq("Y").astype("int8")
    return joined, target


def engineer_base_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy().reset_index(drop=True)
    term = result["TERM_CODE"].astype("string")
    year = pd.to_numeric(term.str.slice(0, 4), errors="coerce")
    if year.isna().any():
        examples = result.loc[year.isna(), "TERM_CODE"].head(3).tolist()
        raise ValueError(f"Invalid TERM_CODE values, for example: {examples}")
    result["year"] = year.astype("int16")
    result["season"] = term.str.slice(4).fillna(MISSING)
    result["_term_order"] = (
        result["year"].astype("int64") * 10
        + result["season"].map(SEASON_ORDER).fillna(9).astype("int64")
    )

    subject = result["SUBJECT_ID_SORT"].fillna(MISSING).astype(str)
    result["subject_prefix"] = subject.str.extract(r"^([^.]+)", expand=False)
    number_text = subject.str.extract(r"\.([0-9]+)", expand=False)
    result["subject_number"] = pd.to_numeric(number_text, errors="coerce")
    result["subject_level"] = (result["subject_number"] // 100).clip(upper=99)

    title = result["SUBJECT_TITLE"].fillna("").astype(str)
    result["title_normalized"] = (
        title.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    )
    result["title_prefix"] = (
        result["title_normalized"].str.split().str[:2].str.join(" ")
    )
    return result


def add_historical_label_features(
    fit_rows: pd.DataFrame,
    fit_target: pd.Series,
    apply_rows: pd.DataFrame,
    smoothing: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create prior-label features without using a row's own or future label."""
    fit = fit_rows.copy().reset_index(drop=True)
    apply = apply_rows.copy().reset_index(drop=True)
    y = pd.Series(np.asarray(fit_target, dtype=float), index=fit.index)
    prior = float(y.mean())
    ordered_index = fit.sort_values(
        ["_term_order", "SUBJECT_ID_SORT"], kind="stable"
    ).index

    history = fit.loc[ordered_index, ["SUBJECT_ID_SORT", "season"]].copy()
    history["_target"] = y.loc[ordered_index].to_numpy()

    group_specs = [
        (["SUBJECT_ID_SORT"], "subject"),
        (["SUBJECT_ID_SORT", "season"], "subject_season"),
    ]
    for group_columns, suffix in group_specs:
        grouped = history.groupby(group_columns, sort=False, dropna=False)["_target"]
        count = grouped.cumcount().astype(float)
        prior_sum = grouped.cumsum() - history["_target"]
        prior_last = grouped.shift()

        fit.loc[ordered_index, f"hist_count_{suffix}"] = count.to_numpy()
        fit.loc[ordered_index, f"hist_rate_{suffix}"] = (
            (prior_sum + smoothing * prior) / (count + smoothing)
        ).to_numpy()
        fit.loc[ordered_index, f"hist_last_{suffix}"] = (
            prior_last.fillna(prior).to_numpy()
        )

        stats = history.groupby(
            group_columns, sort=False, dropna=False
        )["_target"].agg(["sum", "count", "last"])
        if len(group_columns) == 1:
            application_keys = pd.Index(apply[group_columns[0]])
        else:
            application_keys = pd.MultiIndex.from_frame(apply[group_columns])
        mapped = stats.reindex(application_keys)
        counts = mapped["count"].fillna(0.0).to_numpy(dtype=float)
        sums = mapped["sum"].fillna(0.0).to_numpy(dtype=float)
        lasts = mapped["last"].fillna(prior).to_numpy(dtype=float)
        apply[f"hist_count_{suffix}"] = counts
        apply[f"hist_rate_{suffix}"] = (
            sums + smoothing * prior
        ) / (counts + smoothing)
        apply[f"hist_last_{suffix}"] = lasts

    return fit, apply


def align_feature_types(
    fit_rows: pd.DataFrame, apply_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use training-only categorical vocabularies; unseen values become missing."""
    fit = fit_rows.copy()
    apply = apply_rows.copy()
    for column in CATEGORICAL_FEATURES:
        fit_values = fit[column].fillna(MISSING).astype(str)
        categories = pd.Index(fit_values.unique())
        dtype = pd.CategoricalDtype(categories=categories)
        fit[column] = fit_values.astype(dtype)
        apply[column] = apply[column].fillna(MISSING).astype(str).astype(dtype)
    for column in NUMERIC_BASE_FEATURES + HISTORICAL_LABEL_FEATURES:
        fit[column] = pd.to_numeric(fit[column], errors="coerce")
        apply[column] = pd.to_numeric(apply[column], errors="coerce")
    return fit, apply


def fit_model(rows: pd.DataFrame, target: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=650,
        learning_rate=0.06,
        num_leaves=63,
        min_child_samples=40,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        rows[FEATURES],
        np.asarray(target, dtype=np.int8),
        categorical_feature=CATEGORICAL_FEATURES,
    )
    return model


def select_threshold(target: pd.Series, probability: np.ndarray) -> tuple[float, float]:
    best_score = -1.0
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.651, 0.005):
        score = f1_score(target, probability >= threshold, average="macro")
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def time_validation(
    base_rows: pd.DataFrame, target: pd.Series
) -> tuple[float, dict[str, float | int]]:
    """Validate on the latest two years, then return the selected threshold."""
    maximum_year = int(base_rows["year"].max())
    validation_start = maximum_year - 1
    fit_mask = base_rows["year"] < validation_start
    validation_mask = ~fit_mask
    if fit_mask.sum() == 0 or validation_mask.sum() == 0:
        raise ValueError("At least three years are needed for time validation")

    fit_rows, validation_rows = add_historical_label_features(
        base_rows.loc[fit_mask],
        target.loc[fit_mask],
        base_rows.loc[validation_mask],
    )
    fit_rows, validation_rows = align_feature_types(fit_rows, validation_rows)
    model = fit_model(fit_rows, target.loc[fit_mask])
    probability = model.predict_proba(validation_rows[FEATURES])[:, 1]
    threshold, macro_f1 = select_threshold(target.loc[validation_mask], probability)
    prediction = probability >= threshold
    positive_f1 = f1_score(target.loc[validation_mask], prediction, pos_label=1)
    metrics: dict[str, float | int] = {
        "fit_rows": int(fit_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "validation_start_year": validation_start,
        "threshold": threshold,
        "macro_f1": macro_f1,
        "positive_f1": float(positive_f1),
        "predicted_positive_rate": float(prediction.mean()),
        "actual_positive_rate": float(target.loc[validation_mask].mean()),
    }
    return threshold, metrics


def print_validation(metrics: dict[str, float | int]) -> None:
    print(
        "Time validation "
        f"(years >= {metrics['validation_start_year']}, "
        f"n={metrics['validation_rows']}):"
    )
    print(
        f"  macro F1={metrics['macro_f1']:.5f}, "
        f"Y-class F1={metrics['positive_f1']:.5f}, "
        f"threshold={metrics['threshold']:.3f}"
    )
    print(
        f"  positive rate: predicted={metrics['predicted_positive_rate']:.4f}, "
        f"actual={metrics['actual_positive_rate']:.4f}"
    )


def predict_test(
    train_base: pd.DataFrame,
    train_target: pd.Series,
    test_dir: Path,
    threshold: float,
) -> pd.DataFrame:
    """Fit on all training rows and predict test summary rows in source order."""
    test_keys_and_features = load_feature_rows(test_dir)
    test_base = engineer_base_features(test_keys_and_features)
    fit_rows, test_rows = add_historical_label_features(
        train_base, train_target, test_base
    )
    fit_rows, test_rows = align_feature_types(fit_rows, test_rows)
    model = fit_model(fit_rows, train_target)
    probability = model.predict_proba(test_rows[FEATURES])[:, 1]
    prediction = np.where(probability >= threshold, "Y", "N")
    return pd.DataFrame(
        {
            "TERM_CODE": test_keys_and_features["TERM_CODE"].to_numpy(),
            "SUBJECT_ID_SORT": test_keys_and_features["SUBJECT_ID_SORT"].to_numpy(),
            TARGET: prediction,
        }
    )


def evaluate_predictions(predictions: pd.DataFrame, gold_path: Path) -> None:
    """Read held-out gold only after prediction and report evaluation metrics."""
    gold = pd.read_csv(gold_path, dtype={TARGET: "string"})
    required = set(KEYS + [TARGET])
    if not required.issubset(gold.columns):
        raise ValueError(f"{gold_path} must contain {sorted(required)}")
    gold = gold[KEYS + [TARGET]]
    _check_unique(gold, gold_path)
    _check_unique(predictions, Path("predictions"))

    prediction_keys = pd.MultiIndex.from_frame(predictions[KEYS])
    gold_keys = pd.MultiIndex.from_frame(gold[KEYS])
    if not prediction_keys.equals(gold_keys):
        missing = len(gold_keys.difference(prediction_keys))
        extra = len(prediction_keys.difference(gold_keys))
        if missing or extra:
            raise ValueError(
                f"Prediction/gold key mismatch (gold-only={missing}, "
                f"prediction-only={extra})"
            )
    scored = predictions.merge(
        gold,
        on=KEYS,
        how="inner",
        suffixes=("_PRED", "_TRUE"),
        validate="one_to_one",
    )
    y_true = scored[f"{TARGET}_TRUE"].eq("Y")
    y_pred = scored[f"{TARGET}_PRED"].eq("Y")
    print(f"Test macro F1: {f1_score(y_true, y_pred, average='macro'):.6f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred, labels=[False, True]))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[False, True],
            target_names=["N", "Y"],
            digits=5,
            zero_division=0,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("table_splits/train"),
        help="Training table directory (default: table_splits/train)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Held-out table directory. Omit during training/validation.",
    )
    parser.add_argument(
        "--test-gold",
        type=Path,
        help="Optional gold_enrollment_test.csv, read only after prediction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Prediction CSV path (default: predictions.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test_gold is not None and args.test_dir is None:
        raise ValueError("--test-gold requires --test-dir")

    train_rows, train_target = load_training_rows(args.train_dir)
    train_base = engineer_base_features(train_rows)
    threshold, validation_metrics = time_validation(train_base, train_target)
    print_validation(validation_metrics)

    if args.test_dir is None:
        print("No --test-dir supplied; validation completed without reading test data.")
        return

    predictions = predict_test(
        train_base,
        train_target,
        args.test_dir,
        threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(
        f"Wrote {len(predictions)} predictions to {args.output} "
        f"(Y rate={predictions[TARGET].eq('Y').mean():.4f})"
    )
    if args.test_gold is not None:
        evaluate_predictions(predictions, args.test_gold)


if __name__ == "__main__":
    main()
