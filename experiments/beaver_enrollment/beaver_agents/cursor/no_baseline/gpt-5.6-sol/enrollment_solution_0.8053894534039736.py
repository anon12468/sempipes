#!/usr/bin/env python3
"""Leakage-safe high-enrollment training, prediction, and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


KEYS = ["TERM_CODE", "SUBJECT_ID_SORT"]
TARGET = "HIGH_ENROLLMENT"
SEASON_ORDER = {"FA": 0, "JA": 1, "10": 1, "SP": 2, "SU": 3}

CATEGORICAL_FEATURES = [
    "season",
    "subject_prefix",
    "subject_number",
    "subject_level",
    "COURSE_NUMBER",
    "SUBJECT_TITLE",
    "MASTER_SUBJECT_ID_SORT",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "RESPONSIBLE_FACULTY_NAME",
]

NUMERIC_FEATURES = [
    "year",
    "subject_number_numeric",
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
    "title_len",
    "lag1_enrollment",
    "lag2_enrollment",
    "roll3_enrollment_mean",
    "lag1_enrollment_pct_dept",
    "lag2_enrollment_pct_dept",
    "roll3_enrollment_mean_pct_dept",
    "lag1_enrollment_log",
    "lag2_enrollment_log",
    "roll3_enrollment_mean_log",
    "dept_offering_count",
    "course_prev_y",
    "course_prior_count",
    "course_prior_rate",
    "course_same_season_prev_y",
    "section_rows",
    "master_section_count",
    "lecture_section_count",
    "lab_section_count",
    "recitation_section_count",
    "design_section_count",
    "repeatable_section_count",
    "has_meet_time",
    "has_meet_place",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

SECTION_FLAGS = {
    "IS_MASTER_SECTION": "master_section_count",
    "IS_LECTURE_SECTION": "lecture_section_count",
    "IS_LAB_SECTION": "lab_section_count",
    "IS_RECITATION_SECTION": "recitation_section_count",
    "IS_DESIGN_SECTION": "design_section_count",
    "IS_REPEATABLE_SUBJECT": "repeatable_section_count",
}


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def _normalize_target(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.upper()
    mapping = {
        "Y": 1,
        "YES": 1,
        "TRUE": 1,
        "1": 1,
        "1.0": 1,
        "N": 0,
        "NO": 0,
        "FALSE": 0,
        "0": 0,
        "0.0": 0,
    }
    result = normalized.map(mapping)
    if result.isna().any():
        bad = sorted(normalized[result.isna()].unique().tolist())
        raise ValueError(f"Unsupported target values: {bad}")
    return result.astype("int8")


def _term_parts(term: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    text = term.astype(str)
    year = pd.to_numeric(text.str[:4], errors="coerce").fillna(0).astype("int16")
    season = text.str[4:].fillna("")
    order = year.astype("int32") * 10 + season.map(SEASON_ORDER).fillna(4).astype("int32")
    return year, season, order


def _read_keys(path: Path) -> pd.DataFrame:
    _require_file(path)
    keys = pd.read_csv(path, usecols=KEYS, dtype=str)
    if keys.duplicated(KEYS).any():
        raise ValueError(f"Duplicate prediction keys in {path}")
    return keys


def _load_section_features(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "SUBJECT_OFFERED.csv"
    if not path.is_file():
        return pd.DataFrame(columns=KEYS + list(SECTION_FLAGS.values()) + [
            "section_rows",
            "has_meet_time",
            "has_meet_place",
        ])

    columns = KEYS + list(SECTION_FLAGS) + ["MEET_TIME", "MEET_PLACE"]
    sections = pd.read_csv(path, usecols=columns, dtype=str, low_memory=False)
    for source, output in SECTION_FLAGS.items():
        sections[output] = sections[source].eq("Y").astype("int8")
    sections["section_rows"] = 1
    sections["has_meet_time"] = sections["MEET_TIME"].notna().astype("int8")
    sections["has_meet_place"] = sections["MEET_PLACE"].notna().astype("int8")
    aggregate_columns = [
        "section_rows",
        *SECTION_FLAGS.values(),
        "has_meet_time",
        "has_meet_place",
    ]
    return sections.groupby(KEYS, as_index=False, sort=False)[aggregate_columns].sum()


def load_base_frame(data_dir: Path, keys: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load only features available for a future offering."""

    summary_path = data_dir / "SUBJECT_OFFERED_SUMMARY.csv"
    history_path = data_dir / "ENROLLMENT_HISTORY_FEATURES.csv"
    _require_file(summary_path)
    _require_file(history_path)

    summary = pd.read_csv(summary_path, dtype={key: str for key in KEYS}, low_memory=False)
    history = pd.read_csv(history_path, dtype={key: str for key in KEYS}, low_memory=False)
    if summary.duplicated(KEYS).any() or history.duplicated(KEYS).any():
        raise ValueError("Feature tables must contain one row per prediction key")

    if keys is None:
        keys = summary[KEYS].copy()
    frame = keys.merge(summary, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not frame["_merge"].eq("both").all():
        missing = frame.loc[frame["_merge"].ne("both"), KEYS].head().to_dict("records")
        raise ValueError(f"Keys missing from SUBJECT_OFFERED_SUMMARY.csv: {missing}")
    frame = frame.drop(columns="_merge")
    frame = frame.merge(history, on=KEYS, how="left", validate="one_to_one")
    frame = frame.merge(_load_section_features(data_dir), on=KEYS, how="left")

    frame["year"], frame["season"], frame["term_order"] = _term_parts(frame["TERM_CODE"])
    split_subject = frame["SUBJECT_ID_SORT"].str.split(".", n=1, expand=True)
    frame["subject_prefix"] = split_subject[0].fillna("")
    frame["subject_number"] = split_subject[1].fillna("") if split_subject.shape[1] > 1 else ""
    frame["subject_number_numeric"] = pd.to_numeric(
        frame["subject_number"].str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0],
        errors="coerce",
    )
    frame["subject_level"] = frame["subject_number"].str.extract(r"(\d)")[0].fillna("X")
    frame["title_len"] = frame["SUBJECT_TITLE"].fillna("").str.len()

    enrollment_columns = ["lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]
    for column in enrollment_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        frame[f"{column}_pct_dept"] = (
            frame.groupby(["TERM_CODE", "OFFER_DEPT_CODE"], dropna=False)[column]
            .rank(pct=True, method="average")
            .fillna(0.0)
        )
        frame[f"{column}_log"] = np.log1p(frame[column].clip(lower=0))

    frame["dept_offering_count"] = frame.groupby(
        ["TERM_CODE", "OFFER_DEPT_CODE"], dropna=False
    )["SUBJECT_ID_SORT"].transform("size")
    return frame


def make_label_history(frame: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    history = frame[["SUBJECT_ID_SORT", "season", "term_order", "TERM_CODE"]].copy()
    history["y"] = np.asarray(y, dtype="int8")
    history = history.sort_values(["term_order", "TERM_CODE", "SUBJECT_ID_SORT"])

    by_course = history.groupby("SUBJECT_ID_SORT", sort=False)["y"].agg(
        course_prev_y="last",
        course_prior_count="size",
        course_prior_rate="mean",
    )
    by_course_season = history.groupby(
        ["SUBJECT_ID_SORT", "season"], sort=False
    )["y"].last()
    return {
        "by_course": by_course,
        "by_course_season": by_course_season,
    }


def apply_label_history(frame: pd.DataFrame, history: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    history_columns = [
        "course_prev_y",
        "course_prior_count",
        "course_prior_rate",
        "course_same_season_prev_y",
    ]
    result = result.drop(columns=history_columns, errors="ignore")
    result = result.join(history["by_course"], on="SUBJECT_ID_SORT")
    same_season = pd.MultiIndex.from_frame(result[["SUBJECT_ID_SORT", "season"]])
    result["course_same_season_prev_y"] = history["by_course_season"].reindex(
        same_season
    ).to_numpy()
    defaults = {
        "course_prev_y": -1,
        "course_prior_count": 0,
        "course_prior_rate": -1,
        "course_same_season_prev_y": -1,
    }
    for column, default in defaults.items():
        result[column] = result[column].fillna(default)
    return result


def add_expanding_label_history(frame: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Add strictly prior target statistics to model-training rows."""

    result = frame.copy()
    result["_row_id"] = np.arange(len(result))
    result["_y"] = np.asarray(y, dtype="int8")
    result = result.sort_values(["term_order", "TERM_CODE", "SUBJECT_ID_SORT"])

    course_group = result.groupby("SUBJECT_ID_SORT", sort=False)["_y"]
    result["course_prev_y"] = course_group.shift()
    result["course_prior_count"] = result.groupby(
        "SUBJECT_ID_SORT", sort=False
    ).cumcount()
    numerator = course_group.cumsum() - result["_y"]
    denominator = result["course_prior_count"].replace(0, np.nan)
    result["course_prior_rate"] = numerator / denominator
    result["course_same_season_prev_y"] = result.groupby(
        ["SUBJECT_ID_SORT", "season"], sort=False
    )["_y"].shift()

    result = result.sort_values("_row_id").drop(columns=["_row_id", "_y"])
    defaults = {
        "course_prev_y": -1,
        "course_prior_count": 0,
        "course_prior_rate": -1,
        "course_same_season_prev_y": -1,
    }
    for column, default in defaults.items():
        result[column] = result[column].fillna(default)
    return result


def prepare_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame.copy()
    for column in FEATURES:
        if column not in matrix:
            matrix[column] = np.nan
    for column in CATEGORICAL_FEATURES:
        matrix[column] = matrix[column].fillna("__MISSING__").astype("category")
    for column in NUMERIC_FEATURES:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").fillna(0.0)
    return matrix[FEATURES]


def _new_model(n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=48,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def _best_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.60, 111)
    scores = np.array(
        [f1_score(y_true, probability >= threshold, average="macro") for threshold in candidates]
    )
    best = int(scores.argmax())
    return float(candidates[best]), float(scores[best])


def validate(
    train_dir: Path,
    validation_years: int = 2,
    n_estimators: int = 800,
) -> dict[str, Any]:
    gold_path = train_dir / "gold_enrollment_train.csv"
    _require_file(gold_path)
    gold = pd.read_csv(gold_path, dtype={key: str for key in KEYS})
    y = _normalize_target(gold[TARGET])
    frame = load_base_frame(train_dir, gold[KEYS])
    frame = add_expanding_label_history(frame, y)

    available_years = sorted(year for year in frame["year"].unique() if year > 0)
    if validation_years < 1 or validation_years >= len(available_years):
        raise ValueError("validation_years must leave at least one earlier training year")
    holdout_years = [int(year) for year in available_years[-validation_years:]]
    is_valid = frame["year"].isin(holdout_years)

    # Simulate a real temporal cutoff: held-out rows may use labels only from earlier years.
    cutoff_history = make_label_history(frame.loc[~is_valid], y.loc[~is_valid])
    validation_frame = apply_label_history(frame.loc[is_valid], cutoff_history)
    frame.loc[is_valid, [
        "course_prev_y",
        "course_prior_count",
        "course_prior_rate",
        "course_same_season_prev_y",
    ]] = validation_frame[[
        "course_prev_y",
        "course_prior_count",
        "course_prior_rate",
        "course_same_season_prev_y",
    ]]

    x = prepare_matrix(frame)
    model = _new_model(n_estimators)
    model.fit(
        x.loc[~is_valid],
        y.loc[~is_valid],
        categorical_feature=CATEGORICAL_FEATURES,
        eval_set=[(x.loc[is_valid], y.loc[is_valid])],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    probability = model.predict_proba(x.loc[is_valid])[:, 1]
    threshold, macro_f1 = _best_threshold(y.loc[is_valid], probability)
    prediction = probability >= threshold
    negative_f1, positive_f1 = f1_score(
        y.loc[is_valid], prediction, average=None, labels=[0, 1]
    )
    tn, fp, fn, tp = confusion_matrix(y.loc[is_valid], prediction, labels=[0, 1]).ravel()
    return {
        "validation_years": holdout_years,
        "training_rows": int((~is_valid).sum()),
        "validation_rows": int(is_valid.sum()),
        "validation_positive_rate": float(y.loc[is_valid].mean()),
        "best_iteration": int(model.best_iteration_ or n_estimators),
        "threshold": threshold,
        "macro_f1": macro_f1,
        "negative_f1": float(negative_f1),
        "positive_f1": float(positive_f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train(
    train_dir: Path,
    model_path: Path,
    validation_years: int = 2,
    n_estimators: int = 800,
) -> dict[str, Any]:
    """Validate, refit on all training rows, and save the model artifact."""

    report = validate(train_dir, validation_years, n_estimators)
    gold = pd.read_csv(
        train_dir / "gold_enrollment_train.csv",
        dtype={key: str for key in KEYS},
    )
    y = _normalize_target(gold[TARGET])
    base_frame = load_base_frame(train_dir, gold[KEYS])
    train_frame = add_expanding_label_history(base_frame, y)
    matrix = prepare_matrix(train_frame)

    final_model = _new_model(report["best_iteration"])
    final_model.fit(matrix, y, categorical_feature=CATEGORICAL_FEATURES)
    artifact = {
        "model": final_model,
        "threshold": report["threshold"],
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "label_history": make_label_history(base_frame, y),
        "validation_report": report,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    report["model_path"] = str(model_path)
    report["final_training_rows"] = len(gold)
    return report


def predict(
    model_path: Path,
    test_dir: Path,
    output_path: Path,
    keys_path: Path | None = None,
) -> pd.DataFrame:
    """Predict without reading the target column from any test gold file."""

    artifact = joblib.load(model_path)
    keys = _read_keys(keys_path) if keys_path else None
    frame = load_base_frame(test_dir, keys)
    frame = apply_label_history(frame, artifact["label_history"])
    probability = artifact["model"].predict_proba(prepare_matrix(frame))[:, 1]
    prediction = np.where(probability >= artifact["threshold"], "Y", "N")

    result = frame[KEYS].copy()
    result[TARGET] = prediction
    result["HIGH_ENROLLMENT_PROBABILITY"] = probability
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def evaluate(predictions_path: Path, gold_path: Path) -> dict[str, Any]:
    predictions = pd.read_csv(predictions_path, dtype={key: str for key in KEYS})
    gold = pd.read_csv(gold_path, dtype={key: str for key in KEYS})
    for name, table in [("predictions", predictions), ("gold", gold)]:
        if table.duplicated(KEYS).any():
            raise ValueError(f"Duplicate keys in {name}")

    joined = gold[KEYS + [TARGET]].merge(
        predictions[KEYS + [TARGET]],
        on=KEYS,
        how="outer",
        suffixes=("_gold", "_prediction"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        counts = joined["_merge"].value_counts().to_dict()
        raise ValueError(f"Prediction/gold key mismatch: {counts}")

    y_true = _normalize_target(joined[f"{TARGET}_gold"])
    y_pred = _normalize_target(joined[f"{TARGET}_prediction"])
    negative_f1, positive_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "rows": len(joined),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "negative_f1": float(negative_f1),
        "positive_f1": float(positive_f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Run time-based validation")
    validate_parser.add_argument("--train-dir", type=_path, required=True)
    validate_parser.add_argument("--validation-years", type=int, default=2)
    validate_parser.add_argument("--n-estimators", type=int, default=800)

    train_parser = subparsers.add_parser("train", help="Validate and save a fitted model")
    train_parser.add_argument("--train-dir", type=_path, required=True)
    train_parser.add_argument("--model-path", type=_path, required=True)
    train_parser.add_argument("--validation-years", type=int, default=2)
    train_parser.add_argument("--n-estimators", type=int, default=800)

    predict_parser = subparsers.add_parser("predict", help="Predict test summary rows")
    predict_parser.add_argument("--model-path", type=_path, required=True)
    predict_parser.add_argument("--test-dir", type=_path, required=True)
    predict_parser.add_argument("--output", type=_path, required=True)
    predict_parser.add_argument(
        "--keys-file",
        type=_path,
        help="Optional CSV whose TERM_CODE/SUBJECT_ID_SORT columns define prediction rows",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate saved predictions")
    evaluate_parser.add_argument("--predictions", type=_path, required=True)
    evaluate_parser.add_argument("--gold", type=_path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        report = validate(args.train_dir, args.validation_years, args.n_estimators)
    elif args.command == "train":
        report = train(
            args.train_dir,
            args.model_path,
            args.validation_years,
            args.n_estimators,
        )
    elif args.command == "predict":
        result = predict(args.model_path, args.test_dir, args.output, args.keys_file)
        report = {
            "rows": len(result),
            "positive_predictions": int(result[TARGET].eq("Y").sum()),
            "output": str(args.output),
        }
    else:
        report = evaluate(args.predictions, args.gold)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
