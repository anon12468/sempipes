#!/usr/bin/env python3
"""Train and evaluate a leakage-safe high-enrollment classifier.

Running without arguments reads only ``table_splits/train`` and reports
time-based validation macro F1. To score a future split explicitly:

    python pipeline.py \
        --test-data-dir /path/to/test \
        --gold-test /path/to/gold_enrollment_test.csv \
        --output predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

# Paths are anchored to this file, so the script can be run from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DATA_DIR = PROJECT_ROOT / "table_splits" / "train"
GOLD_TRAIN_PATH = TRAIN_DATA_DIR / "gold_enrollment_train.csv"
TEST_DATA_DIR: Path | None = None
GOLD_TEST_PATH: Path | None = None

KEY_COLS = ["TERM_CODE", "SUBJECT_ID_SORT"]
TARGET_COL = "HIGH_ENROLLMENT"
VAL_TERM_RANK_THRESHOLD = 20150  # pre-2015 fit; 2015+ validation
RANDOM_SEED = 42

OFFERED_TABLE = "SUBJECT_OFFERED.csv"
SUMMARY_TABLE = "SUBJECT_OFFERED_SUMMARY.csv"
HISTORY_TABLE = "ENROLLMENT_HISTORY_FEATURES.csv"

HISTORY_COLS = ["lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]
UNIT_COLS = ["TOTAL_UNITS", "LECTURE_UNITS", "LAB_UNITS", "PREPARATION_UNITS"]
SECTION_FLAG_COLS = [
    "IS_MASTER_SECTION",
    "IS_LECTURE_SECTION",
    "IS_LAB_SECTION",
    "IS_RECITATION_SECTION",
    "IS_DESIGN_SECTION",
    "EVALUATE_THIS_SUBJECT",
    "IS_REPEATABLE_SUBJECT",
]
SECTION_AGG_COLS = [
    "section_count",
    "master_sections",
    "lecture_sections",
    "lab_sections",
    "recitation_sections",
    "design_sections",
    "faculty_count",
    "meet_time_count",
    "meet_place_count",
    "evaluated_sections",
    "repeatable_sections",
]
CATEGORICAL_FEATURES = [
    "SUBJECT_ID_SORT",
    "SUBJECT_TITLE",
    "subject_prefix",
    "course_level",
    "term_season",
    "COURSE_NUMBER",
    "MASTER_SUBJECT_ID_SORT",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "RESPONSIBLE_FACULTY_NAME",
]
NUMERIC_FEATURES = [
    "term_year",
    "subject_number",
    *UNIT_COLS,
    *HISTORY_COLS,
    "log_lag1_enrollment",
    "log_lag2_enrollment",
    "log_roll3_enrollment_mean",
    "lag1_enrollment_dept_pct",
    "lag2_enrollment_dept_pct",
    "roll3_enrollment_mean_dept_pct",
    "history_mean",
    "history_max",
    "history_trend",
    "history_nonzero_count",
    "dept_term_size",
    *SECTION_AGG_COLS,
]


def term_to_rank(series: pd.Series) -> pd.Series:
    """Map MIT-style term codes to sortable integers."""
    text = series.astype("string").fillna("")
    year = pd.to_numeric(text.str.slice(0, 4), errors="coerce").fillna(0)
    season = text.str.slice(4, 6).map({"JA": 1, "SP": 2, "SU": 3, "FA": 4}).fillna(9)
    return year * 10 + season


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in KEY_COLS:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("").str.strip()
    return out


def read_table(data_dir: Path, name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing table: {path}")
    return normalize_keys(pd.read_csv(path, usecols=usecols, low_memory=False))


def read_gold(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing gold labels: {path}")
    gold = normalize_keys(pd.read_csv(path, dtype={c: "string" for c in KEY_COLS}))
    missing = set(KEY_COLS) - set(gold.columns)
    if missing:
        raise ValueError(f"{path} is missing key columns: {sorted(missing)}")
    if gold.duplicated(KEY_COLS).any():
        raise ValueError(f"{path} contains duplicate prediction keys")
    return gold


def aggregate_offered_sections(data_dir: Path) -> pd.DataFrame:
    """Collapse section rows to course-term scheduling features."""
    requested = KEY_COLS + SECTION_FLAG_COLS + [
        "RESPONSIBLE_FACULTY_NAME",
        "MEET_TIME",
        "MEET_PLACE",
    ]
    offered_path = data_dir / OFFERED_TABLE
    header = pd.read_csv(offered_path, nrows=0).columns
    usecols = [c for c in requested if c in header]
    offered = read_table(data_dir, OFFERED_TABLE, usecols=usecols)

    for col in SECTION_FLAG_COLS:
        if col not in offered:
            offered[col] = 0
        else:
            offered[col] = (
                offered[col]
                .astype("string")
                .str.strip()
                .str.upper()
                .eq("Y")
                .fillna(False)
                .astype("int8")
            )
    for col in ["RESPONSIBLE_FACULTY_NAME", "MEET_TIME", "MEET_PLACE"]:
        if col not in offered:
            offered[col] = pd.NA

    return (
        offered.groupby(KEY_COLS, as_index=False, dropna=False)
        .agg(
            section_count=("SUBJECT_ID_SORT", "size"),
            master_sections=("IS_MASTER_SECTION", "sum"),
            lecture_sections=("IS_LECTURE_SECTION", "sum"),
            lab_sections=("IS_LAB_SECTION", "sum"),
            recitation_sections=("IS_RECITATION_SECTION", "sum"),
            design_sections=("IS_DESIGN_SECTION", "sum"),
            faculty_count=("RESPONSIBLE_FACULTY_NAME", "nunique"),
            meet_time_count=("MEET_TIME", "nunique"),
            meet_place_count=("MEET_PLACE", "nunique"),
            evaluated_sections=("EVALUATE_THIS_SUBJECT", "sum"),
            repeatable_sections=("IS_REPEATABLE_SUBJECT", "sum"),
        )
        .reset_index(drop=True)
    )


def load_course_offerings(data_dir: Path, prediction_keys: pd.DataFrame) -> pd.DataFrame:
    """Create one feature row per requested key without current-term enrollment."""
    summary = read_table(data_dir, SUMMARY_TABLE).drop_duplicates(KEY_COLS)
    history = read_table(data_dir, HISTORY_TABLE).drop_duplicates(KEY_COLS)
    sections = aggregate_offered_sections(data_dir)

    # Starting from prediction_keys guarantees exact output coverage and order.
    df = normalize_keys(prediction_keys[KEY_COLS]).merge(
        summary, on=KEY_COLS, how="left", validate="one_to_one"
    )
    if df["OFFER_DEPT_CODE"].isna().any():
        count = int(df["OFFER_DEPT_CODE"].isna().sum())
        raise ValueError(f"{count} requested keys are missing from {SUMMARY_TABLE}")
    df = df.merge(history, on=KEY_COLS, how="left", validate="one_to_one")
    df = df.merge(sections, on=KEY_COLS, how="left", validate="one_to_one")

    df["term_rank"] = term_to_rank(df["TERM_CODE"])
    df["term_year"] = pd.to_numeric(
        df["TERM_CODE"].astype("string").str.slice(0, 4), errors="coerce"
    )
    df["term_season"] = df["TERM_CODE"].astype("string").str.slice(4, 6)
    df["subject_prefix"] = (
        df["SUBJECT_ID_SORT"].astype("string").str.split(".", n=1).str[0]
    )
    number_text = df["SUBJECT_ID_SORT"].astype("string").str.extract(r"\.(\d+)")[0]
    df["subject_number"] = pd.to_numeric(number_text, errors="coerce")
    df["course_level"] = (df["subject_number"] // 100).astype("Int64").astype("string")

    raw_numeric = [*HISTORY_COLS, *UNIT_COLS, *SECTION_AGG_COLS, "subject_number"]
    for col in raw_numeric:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in HISTORY_COLS:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
        df[f"{col}_dept_pct"] = df.groupby(
            ["TERM_CODE", "OFFER_DEPT_CODE"], dropna=False
        )[col].rank(method="average", pct=True)

    df["history_mean"] = df[HISTORY_COLS].mean(axis=1)
    df["history_max"] = df[HISTORY_COLS].max(axis=1)
    df["history_trend"] = df["lag1_enrollment"] - df["lag2_enrollment"]
    df["history_nonzero_count"] = df[HISTORY_COLS].gt(0).sum(axis=1)
    df["dept_term_size"] = df.groupby(
        ["TERM_CODE", "OFFER_DEPT_CODE"], dropna=False
    )["SUBJECT_ID_SORT"].transform("size")

    return df.reset_index(drop=True)


def build_labeled_frame(data_dir: Path, gold: pd.DataFrame) -> pd.DataFrame:
    required = set(KEY_COLS + [TARGET_COL])
    if not required.issubset(gold.columns):
        raise ValueError(f"Training gold must contain {sorted(required)}")
    labeled = load_course_offerings(data_dir, gold[KEY_COLS])
    labeled[TARGET_COL] = gold[TARGET_COL].astype("string").str.strip().str.upper()
    if not labeled[TARGET_COL].isin(["Y", "N"]).all():
        raise ValueError("Training labels must contain only Y or N")
    labeled["y"] = labeled[TARGET_COL].eq("Y").astype("int8")
    return labeled


def prepare_features(
    train_df: pd.DataFrame, predict_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str], list[str]]:
    feature_cols = [
        c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c in train_df.columns
    ]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    train_x = train_df[feature_cols].copy()
    predict_x = predict_df[feature_cols].copy() if predict_df is not None else None

    for col in cat_cols:
        train_x[col] = train_x[col].astype("string").fillna("__MISSING__").astype(str)
        if predict_x is not None:
            predict_x[col] = (
                predict_x[col].astype("string").fillna("__MISSING__").astype(str)
            )
    for col in (c for c in feature_cols if c not in cat_cols):
        train_x[col] = pd.to_numeric(train_x[col], errors="coerce").fillna(0.0)
        if predict_x is not None:
            predict_x[col] = pd.to_numeric(
                predict_x[col], errors="coerce"
            ).fillna(0.0)
    return train_x, predict_x, feature_cols, cat_cols


def make_model(iterations: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=8,
        learning_rate=0.10,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def latest_subject_labels(reference: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Return each target course's most recent known high-enrollment status."""
    latest = (
        reference.sort_values(["term_rank", "TERM_CODE"])
        .drop_duplicates("SUBJECT_ID_SORT", keep="last")
        .set_index("SUBJECT_ID_SORT")["y"]
    )
    return (
        target["SUBJECT_ID_SORT"].map(latest).fillna(0.0).astype(float).to_numpy()
    )


def tune_decision_rule(
    y_true: pd.Series, probability: np.ndarray, prior_label: np.ndarray
) -> tuple[float, float, float]:
    """Tune a small history blend and threshold for validation macro F1."""
    best_f1, best_threshold, best_blend = -1.0, 0.5, 0.0
    y = y_true.to_numpy()
    for blend in np.linspace(0.0, 0.40, 9):
        score = (1.0 - blend) * probability + blend * prior_label
        thresholds = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 501)))
        for threshold in thresholds:
            pred = score >= threshold
            metric = f1_score(y, pred, average="macro", zero_division=0)
            if metric > best_f1:
                best_f1 = float(metric)
                best_threshold = float(threshold)
                best_blend = float(blend)
    return best_threshold, best_blend, best_f1


def fit_validation_model(
    train_df: pd.DataFrame, max_iterations: int
) -> tuple[float, float, float, int]:
    fit_mask = train_df["term_rank"] < VAL_TERM_RANK_THRESHOLD
    if not fit_mask.any() or fit_mask.all():
        raise ValueError("Time split produced an empty fit or validation partition")

    x, _, feature_cols, cat_cols = prepare_features(train_df)
    model = make_model(max_iterations)
    model.fit(
        x.loc[fit_mask, feature_cols],
        train_df.loc[fit_mask, "y"],
        cat_features=cat_cols,
        eval_set=(x.loc[~fit_mask, feature_cols], train_df.loc[~fit_mask, "y"]),
        early_stopping_rounds=70,
    )
    probability = model.predict_proba(x.loc[~fit_mask, feature_cols])[:, 1]
    prior_label = latest_subject_labels(
        train_df.loc[fit_mask], train_df.loc[~fit_mask]
    )
    threshold, blend, val_f1 = tune_decision_rule(
        train_df.loc[~fit_mask, "y"], probability, prior_label
    )
    iterations = max(1, int(model.tree_count_))
    return threshold, blend, val_f1, iterations


def fit_full_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    threshold: float,
    history_blend: float,
    iterations: int,
) -> pd.Series:
    train_x, predict_x, feature_cols, cat_cols = prepare_features(
        train_df, predict_df
    )
    assert predict_x is not None
    model = make_model(iterations)
    model.fit(
        train_x[feature_cols],
        train_df["y"],
        cat_features=cat_cols,
    )
    probability = model.predict_proba(predict_x[feature_cols])[:, 1]
    prior_label = latest_subject_labels(train_df, predict_df)
    score = (1.0 - history_blend) * probability + history_blend * prior_label
    return pd.Series((score >= threshold).astype("int8"), index=predict_df.index)


def submission_frame(predict_df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    out = predict_df[KEY_COLS].copy()
    out[TARGET_COL] = pred.map({1: "Y", 0: "N"})
    if out[TARGET_COL].isna().any() or out.duplicated(KEY_COLS).any():
        raise ValueError("Predictions are incomplete or contain duplicate keys")
    return out


def evaluate_macro_f1(gold: pd.DataFrame, pred: pd.DataFrame) -> float:
    gold = normalize_keys(gold)
    pred = normalize_keys(pred)
    labeled_gold = gold[
        gold[TARGET_COL].astype("string").str.strip().str.upper().isin(["Y", "N"])
    ].copy()
    labeled_gold[TARGET_COL] = (
        labeled_gold[TARGET_COL].astype("string").str.strip().str.upper()
    )
    pred[TARGET_COL] = pred[TARGET_COL].astype("string").str.strip().str.upper()
    merged = labeled_gold.merge(
        pred,
        on=KEY_COLS,
        how="left",
        suffixes=("_gold", "_pred"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Test gold contains no Y/N labels to evaluate")
    if not merged[f"{TARGET_COL}_pred"].isin(["Y", "N"]).all():
        raise ValueError("A prediction is missing for at least one gold key")
    y_true = merged[f"{TARGET_COL}_gold"].eq("Y").astype(int)
    y_pred = merged[f"{TARGET_COL}_pred"].eq("Y").astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data-dir", type=Path, default=TRAIN_DATA_DIR)
    parser.add_argument("--gold-train", type=Path, default=GOLD_TRAIN_PATH)
    parser.add_argument("--test-data-dir", type=Path, default=TEST_DATA_DIR)
    parser.add_argument("--gold-test", type=Path, default=GOLD_TEST_PATH)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "predictions.csv")
    parser.add_argument("--max-iterations", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.test_data_dir is None) != (args.gold_test is None):
        raise ValueError("--test-data-dir and --gold-test must be supplied together")

    train_gold = read_gold(args.gold_train)
    train_df = build_labeled_frame(args.train_data_dir, train_gold)
    threshold, blend, val_f1, iterations = fit_validation_model(
        train_df, args.max_iterations
    )
    print(
        f"Val macro F1: {val_f1:.4f} "
        f"(threshold={threshold:.4f}, history_blend={blend:.2f}, "
        f"iterations={iterations})"
    )

    # The default validation-only path never touches a held-out directory.
    if args.test_data_dir is None:
        return

    test_gold = read_gold(args.gold_test)
    predict_df = load_course_offerings(args.test_data_dir, test_gold[KEY_COLS])
    test_pred = fit_full_and_predict(
        train_df, predict_df, threshold, blend, iterations
    )
    submission = submission_frame(predict_df, test_pred)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output}")

    if TARGET_COL in test_gold.columns:
        test_f1 = evaluate_macro_f1(test_gold, submission)
        print(f"Test macro F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()

