#!/usr/bin/env python3
"""
Self-contained enrollment forecasting pipeline.

With no arguments, trains only on table_splits/train and prints temporal
validation macro F1. To predict and evaluate a future split:

  python pipeline.py --test-data-dir table_splits/test \
    --gold-test-path table_splits/test/gold_enrollment_test.csv

Paths resolve from this file's location, so the script can run from any cwd.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# --- Paths (anchored to repo root via this file) ---
PROJECT_ROOT = Path(__file__).resolve().parents[0]
TRAIN_DATA_DIR = PROJECT_ROOT / "table_splits" / "train"
TEST_DATA_DIR = None
GOLD_TRAIN_PATH = PROJECT_ROOT / "table_splits" / "train" / "gold_enrollment_train.csv"
GOLD_TEST_PATH = None
VAL_TERM_RANK_THRESHOLD = 20150  # terms before 2015 = fit; 2015+ within train = validation

KEY_COLS = ["TERM_CODE", "SUBJECT_ID_SORT"]
TARGET_COL = "HIGH_ENROLLMENT"
LEAKAGE_COLUMNS = {
    TARGET_COL,
    "NUM_ENROLLED_STUDENTS",
    "SUBJECT_ENROLLMENT_NUMBER",
    "SECTION_ENROLLMENT_NUMBER",
    "CLUSTER_ENROLLMENT_NUMBER",
    "y",
}
OFFERED_TABLE = "SUBJECT_OFFERED.csv"
SUMMARY_TABLE = "SUBJECT_OFFERED_SUMMARY.csv"
HISTORY_TABLE = "ENROLLMENT_HISTORY_FEATURES.csv"

OFFERED_COLUMNS = [
    *KEY_COLS,
    "SUBJECT_TITLE",
    "MASTER_SUBJECT_ID_SORT",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "IS_REPEATABLE_SUBJECT",
    "EVALUATE_THIS_SUBJECT",
]
SUMMARY_COLUMNS = [
    *KEY_COLS,
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
]
HISTORY_COLUMNS = [
    *KEY_COLS,
    "lag1_enrollment",
    "lag2_enrollment",
    "roll3_enrollment_mean",
]


# --- Data helpers (inlined so this script has no local imports) ---

def term_to_rank(series: pd.Series) -> pd.Series:
    year = pd.to_numeric(series.astype(str).str.slice(0, 4), errors="coerce").fillna(0)
    season = series.astype(str).str.slice(4, 6).map({"JA": 1, "SP": 2, "SU": 3, "FA": 4}).fillna(9)
    return year * 10 + season


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in KEY_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    return out


def read_table(
    data_dir: Path,
    name: str,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    selected = None if usecols is None else lambda col: col in set(usecols)
    return normalize_keys(pd.read_csv(path, usecols=selected, low_memory=False))


def load_course_offerings(data_dir: Path) -> pd.DataFrame:
    """Load leakage-safe offering data and construct course/group features."""
    offered = read_table(data_dir, OFFERED_TABLE, OFFERED_COLUMNS)
    missing_keys = [col for col in KEY_COLS if col not in offered.columns]
    if missing_keys:
        raise ValueError(f"{OFFERED_TABLE} is missing key columns: {missing_keys}")
    df = offered.drop_duplicates(KEY_COLS).copy()

    if (data_dir / SUMMARY_TABLE).exists():
        summary = read_table(data_dir, SUMMARY_TABLE, SUMMARY_COLUMNS)
        extra = [col for col in SUMMARY_COLUMNS if col in summary.columns]
        summary = summary[extra].drop_duplicates(KEY_COLS)
        df = df.merge(summary, on=KEY_COLS, how="left", validate="one_to_one")

    history_path = data_dir / HISTORY_TABLE
    if history_path.exists():
        history = read_table(data_dir, HISTORY_TABLE, HISTORY_COLUMNS)
        history = history.drop_duplicates(KEY_COLS)
        df = df.merge(history, on=KEY_COLS, how="left", validate="one_to_one")

    optional_columns = set(OFFERED_COLUMNS + SUMMARY_COLUMNS + HISTORY_COLUMNS) - set(KEY_COLS)
    for col in optional_columns:
        if col not in df.columns:
            df[col] = pd.NA

    df["term_rank"] = term_to_rank(df["TERM_CODE"])
    df["term_year"] = pd.to_numeric(df["TERM_CODE"].astype(str).str[:4], errors="coerce")
    df["term_season"] = df["TERM_CODE"].astype(str).str[4:6]

    numeric_source_cols = [
        "lag1_enrollment",
        "lag2_enrollment",
        "roll3_enrollment_mean",
        "TOTAL_UNITS",
        "LECTURE_UNITS",
        "LAB_UNITS",
        "PREPARATION_UNITS",
    ]
    for col in numeric_source_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    history_cols = ["lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]
    df[history_cols] = df[history_cols].fillna(0.0)

    # Enrollment history summaries are more robust than any one lag.
    df["history_max"] = df[history_cols].max(axis=1)
    df["history_min"] = df[history_cols].min(axis=1)
    df["history_nonzero_count"] = (df[history_cols] > 0).sum(axis=1)
    df["history_trend"] = df["lag1_enrollment"] - df["lag2_enrollment"]
    df["history_vs_roll3"] = df["lag1_enrollment"] - df["roll3_enrollment_mean"]

    # Extract reusable structure from course identifiers without treating each
    # high-cardinality identifier as an arbitrary ordered number.
    subject_parts = df["SUBJECT_ID_SORT"].astype(str).str.split(".", n=1, expand=True)
    df["subject_prefix"] = subject_parts[0]
    subject_suffix = subject_parts[1] if subject_parts.shape[1] > 1 else pd.Series("", index=df.index)
    first_digits = subject_suffix.fillna("").str.extract(r"(\d+)", expand=False)
    df["subject_level"] = pd.to_numeric(first_digits.str[:2], errors="coerce")
    df["subject_has_letter"] = subject_suffix.fillna("").str.contains(
        r"[A-Za-z]", regex=True
    ).astype("int8")
    titles = df["SUBJECT_TITLE"].fillna("").astype(str)
    df["title_length"] = titles.str.len()
    df["title_word_count"] = titles.str.count(r"\S+")

    # The target is relative to other courses in a department and term. These
    # features expose the analogous relative position of historical demand.
    dept_group = [df["TERM_CODE"], df["OFFER_DEPT_CODE"].fillna("__MISSING__")]
    school_group = [df["TERM_CODE"], df["OFFER_SCHOOL_NAME"].fillna("__MISSING__")]
    for col in ["roll3_enrollment_mean", "history_max"]:
        df[f"{col}_dept_pct"] = df[col].groupby(dept_group).rank(pct=True)
        df[f"{col}_school_pct"] = df[col].groupby(school_group).rank(pct=True)
    df["dept_term_size"] = df.groupby(
        ["TERM_CODE", "OFFER_DEPT_CODE"], dropna=False
    )["SUBJECT_ID_SORT"].transform("size")
    df["dept_history_positive_count"] = (
        df["roll3_enrollment_mean"]
        .gt(0)
        .groupby(dept_group)
        .transform("sum")
    )
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in [
            "lag1_enrollment",
            "lag2_enrollment",
            "roll3_enrollment_mean",
            "history_max",
            "history_min",
            "history_nonzero_count",
            "history_trend",
            "history_vs_roll3",
            "roll3_enrollment_mean_dept_pct",
            "roll3_enrollment_mean_school_pct",
            "history_max_dept_pct",
            "history_max_school_pct",
            "dept_term_size",
            "dept_history_positive_count",
            "OFFER_DEPT_CODE",
            "OFFER_SCHOOL_NAME",
            "subject_prefix",
            "subject_level",
            "subject_has_letter",
            "term_year",
            "term_season",
            "TOTAL_UNITS",
            "LECTURE_UNITS",
            "LAB_UNITS",
            "PREPARATION_UNITS",
            "CLUSTER_TYPE",
            "HGN_CODE",
            "IS_REPEATABLE_SUBJECT",
            "EVALUATE_THIS_SUBJECT",
            "title_length",
            "title_word_count",
        ]
        if c in df.columns and c not in LEAKAGE_COLUMNS
    ]
    return cols


def labels_to_y(series: pd.Series) -> pd.Series:
    return (series.astype(str) == "Y").astype(int)


def read_gold(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing gold labels: {path}")
    return normalize_keys(pd.read_csv(path))


def build_labeled_frame(data_dir: Path, gold: pd.DataFrame) -> pd.DataFrame:
    courses = load_course_offerings(data_dir)
    if gold.duplicated(KEY_COLS).any():
        raise ValueError("Gold labels contain duplicate prediction keys.")
    labeled = gold[KEY_COLS + [TARGET_COL]].merge(
        courses, on=KEY_COLS, how="left", validate="one_to_one"
    )
    labeled["y"] = labels_to_y(labeled[TARGET_COL]).astype(int)
    labeled["term_rank"] = term_to_rank(labeled["TERM_CODE"])
    return labeled


def build_prediction_frame(data_dir: Path, keys: pd.DataFrame) -> pd.DataFrame:
    """Return exactly one feature row per prediction key, preserving key order."""
    if keys.duplicated(KEY_COLS).any():
        raise ValueError("Prediction keys contain duplicates.")
    courses = load_course_offerings(data_dir)
    return keys[KEY_COLS].merge(
        courses, on=KEY_COLS, how="left", validate="one_to_one", sort=False
    )


def evaluate_macro_f1(gold: pd.DataFrame, pred: pd.DataFrame) -> float:
    gold = normalize_keys(gold.copy())
    pred = normalize_keys(pred.copy())
    gold[TARGET_COL] = gold[TARGET_COL].astype(str).str.strip().str.upper()
    pred[TARGET_COL] = pred[TARGET_COL].astype(str).str.strip().str.upper()

    if gold.duplicated(KEY_COLS).any() or pred.duplicated(KEY_COLS).any():
        raise ValueError("Gold and predictions must have unique keys.")
    gold = gold[gold[TARGET_COL].isin(["Y", "N"])]
    merged = gold.merge(
        pred, on=KEY_COLS, how="left", suffixes=("_gold", "_pred"), validate="one_to_one"
    )
    pred_col = f"{TARGET_COL}_pred"
    if len(merged) != len(pred) or not merged[pred_col].isin(["Y", "N"]).all():
        raise ValueError("Predictions must contain one valid Y/N row per gold key.")

    y_true = (merged[f"{TARGET_COL}_gold"] == "Y").astype(int)
    y_pred = (merged[pred_col] == "Y").astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# --- Model ---

def make_model(x_train: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(x_train[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols)
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_cols,
            )
        )

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=42,
                    max_iter=300,
                    max_depth=10,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                ),
            ),
        ]
    )


def fit_threshold(y_true: pd.Series, prob_val: pd.Series) -> float:
    """Choose the probability cutoff that maximizes validation macro F1."""
    if len(prob_val) == 0 or y_true.nunique() < 2:
        return 0.5

    y_sorted = y_true.to_numpy()[np.argsort(-prob_val.to_numpy(), kind="stable")]
    prob_sorted = np.sort(prob_val.to_numpy())[::-1]
    # Evaluate only at the end of each tie: p >= threshold then includes the
    # complete tie and exactly corresponds to the confusion counts below.
    tie_ends = np.r_[prob_sorted[:-1] != prob_sorted[1:], True]
    predicted_positive = np.arange(1, len(y_sorted) + 1)[tie_ends]
    true_positive = np.cumsum(y_sorted)[tie_ends]
    false_positive = predicted_positive - true_positive
    total_positive = y_sorted.sum()
    total_negative = len(y_sorted) - total_positive
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive

    positive_denominator = 2 * true_positive + false_positive + false_negative
    negative_denominator = 2 * true_negative + false_positive + false_negative
    positive_f1 = np.divide(
        2 * true_positive,
        positive_denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=positive_denominator != 0,
    )
    negative_f1 = np.divide(
        2 * true_negative,
        negative_denominator,
        out=np.zeros_like(true_negative, dtype=float),
        where=negative_denominator != 0,
    )
    macro_f1 = (positive_f1 + negative_f1) / 2
    return float(prob_sorted[tie_ends][int(np.argmax(macro_f1))])


def train_validate_refit(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Pipeline, float, float | None]:
    """Tune on future terms, then refit on every available training row."""
    fit_mask = train_df["term_rank"] < VAL_TERM_RANK_THRESHOLD
    val_mask = ~fit_mask
    if not fit_mask.any():
        raise ValueError("The validation cutoff leaves no rows for model fitting.")

    validation_model = make_model(train_df.loc[fit_mask], feature_cols)
    validation_model.fit(
        train_df.loc[fit_mask, feature_cols],
        train_df.loc[fit_mask, "y"],
    )

    if val_mask.any():
        y_val = train_df.loc[val_mask, "y"]
        prob_val = pd.Series(
            validation_model.predict_proba(train_df.loc[val_mask, feature_cols])[:, 1],
            index=train_df.index[val_mask],
        )
        threshold = fit_threshold(y_val, prob_val)
        val_pred = (prob_val >= threshold).astype(int)
        val_macro_f1 = float(
            f1_score(y_val, val_pred, average="macro", zero_division=0)
        )
    else:
        threshold = 0.5
        val_macro_f1 = None

    final_model = make_model(train_df, feature_cols)
    final_model.fit(train_df[feature_cols], train_df["y"])
    return final_model, threshold, val_macro_f1


def train_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.Series, float, float | None]:
    model, threshold, val_macro_f1 = train_validate_refit(train_df, feature_cols)
    for col in feature_cols:
        if col not in predict_df.columns:
            predict_df[col] = pd.NA
    prob_pred = pd.Series(model.predict_proba(predict_df[feature_cols])[:, 1], index=predict_df.index)
    pred = (prob_pred >= threshold).astype(int)
    return pred, threshold, val_macro_f1


def submission_frame(predict_df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    out = predict_df[KEY_COLS].copy()
    out[TARGET_COL] = pred.map({1: "Y", 0: "N"})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate, and optionally evaluate enrollment predictions."
    )
    parser.add_argument("--train-data-dir", type=Path, default=TRAIN_DATA_DIR)
    parser.add_argument("--gold-train-path", type=Path, default=GOLD_TRAIN_PATH)
    parser.add_argument("--test-data-dir", type=Path, default=TEST_DATA_DIR)
    parser.add_argument("--gold-test-path", type=Path, default=GOLD_TEST_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "predictions.csv",
        help="Where to write test predictions when test inputs are supplied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = build_labeled_frame(
        args.train_data_dir,
        read_gold(args.gold_train_path),
    )
    feature_cols = feature_columns(train_df)
    if not feature_cols:
        raise RuntimeError("No features available; check ENROLLMENT_HISTORY_FEATURES.csv exists.")

    # Complete training and temporal validation before looking at any test file.
    model, threshold, val_macro_f1 = train_validate_refit(train_df, feature_cols)
    if val_macro_f1 is not None:
        print(f"Val macro F1:  {val_macro_f1:.4f}")
        print(f"Probability threshold: {threshold:.6f}")

    if (args.test_data_dir is None) != (args.gold_test_path is None):
        raise ValueError(
            "--test-data-dir and --gold-test-path must either both be supplied or both omitted."
        )
    if args.test_data_dir is None:
        print("No test paths supplied; test data was not read.")
        return

    test_gold = read_gold(args.gold_test_path)
    predict_df = build_prediction_frame(args.test_data_dir, test_gold[KEY_COLS])
    for col in feature_cols:
        if col not in predict_df.columns:
            predict_df[col] = pd.NA
    test_probability = pd.Series(
        model.predict_proba(predict_df[feature_cols])[:, 1],
        index=predict_df.index,
    )
    test_pred = (test_probability >= threshold).astype(int)
    submission = submission_frame(predict_df, test_pred)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)

    test_macro_f1 = evaluate_macro_f1(test_gold, submission)
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"Predictions written to: {args.output_path}")


if __name__ == "__main__":
    main()

