#!/usr/bin/env python3
"""
Improved pipeline for the course enrollment forecasting task.

Predicts whether each course offering will have high enrollment (top quartile relative
to other offerings in the same department and term).

Features engineered:
- Aggregations over course sections: section counts, lecture, lab, recitation, design,
  master counts, meeting time / place indicators and frequencies, repeatable flags,
  and subject evaluation indicators.
- Summary table units: lecture, lab, preparation, and total units, plus unit ratios.
- Cluster & linkage features: joint offerings, cluster type, cluster size, and master subject matching.
- History lag features: lag1, lag2, roll3 mean, log-transforms, zero-history indicators.
- Subject ID structural decomposition: course prefixes, Harvard/cross-reg indicators,
  joint-course flags ('J'), undergraduate suffixes ('U'), course level numbers.
- Course title NLP keywords: flags for thesis, research, seminar, graduate, independent study,
  intro, lab, topics, workshop, title length and word counts.
- Within-department-term relative context: department-level mean, max, ratio, and percentile
  ranks for historical enrollment, section count, recitation count, and department offering count.
- LightGBM gradient boosting with calibrated thresholding tuned on temporal validation slices.

Usage:
  # Development validation run (uses table_splits/train only):
  python pipeline.py

  # Evaluation on held-out test split when available:
  python pipeline.py --test-data-dir table_splits/test --gold-test-path table_splits/test/gold_enrollment_test.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

# --- Paths (anchored to repo root via this file) ---
PROJECT_ROOT = Path(__file__).resolve().parents[0]
TRAIN_DATA_DIR = PROJECT_ROOT / "table_splits" / "train"
TEST_DATA_DIR: Path | None = Path("/Users/USER/BEAVER/test/")
GOLD_TRAIN_PATH = PROJECT_ROOT / "table_splits" / "train" / "gold_enrollment_train.csv"
GOLD_TEST_PATH: Path | None = Path("/Users/USER/BEAVER/eval/gold_enrollment_test.csv")
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

CATEGORICAL_FEATURE_NAMES = [
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "term_season",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "COURSE_NUMBER",
    "subj_prefix",
]

NUMERIC_FEATURE_NAMES = [
    "lag1_enrollment",
    "lag2_enrollment",
    "roll3_enrollment_mean",
    "is_zero_roll3",
    "is_zero_lag1",
    "log_roll3",
    "log_lag1",
    "log_lag2",
    "n_sections",
    "n_lecture",
    "n_lab",
    "n_recitation",
    "n_design",
    "n_master_sec",
    "has_meet_time",
    "n_meet_time",
    "has_meet_place",
    "n_meet_place",
    "is_repeatable",
    "evaluate_this",
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
    "lecture_unit_ratio",
    "lab_unit_ratio",
    "prep_unit_ratio",
    "is_harvard",
    "is_joint_j",
    "is_undergrad_u",
    "subj_num",
    "is_low_num",
    "is_mid_num",
    "title_len",
    "title_words",
    "title_has_thesis",
    "title_has_research",
    "title_has_grad",
    "title_has_special",
    "title_has_seminar",
    "title_has_indep",
    "title_has_intro",
    "title_has_lab",
    "title_has_topics",
    "title_has_workshop",
    "is_master_subj",
    "is_joint_cluster",
    "cluster_size",
    "term_year",
    "is_regular_term",
    "dept_roll3_mean",
    "dept_roll3_max",
    "dept_roll3_ratio",
    "dept_roll3_pct",
    "dept_sections_mean",
    "dept_sections_ratio",
    "dept_sections_pct",
    "dept_recitations_mean",
    "dept_recitations_ratio",
    "dept_offerings_count",
]


# --- Data helpers ---

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


def read_table(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return normalize_keys(pd.read_csv(path, low_memory=False))


def load_course_offerings(data_dir: Path) -> pd.DataFrame:
    """
    Loads course offerings with rich feature engineering, feature extraction,
    and feature aggregation from SUBJECT_OFFERED, SUBJECT_OFFERED_SUMMARY,
    and ENROLLMENT_HISTORY_FEATURES.
    """
    # 1. Base offerings and section-level aggregation from SUBJECT_OFFERED
    offered = read_table(data_dir, OFFERED_TABLE)

    # Boolean flags for section characteristics
    offered["is_lecture"] = (offered["IS_LECTURE_SECTION"] == "Y").astype(np.int8)
    offered["is_lab"] = (offered["IS_LAB_SECTION"] == "Y").astype(np.int8)
    offered["is_recitation"] = (offered["IS_RECITATION_SECTION"] == "Y").astype(np.int8)
    offered["is_design"] = (offered["IS_DESIGN_SECTION"] == "Y").astype(np.int8)
    offered["is_master_sec"] = (offered["IS_MASTER_SECTION"] == "Y").astype(np.int8)
    offered["has_meet_time"] = offered["MEET_TIME"].notna().astype(np.int8)
    offered["has_meet_place"] = offered["MEET_PLACE"].notna().astype(np.int8)
    offered["is_repeatable"] = (offered["IS_REPEATABLE_SUBJECT"] == "Y").astype(np.int8)
    offered["evaluate_this"] = (offered["EVALUATE_THIS_SUBJECT"] == "Y").astype(np.int8)

    # Aggregate sections per course offering (TERM_CODE, SUBJECT_ID_SORT)
    sec_agg = offered.groupby(KEY_COLS, as_index=False).agg(
        n_sections=("IS_MASTER_SECTION", "count"),
        n_lecture=("is_lecture", "sum"),
        n_lab=("is_lab", "sum"),
        n_recitation=("is_recitation", "sum"),
        n_design=("is_design", "sum"),
        n_master_sec=("is_master_sec", "sum"),
        has_meet_time=("has_meet_time", "max"),
        n_meet_time=("has_meet_time", "sum"),
        has_meet_place=("has_meet_place", "max"),
        n_meet_place=("has_meet_place", "sum"),
        is_repeatable=("is_repeatable", "max"),
        evaluate_this=("evaluate_this", "max"),
    )

    keep_offered = [
        c
        for c in KEY_COLS + ["OFFER_DEPT_CODE", "OFFER_SCHOOL_NAME", "SUBJECT_TITLE"]
        if c in offered.columns
    ]
    df = offered[keep_offered].drop_duplicates(KEY_COLS).copy()
    df = df.merge(sec_agg, on=KEY_COLS, how="left")

    # 2. Merge summary table for course units, clusters, and department details
    summary_path = data_dir / SUMMARY_TABLE
    if summary_path.exists():
        summary = read_table(data_dir, SUMMARY_TABLE)
        extra = [
            c
            for c in [
                "TOTAL_UNITS",
                "LECTURE_UNITS",
                "LAB_UNITS",
                "PREPARATION_UNITS",
                "CLUSTER_TYPE",
                "CLUSTER_LIST",
                "MASTER_SUBJECT_ID_SORT",
                "HGN_CODE",
                "COURSE_NUMBER",
                "OFFER_DEPT_CODE",
                "OFFER_SCHOOL_NAME",
                "SUBJECT_TITLE",
            ]
            if c in summary.columns
        ]
        summary_dedup = summary[KEY_COLS + [c for c in extra if c not in KEY_COLS]].drop_duplicates(KEY_COLS)
        # Avoid column collisions during merge
        for col in ["OFFER_DEPT_CODE", "OFFER_SCHOOL_NAME", "SUBJECT_TITLE"]:
            if col in summary_dedup.columns and col in df.columns:
                df[col] = df[col].combine_first(summary_dedup[col])
                summary_dedup = summary_dedup.drop(columns=[col])
        df = df.merge(summary_dedup, on=KEY_COLS, how="left")

    # 3. Merge lag history features
    history_path = data_dir / HISTORY_TABLE
    if history_path.exists():
        history = normalize_keys(pd.read_csv(history_path, low_memory=False))
        df = df.merge(history, on=KEY_COLS, how="left")

    # Clean and fill history lag numeric features
    for col in ["lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df["is_zero_roll3"] = (df["roll3_enrollment_mean"] == 0).astype(np.int8)
    df["is_zero_lag1"] = (df["lag1_enrollment"] == 0).astype(np.int8)
    df["log_roll3"] = np.log1p(df["roll3_enrollment_mean"].clip(lower=0))
    df["log_lag1"] = np.log1p(df["lag1_enrollment"].clip(lower=0))
    df["log_lag2"] = np.log1p(df["lag2_enrollment"].clip(lower=0))

    # Unit calculations and ratios
    for col in ["TOTAL_UNITS", "LECTURE_UNITS", "LAB_UNITS", "PREPARATION_UNITS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df["lecture_unit_ratio"] = df["LECTURE_UNITS"] / (df["TOTAL_UNITS"] + 1.0)
    df["lab_unit_ratio"] = df["LAB_UNITS"] / (df["TOTAL_UNITS"] + 1.0)
    df["prep_unit_ratio"] = df["PREPARATION_UNITS"] / (df["TOTAL_UNITS"] + 1.0)

    # Subject ID structure parsing
    parts = df["SUBJECT_ID_SORT"].astype(str).str.split(".", expand=True)
    df["subj_prefix"] = parts[0]
    df["subj_suffix"] = parts[1].fillna("") if parts.shape[1] > 1 else ""
    df["is_harvard"] = df["subj_prefix"].str.startswith("HA").astype(np.int8)
    df["is_joint_j"] = df["SUBJECT_ID_SORT"].astype(str).str.endswith("J").astype(np.int8)
    df["is_undergrad_u"] = df["SUBJECT_ID_SORT"].astype(str).str.endswith("U").astype(np.int8)
    numeric_part = pd.to_numeric(df["subj_suffix"].str.extract(r"(\d+)")[0], errors="coerce").fillna(0)
    df["subj_num"] = numeric_part
    df["is_low_num"] = (numeric_part < 100).astype(np.int8)
    df["is_mid_num"] = ((numeric_part >= 100) & (numeric_part < 500)).astype(np.int8)

    # Subject title keyword extraction
    title = df["SUBJECT_TITLE"].fillna("").astype(str).str.upper()
    df["title_len"] = title.str.len()
    df["title_words"] = title.str.split().str.len()
    df["title_has_thesis"] = title.str.contains("THESIS").astype(np.int8)
    df["title_has_research"] = title.str.contains("RESEARCH").astype(np.int8)
    df["title_has_grad"] = title.str.contains("GRADUATE").astype(np.int8)
    df["title_has_special"] = title.str.contains("SPECIAL").astype(np.int8)
    df["title_has_seminar"] = title.str.contains("SEMINAR").astype(np.int8)
    df["title_has_indep"] = title.str.contains("INDEPENDENT").astype(np.int8)
    df["title_has_intro"] = title.str.contains("INTRO").astype(np.int8)
    df["title_has_lab"] = title.str.contains("LAB").astype(np.int8)
    df["title_has_topics"] = title.str.contains("TOPICS").astype(np.int8)
    df["title_has_workshop"] = title.str.contains("WORKSHOP").astype(np.int8)

    # Cluster features
    if "MASTER_SUBJECT_ID_SORT" in df.columns:
        df["is_master_subj"] = (df["SUBJECT_ID_SORT"] == df["MASTER_SUBJECT_ID_SORT"]).astype(np.int8)
    else:
        df["is_master_subj"] = 0
    df["is_joint_cluster"] = (
        (df["CLUSTER_TYPE"] == "J").astype(np.int8) if "CLUSTER_TYPE" in df.columns else 0
    )
    if "CLUSTER_LIST" in df.columns:
        df["cluster_size"] = df["CLUSTER_LIST"].fillna("").astype(str).apply(
            lambda s: len(s.split(",")) if s else 0
        )
    else:
        df["cluster_size"] = 0

    # Term features
    df["term_rank"] = term_to_rank(df["TERM_CODE"])
    df["term_year"] = pd.to_numeric(df["TERM_CODE"].astype(str).str[:4], errors="coerce").fillna(0).astype(int)
    df["term_season"] = df["TERM_CODE"].astype(str).str[4:6]
    df["is_regular_term"] = df["term_season"].isin(["FA", "SP"]).astype(np.int8)

    # Fill default for section counts if missing
    for c in [
        "n_sections", "n_lecture", "n_lab", "n_recitation", "n_design", "n_master_sec",
        "has_meet_time", "n_meet_time", "has_meet_place", "n_meet_place",
        "is_repeatable", "evaluate_this",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Within-department and within-term contextual aggregations
    dept_term = df.groupby(["TERM_CODE", "OFFER_DEPT_CODE"], observed=False)
    df["dept_roll3_mean"] = dept_term["roll3_enrollment_mean"].transform("mean")
    df["dept_roll3_max"] = dept_term["roll3_enrollment_mean"].transform("max")
    df["dept_roll3_ratio"] = df["roll3_enrollment_mean"] / (df["dept_roll3_mean"] + 1.0)
    df["dept_roll3_pct"] = dept_term["roll3_enrollment_mean"].rank(pct=True)

    df["dept_sections_mean"] = dept_term["n_sections"].transform("mean")
    df["dept_sections_ratio"] = df["n_sections"] / (df["dept_sections_mean"] + 0.1)
    df["dept_sections_pct"] = dept_term["n_sections"].rank(pct=True)

    df["dept_recitations_mean"] = dept_term["n_recitation"].transform("mean")
    df["dept_recitations_ratio"] = df["n_recitation"] / (df["dept_recitations_mean"] + 0.1)
    df["dept_offerings_count"] = dept_term["SUBJECT_ID_SORT"].transform("count")

    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns candidate feature columns present in df, excluding leakage columns."""
    candidate_cols = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
    return [c for c in candidate_cols if c in df.columns and c not in LEAKAGE_COLUMNS]


def labels_to_y(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.upper() == "Y").astype(int)


def read_gold(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing gold labels: {path}")
    return normalize_keys(pd.read_csv(path))


def build_labeled_frame(data_dir: Path, gold: pd.DataFrame) -> pd.DataFrame:
    courses = load_course_offerings(data_dir)
    labeled = gold[KEY_COLS + [TARGET_COL]].merge(courses, on=KEY_COLS, how="inner")
    labeled["y"] = labels_to_y(labeled[TARGET_COL]).astype(int)
    labeled["term_rank"] = term_to_rank(labeled["TERM_CODE"])
    return labeled


def evaluate_macro_f1(gold: pd.DataFrame, pred: pd.DataFrame) -> float:
    gold = normalize_keys(gold.copy())
    pred = normalize_keys(pred.copy())
    gold[TARGET_COL] = gold[TARGET_COL].astype(str).str.strip().str.upper()
    pred[TARGET_COL] = pred[TARGET_COL].astype(str).str.strip().str.upper()

    gold = gold[gold[TARGET_COL].isin(["Y", "N"])]
    merged = gold.merge(pred, on=KEY_COLS, how="left", suffixes=("_gold", "_pred"))
    scored = merged[merged[f"{TARGET_COL}_pred"].isin(["Y", "N"])]
    if scored.empty:
        return 0.0

    y_true = (scored[f"{TARGET_COL}_gold"] == "Y").astype(int)
    y_pred = (scored[f"{TARGET_COL}_pred"] == "Y").astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# --- Model and Threshold Calibration ---

def make_model(
    x_train: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = 42,
):
    """
    Creates an optimized LightGBM classifier with categorical handling,
    or falls back to HistGradientBoostingClassifier pipeline if lightgbm is missing.
    """
    if HAS_LIGHTGBM:
        return LGBMClassifier(
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        )

    # Fallback to HistGradientBoostingClassifier pipeline
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
                    random_state=random_state,
                    max_iter=300,
                    max_depth=10,
                    learning_rate=0.05,
                ),
            ),
        ]
    )


def calibrate_threshold(
    y_true: pd.Series,
    prob: pd.Series,
    min_th: float = 0.20,
    max_th: float = 0.60,
    step: float = 0.01,
) -> float:
    """Finds the decision threshold maximizing macro F1 score on validation predictions."""
    if len(y_true) == 0 or len(prob) == 0:
        return 0.35
    best_th = 0.35
    best_score = 0.0
    for th in np.arange(min_th, max_th, step):
        pred = (prob >= th).astype(int)
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_th = float(th)
    return best_th


def fit_threshold(
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    term_rank_series: pd.Series,
    feature_cols: list[str],
) -> float:
    """
    Calibrates classification threshold on the most recent temporal slice of fit data
    (e.g., academic terms 2012 <= year < 2015) to avoid threshold leakage into evaluation split.
    """
    # Use 20120-20150 as internal temporal tuning slice if available
    internal_val_mask = term_rank_series >= 20120
    internal_fit_mask = term_rank_series < 20120

    if internal_val_mask.sum() > 500 and internal_fit_mask.sum() > 500:
        model_int = make_model(x_fit.loc[internal_fit_mask], feature_cols, random_state=42)
        model_int.fit(x_fit.loc[internal_fit_mask], y_fit.loc[internal_fit_mask])
        prob_int_val = pd.Series(
            model_int.predict_proba(x_fit.loc[internal_val_mask])[:, 1],
            index=x_fit.loc[internal_val_mask].index,
        )
        return calibrate_threshold(y_fit.loc[internal_val_mask], prob_int_val)

    # Fallback to empirical positive rate percentile
    train_pos_rate = float(y_fit.mean())
    return max(0.20, min(0.50, 1.0 - train_pos_rate * 4.0))


def train_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.Series, float, float | None]:
    """
    Fits model on terms before VAL_TERM_RANK_THRESHOLD, evaluates macro F1
    on the held-out validation slice, and generates predictions on predict_df.
    """
    fit_mask = train_df["term_rank"] < VAL_TERM_RANK_THRESHOLD
    val_mask = ~fit_mask

    # Ensure categorical dtypes for LightGBM
    x_train = train_df[feature_cols].copy()
    x_predict = predict_df[feature_cols].copy()

    for col in CATEGORICAL_FEATURE_NAMES:
        if col in x_train.columns:
            x_train[col] = x_train[col].astype("category")
        if col in x_predict.columns:
            x_predict[col] = x_predict[col].astype("category")

    x_fit = x_train.loc[fit_mask]
    y_fit = train_df.loc[fit_mask, "y"]
    x_val = x_train.loc[val_mask]
    y_val = train_df.loc[val_mask, "y"]

    # Fit classification threshold on internal temporal slice
    threshold = fit_threshold(x_fit, y_fit, train_df.loc[fit_mask, "term_rank"], feature_cols)

    # Train final model on fit slice
    model = make_model(x_fit, feature_cols)
    model.fit(x_fit, y_fit)

    # Evaluate validation split if present
    if len(x_val):
        prob_val = pd.Series(model.predict_proba(x_val)[:, 1], index=x_val.index)
        val_pred = (prob_val >= threshold).astype(int)
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
    else:
        val_macro_f1 = None

    # Predict on target predict_df
    prob_pred = pd.Series(model.predict_proba(x_predict)[:, 1], index=x_predict.index)
    pred = (prob_pred >= threshold).astype(int)
    return pred, threshold, val_macro_f1


def submission_frame(predict_df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    out = predict_df[KEY_COLS].copy()
    out[TARGET_COL] = pred.map({1: "Y", 0: "N"})
    return out


def predict_and_evaluate_test(
    train_df: pd.DataFrame,
    test_data_dir: Path,
    gold_test_path: Path,
    feature_cols: list[str],
    output_predictions_path: Path | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Predicts course offerings in test_data_dir and evaluates macro F1 against gold_test_path.
    Optionally saves the predictions to output_predictions_path.
    """
    gold_test = read_gold(gold_test_path)
    test_offerings = load_course_offerings(test_data_dir)
    predict_df = gold_test[KEY_COLS].merge(test_offerings, on=KEY_COLS, how="inner")

    test_pred, threshold, _ = train_and_predict(train_df, predict_df, feature_cols)
    sub = submission_frame(predict_df, test_pred)

    if output_predictions_path is not None:
        sub.to_csv(output_predictions_path, index=False)
        print(f"Saved test predictions to {output_predictions_path}")

    macro_f1 = evaluate_macro_f1(gold_test, sub)
    return sub, macro_f1


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Course enrollment forecasting pipeline.")
    parser.add_argument(
        "--train-data-dir",
        type=Path,
        default=TRAIN_DATA_DIR,
        help="Path to training data directory.",
    )
    parser.add_argument(
        "--gold-train-path",
        type=Path,
        default=GOLD_TRAIN_PATH,
        help="Path to gold training labels CSV.",
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=TEST_DATA_DIR,
        help="Path to test data directory.",
    )
    parser.add_argument(
        "--gold-test-path",
        type=Path,
        default=GOLD_TEST_PATH,
        help="Path to gold test labels CSV.",
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=None,
        help="Path to save predictions CSV.",
    )
    opts = parser.parse_args(args)

    print(f"Loading training data from {opts.train_data_dir}...")
    train_df = build_labeled_frame(opts.train_data_dir, read_gold(opts.gold_train_path))
    feature_cols = feature_columns(train_df)
    if not feature_cols:
        raise RuntimeError("No features available; check training tables.")
    print(f"Engineered {len(feature_cols)} features across {len(train_df)} training records.")

    # Validation evaluation on the development time split
    val_mask = train_df["term_rank"] >= VAL_TERM_RANK_THRESHOLD
    val_predict_df = train_df.loc[val_mask].copy()

    val_pred, threshold, val_macro_f1 = train_and_predict(train_df, val_predict_df, feature_cols)
    if val_macro_f1 is not None:
        print(f"Calibrated threshold: {threshold:.4f}")
        print(f"Val macro F1:         {val_macro_f1:.4f}")

    # Evaluate test split if test tables and gold labels are specified
    if opts.test_data_dir is not None and opts.gold_test_path is not None:
        if opts.test_data_dir.exists() and opts.gold_test_path.exists():
            print(f"\nEvaluating test split from {opts.test_data_dir}...")
            _, test_macro_f1 = predict_and_evaluate_test(
                train_df=train_df,
                test_data_dir=opts.test_data_dir,
                gold_test_path=opts.gold_test_path,
                feature_cols=feature_cols,
                output_predictions_path=opts.output_predictions,
            )
            print(f"Test macro F1:        {test_macro_f1:.4f}")
        else:
            print("\nSpecified test directory or gold test path does not exist.")
    else:
        print(
            "\nNote: Test data directory / gold test path not provided (default None during training).\n"
            "To evaluate on test split, specify: --test-data-dir <dir> --gold-test-path <path>"
        )


if __name__ == "__main__":
    main()
