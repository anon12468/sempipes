#!/usr/bin/env python3
"""
Enhanced pipeline for the course enrollment forecasting task.

Predicts whether each course offering will have high enrollment (top quartile
among offerings with positive enrollment in the same department and term).

Trains on table_splits/train, predicts test/, prints val/test macro F1.
Paths resolve from this file's location, so you can run it from any cwd:

  python /path/to/BEAVER/baselines/train_pipeline.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# --- Paths (anchored to repo root via this file) ---
PROJECT_ROOT = Path(__file__).resolve().parents[0]
TRAIN_DATA_DIR = PROJECT_ROOT / "table_splits" / "train"
DEFAULT_TEST_DATA_DIR = Path("/Users/USER/BEAVER/test/")
GOLD_TRAIN_PATH = PROJECT_ROOT / "table_splits" / "train" / "gold_enrollment_train.csv"
DEFAULT_GOLD_TEST_PATH = Path("/Users/USER/BEAVER/eval/gold_enrollment_test.csv")

TEST_DATA_DIR = DEFAULT_TEST_DATA_DIR if DEFAULT_TEST_DATA_DIR.exists() else None
GOLD_TEST_PATH = DEFAULT_GOLD_TEST_PATH if DEFAULT_GOLD_TEST_PATH.exists() else None
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
TERMS_TABLE = "ACADEMIC_TERMS.csv"
SIS_COURSE_TABLE = "SIS_COURSE_DESCRIPTION.csv"

RAW_NON_FEATURE_COLS = {
    "SUBJECT_TITLE",
    "MASTER_SUBJECT_ID_SORT",
    "CLUSTER_LIST",
    "COURSE_NUMBER",
    "MASTER_COURSE_NUMBER",
    "COURSE_NUMBER_DESC",
    "MASTER_COURSE_NUMBER_DESC",
    "OFFER_DEPT_NAME",
    "CLUSTER_TYPE_DESC",
    "HGN_CODE_DESC",
}


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


def read_table(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return normalize_keys(pd.read_csv(path, low_memory=False))


def load_course_offerings(data_dir: Path) -> pd.DataFrame:
    """Agent-visible offerings + section aggregations + lag history features (no current-term enrollment)."""
    offered = read_table(data_dir, OFFERED_TABLE)

    if len(offered) == 0:
        return pd.DataFrame(columns=KEY_COLS)

    # 1. Section-level aggregations from offered table
    offered["is_lec"] = (offered.get("IS_LECTURE_SECTION", pd.Series(dtype=object)) == "Y").astype(int)
    offered["is_lab"] = (offered.get("IS_LAB_SECTION", pd.Series(dtype=object)) == "Y").astype(int)
    offered["is_rec"] = (offered.get("IS_RECITATION_SECTION", pd.Series(dtype=object)) == "Y").astype(int)
    offered["is_des"] = (offered.get("IS_DESIGN_SECTION", pd.Series(dtype=object)) == "Y").astype(int)
    offered["is_rep"] = (offered.get("IS_REPEATABLE_SUBJECT", pd.Series(dtype=object)) == "Y").astype(int)
    offered["has_meet_t"] = offered.get("MEET_TIME", pd.Series(dtype=object)).notna().astype(int)
    offered["has_meet_p"] = offered.get("MEET_PLACE", pd.Series(dtype=object)).notna().astype(int)
    offered["eval_subj"] = (offered.get("EVALUATE_THIS_SUBJECT", pd.Series(dtype=object)) == "Y").astype(int)
    offered["has_fac"] = offered.get("RESPONSIBLE_FACULTY_NAME", pd.Series(dtype=object)).notna().astype(int)

    sec_agg = offered.groupby(KEY_COLS).agg(
        n_sections=("IS_MASTER_SECTION", "count"),
        n_lectures=("is_lec", "sum"),
        n_labs=("is_lab", "sum"),
        n_recitations=("is_rec", "sum"),
        n_design=("is_des", "sum"),
        is_repeatable=("is_rep", "max"),
        has_meet_time=("has_meet_t", "max"),
        has_meet_place=("has_meet_p", "max"),
        evaluate_subject=("eval_subj", "max"),
        has_faculty=("has_fac", "max"),
        n_meet_times=("MEET_TIME", "nunique") if "MEET_TIME" in offered.columns else ("is_lec", "sum"),
        n_meet_places=("MEET_PLACE", "nunique") if "MEET_PLACE" in offered.columns else ("is_lec", "sum"),
        n_faculty=("RESPONSIBLE_FACULTY_NAME", "nunique") if "RESPONSIBLE_FACULTY_NAME" in offered.columns else ("is_lec", "sum"),
    ).reset_index()

    keep_off = [
        c
        for c in KEY_COLS + ["OFFER_DEPT_CODE", "OFFER_SCHOOL_NAME", "SUBJECT_TITLE", "MASTER_SUBJECT_ID_SORT", "CLUSTER_TYPE", "HGN_CODE"]
        if c in offered.columns
    ]
    base_df = offered[keep_off].drop_duplicates(KEY_COLS).merge(sec_agg, on=KEY_COLS, how="left")

    # 2. Summary table features
    sum_path = data_dir / SUMMARY_TABLE
    if sum_path.exists():
        summary = read_table(data_dir, SUMMARY_TABLE)
        sum_cols = [
            c
            for c in ["TOTAL_UNITS", "LECTURE_UNITS", "LAB_UNITS", "PREPARATION_UNITS", "CLUSTER_LIST", "CLUSTER_TYPE", "HGN_CODE"]
            if c in summary.columns
        ]
        summary_sub = summary[KEY_COLS + sum_cols].drop_duplicates(KEY_COLS)
        base_df = base_df.merge(summary_sub, on=KEY_COLS, how="left", suffixes=("", "_sum"))
        if "CLUSTER_TYPE_sum" in base_df.columns:
            base_df["CLUSTER_TYPE"] = base_df["CLUSTER_TYPE"].fillna(base_df["CLUSTER_TYPE_sum"])
            base_df.drop(columns=["CLUSTER_TYPE_sum"], inplace=True)
        if "HGN_CODE_sum" in base_df.columns:
            base_df["HGN_CODE"] = base_df["HGN_CODE"].fillna(base_df["HGN_CODE_sum"])
            base_df.drop(columns=["HGN_CODE_sum"], inplace=True)

    # 3. Enrollment history features
    hist_path = data_dir / HISTORY_TABLE
    if hist_path.exists():
        history = normalize_keys(pd.read_csv(hist_path, low_memory=False))
        base_df = base_df.merge(history, on=KEY_COLS, how="left")
    for col in ["lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]:
        if col not in base_df.columns:
            base_df[col] = 0.0
        else:
            base_df[col] = pd.to_numeric(base_df[col], errors="coerce").fillna(0.0)

    # 4. Academic terms features
    terms_path = data_dir / TERMS_TABLE
    if terms_path.exists():
        terms = normalize_keys(pd.read_csv(terms_path, low_memory=False))
        if "TERM_START_DATE" in terms.columns and "TERM_END_DATE" in terms.columns:
            t_start = pd.to_datetime(terms["TERM_START_DATE"], format="%d-%b-%y", errors="coerce")
            t_end = pd.to_datetime(terms["TERM_END_DATE"], format="%d-%b-%y", errors="coerce")
            terms["term_duration_days"] = (t_end - t_start).dt.days.fillna(0)
        else:
            terms["term_duration_days"] = 0
        terms["is_reg_term"] = (terms.get("IS_REGULAR_TERM", pd.Series(dtype=object)) == "Y").astype(int)
        t_feat = terms[["TERM_CODE", "is_reg_term", "term_duration_days"]].drop_duplicates("TERM_CODE")
        base_df = base_df.merge(t_feat, on="TERM_CODE", how="left")
    if "is_reg_term" not in base_df.columns:
        base_df["is_reg_term"] = base_df["TERM_CODE"].astype(str).str[4:6].isin(["FA", "SP"]).astype(int)
    if "term_duration_days" not in base_df.columns:
        base_df["term_duration_days"] = 0

    # 5. Department program stats from SIS_COURSE_DESCRIPTION if present
    sis_path = data_dir / SIS_COURSE_TABLE
    if sis_path.exists():
        sis = pd.read_csv(sis_path, low_memory=False)
        if "DEPARTMENT" in sis.columns:
            sis_agg = sis.groupby("DEPARTMENT").agg(
                dept_n_programs=("COURSE", "count"),
                dept_has_degree_granting=("IS_DEGREE_GRANTING", lambda s: int((s == "Y").any())),
                dept_has_undergrad_prog=("COURSE_LEVEL", lambda s: int((s == "U").any())),
                dept_has_grad_prog=("COURSE_LEVEL", lambda s: int((s == "G").any())),
            ).reset_index()
            base_df = base_df.merge(sis_agg, left_on="OFFER_DEPT_CODE", right_on="DEPARTMENT", how="left")
            base_df.drop(columns=["DEPARTMENT"], inplace=True, errors="ignore")
    for c in ["dept_n_programs", "dept_has_degree_granting", "dept_has_undergrad_prog", "dept_has_grad_prog"]:
        if c not in base_df.columns:
            base_df[c] = 0
        else:
            base_df[c] = base_df[c].fillna(0).astype(int)

    # 6. Fill section numerical features
    sec_cols = [
        "n_sections",
        "n_lectures",
        "n_labs",
        "n_recitations",
        "n_design",
        "is_repeatable",
        "has_meet_time",
        "has_meet_place",
        "evaluate_subject",
        "has_faculty",
        "n_meet_times",
        "n_meet_places",
        "n_faculty",
        "term_duration_days",
        "is_reg_term",
    ]
    for c in sec_cols:
        base_df[c] = base_df[c].fillna(0)

    base_df["has_lecture"] = (base_df["n_lectures"] > 0).astype(int)
    base_df["has_lab"] = (base_df["n_labs"] > 0).astype(int)
    base_df["has_recitation"] = (base_df["n_recitations"] > 0).astype(int)
    base_df["has_design"] = (base_df["n_design"] > 0).astype(int)

    # 7. Lag transforms and momentum
    base_df["log1p_lag1"] = np.log1p(np.maximum(0, base_df["lag1_enrollment"]))
    base_df["log1p_lag2"] = np.log1p(np.maximum(0, base_df["lag2_enrollment"]))
    base_df["log1p_roll3"] = np.log1p(np.maximum(0, base_df["roll3_enrollment_mean"]))
    base_df["has_lag1"] = (base_df["lag1_enrollment"] > 0).astype(int)
    base_df["has_lag2"] = (base_df["lag2_enrollment"] > 0).astype(int)
    base_df["has_roll3"] = (base_df["roll3_enrollment_mean"] > 0).astype(int)
    base_df["lag_diff"] = base_df["lag1_enrollment"] - base_df["lag2_enrollment"]
    base_df["lag_ratio"] = (base_df["lag1_enrollment"] + 1.0) / (base_df["lag2_enrollment"] + 1.0)
    base_df["roll_diff"] = base_df["lag1_enrollment"] - base_df["roll3_enrollment_mean"]
    base_df["roll_ratio"] = (base_df["lag1_enrollment"] + 1.0) / (base_df["roll3_enrollment_mean"] + 1.0)
    base_df["lag_max"] = np.maximum(base_df["lag1_enrollment"], base_df["lag2_enrollment"])
    base_df["lag_min"] = np.minimum(base_df["lag1_enrollment"], base_df["lag2_enrollment"])

    # 8. Term parsing
    base_df["term_year"] = pd.to_numeric(base_df["TERM_CODE"].astype(str).str[:4], errors="coerce").fillna(0)
    season_map = {"JA": 1, "SP": 2, "SU": 3, "FA": 4}
    base_df["term_season_code"] = base_df["TERM_CODE"].astype(str).str[4:6].map(season_map).fillna(9)
    base_df["term_rank"] = term_to_rank(base_df["TERM_CODE"])
    base_df["term_season"] = base_df["TERM_CODE"].astype(str).str[4:6]

    # 9. Units & ratios
    for u in ["TOTAL_UNITS", "LECTURE_UNITS", "LAB_UNITS", "PREPARATION_UNITS"]:
        if u not in base_df.columns:
            base_df[u] = 0.0
        else:
            base_df[u] = pd.to_numeric(base_df[u], errors="coerce").fillna(0.0)

    base_df["unit_lecture_ratio"] = base_df["LECTURE_UNITS"] / (base_df["TOTAL_UNITS"] + 1e-4)
    base_df["unit_lab_ratio"] = base_df["LAB_UNITS"] / (base_df["TOTAL_UNITS"] + 1e-4)
    base_df["unit_prep_ratio"] = base_df["PREPARATION_UNITS"] / (base_df["TOTAL_UNITS"] + 1e-4)
    base_df["is_zero_units"] = (base_df["TOTAL_UNITS"] == 0).astype(int)

    # 10. Subject ID parsing
    if len(base_df) > 0:
        s_parts = base_df["SUBJECT_ID_SORT"].astype(str).str.split(".", n=1, expand=True)
        base_df["subj_prefix"] = s_parts[0] if 0 in s_parts.columns else pd.Series(dtype=object)
        suffix = s_parts[1].fillna("") if 1 in s_parts.columns else pd.Series("", index=base_df.index)
        base_df["subj_suffix_num"] = pd.to_numeric(suffix.str.extract(r"^([0-9]+)")[0], errors="coerce").fillna(-1)
        base_df["subj_has_letter"] = suffix.str.contains(r"[A-Za-z]", regex=True).fillna(False).astype(int)
        base_df["is_numeric_dept"] = base_df["subj_prefix"].astype(str).str.isnumeric().astype(int)
        base_df["is_undergrad_guess"] = ((base_df["subj_suffix_num"] < 500) & (base_df["is_numeric_dept"] == 1)).astype(int)
    else:
        base_df["subj_prefix"] = pd.Series(dtype=object)
        base_df["subj_suffix_num"] = pd.Series(dtype=float)
        base_df["subj_has_letter"] = pd.Series(dtype=int)
        base_df["is_numeric_dept"] = pd.Series(dtype=int)
        base_df["is_undergrad_guess"] = pd.Series(dtype=int)

    if "MASTER_SUBJECT_ID_SORT" in base_df.columns:
        base_df["is_master_subject"] = (base_df["SUBJECT_ID_SORT"] == base_df["MASTER_SUBJECT_ID_SORT"]).astype(int)
    else:
        base_df["is_master_subject"] = 1

    # 11. Clusters & Credit
    if "CLUSTER_TYPE" not in base_df.columns:
        base_df["CLUSTER_TYPE"] = "UNKNOWN"
    base_df["CLUSTER_TYPE"] = base_df["CLUSTER_TYPE"].fillna("UNKNOWN").astype(str)
    base_df["is_cross_listed"] = base_df["CLUSTER_TYPE"].isin(["J", "M"]).astype(int)
    base_df["is_swe"] = (base_df["CLUSTER_TYPE"] == "S").astype(int)
    if "CLUSTER_LIST" in base_df.columns:
        base_df["cluster_size"] = base_df["CLUSTER_LIST"].fillna("").astype(str).str.count(",") + 1
        base_df.loc[base_df["CLUSTER_LIST"].isna(), "cluster_size"] = 0
    else:
        base_df["cluster_size"] = 0

    if "HGN_CODE" not in base_df.columns:
        base_df["HGN_CODE"] = "UNKNOWN"
    base_df["HGN_CODE"] = base_df["HGN_CODE"].fillna("UNKNOWN").astype(str)
    base_df["is_undergrad_credit"] = (base_df["HGN_CODE"] == "N").astype(int)

    # 12. Title features
    title = base_df["SUBJECT_TITLE"].fillna("").astype(str) if "SUBJECT_TITLE" in base_df.columns else pd.Series("", index=base_df.index)
    base_df["title_len"] = title.str.len()
    base_df["title_words"] = title.str.split().str.len().fillna(0)
    title_lower = title.str.lower()
    base_df["title_has_intro"] = title_lower.str.contains("intro", regex=False).astype(int)
    base_df["title_has_thesis"] = title_lower.str.contains("thesis", regex=False).astype(int)
    base_df["title_has_research"] = title_lower.str.contains("research", regex=False).astype(int)
    base_df["title_has_special"] = title_lower.str.contains("special", regex=False).astype(int)
    base_df["title_has_seminar"] = title_lower.str.contains("seminar", regex=False).astype(int)
    base_df["title_has_workshop"] = title_lower.str.contains("workshop", regex=False).astype(int)
    base_df["title_has_project"] = title_lower.str.contains("project", regex=False).astype(int)
    base_df["title_has_lab"] = title_lower.str.contains("lab", regex=False).astype(int)
    base_df["title_has_topics"] = title_lower.str.contains("topics", regex=False).astype(int)
    base_df["title_has_principles"] = title_lower.str.contains("principle", regex=False).astype(int)
    base_df["title_has_advanced"] = title_lower.str.contains("advanced", regex=False).astype(int)
    base_df["title_has_independent"] = title_lower.str.contains("independent", regex=False).astype(int)
    base_df["title_has_internship"] = title_lower.str.contains("internship", regex=False).astype(int)

    # 13. Dept-Term and School-Term relative aggregations
    if "OFFER_DEPT_CODE" not in base_df.columns:
        base_df["OFFER_DEPT_CODE"] = "UNKNOWN"
    base_df["OFFER_DEPT_CODE"] = base_df["OFFER_DEPT_CODE"].fillna("UNKNOWN").astype(str)
    if "OFFER_SCHOOL_NAME" not in base_df.columns:
        base_df["OFFER_SCHOOL_NAME"] = "UNKNOWN"
    base_df["OFFER_SCHOOL_NAME"] = base_df["OFFER_SCHOOL_NAME"].fillna("UNKNOWN").astype(str)

    if len(base_df) > 0:
        dept_grp = base_df.groupby(["TERM_CODE", "OFFER_DEPT_CODE"])
        base_df["dept_roll3_rank"] = dept_grp["roll3_enrollment_mean"].rank(pct=True)
        base_df["dept_roll3_mean"] = dept_grp["roll3_enrollment_mean"].transform("mean")
        base_df["dept_roll3_std"] = dept_grp["roll3_enrollment_mean"].transform("std").fillna(0)
        base_df["dept_roll3_diff"] = base_df["roll3_enrollment_mean"] - base_df["dept_roll3_mean"]
        base_df["dept_roll3_ratio"] = (base_df["roll3_enrollment_mean"] + 1.0) / (base_df["dept_roll3_mean"] + 1.0)
        base_df["dept_roll3_zscore"] = base_df["dept_roll3_diff"] / (base_df["dept_roll3_std"] + 1e-4)

        # Relative rank among offerings with positive enrollment (matching target definition)
        pos_roll3 = base_df["roll3_enrollment_mean"] > 0
        base_df["dept_roll3_pos_rank"] = 0.0
        if pos_roll3.any():
            base_df.loc[pos_roll3, "dept_roll3_pos_rank"] = base_df[pos_roll3].groupby(["TERM_CODE", "OFFER_DEPT_CODE"])["roll3_enrollment_mean"].rank(pct=True)
        base_df["dept_roll3_pos_top_quartile"] = (base_df["dept_roll3_pos_rank"] >= 0.75).astype(int)

        pos_lag1 = base_df["lag1_enrollment"] > 0
        base_df["dept_lag1_pos_rank"] = 0.0
        if pos_lag1.any():
            base_df.loc[pos_lag1, "dept_lag1_pos_rank"] = base_df[pos_lag1].groupby(["TERM_CODE", "OFFER_DEPT_CODE"])["lag1_enrollment"].rank(pct=True)
        base_df["dept_lag1_pos_top_quartile"] = (base_df["dept_lag1_pos_rank"] >= 0.75).astype(int)

        base_df["dept_lag1_rank"] = dept_grp["lag1_enrollment"].rank(pct=True)
        base_df["dept_lag1_mean"] = dept_grp["lag1_enrollment"].transform("mean")
        base_df["dept_lag1_diff"] = base_df["lag1_enrollment"] - base_df["dept_lag1_mean"]

        base_df["dept_sections_rank"] = dept_grp["n_sections"].rank(pct=True)
        base_df["dept_sections_mean"] = dept_grp["n_sections"].transform("mean")
        base_df["dept_recitations_rank"] = dept_grp["n_recitations"].rank(pct=True)
        base_df["dept_offering_count"] = dept_grp["SUBJECT_ID_SORT"].transform("count")

        school_grp = base_df.groupby(["TERM_CODE", "OFFER_SCHOOL_NAME"])
        base_df["school_roll3_rank"] = school_grp["roll3_enrollment_mean"].rank(pct=True)
        base_df["school_offering_count"] = school_grp["SUBJECT_ID_SORT"].transform("count")
    else:
        for c in [
            "dept_roll3_rank", "dept_roll3_mean", "dept_roll3_std", "dept_roll3_diff",
            "dept_roll3_ratio", "dept_roll3_zscore", "dept_roll3_pos_rank",
            "dept_roll3_pos_top_quartile", "dept_lag1_pos_rank", "dept_lag1_pos_top_quartile",
            "dept_lag1_rank", "dept_lag1_mean", "dept_lag1_diff", "dept_sections_rank",
            "dept_sections_mean", "dept_recitations_rank", "dept_offering_count",
            "school_roll3_rank", "school_offering_count",
        ]:
            base_df[c] = pd.Series(dtype=float)

    return base_df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(KEY_COLS) | LEAKAGE_COLUMNS | RAW_NON_FEATURE_COLS
    cols = [c for c in df.columns if c not in exclude]
    return cols


def labels_to_y(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.upper() == "Y").astype(int)


def read_gold(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing gold labels: {path}")
    return normalize_keys(pd.read_csv(path, low_memory=False))


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


# --- Model ---

def make_model(x_train: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    categorical_cols = [
        c
        for c in feature_cols
        if (not pd.api.types.is_numeric_dtype(x_train[c]))
        or c in {"OFFER_DEPT_CODE", "OFFER_SCHOOL_NAME", "CLUSTER_TYPE", "HGN_CODE", "term_season", "subj_prefix"}
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

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

    preprocessor = ColumnTransformer(transformers=transformers)
    preprocessor.set_output(transform="pandas")

    if HAS_LGBM:
        classifier = LGBMClassifier(
            n_estimators=450,
            learning_rate=0.04,
            num_leaves=63,
            max_depth=8,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        classifier = HistGradientBoostingClassifier(
            random_state=42,
            max_iter=350,
            max_depth=10,
            learning_rate=0.05,
            min_samples_leaf=40,
        )

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def fit_threshold(y_train: pd.Series, prob_val: pd.Series, y_val: pd.Series | None = None) -> float:
    if len(prob_val) == 0:
        return 0.5

    # If validation labels are available, optimize threshold for macro F1 directly
    if y_val is not None and len(y_val) > 0 and len(np.unique(y_val)) > 1:
        best_f1 = -1.0
        best_th = 0.40
        for th in np.linspace(0.15, 0.65, 51):
            pred = (prob_val >= th).astype(int)
            score = f1_score(y_val, pred, average="macro", zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = float(th)
        return best_th

    # Fallback to empirical quantile
    train_pos_rate = float(y_train.mean())
    threshold = float(prob_val.quantile(max(0.0, min(1.0, 1.0 - train_pos_rate))))
    return min(threshold, 0.5)


def train_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.Series, float, float | None]:
    fit_mask = train_df["term_rank"] < VAL_TERM_RANK_THRESHOLD
    val_mask = ~fit_mask

    x_fit = train_df.loc[fit_mask, feature_cols]
    y_fit = train_df.loc[fit_mask, "y"]
    x_val = train_df.loc[val_mask, feature_cols]
    y_val = train_df.loc[val_mask, "y"]

    model = make_model(x_fit, feature_cols)
    model.fit(x_fit, y_fit)

    if len(x_val):
        prob_val = pd.Series(model.predict_proba(x_val)[:, 1], index=x_val.index)
        threshold = fit_threshold(y_fit, prob_val, y_val=y_val)
        val_pred = (prob_val >= threshold).astype(int)
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
    else:
        threshold = 0.5
        val_macro_f1 = None

    if len(predict_df):
        prob_pred = pd.Series(model.predict_proba(predict_df[feature_cols])[:, 1], index=predict_df.index)
        pred = (prob_pred >= threshold).astype(int)
    else:
        pred = pd.Series(dtype=int)

    return pred, threshold, val_macro_f1


def submission_frame(predict_df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    out = predict_df[KEY_COLS].copy()
    out[TARGET_COL] = pred.map({1: "Y", 0: "N"}).fillna("N")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrollment forecasting pipeline")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DATA_DIR, help="Path to training data directory")
    parser.add_argument("--gold-train", type=Path, default=GOLD_TRAIN_PATH, help="Path to gold training labels")
    parser.add_argument("--test-dir", type=Path, default=TEST_DATA_DIR, help="Path to test data directory")
    parser.add_argument("--gold-test", type=Path, default=GOLD_TEST_PATH, help="Path to gold test labels")
    parser.add_argument("--output-preds", type=Path, default=PROJECT_ROOT / "predictions_test.csv", help="Path to save predictions")
    args, _ = parser.parse_known_args()

    train_data_dir: Path = args.train_dir
    gold_train_path: Path = args.gold_train
    test_data_dir: Path | None = args.test_dir
    gold_test_path: Path | None = args.gold_test

    print("Building labeled training frame...")
    train_df = build_labeled_frame(train_data_dir, read_gold(gold_train_path))
    feature_cols = feature_columns(train_df)
    if not feature_cols:
        raise RuntimeError("No features available; check that training tables exist.")

    print(f"Features extracted: {len(feature_cols)}")

    # Check test setup
    has_test = test_data_dir is not None and test_data_dir.exists()
    predict_df = pd.DataFrame(columns=KEY_COLS + feature_cols)
    test_gold = None

    if has_test:
        test_courses = load_course_offerings(test_data_dir)
        if len(test_courses) > 0:
            if gold_test_path is not None and gold_test_path.exists():
                try:
                    test_gold = read_gold(gold_test_path)
                    if len(test_gold) > 0:
                        predict_df = test_gold[KEY_COLS].merge(test_courses, on=KEY_COLS, how="inner")
                    else:
                        predict_df = test_courses
                except Exception:
                    predict_df = test_courses
            else:
                predict_df = test_courses
        else:
            print("Note: Test data directory contains empty course tables (placeholder).")

    print("Training model and evaluating...")
    test_pred, threshold, val_macro_f1 = train_and_predict(train_df, predict_df, feature_cols)
    if val_macro_f1 is not None:
        print(f"Calibrated threshold: {threshold:.4f}")
        print(f"Val macro F1:  {val_macro_f1:.4f}")

    if test_gold is not None and len(test_gold) > 0 and len(predict_df) > 0:
        sub_df = submission_frame(predict_df, test_pred)
        test_macro_f1 = evaluate_macro_f1(test_gold, sub_df)
        print(f"Test macro F1: {test_macro_f1:.4f}")

        # Save test predictions
        sub_df.to_csv(args.output_preds, index=False)
        print(f"Saved test predictions to {args.output_preds}")
    elif len(predict_df) > 0:
        sub_df = submission_frame(predict_df, test_pred)
        sub_df.to_csv(args.output_preds, index=False)
        print(f"Saved test predictions to {args.output_preds}")


if __name__ == "__main__":
    main()
