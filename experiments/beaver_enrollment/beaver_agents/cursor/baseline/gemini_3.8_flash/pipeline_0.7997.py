from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DATA_DIR = PROJECT_ROOT / "table_splits" / "train"
GOLD_TRAIN_PATH = TRAIN_DATA_DIR / "gold_enrollment_train.csv"
DEFAULT_TEST_DIR = Path("/Users/USER/BEAVER/test/")
DEFAULT_GOLD_TEST = Path("/Users/USER/BEAVER/eval/gold_enrollment_test.csv")

TEST_DATA_DIR: Path | None = DEFAULT_TEST_DIR if DEFAULT_TEST_DIR.exists() else None
GOLD_TEST_PATH: Path | None = DEFAULT_GOLD_TEST if DEFAULT_GOLD_TEST.exists() else None

KEY_COLS = ["TERM_CODE", "SUBJECT_ID_SORT"]
TARGET_COL = "HIGH_ENROLLMENT"
LEAKAGE_COLUMNS = [
    "y",
    "HIGH_ENROLLMENT",
    "ENROLLMENT_COUNT",
    "OFFERING_STATUS",
    "CURRENT_STATUS",
    "STATUS",
    "DROP_COUNT",
    "PRE_ENROLLMENT_COUNT",
    "WAITLIST_COUNT",
]

OFFERED_TABLE = "SUBJECT_OFFERED.csv"
SUMMARY_TABLE = "SUBJECT_OFFERED_SUMMARY.csv"
HISTORY_TABLE = "ENROLLMENT_HISTORY_FEATURES.csv"
ACADEMIC_TERMS_TABLE = "ACADEMIC_TERMS.csv"

VAL_TERM_RANK_THRESHOLD = 20150


class NativeLGBMClassifier(BaseEstimator, ClassifierMixin):
    """LGBMClassifier wrapper that preserves native categorical feature handling."""

    def __init__(
        self,
        cat_cols: list[str] | None = None,
        n_estimators: int = 450,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = 9,
        colsample_bytree: float = 0.85,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        random_state: int = 42,
        n_jobs: int = 4,
        verbose: int = -1,
    ):
        self.cat_cols = cat_cols or []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.model_ = None
        self.cat_categories_: dict[str, pd.Index] = {}

    def _prepare_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cat_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype("category")
                self.cat_categories_[col] = X_out[col].cat.categories
        return X_out

    def _prepare_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cat_cols:
            if col in X_out.columns:
                train_cats = self.cat_categories_.get(col, None)
                if train_cats is not None:
                    X_out[col] = pd.Categorical(X_out[col], categories=train_cats)
                else:
                    X_out[col] = X_out[col].astype("category")
        return X_out

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> NativeLGBMClassifier:
        X_prep = self._prepare_fit(X)
        self.model_ = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        self.model_.fit(X_prep, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0, 2))
        X_prep = self._prepare_predict(X)
        return self.model_.predict_proba(X_prep)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.empty(0, dtype=int)
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def term_to_rank(series: pd.Series) -> pd.Series:
    """Parse academic term codes like 2014FA -> 20142, 2015SP -> 20151, 2015JA -> 20150."""
    raw = series.astype(str).str.strip().str.upper()
    year = pd.to_numeric(raw.str.slice(0, 4), errors="coerce").fillna(0).astype(int)
    season_code = raw.str.slice(4, 6)
    season_map = {"JA": 0, "SP": 1, "SU": 2, "FA": 3}
    season_num = season_code.map(season_map).fillna(0).astype(int)
    return year * 10 + season_num


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    for col in KEY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def read_table(data_dir: Path | str, name: str) -> pd.DataFrame | None:
    path = Path(data_dir) / name
    if not path.is_file():
        return None
    return pd.read_csv(path, low_memory=False)


def aggregate_sections(offered: pd.DataFrame) -> pd.DataFrame:
    """Vectorized aggregation of section-level records into offering-level features."""
    df = offered.copy()
    normalize_keys(df)

    df["is_sec"] = 1
    df["is_lec"] = (df["IS_LECTURE_SECTION"] == "Y").astype(np.int8) if "IS_LECTURE_SECTION" in df else 0
    df["is_lab"] = (df["IS_LAB_SECTION"] == "Y").astype(np.int8) if "IS_LAB_SECTION" in df else 0
    df["is_rec"] = (df["IS_RECITATION_SECTION"] == "Y").astype(np.int8) if "IS_RECITATION_SECTION" in df else 0
    df["is_des"] = (df["IS_DESIGN_SECTION"] == "Y").astype(np.int8) if "IS_DESIGN_SECTION" in df else 0
    df["has_time"] = df["MEET_TIME"].notna().astype(np.int8) if "MEET_TIME" in df else 0
    df["has_place"] = df["MEET_PLACE"].notna().astype(np.int8) if "MEET_PLACE" in df else 0
    df["is_eval"] = (df["EVALUATE_THIS_SUBJECT"] == "Y").astype(np.int8) if "EVALUATE_THIS_SUBJECT" in df else 0
    df["is_ose"] = (df["IS_OSE_SUBJECT"] == "Y").astype(np.int8) if "IS_OSE_SUBJECT" in df else 0
    df["is_rep"] = (df["IS_REPEATABLE_SUBJECT"] == "Y").astype(np.int8) if "IS_REPEATABLE_SUBJECT" in df else 0
    df["has_fac"] = df["RESPONSIBLE_FACULTY_NAME"].notna().astype(np.int8) if "RESPONSIBLE_FACULTY_NAME" in df else 0

    agg = df.groupby(KEY_COLS, as_index=False).agg(
        n_sections=("is_sec", "sum"),
        has_lecture=("is_lec", "max"),
        n_lectures=("is_lec", "sum"),
        has_lab=("is_lab", "max"),
        n_labs=("is_lab", "sum"),
        has_recitation=("is_rec", "max"),
        n_recitations=("is_rec", "sum"),
        has_design=("is_des", "max"),
        has_meet_time=("has_time", "max"),
        has_meet_place=("has_place", "max"),
        evaluate_subject=("is_eval", "max"),
        is_ose=("is_ose", "max"),
        is_repeatable=("is_rep", "max"),
        has_faculty=("has_fac", "max"),
    )
    return agg


def _extract_suffix_features(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Parse subject suffix (course level) into numeric and level indicator signals."""
    s = series.astype(str).str.strip()
    extracted_num = s.str.extract(r"(\d+)")[0]
    suffix_num = pd.to_numeric(extracted_num, errors="coerce").fillna(-1.0)
    is_undergrad = ((suffix_num > 0) & (suffix_num < 500)).astype(np.int8)
    is_grad = (suffix_num >= 500).astype(np.int8)
    is_alpha = s.str.contains(r"[A-Za-z]", regex=True).astype(np.int8)
    return suffix_num, is_undergrad, is_grad, is_alpha


def _extract_title_features(series: pd.Series) -> pd.DataFrame:
    """Extract semantic and keyword signals from subject title."""
    titles = series.fillna("").astype(str).str.lower()
    return pd.DataFrame(
        {
            "title_is_thesis": titles.str.contains(r"\bthesis\b|\bdissertation\b", regex=True).astype(np.int8),
            "title_is_intro": titles.str.contains(r"\bintro|\bprinciple|\bfundamental", regex=True).astype(np.int8),
            "title_is_seminar": titles.str.contains(r"\bseminar\b|\bcolloquium\b", regex=True).astype(np.int8),
            "title_is_research": titles.str.contains(r"\bresearch\b|\bindependent study\b", regex=True).astype(np.int8),
            "title_is_special": titles.str.contains(r"\bspecial subject\b|\bspecial topics\b|\bworkshop\b", regex=True).astype(np.int8),
            "title_is_project": titles.str.contains(r"\bproject\b|\blab\b|\blaboratory\b", regex=True).astype(np.int8),
            "title_len": series.fillna("").astype(str).str.len(),
        }
    )


def load_course_offerings(data_dir: Path | str) -> pd.DataFrame:
    """Load, merge, and engineer multi-table catalog and history features for course offerings."""
    offered_raw = read_table(data_dir, OFFERED_TABLE)
    if offered_raw is None:
        raise FileNotFoundError(f"Missing required table: {Path(data_dir) / OFFERED_TABLE}")

    # Use first section occurrence per course offering for offering metadata
    normalize_keys(offered_raw)
    base_offerings = offered_raw.drop_duplicates(subset=KEY_COLS).copy()

    # Aggregate section counts and indicators
    sec_agg = aggregate_sections(offered_raw)
    merged = base_offerings.merge(sec_agg, on=KEY_COLS, how="left")

    # Merge summary table if present (enrich with any columns not already present)
    summary = read_table(data_dir, SUMMARY_TABLE)
    if summary is not None:
        normalize_keys(summary)
        summary_cols = [c for c in summary.columns if c not in merged.columns or c in KEY_COLS]
        if len(summary_cols) > len(KEY_COLS):
            merged = merged.merge(summary[summary_cols].drop_duplicates(subset=KEY_COLS), on=KEY_COLS, how="left")

    # Merge enrollment history features table
    history = read_table(data_dir, HISTORY_TABLE)
    if history is not None:
        normalize_keys(history)
        hist_cols = [c for c in history.columns if c not in merged.columns or c in KEY_COLS]
        merged = merged.merge(history[hist_cols].drop_duplicates(subset=KEY_COLS), on=KEY_COLS, how="left")

    # Merge academic terms table if present
    terms = read_table(data_dir, ACADEMIC_TERMS_TABLE)
    if terms is not None and "TERM_CODE" in terms.columns:
        terms["TERM_CODE"] = terms["TERM_CODE"].astype(str).str.strip()
        term_cols = [c for c in ["TERM_CODE", "IS_REGULAR_TERM"] if c in terms.columns]
        merged = merged.merge(terms[term_cols].drop_duplicates(subset=["TERM_CODE"]), on="TERM_CODE", how="left")

    # Time signals
    merged["term_rank"] = term_to_rank(merged["TERM_CODE"])
    merged["term_season"] = merged["TERM_CODE"].astype(str).str.strip().str.slice(4, 6)
    merged["term_year"] = pd.to_numeric(
        merged["TERM_CODE"].astype(str).str.strip().str.slice(0, 4), errors="coerce"
    ).fillna(0).astype(int)

    # Unit breakdowns & ratios
    for ucol in ["TOTAL_UNITS", "LECTURE_UNITS", "LAB_UNITS"]:
        if ucol in merged.columns:
            merged[ucol] = pd.to_numeric(merged[ucol], errors="coerce").fillna(0.0)
        else:
            merged[ucol] = 0.0

    if "PREPARATION_UNITS" in merged.columns:
        merged["PREPARATION_UNITS"] = pd.to_numeric(merged["PREPARATION_UNITS"], errors="coerce").fillna(0.0)
    else:
        merged["PREPARATION_UNITS"] = np.maximum(0.0, merged["TOTAL_UNITS"] - merged["LECTURE_UNITS"] - merged["LAB_UNITS"])

    safe_total = np.where(merged["TOTAL_UNITS"] > 0, merged["TOTAL_UNITS"], 1.0)
    merged["units_lec_ratio"] = merged["LECTURE_UNITS"] / safe_total
    merged["units_lab_ratio"] = merged["LAB_UNITS"] / safe_total
    merged["units_prep_ratio"] = merged["PREPARATION_UNITS"] / safe_total
    merged["is_zero_units"] = (merged["TOTAL_UNITS"] == 0).astype(np.int8)

    # Structural identifiers & cluster signals
    if "MASTER_SUBJECT_ID_SORT" in merged.columns:
        merged["is_master_course"] = (merged["SUBJECT_ID_SORT"] == merged["MASTER_SUBJECT_ID_SORT"]).astype(np.int8)
    else:
        merged["is_master_course"] = 1

    if "CLUSTER_LIST" in merged.columns:
        merged["cluster_size"] = merged["CLUSTER_LIST"].fillna("").astype(str).apply(
            lambda s: len(s.split(",")) if s else 1
        )
    else:
        merged["cluster_size"] = 1

    # Subject suffix / course level signals
    if "SUBJECT_SUFFIX" in merged.columns:
        s_num, s_ug, s_grad, s_alpha = _extract_suffix_features(merged["SUBJECT_SUFFIX"])
        merged["suffix_num"] = s_num
        merged["is_undergrad_level"] = s_ug
        merged["is_grad_level"] = s_grad
        merged["is_alpha_suffix"] = s_alpha
    else:
        merged["suffix_num"] = -1.0
        merged["is_undergrad_level"] = 0
        merged["is_grad_level"] = 0
        merged["is_alpha_suffix"] = 0

    # Title semantic signals
    if "SUBJECT_TITLE" in merged.columns:
        title_feats = _extract_title_features(merged["SUBJECT_TITLE"])
        for c in title_feats.columns:
            merged[c] = title_feats[c]

    # Enrollment history: handle both naming schemas
    lag1_col = "lag1_enrollment" if "lag1_enrollment" in merged.columns else ("ENROLLMENT_LAG_1" if "ENROLLMENT_LAG_1" in merged.columns else None)
    lag2_col = "lag2_enrollment" if "lag2_enrollment" in merged.columns else ("ENROLLMENT_LAG_2" if "ENROLLMENT_LAG_2" in merged.columns else None)
    roll3_col = "roll3_enrollment_mean" if "roll3_enrollment_mean" in merged.columns else ("ROLLING_MEAN_ENROLLMENT_3" if "ROLLING_MEAN_ENROLLMENT_3" in merged.columns else None)

    lag1 = pd.to_numeric(merged[lag1_col], errors="coerce") if lag1_col else pd.Series(np.nan, index=merged.index)
    lag2 = pd.to_numeric(merged[lag2_col], errors="coerce") if lag2_col else pd.Series(np.nan, index=merged.index)
    roll3 = pd.to_numeric(merged[roll3_col], errors="coerce") if roll3_col else pd.Series(np.nan, index=merged.index)

    merged["has_lag1"] = lag1.notna().astype(np.int8)
    merged["has_lag2"] = lag2.notna().astype(np.int8)
    merged["has_roll3"] = roll3.notna().astype(np.int8)

    merged["log_lag1"] = np.log1p(np.maximum(0.0, lag1.fillna(0.0)))
    merged["log_lag2"] = np.log1p(np.maximum(0.0, lag2.fillna(0.0)))
    merged["log_roll3"] = np.log1p(np.maximum(0.0, roll3.fillna(0.0)))
    merged["enrollment_trend"] = lag1.fillna(0.0) - lag2.fillna(0.0)

    # Department offering & cross-sectional term aggregations
    dept_col = "OFFER_DEPT_CODE" if "OFFER_DEPT_CODE" in merged.columns else ("COURSE_NUMBER" if "COURSE_NUMBER" in merged.columns else None)
    if dept_col is not None:
        merged["_dept_clean"] = merged[dept_col].astype(str).str.strip()
    else:
        merged["_dept_clean"] = "UNKNOWN"

    dt_grp = merged.groupby(["TERM_CODE", "_dept_clean"])
    merged["dept_offering_count"] = dt_grp["SUBJECT_ID_SORT"].transform("count")

    # Rolling mean aggregations within dept-term
    merged["dept_roll3_mean"] = dt_grp["log_roll3"].transform("mean")
    merged["dept_roll3_diff_mean"] = merged["log_roll3"] - merged["dept_roll3_mean"]
    merged["dept_roll3_pct"] = dt_grp["log_roll3"].rank(pct=True)

    pos_roll = roll3.where(roll3 > 0)
    merged["dept_roll3_pos_q75"] = merged.assign(_pos=pos_roll).groupby(["TERM_CODE", "_dept_clean"])["_pos"].transform(
        lambda s: s.quantile(0.75) if s.notna().any() else 0.0
    )
    merged["dept_roll3_pos_pct"] = merged.assign(_pos=pos_roll).groupby(["TERM_CODE", "_dept_clean"])["_pos"].rank(pct=True).fillna(0.0)
    merged["is_above_dept_roll3_pos_q75"] = (
        (roll3 >= merged["dept_roll3_pos_q75"]) & (merged["dept_roll3_pos_q75"] > 0)
    ).astype(np.int8)

    # Lag1 aggregations within dept-term
    merged["dept_lag1_mean"] = dt_grp["log_lag1"].transform("mean")
    merged["dept_lag1_diff_mean"] = merged["log_lag1"] - merged["dept_lag1_mean"]
    merged["dept_lag1_pct"] = dt_grp["log_lag1"].rank(pct=True)

    pos_lag1 = lag1.where(lag1 > 0)
    merged["dept_lag1_pos_q75"] = merged.assign(_pos=pos_lag1).groupby(["TERM_CODE", "_dept_clean"])["_pos"].transform(
        lambda s: s.quantile(0.75) if s.notna().any() else 0.0
    )
    merged["is_above_dept_lag1_pos_q75"] = (
        (lag1 >= merged["dept_lag1_pos_q75"]) & (merged["dept_lag1_pos_q75"] > 0)
    ).astype(np.int8)

    merged = merged.drop(columns=["_dept_clean"])
    return merged


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Select model features, excluding identifiers and leakage fields."""
    excluded = set(KEY_COLS + LEAKAGE_COLUMNS + ["term_rank"])
    cols: list[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if c in ["SUBJECT_TITLE", "CLUSTER_LIST", "MASTER_SUBJECT_ID_SORT"]:
            continue
        if df[c].dtype.kind in "biufc":
            cols.append(c)
        elif df[c].dtype == "object":
            sample = df[c].dropna().astype(str)
            if len(sample) == 0:
                continue
            cardinality = sample.nunique()
            if cardinality <= 300:
                cols.append(c)
    return sorted(cols)


def labels_to_y(series: pd.Series) -> pd.Series:
    """Map HIGH_ENROLLMENT 'Y'/'N' to 1/0."""
    return (series.astype(str).str.strip().str.upper() == "Y").astype(int)


def read_gold(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Gold labels file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    for c in KEY_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def build_labeled_frame(data_dir: Path | str, gold: pd.DataFrame) -> pd.DataFrame:
    """Join gold labels to course catalog and history features."""
    offerings = load_course_offerings(data_dir)
    labeled = gold.merge(offerings, on=KEY_COLS, how="inner")
    labeled["y"] = labels_to_y(labeled[TARGET_COL])
    return labeled


def evaluate_macro_f1(gold: pd.DataFrame, pred: pd.DataFrame) -> float:
    """Compute Macro F1 matching gold and pred tables by key columns."""
    for c in KEY_COLS:
        gold[c] = gold[c].astype(str).str.strip()
        pred[c] = pred[c].astype(str).str.strip()
    merged = gold[KEY_COLS + [TARGET_COL]].merge(
        pred[KEY_COLS + [TARGET_COL]],
        on=KEY_COLS,
        suffixes=("_true", "_pred"),
        how="inner",
    )
    if len(merged) == 0:
        return 0.0
    y_true = labels_to_y(merged[TARGET_COL + "_true"])
    y_pred = labels_to_y(merged[TARGET_COL + "_pred"])
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def make_model(x_train: pd.DataFrame, feature_cols: list[str]) -> BaseEstimator:
    """Construct an optimized gradient boosting estimator with categorical support."""
    cat_cols = [c for c in feature_cols if x_train[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    if HAS_LIGHTGBM:
        return NativeLGBMClassifier(
            cat_cols=cat_cols,
            n_estimators=450,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=9,
            colsample_bytree=0.85,
            subsample=0.8,
            subsample_freq=1,
            random_state=42,
            n_jobs=4,
            verbose=-1,
        )

    # Fallback to HistGradientBoostingClassifier if lightgbm is unavailable
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=63,
        max_depth=9,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", hgb)])


def fit_threshold(
    y_val: pd.Series | None,
    prob_val: pd.Series,
    y_train_fallback: pd.Series | None = None,
) -> float:
    """Find the probability threshold that maximizes Macro F1 on validation slice."""
    if len(prob_val) == 0:
        return 0.38
    if y_val is not None and len(y_val) == len(prob_val) and len(y_val) > 0:
        best_threshold = 0.38
        best_score = -1.0
        for threshold in np.linspace(0.15, 0.60, 91):
            pred_binary = (prob_val >= threshold).astype(int)
            score = float(f1_score(y_val, pred_binary, average="macro", zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return best_threshold
    if y_train_fallback is not None and len(y_train_fallback) > 0:
        train_pos_rate = float(y_train_fallback.mean())
        threshold = float(prob_val.quantile(max(0.0, min(1.0, 1.0 - train_pos_rate))))
        return min(threshold, 0.5)
    return 0.38


def train_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.Series, float, float | None]:
    """Train gradient booster, tune decision threshold on validation slice, and predict."""
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
        threshold = fit_threshold(y_val, prob_val, y_fit)
        val_pred = (prob_val >= threshold).astype(int)
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
    else:
        threshold = 0.38
        val_macro_f1 = None

    if len(predict_df) > 0:
        prob_pred = pd.Series(model.predict_proba(predict_df[feature_cols])[:, 1], index=predict_df.index)
        pred = (prob_pred >= threshold).astype(int)
    else:
        pred = pd.Series([], dtype=int, index=predict_df.index)

    return pred, threshold, val_macro_f1


def submission_frame(predict_df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Format binary predictions as 'Y'/'N' table matching gold submission format."""
    pred_str = pred.map({1: "Y", 0: "N"}).fillna("N")
    out = predict_df[KEY_COLS].copy()
    out[TARGET_COL] = pred_str.values
    return out


def predict_and_evaluate(
    gold_test_path: Path | str,
    test_data_dir: Path | str,
    train_df: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
) -> float:
    """Predict on test split course offerings and evaluate Macro F1 using test gold labels."""
    gold_test_path = Path(gold_test_path)
    test_data_dir = Path(test_data_dir)

    print(f"Reading test gold labels: {gold_test_path}")
    test_gold = read_gold(gold_test_path)

    print(f"Loading test course offerings: {test_data_dir}")
    test_offerings = load_course_offerings(test_data_dir)
    predict_df = test_gold[KEY_COLS].merge(test_offerings, on=KEY_COLS, how="inner")

    if train_df is None:
        print(f"Loading train data: {TRAIN_DATA_DIR}")
        train_df = build_labeled_frame(TRAIN_DATA_DIR, read_gold(GOLD_TRAIN_PATH))

    feature_cols = feature_columns(train_df)
    test_pred, threshold, val_macro_f1 = train_and_predict(train_df, predict_df, feature_cols)

    if val_macro_f1 is not None:
        print(f"Validation Macro F1: {val_macro_f1:.4f} (threshold: {threshold:.4f})")

    sub_df = submission_frame(predict_df, test_pred)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")

    if len(test_gold) > 0 and TARGET_COL in test_gold.columns:
        test_macro_f1 = evaluate_macro_f1(test_gold, sub_df)
        print(f"Test Macro F1:       {test_macro_f1:.4f}")
        return test_macro_f1

    print("Test gold set has no labeled rows to score; prediction frame successfully generated.")
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Course enrollment forecasting pipeline")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DATA_DIR, help="Path to training tables directory")
    parser.add_argument("--gold-train", type=Path, default=GOLD_TRAIN_PATH, help="Path to gold training labels CSV")
    parser.add_argument("--test-dir", type=Path, default=TEST_DATA_DIR, help="Path to test tables directory")
    parser.add_argument("--gold-test", type=Path, default=GOLD_TEST_PATH, help="Path to gold test labels CSV")
    parser.add_argument("--output-pred", type=Path, default=None, help="Optional output path for predictions CSV")
    args, _ = parser.parse_known_args()

    train_data_dir = args.train_dir or TRAIN_DATA_DIR
    gold_train_path = args.gold_train or GOLD_TRAIN_PATH
    test_data_dir = args.test_dir
    gold_test_path = args.gold_test

    print(f"Loading training data from {train_data_dir}...")
    train_df = build_labeled_frame(train_data_dir, read_gold(gold_train_path))
    feature_cols = feature_columns(train_df)
    if not feature_cols:
        raise RuntimeError("No features available; check input tables.")
    print(f"Engineered {len(feature_cols)} features for model training.")

    # Check if a non-empty test set is provided
    has_test_data = False
    if gold_test_path is not None and test_data_dir is not None:
        if Path(gold_test_path).is_file() and Path(test_data_dir).is_dir():
            test_gold = read_gold(gold_test_path)
            if len(test_gold) > 0:
                has_test_data = True

    if has_test_data:
        predict_and_evaluate(
            gold_test_path=gold_test_path,
            test_data_dir=test_data_dir,
            train_df=train_df,
            output_path=args.output_pred,
        )
    else:
        # Development mode: validate on the time-based validation slice (term_rank >= 20150)
        print("Evaluating on time-based validation slice (term_rank >= 20150)...")
        fit_mask = train_df["term_rank"] < VAL_TERM_RANK_THRESHOLD
        val_df = train_df.loc[~fit_mask].copy()
        pred, thresh, val_macro_f1 = train_and_predict(train_df, val_df, feature_cols)
        if val_macro_f1 is not None:
            print(f"Validation Macro F1: {val_macro_f1:.4f} (tuned threshold: {thresh:.4f})")


if __name__ == "__main__":
    main()
