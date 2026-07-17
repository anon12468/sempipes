#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
from skrub import DataOp, TableVectorizer

import sempipes
from experiments.beaver_enrollment.config import (
    GOLD_TEST_PATH,
    GOLD_TRAIN_PATH,
    SCRIPTS_DIR,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    VAL_TERM_RANK_THRESHOLD,
)

sys.path.insert(0, str(SCRIPTS_DIR))
from enrollment_common import (  # noqa: E402
    KEY_COLS,
    TARGET_COL,
    labels_to_y,
    normalize_keys,
    term_to_rank,
)
from evaluate_submission import evaluate as evaluate_submission  # noqa: E402

warnings.filterwarnings("ignore")

JOIN_KEY = "SUBJECT_TERM_KEY"
OP_AGG = "aggregate_section_features"
OP_GEN = "enrollment_derived_features"

DROP_BEFORE_MODEL = [*KEY_COLS, JOIN_KEY, "term_rank", "SUBJECT_TITLE"]

SUMMARY_FEATURE_COLS = [
    "COURSE_NUMBER",
    "MASTER_SUBJECT_ID_SORT",
    "CLUSTER_TYPE",
    "CLUSTER_TYPE_DESC",
    "CLUSTER_LIST",
    "HGN_CODE",
    "HGN_CODE_DESC",
    "OFFER_DEPT_CODE",
    "OFFER_DEPT_NAME",
    "OFFER_SCHOOL_NAME",
    "RESPONSIBLE_FACULTY_NAME",
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
]

OFFERED_FEATURE_COLS = [
    "MASTER_COURSE_NUMBER_SORT",
    "COURSE_NUMBER_SORT",
    "COURSE_NUMBER_DESC",
    "FORM_TYPE",
    "FORM_TYPE_DESC",
    "EVALUATE_THIS_SUBJECT",
    "IS_OSE_SUBJECT",
    "IS_REPEATABLE_SUBJECT",
]

TERM_FEATURE_COLS = [
    "IS_REGULAR_TERM",
    "DEGREE_YEAR",
]

X_DESCRIPTION = (
    "One row per course offering, keyed by (TERM_CODE, SUBJECT_ID_SORT). Columns describe "
    "the offering's owning department (OFFER_DEPT_CODE) and school (OFFER_SCHOOL_NAME), "
    "course/catalog identity (COURSE_NUMBER, MASTER_SUBJECT_ID_SORT, HGN/cluster/form fields), "
    "scheduled units (TOTAL_UNITS, LECTURE_UNITS, LAB_UNITS, PREPARATION_UNITS), term metadata, "
    "and lagged historical enrollment of the same subject in prior terms "
    "(lag1_enrollment, lag2_enrollment, roll3_enrollment_mean). Labels are binary: Y if the "
    "offering reaches the positive-enrollment 75th percentile threshold within its "
    "department-term, with department-level fallback. N otherwise. High-enrollment courses "
    "tend to be large lecture offerings, recurring gateway/catalog courses, and subjects with "
    "strong recent enrollment momentum."
)
SECTIONS_DESCRIPTION = (
    "Raw SUBJECT_OFFERED rows: potentially several physical sections per course offering "
    "(IS_MASTER_SECTION, IS_LECTURE_SECTION, IS_LAB_SECTION, IS_RECITATION_SECTION, "
    "IS_DESIGN_SECTION flags; RESPONSIBLE_FACULTY_NAME; MEET_TIME; MEET_PLACE; HGN_CODE; "
    "FORM_TYPE; CLUSTER_TYPE; OSE/repeatable/evaluate flags). Multiple rows share the same "
    "SUBJECT_TERM_KEY. Join on SUBJECT_TERM_KEY to aggregate section-level detail up to the "
    "offering row."
)

AGG_PROMPT = """
Aggregate the multiple section rows per course offering (group by SUBJECT_TERM_KEY) into
numeric features that help predict top-quartile enrollment within a department-term.

REQUIRED: always emit a column named exactly ``total_sections`` (row count per offering).
Downstream operators depend on this stable name. You may add other section-count columns,
but ``total_sections`` must always be present.

High-value aggregation families (prioritize counts, distinct counts, and proportions):

1. Section scale and mix
   - ``total_sections`` plus counts of lecture / lab / recitation / design / master sections
   - proportions of each section type (e.g. prop_lecture_sections, prop_lab_sections)
   - lecture-heavy or multi-section offerings often attract more students

2. Staffing and scheduling footprint
   - num_distinct_faculty (RESPONSIBLE_FACULTY_NAME), num_distinct_meet_places, num_distinct_meet_times
   - counts of sections missing faculty, meet time, or meet place (data-quality / TBA signals)
   - sections with MEET_TIME like "to be arranged" or missing place

3. Curriculum / catalog structure
   - num_distinct_CLUSTER_TYPE, HGN_CODE, HGN_CODE_DESC, FORM_TYPE, FORM_TYPE_DESC values
     across sections
   - counts of specific cluster or HGN categories if they separate large gateway courses
     from small seminars (e.g. standard cluster vs HASS HGN sections)
   - counts/proportions of IS_OSE_SUBJECT, IS_REPEATABLE_SUBJECT, and EVALUATE_THIS_SUBJECT
     flags if present

Prefer simple pandas groupby aggregations. Use fillna(0) for counts. Output only numeric
columns with clear snake_case names. Do not leak current-term enrollment.
"""

GEN_PROMPT = """
Derive additional numeric features from the joined offering frame to help a tree-based
classifier predict top-quartile enrollment within a department-term (~25% positive class).

High-value derived-feature families:

1. Enrollment history and momentum (always present: lag1_enrollment, lag2_enrollment,
   roll3_enrollment_mean)
   - growth or difference: lag1 minus lag2, or (lag1 - lag2) / max(lag2, 1)
   - ratio to rolling mean: lag1 / max(roll3_enrollment_mean, 1)
   - deviation from roll3 mean (absolute or percent)
   - is_new_offering / no_history flags when all lags are zero
   - has_recent_growth when lag1 > lag2 and lag2 > 0

2. Unit mix (TOTAL_UNITS, LECTURE_UNITS, LAB_UNITS when present)
   - lecture_units / total_units, lab_units / total_units, preparation_units / total_units,
     other_units fraction
   - imbalance between lecture and lab units

3. Cross-feature interactions with aggregated section columns (optional — only if present)
   - lag1_enrollment per section, per distinct faculty, or per lecture section
   - enrollment momentum × section scale
   - faculty_per_section or sections_per_faculty

4. Department-term relative features (very high value; the label is top quartile
   within OFFER_DEPT_CODE + TERM_CODE)
   - within each (TERM_CODE, OFFER_DEPT_CODE), compute percentile/rank features for
     lag1_enrollment, roll3_enrollment_mean, lag1 / roll3 ratio, TOTAL_UNITS, and
     total_sections / section count when available
   - ratios to the department-term median or mean, e.g. lag1_enrollment /
     max(dept_term_median_lag1, 1), roll3_enrollment_mean / max(dept_term_mean_roll3, 1)
   - binary high-relative indicators such as lag1_enrollment above department-term
     median or in the top quartile
   - interactions such as lag1_percentile * section_count, roll3_percentile *
     lecture_unit_fraction, or dept_term_relative_lag1 * recent_growth
   - Use groupby transforms only on current feature rows and historical lag columns;
     do not use labels or current-term enrollment.

5. Catalog and course identity features (all are visible at prediction time)
   - parse COURSE_NUMBER / COURSE_NUMBER_SORT into a numeric course level and flags for
     intro/advanced/graduate-looking numbers when possible
   - indicators for whether SUBJECT_ID_SORT equals MASTER_SUBJECT_ID_SORT, whether a master
     subject exists, and simple string-length/hash-free numeric properties of subject/course ids
   - numeric flags from IS_OSE_SUBJECT, IS_REPEATABLE_SUBJECT, EVALUATE_THIS_SUBJECT,
     IS_REGULAR_TERM, and missingness indicators for catalog fields
   - HGN/cluster/form signals: simple one-vs-rest numeric indicators for common values only
     when the columns exist; do not create high-cardinality text encodings manually
   - interactions between catalog/course level and total_sections, lecture fraction, and
     lag1_enrollment

CRITICAL — column safety (aggregated column names change across pipeline versions):
- Never assume a column exists; always check ``col in df.columns`` before using it.
- For department-term relative features, first check that both ``TERM_CODE`` and
  ``OFFER_DEPT_CODE`` exist. Use ``df.groupby(["TERM_CODE", "OFFER_DEPT_CODE"])``
  transforms, preserve the original row order, and fill missing / singleton-group
  values with 0.
- For section count, resolve once at the top of the function:
    section_cols = [c for c in ("total_sections", "num_sections", "total_num_sections") if c in df.columns]
    n_sections = df[section_cols[0]] if section_cols else 1
  Use ``n_sections`` (not a hard-coded column name) in per-section ratios.
- For optional agg columns (e.g. num_distinct_faculty, num_lecture_sections), only use them
  inside ``if col in df.columns`` blocks; skip that feature otherwise.
- Prefer features built only from lag/unit columns when agg columns are absent.

Rules:
- Do NOT use absolute calendar year or term_rank (must generalise to future terms).
- term_season is allowed for seasonality.
- Keep all existing columns; add only new numeric features.
- Guard divisions with max(denominator, 1) or fillna(0) to avoid inf/NaN.
- Prefer interpretable ratios and differences over opaque transforms.
"""

SECTION_COLS = [
    *KEY_COLS,
    "IS_MASTER_SECTION",
    "IS_LECTURE_SECTION",
    "IS_LAB_SECTION",
    "IS_RECITATION_SECTION",
    "IS_DESIGN_SECTION",
    "RESPONSIBLE_FACULTY_NAME",
    "MEET_TIME",
    "MEET_PLACE",
    "HGN_CODE",
    "HGN_CODE_DESC",
    "FORM_TYPE",
    "FORM_TYPE_DESC",
    "CLUSTER_TYPE",
    "CLUSTER_TYPE_DESC",
    "EVALUATE_THIS_SUBJECT",
    "IS_OSE_SUBJECT",
    "IS_REPEATABLE_SUBJECT",
]


def _read_gold(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet" and path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing gold labels: {path} / {csv_path}")


def _add_join_key(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_keys(df.copy())
    out[JOIN_KEY] = out["TERM_CODE"].astype(str) + "||" + out["SUBJECT_ID_SORT"].astype(str)
    return out


@lru_cache(maxsize=4)
def _load_sections_cached(data_dir_str: str) -> pd.DataFrame:
    """Load section rows; cached because the same table is reused across fit/val/test."""
    data_dir = Path(data_dir_str)
    path = data_dir / "SUBJECT_OFFERED.csv"
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in SECTION_COLS if c in header]
    sections = pd.read_csv(path, usecols=usecols, low_memory=False)
    return _add_join_key(sections)


def load_sections(data_dir: Path) -> pd.DataFrame:
    return _load_sections_cached(str(data_dir.resolve()))


def _read_csv_with_existing_columns(path: Path, desired_cols: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in desired_cols if c in header]
    if not usecols:
        return None
    return normalize_keys(pd.read_csv(path, usecols=usecols, low_memory=False))


def _load_extended_agent_course_offerings(data_dir: Path) -> pd.DataFrame:
    """Agent-visible offering frame with non-leaky catalog, term, and history fields."""
    summary = _read_csv_with_existing_columns(
        data_dir / "SUBJECT_OFFERED_SUMMARY.csv",
        [*KEY_COLS, "SUBJECT_TITLE", *SUMMARY_FEATURE_COLS],
    )
    offered = _read_csv_with_existing_columns(
        data_dir / "SUBJECT_OFFERED.csv",
        [*KEY_COLS, "SUBJECT_TITLE", *SUMMARY_FEATURE_COLS, *OFFERED_FEATURE_COLS],
    )

    if summary is not None:
        df = summary.drop_duplicates(KEY_COLS, keep="first").copy()
        if offered is not None:
            offered_extra_cols = [c for c in [*KEY_COLS, *OFFERED_FEATURE_COLS] if c in offered.columns]
            offered_extra = offered[offered_extra_cols].drop_duplicates(KEY_COLS, keep="first")
            df = df.merge(offered_extra, on=KEY_COLS, how="left")
    elif offered is not None:
        df = offered.drop_duplicates(KEY_COLS, keep="first").copy()
    else:
        raise FileNotFoundError(f"Missing SUBJECT_OFFERED(_SUMMARY) tables in {data_dir}")

    history = _read_csv_with_existing_columns(
        data_dir / "ENROLLMENT_HISTORY_FEATURES.csv",
        [*KEY_COLS, "lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"],
    )
    if history is not None:
        df = df.merge(history.drop_duplicates(KEY_COLS, keep="first"), on=KEY_COLS, how="left")

    terms = _read_csv_with_existing_columns(data_dir / "ACADEMIC_TERMS.csv", ["TERM_CODE", *TERM_FEATURE_COLS])
    if terms is not None:
        df = df.merge(terms.drop_duplicates("TERM_CODE", keep="first"), on="TERM_CODE", how="left")

    for col in [
        "TOTAL_UNITS",
        "LECTURE_UNITS",
        "LAB_UNITS",
        "PREPARATION_UNITS",
        "lag1_enrollment",
        "lag2_enrollment",
        "roll3_enrollment_mean",
        "DEGREE_YEAR",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    df["term_rank"] = term_to_rank(df["TERM_CODE"])
    df["term_season"] = df["TERM_CODE"].astype(str).str.slice(4, 6)
    return df.reset_index(drop=True)


@lru_cache(maxsize=4)
def _build_offering_frame_cached(data_dir_str: str) -> pd.DataFrame:
    df = _load_extended_agent_course_offerings(Path(data_dir_str))
    return _add_join_key(df)


def build_offering_frame(data_dir: Path) -> pd.DataFrame:
    return _build_offering_frame_cached(str(data_dir.resolve()))


def build_labeled_frame(data_dir: Path, gold: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    offerings = build_offering_frame(data_dir)
    gold = normalize_keys(gold[KEY_COLS + [TARGET_COL]])
    labeled = gold.merge(offerings, on=KEY_COLS, how="inner")
    y = labels_to_y(labeled[TARGET_COL]).astype(int)
    meta = labeled[KEY_COLS].copy()
    meta["term_rank"] = labeled["term_rank"]
    x = labeled.drop(columns=[TARGET_COL])
    return x.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True)


def build_predict_frame(data_dir: Path, gold_keys: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    offerings = build_offering_frame(data_dir)
    keys = normalize_keys(gold_keys[KEY_COLS])
    predict_df = keys.merge(offerings, on=KEY_COLS, how="inner")
    meta = predict_df[KEY_COLS].copy()
    return predict_df.reset_index(drop=True), meta.reset_index(drop=True)


def build_pipeline(seed: int = 42) -> DataOp:
    data = skrub.var("data").skb.set_description(X_DESCRIPTION)
    sections = skrub.var("sections").skb.set_description(SECTIONS_DESCRIPTION)
    labels = skrub.var("labels")

    x = data.skb.mark_as_X()
    y = labels.skb.mark_as_y()

    aggregated = x.sem_agg_features(
        sections,
        left_on=JOIN_KEY,
        right_on=JOIN_KEY,
        nl_prompt=AGG_PROMPT,
        name=OP_AGG,
        how_many=12,
    )

    engineered = aggregated.sem_gen_features(
        nl_prompt=GEN_PROMPT,
        name=OP_GEN,
        how_many=16,
    )

    features = engineered.drop(columns=DROP_BEFORE_MODEL, errors="ignore")

    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
    vectorizer = TableVectorizer(low_cardinality=ordinal, high_cardinality=ordinal)
    encoded = features.skb.apply(vectorizer)

    classifier = HistGradientBoostingClassifier(
        random_state=seed,
        max_iter=300,
        max_depth=10,
        learning_rate=0.05,
    )
    return encoded.skb.apply(classifier, y=y)


def _fit_threshold(y_train: pd.Series, prob_val: pd.Series) -> float:
    train_pos_rate = float(y_train.mean())
    if len(prob_val) == 0:
        return 0.5
    threshold = float(prob_val.quantile(max(0.0, min(1.0, 1.0 - train_pos_rate))))
    return min(threshold, 0.5)


def _proba_positive(learner, env: Dict[str, object]) -> np.ndarray:
    proba = learner.predict_proba(env)
    proba = np.asarray(proba)
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()


def submission_frame(meta: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    out = meta[KEY_COLS].copy()
    out[TARGET_COL] = pd.Series(pred, index=meta.index).map({1: "Y", 0: "N"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="BEAVER enrollment sempipes pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the classifier")
    args = parser.parse_args()

    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 0.0},
        )
    )

    print(f"Seed: {args.seed}")
    print("Loading training data...")
    gold_train = _read_gold(GOLD_TRAIN_PATH)
    x_train, y_train, meta_train = build_labeled_frame(TRAIN_DATA_DIR, gold_train)
    train_sections = load_sections(TRAIN_DATA_DIR)

    fit_mask = meta_train["term_rank"] < VAL_TERM_RANK_THRESHOLD
    val_mask = ~fit_mask

    x_fit = x_train.loc[fit_mask].reset_index(drop=True)
    y_fit = y_train.loc[fit_mask].reset_index(drop=True)
    x_val = x_train.loc[val_mask].reset_index(drop=True)
    y_val = y_train.loc[val_mask].reset_index(drop=True)

    pipeline = build_pipeline(seed=args.seed)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)

    def env(x: pd.DataFrame, y: pd.Series | None, sections: pd.DataFrame) -> Dict[str, object]:
        e = pipeline.skb.get_data()
        e["data"] = x
        e["sections"] = sections
        e["labels"] = y.to_numpy() if y is not None else np.zeros(len(x), dtype=int)
        return e

    print("Fitting sempipes pipeline (LLM feature generation + classifier)...")
    learner.fit(env(x_fit, y_fit, train_sections))

    print("Scoring validation split...")
    prob_val = _proba_positive(learner, env(x_val, y_val, train_sections))
    threshold = _fit_threshold(y_fit, pd.Series(prob_val))
    val_pred = (prob_val >= threshold).astype(int)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    print(f"Validation macro F1: {val_macro_f1:.4f} (threshold={threshold:.4f})")

    print("Loading test data and predicting...")
    gold_test = _read_gold(GOLD_TEST_PATH)
    x_test, meta_test = build_predict_frame(TEST_DATA_DIR, gold_test)
    test_sections = load_sections(TEST_DATA_DIR)

    prob_test = _proba_positive(learner, env(x_test, None, test_sections))
    test_pred = (prob_test >= threshold).astype(int)

    test_eval = evaluate_submission(gold_test, submission_frame(meta_test, test_pred))
    print(f"Test macro F1: {test_eval['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
