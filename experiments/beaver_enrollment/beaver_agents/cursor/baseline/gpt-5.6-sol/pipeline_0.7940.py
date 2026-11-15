#!/usr/bin/env python3
"""
Self-contained reference baseline for the enrollment forecasting task.

Trains on table_splits/train, predicts test/, prints val/test macro F1.
Paths resolve from this file's location, so you can run it from any cwd:

  python /path/to/BEAVER/baselines/train_pipeline.py
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
TEST_DATA_DIR = Path("/Users/USER/BEAVER/test")
GOLD_TRAIN_PATH = PROJECT_ROOT / "table_splits" / "train" / "gold_enrollment_train.csv"
GOLD_TEST_PATH = Path("/Users/USER/BEAVER/test/gold_enrollment_test.csv")
CALIBRATION_TERM_RANK_THRESHOLD = 20150  # 2015-2016 tunes the decision threshold
VAL_TERM_RANK_THRESHOLD = 20170  # 2017+ is the untouched time-based validation slice

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
HISTORY_COLUMNS = ["lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]
OFFER_AGG_COLUMNS = [
    "section_count",
    "meet_time_count",
    "meet_place_count",
    "instructor_count",
    "is_master_section_count",
    "is_lecture_section_count",
    "is_lab_section_count",
    "is_recitation_section_count",
    "is_design_section_count",
    "is_repeatable_subject_count",
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


def read_table(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return normalize_keys(pd.read_csv(path, low_memory=False))


def load_course_offerings(data_dir: Path) -> pd.DataFrame:
    """Load one row per offered subject and build enrollment-safe features."""
    summary = read_table(data_dir, SUMMARY_TABLE)
    if summary.duplicated(KEY_COLS).any():
        raise ValueError(f"{SUMMARY_TABLE} contains duplicate prediction keys")

    safe_summary_columns = [
        "COURSE_NUMBER",
        "SUBJECT_TITLE",
        "MASTER_SUBJECT_ID_SORT",
        "CLUSTER_TYPE",
        "HGN_CODE",
        "OFFER_DEPT_CODE",
        "OFFER_DEPT_NAME",
        "OFFER_SCHOOL_NAME",
        "RESPONSIBLE_FACULTY_NAME",
        "TOTAL_UNITS",
        "LECTURE_UNITS",
        "LAB_UNITS",
        "PREPARATION_UNITS",
    ]
    keep = KEY_COLS + [c for c in safe_summary_columns if c in summary.columns]
    df = summary[keep].copy()

    offered_path = data_dir / OFFERED_TABLE
    if offered_path.exists():
        offered = read_table(data_dir, OFFERED_TABLE)
        flag_columns = [
            "IS_MASTER_SECTION",
            "IS_LECTURE_SECTION",
            "IS_LAB_SECTION",
            "IS_RECITATION_SECTION",
            "IS_DESIGN_SECTION",
            "IS_REPEATABLE_SUBJECT",
        ]
        for col in flag_columns:
            if col in offered.columns:
                offered[col] = (
                    offered[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map({"Y": 1.0, "N": 0.0, "1": 1.0, "0": 0.0})
                )

        aggregations: dict[str, tuple[str, str]] = {
            "section_count": ("SUBJECT_ID_SORT", "size"),
        }
        for output, source in [
            ("meet_time_count", "MEET_TIME"),
            ("meet_place_count", "MEET_PLACE"),
            ("instructor_count", "RESPONSIBLE_FACULTY_NAME"),
        ]:
            if source in offered.columns:
                aggregations[output] = (source, "nunique")
        for col in flag_columns:
            if col in offered.columns:
                aggregations[f"{col.lower()}_count"] = (col, "sum")

        offered_agg = (
            offered.groupby(KEY_COLS, sort=False, dropna=False)
            .agg(**aggregations)
            .reset_index()
        )
        df = df.merge(offered_agg, on=KEY_COLS, how="left", validate="one_to_one")

    for col in OFFER_AGG_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    history_path = data_dir / HISTORY_TABLE
    if history_path.exists():
        history = normalize_keys(pd.read_csv(history_path, low_memory=False))
        history = history[KEY_COLS + [c for c in HISTORY_COLUMNS if c in history.columns]]
        history = history.drop_duplicates(KEY_COLS)
        df = df.merge(history, on=KEY_COLS, how="left", validate="one_to_one")

    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["term_rank"] = term_to_rank(df["TERM_CODE"])
    df["term_year"] = pd.to_numeric(df["TERM_CODE"].astype(str).str[:4], errors="coerce")
    df["term_season"] = df["TERM_CODE"].astype(str).str[4:6]
    for col in ["TOTAL_UNITS", "LECTURE_UNITS", "LAB_UNITS", "PREPARATION_UNITS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # The label is department-relative. These features express prior enrollment
    # on the same scale without using current-term enrollment or labels.
    group_cols = ["TERM_CODE", "OFFER_DEPT_CODE"]
    for col in HISTORY_COLUMNS:
        df[f"{col}_dept_rank"] = df.groupby(group_cols, dropna=False)[col].rank(pct=True)
        positive_history = df[col].where(df[col] > 0)
        df[f"{col}_positive_dept_rank"] = (
            positive_history.groupby([df[c] for c in group_cols], dropna=False)
            .rank(pct=True)
            .fillna(0.0)
        )
        df[f"{col}_positive"] = (df[col] > 0).astype("int8")
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        *HISTORY_COLUMNS,
        "OFFER_DEPT_CODE",
        "OFFER_SCHOOL_NAME",
        "term_year",
        "term_season",
        "TOTAL_UNITS",
        "LECTURE_UNITS",
        "LAB_UNITS",
        "CLUSTER_TYPE",
        *[
            f"{col}{suffix}"
            for col in HISTORY_COLUMNS
            for suffix in ("_dept_rank", "_positive_dept_rank", "_positive")
        ],
        *OFFER_AGG_COLUMNS,
    ]
    return [c for c in candidates if c in df.columns and c not in LEAKAGE_COLUMNS]


def labels_to_y(series: pd.Series) -> pd.Series:
    return (series.astype(str) == "Y").astype(int)


def read_gold(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing gold labels: {path}")
    return normalize_keys(pd.read_csv(path))


def build_labeled_frame(data_dir: Path, gold: pd.DataFrame) -> pd.DataFrame:
    courses = load_course_offerings(data_dir)
    if gold.duplicated(KEY_COLS).any():
        raise ValueError("Gold labels contain duplicate prediction keys")
    labeled = gold[KEY_COLS + [TARGET_COL]].merge(
        courses,
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = labeled["_merge"].ne("both")
    if missing.any():
        raise ValueError(f"{int(missing.sum())} gold keys are missing from {SUMMARY_TABLE}")
    labeled = labeled.drop(columns="_merge")
    labeled["y"] = labels_to_y(labeled[TARGET_COL]).astype(int)
    labeled["term_rank"] = term_to_rank(labeled["TERM_CODE"])
    return labeled


def evaluate_macro_f1(gold: pd.DataFrame, pred: pd.DataFrame) -> float:
    gold = normalize_keys(gold.copy())
    pred = normalize_keys(pred.copy())
    if gold.duplicated(KEY_COLS).any() or pred.duplicated(KEY_COLS).any():
        raise ValueError("Gold and prediction keys must be unique")
    gold[TARGET_COL] = gold[TARGET_COL].astype(str).str.strip().str.upper()
    pred[TARGET_COL] = pred[TARGET_COL].astype(str).str.strip().str.upper()
    if not gold[TARGET_COL].isin(["Y", "N"]).all():
        raise ValueError("Gold labels must be Y or N")
    if not pred[TARGET_COL].isin(["Y", "N"]).all():
        raise ValueError("Predictions must be Y or N")

    merged = gold.merge(
        pred,
        on=KEY_COLS,
        how="left",
        suffixes=("_gold", "_pred"),
        validate="one_to_one",
        indicator=True,
    )
    if merged["_merge"].ne("both").any() or len(pred) != len(gold):
        raise ValueError("Predictions must contain exactly one row per gold key")

    y_true = (merged[f"{TARGET_COL}_gold"] == "Y").astype(int)
    y_pred = (merged[f"{TARGET_COL}_pred"] == "Y").astype(int)
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
                    max_iter=400,
                    max_depth=10,
                    learning_rate=0.04,
                    l2_regularization=0.5,
                ),
            ),
        ]
    )


def fit_threshold(y_true: pd.Series, probabilities: pd.Series) -> float:
    """Choose a macro-F1 threshold on a dedicated chronological slice."""
    if len(probabilities) == 0 or y_true.nunique() < 2:
        return 0.5
    candidates = np.linspace(0.05, 0.80, 151)
    scores = [
        f1_score(y_true, probabilities >= threshold, average="macro", zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def validate_time_split(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[float, float | None]:
    fit_mask = train_df["term_rank"] < CALIBRATION_TERM_RANK_THRESHOLD
    calibration_mask = (
        (train_df["term_rank"] >= CALIBRATION_TERM_RANK_THRESHOLD)
        & (train_df["term_rank"] < VAL_TERM_RANK_THRESHOLD)
    )
    validation_mask = train_df["term_rank"] >= VAL_TERM_RANK_THRESHOLD

    if not fit_mask.any() or not calibration_mask.any():
        return 0.5, None

    model = make_model(train_df.loc[fit_mask, feature_cols], feature_cols)
    model.fit(train_df.loc[fit_mask, feature_cols], train_df.loc[fit_mask, "y"])

    calibration_prob = pd.Series(
        model.predict_proba(train_df.loc[calibration_mask, feature_cols])[:, 1],
        index=train_df.index[calibration_mask],
    )
    threshold = fit_threshold(train_df.loc[calibration_mask, "y"], calibration_prob)

    if not validation_mask.any():
        return threshold, None
    validation_prob = model.predict_proba(train_df.loc[validation_mask, feature_cols])[:, 1]
    validation_pred = (validation_prob >= threshold).astype(int)
    validation_f1 = float(
        f1_score(
            train_df.loc[validation_mask, "y"],
            validation_pred,
            average="macro",
            zero_division=0,
        )
    )
    return threshold, validation_f1


def train_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.Series, float, float | None]:
    threshold, val_macro_f1 = validate_time_split(train_df, feature_cols)
    model = make_model(train_df[feature_cols], feature_cols)
    model.fit(train_df[feature_cols], train_df["y"])
    prob_pred = pd.Series(model.predict_proba(predict_df[feature_cols])[:, 1], index=predict_df.index)
    pred = (prob_pred >= threshold).astype(int)
    return pred, threshold, val_macro_f1


def submission_frame(predict_df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    out = predict_df[KEY_COLS].copy()
    out[TARGET_COL] = pred.to_numpy()
    out[TARGET_COL] = out[TARGET_COL].map({1: "Y", 0: "N"})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data-dir", type=Path, default=TRAIN_DATA_DIR)
    parser.add_argument("--gold-train-path", type=Path, default=GOLD_TRAIN_PATH)
    parser.add_argument("--test-data-dir", type=Path, default=TEST_DATA_DIR)
    parser.add_argument("--gold-test-path", type=Path, default=GOLD_TEST_PATH)
    parser.add_argument("--output-path", type=Path, default=PROJECT_ROOT / "test_predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.test_data_dir is None) != (args.gold_test_path is None):
        raise ValueError("--test-data-dir and --gold-test-path must be supplied together")

    train_df = build_labeled_frame(args.train_data_dir, read_gold(args.gold_train_path))
    feature_cols = feature_columns(train_df)
    if not feature_cols:
        raise RuntimeError("No features available; check ENROLLMENT_HISTORY_FEATURES.csv exists.")

    if args.test_data_dir is None:
        threshold, val_macro_f1 = validate_time_split(train_df, feature_cols)
        if val_macro_f1 is None:
            print("Validation unavailable: chronological slices are empty.")
        else:
            print(f"Validation macro F1: {val_macro_f1:.4f}")
            print(f"Decision threshold:  {threshold:.3f}")
        return

    test_gold = read_gold(args.gold_test_path)
    predict_df = test_gold[KEY_COLS].merge(
        load_course_offerings(args.test_data_dir),
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = predict_df["_merge"].ne("both")
    if missing.any():
        raise ValueError(f"{int(missing.sum())} test keys are missing from {SUMMARY_TABLE}")
    predict_df = predict_df.drop(columns="_merge")
    for col in feature_cols:
        if col not in predict_df.columns:
            predict_df[col] = np.nan

    test_pred, threshold, val_macro_f1 = train_and_predict(train_df, predict_df, feature_cols)
    if val_macro_f1 is not None:
        print(f"Validation macro F1: {val_macro_f1:.4f}")
    print(f"Decision threshold:  {threshold:.3f}")

    submission = submission_frame(predict_df, test_pred)
    submission.to_csv(args.output_path, index=False)
    test_macro_f1 = evaluate_macro_f1(test_gold, submission)
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"Predictions written to: {args.output_path}")


if __name__ == "__main__":
    main()

