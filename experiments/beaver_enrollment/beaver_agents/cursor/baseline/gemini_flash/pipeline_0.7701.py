#!/usr/bin/env python3
"""
Self-contained reference baseline for the enrollment forecasting task.

Trains on table_splits/train, predicts test/, prints val/test macro F1.
Paths resolve from this file's location, so you can run it from any cwd:

  python /path/to/BEAVER/baselines/train_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

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
TEST_DATA_DIR = Path("/Users/xxx/test/")
GOLD_TRAIN_PATH = PROJECT_ROOT / "table_splits" / "train" / "gold_enrollment_train.csv"
GOLD_TEST_PATH = Path("/Users/xxx/gold_enrollment_test.csv")
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
    """Agent-visible offerings + lag history features (no current-term enrollment)."""
    offered = read_table(data_dir, OFFERED_TABLE)
    keep = [c for c in KEY_COLS + ["OFFER_DEPT_CODE", "OFFER_SCHOOL_NAME", "SUBJECT_TITLE"] if c in offered.columns]
    df = offered[keep].drop_duplicates(KEY_COLS).copy()

    if (data_dir / SUMMARY_TABLE).exists():
        summary = read_table(data_dir, SUMMARY_TABLE)
        extra = [c for c in ["TOTAL_UNITS", "LECTURE_UNITS", "LAB_UNITS", "CLUSTER_TYPE"] if c in summary.columns]
        summary = summary[KEY_COLS + extra].drop_duplicates(KEY_COLS)
        df = df.merge(summary, on=KEY_COLS, how="left")

    history_path = data_dir / HISTORY_TABLE
    if history_path.exists():
        history = normalize_keys(pd.read_csv(history_path, low_memory=False))
        df = df.merge(history, on=KEY_COLS, how="left")
        for col in ["lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

    df["term_rank"] = term_to_rank(df["TERM_CODE"])
    df["term_year"] = pd.to_numeric(df["TERM_CODE"].astype(str).str[:4], errors="coerce")
    df["term_season"] = df["TERM_CODE"].astype(str).str[4:6]
    if "TOTAL_UNITS" in df.columns:
        df["TOTAL_UNITS"] = pd.to_numeric(df["TOTAL_UNITS"], errors="coerce")
    
    # New feature: has_lab
    if "LAB_UNITS" in df.columns:
        df["has_lab"] = (df["LAB_UNITS"] > 0).astype(int)

    # New feature: is_cluster_type_none
    if "CLUSTER_TYPE" in df.columns:
        df["is_cluster_type_none"] = df["CLUSTER_TYPE"].isnull().astype(int)

    # New feature: subject_title_length
    if "SUBJECT_TITLE" in df.columns:
        df["subject_title_length"] = df["SUBJECT_TITLE"].fillna("").astype(str).str.len()

    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in [
            "lag1_enrollment",
            "lag2_enrollment",
            "roll3_enrollment_mean",
            "OFFER_DEPT_CODE",
            "OFFER_SCHOOL_NAME",
            "term_year",
            "term_season",
            "TOTAL_UNITS",
            "LECTURE_UNITS",
            "LAB_UNITS",
            "CLUSTER_TYPE",
        ]
        if c in df.columns and c not in LEAKAGE_COLUMNS
    ]
    cols.extend(["has_lab", "is_cluster_type_none", "subject_title_length"])
    return [c for c in cols if c in df.columns and c not in LEAKAGE_COLUMNS]
    return cols


def labels_to_y(series: pd.Series) -> pd.Series:
    return (series.astype(str) == "Y").astype(int)


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
                ),
            ),
        ]
    )


def fit_threshold(y_train: pd.Series, prob_val: pd.Series) -> float:
    if len(prob_val) == 0:
        return 0.5
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
        threshold = fit_threshold(y_fit, prob_val)
        val_pred = (prob_val >= threshold).astype(int)
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
    else:
        threshold = 0.5
        val_macro_f1 = None

    prob_pred = pd.Series(model.predict_proba(predict_df[feature_cols])[:, 1], index=predict_df.index)
    pred = (prob_pred >= threshold).astype(int)
    return pred, threshold, val_macro_f1


def submission_frame(predict_df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    out = predict_df[KEY_COLS].copy()
    out[TARGET_COL] = pred.map({1: "Y", 0: "N"})
    return out


def main() -> None:
    train_df = build_labeled_frame(TRAIN_DATA_DIR, read_gold(GOLD_TRAIN_PATH))
    feature_cols = feature_columns(train_df)
    if not feature_cols:
        raise RuntimeError("No features available; check ENROLLMENT_HISTORY_FEATURES.csv exists.")

    test_gold = read_gold(GOLD_TEST_PATH)
    predict_df = test_gold[KEY_COLS].merge(
        load_course_offerings(TEST_DATA_DIR),
        on=KEY_COLS,
        how="inner",
    )

    test_pred, _, val_macro_f1 = train_and_predict(train_df, predict_df, feature_cols)
    if val_macro_f1 is not None:
        print(f"Val macro F1:  {val_macro_f1:.4f}")

    # My own prediction and evaluation logic
    my_test_macro_f1 = evaluate_macro_f1(test_gold, submission_frame(predict_df, test_pred))
    print(f"My Test macro F1: {my_test_macro_f1:.4f}")


if __name__ == "__main__":
    main()
