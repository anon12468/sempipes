#!/usr/bin/env python3
"""Train and evaluate a leakage-safe high-enrollment classifier.

Running without arguments uses only ``table_splits/train`` and reports
time-based validation macro F1. Held-out data is accessed only when both
test paths are supplied explicitly:

  python pipeline.py \
    --test-data-dir /path/to/test \
    --gold-test-path /path/to/test/gold_enrollment_test.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

# --- Paths (anchored to repo root via this file) ---
PROJECT_ROOT = Path(__file__).resolve().parents[0]
TRAIN_DATA_DIR = PROJECT_ROOT / "table_splits" / "train"
GOLD_TRAIN_PATH = PROJECT_ROOT / "table_splits" / "train" / "gold_enrollment_train.csv"
VAL_TERM_RANK_THRESHOLD = 20150  # terms before 2015 = fit; 2015+ within train = validation
RANDOM_SEED = 42
MODEL_ITERATIONS = 700

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

SUMMARY_FEATURES = [
    "COURSE_NUMBER",
    "SUBJECT_TITLE",
    "MASTER_SUBJECT_ID_SORT",
    "CLUSTER_TYPE",
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
    """Load one row per offering using only information known before enrollment."""
    if (data_dir / SUMMARY_TABLE).exists():
        summary = read_table(data_dir, SUMMARY_TABLE)
        keep = [c for c in KEY_COLS + SUMMARY_FEATURES if c in summary.columns]
        df = summary[keep].drop_duplicates(KEY_COLS).copy()
    else:
        offered = read_table(data_dir, OFFERED_TABLE)
        keep = [
            c
            for c in KEY_COLS
            + ["OFFER_DEPT_CODE", "OFFER_SCHOOL_NAME", "SUBJECT_TITLE"]
            if c in offered.columns
        ]
        df = offered[keep].drop_duplicates(KEY_COLS).copy()

    history_path = data_dir / HISTORY_TABLE
    if history_path.exists():
        history = normalize_keys(pd.read_csv(history_path, low_memory=False))
        history = history.drop_duplicates(KEY_COLS)
        df = df.merge(history, on=KEY_COLS, how="left", validate="one_to_one")

    return add_engineered_features(df).reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, course, history, and department-relative features."""
    df = df.copy()
    df["term_rank"] = term_to_rank(df["TERM_CODE"])
    df["term_year"] = pd.to_numeric(df["TERM_CODE"].astype(str).str[:4], errors="coerce")
    df["term_season"] = df["TERM_CODE"].astype(str).str[4:6]

    subject = df["SUBJECT_ID_SORT"].fillna("").astype(str)
    df["subject_prefix"] = subject.str.split(".", n=1).str[0]
    subject_number = subject.str.split(".", n=1).str[-1]
    df["subject_number"] = pd.to_numeric(
        subject_number.str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )
    df["subject_level"] = subject_number.str.extract(r"(\d)")[0].fillna("X")

    title = df.get("SUBJECT_TITLE", pd.Series("", index=df.index)).fillna("").astype(str)
    title_lower = title.str.lower()
    df["title_length"] = title.str.len()
    for token in [
        "intro",
        "seminar",
        "thesis",
        "research",
        "special",
        "laboratory",
        "project",
        "advanced",
    ]:
        df[f"title_has_{token}"] = title_lower.str.contains(token, regex=False).astype("int8")

    numeric_source_cols = [
        "TOTAL_UNITS",
        "LECTURE_UNITS",
        "LAB_UNITS",
        "PREPARATION_UNITS",
    ]
    history_cols = ["lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]
    for col in numeric_source_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in history_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0)

    df["history_mean"] = df[history_cols].mean(axis=1)
    df["history_max"] = df[history_cols].max(axis=1)
    df["history_range"] = df[history_cols].max(axis=1) - df[history_cols].min(axis=1)
    df["history_nonzero_count"] = df[history_cols].gt(0).sum(axis=1)
    df["history_trend"] = df["lag1_enrollment"] - df["lag2_enrollment"]

    if "OFFER_DEPT_CODE" in df.columns:
        department = df["OFFER_DEPT_CODE"].fillna(df["subject_prefix"])
    else:
        department = df["subject_prefix"]
    df["model_department"] = department.fillna("__MISSING__").astype(str)
    group_cols = ["TERM_CODE", "model_department"]
    df["department_term_size"] = df.groupby(group_cols, dropna=False)[
        "SUBJECT_ID_SORT"
    ].transform("size")

    for col in history_cols:
        grouped = df.groupby(group_cols, dropna=False)[col]
        df[f"log_{col}"] = np.log1p(df[col])
        df[f"has_{col}"] = (df[col] > 0).astype("int8")
        df[f"{col}_department_percentile"] = grouped.rank(
            method="average", pct=True
        )
        df[f"{col}_department_mean_ratio"] = df[col] / (
            grouped.transform("mean") + 1.0
        )
        df[f"{col}_department_max_ratio"] = df[col] / (
            grouped.transform("max") + 1.0
        )
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "SUBJECT_ID_SORT",
        "SUBJECT_TITLE",
        "COURSE_NUMBER",
        "MASTER_SUBJECT_ID_SORT",
        "subject_prefix",
        "subject_number",
        "subject_level",
        "model_department",
        "OFFER_DEPT_CODE",
        "OFFER_SCHOOL_NAME",
        "RESPONSIBLE_FACULTY_NAME",
        "CLUSTER_TYPE",
        "HGN_CODE",
        "HGN_CODE_DESC",
        "term_year",
        "term_season",
        "TOTAL_UNITS",
        "LECTURE_UNITS",
        "LAB_UNITS",
        "PREPARATION_UNITS",
        "title_length",
        "history_mean",
        "history_max",
        "history_range",
        "history_nonzero_count",
        "history_trend",
        "department_term_size",
    ]
    engineered_prefixes = (
        "lag",
        "roll3_",
        "log_",
        "has_",
        "title_has_",
    )
    cols = base + [c for c in df.columns if c.startswith(engineered_prefixes)]
    return list(
        dict.fromkeys(
            c for c in cols if c in df.columns and c not in LEAKAGE_COLUMNS
        )
    )


def labels_to_y(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.upper() == "Y").astype(int)


def read_gold(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing gold labels: {path}")
    return normalize_keys(pd.read_csv(path))


def build_labeled_frame(data_dir: Path, gold: pd.DataFrame) -> pd.DataFrame:
    if gold.duplicated(KEY_COLS).any():
        raise ValueError("Gold labels contain duplicate prediction keys.")
    courses = load_course_offerings(data_dir)
    labeled = gold[KEY_COLS + [TARGET_COL]].merge(
        courses,
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )
    labeled["y"] = labels_to_y(labeled[TARGET_COL]).astype(int)
    labeled["term_rank"] = term_to_rank(labeled["TERM_CODE"])
    return labeled


def evaluate_macro_f1(gold: pd.DataFrame, pred: pd.DataFrame) -> float:
    gold = normalize_keys(gold.copy())
    pred = normalize_keys(pred.copy())
    if gold.duplicated(KEY_COLS).any() or pred.duplicated(KEY_COLS).any():
        raise ValueError("Gold and prediction keys must each be unique.")
    gold[TARGET_COL] = gold[TARGET_COL].astype(str).str.strip().str.upper()
    pred[TARGET_COL] = pred[TARGET_COL].astype(str).str.strip().str.upper()

    gold = gold[gold[TARGET_COL].isin(["Y", "N"])]
    merged = gold.merge(
        pred,
        on=KEY_COLS,
        how="left",
        suffixes=("_gold", "_pred"),
        validate="one_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("Predictions are missing one or more gold keys.")
    pred_col = f"{TARGET_COL}_pred"
    if not merged[pred_col].isin(["Y", "N"]).all():
        raise ValueError("Every prediction must be either 'Y' or 'N'.")

    y_true = (merged[f"{TARGET_COL}_gold"] == "Y").astype(int)
    y_pred = (merged[pred_col] == "Y").astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# --- Model ---

def make_model(iterations: int = MODEL_ITERATIONS) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=8,
        learning_rate=0.07,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        l2_leaf_reg=6.0,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )


def prepare_model_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    numeric_medians: dict[str, float],
) -> pd.DataFrame:
    out = df.reindex(columns=feature_cols).copy()
    for col in categorical_cols:
        out[col] = out[col].fillna("__MISSING__").astype(str)
    for col in (c for c in feature_cols if c not in categorical_cols):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(
            numeric_medians.get(col, 0.0)
        )
    return out


def preprocessing_spec(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], dict[str, float]]:
    frame = df.reindex(columns=feature_cols)
    categorical_cols = [
        col for col in feature_cols if not pd.api.types.is_numeric_dtype(frame[col])
    ]
    numeric_medians = {
        col: float(pd.to_numeric(frame[col], errors="coerce").median())
        for col in feature_cols
        if col not in categorical_cols
    }
    numeric_medians = {
        col: value if np.isfinite(value) else 0.0
        for col, value in numeric_medians.items()
    }
    return categorical_cols, numeric_medians


def fit_threshold(y_val: pd.Series, prob_val: pd.Series) -> float:
    """Choose the macro-F1 decision threshold on the time-held-out slice."""
    if len(prob_val) == 0 or y_val.nunique() < 2:
        return 0.5
    candidates = np.linspace(0.05, 0.75, 701)
    scores = [
        f1_score(y_val, prob_val >= threshold, average="macro", zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def train_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame | None,
    feature_cols: list[str],
) -> tuple[pd.Series | None, float, float | None]:
    fit_mask = train_df["term_rank"] < VAL_TERM_RANK_THRESHOLD
    val_mask = ~fit_mask
    if not fit_mask.any():
        raise ValueError("The validation cutoff leaves no rows for model fitting.")

    y_fit = train_df.loc[fit_mask, "y"]
    y_val = train_df.loc[val_mask, "y"]
    categorical_cols, medians = preprocessing_spec(
        train_df.loc[fit_mask], feature_cols
    )
    x_fit = prepare_model_frame(
        train_df.loc[fit_mask], feature_cols, categorical_cols, medians
    )
    x_val = prepare_model_frame(
        train_df.loc[val_mask], feature_cols, categorical_cols, medians
    )

    validation_model = make_model()
    fit_kwargs: dict[str, object] = {"cat_features": categorical_cols}
    if len(x_val):
        fit_kwargs.update(
            {
                "eval_set": (x_val, y_val),
                "early_stopping_rounds": 70,
            }
        )
    validation_model.fit(x_fit, y_fit, **fit_kwargs)

    if len(x_val):
        prob_val = pd.Series(
            validation_model.predict_proba(x_val)[:, 1],
            index=x_val.index,
        )
        threshold = fit_threshold(y_val, prob_val)
        val_pred = (prob_val >= threshold).astype(int)
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
    else:
        threshold = 0.5
        val_macro_f1 = None

    if predict_df is None:
        return None, threshold, val_macro_f1

    best_iteration = validation_model.get_best_iteration()
    final_iterations = (
        max(100, best_iteration + 1) if best_iteration >= 0 else MODEL_ITERATIONS
    )
    categorical_cols, medians = preprocessing_spec(train_df, feature_cols)
    x_all = prepare_model_frame(
        train_df, feature_cols, categorical_cols, medians
    )
    x_predict = prepare_model_frame(
        predict_df, feature_cols, categorical_cols, medians
    )
    final_model = make_model(iterations=final_iterations)
    final_model.fit(x_all, train_df["y"], cat_features=categorical_cols)

    prob_pred = pd.Series(
        final_model.predict_proba(x_predict)[:, 1],
        index=predict_df.index,
    )
    pred = (prob_pred >= threshold).astype(int)
    return pred, threshold, val_macro_f1


def submission_frame(predict_df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    out = predict_df[KEY_COLS].copy()
    out[TARGET_COL] = pred.map({1: "Y", 0: "N"})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        help="Held-out table directory; omitted for training-only validation.",
    )
    parser.add_argument(
        "--gold-test-path",
        type=Path,
        help="Held-out gold CSV used for prediction keys and external evaluation.",
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        help="Optional destination for held-out Y/N predictions.",
    )
    args = parser.parse_args()
    if (args.test_data_dir is None) != (args.gold_test_path is None):
        parser.error("--test-data-dir and --gold-test-path must be supplied together.")
    if args.prediction_output is not None and args.test_data_dir is None:
        parser.error("--prediction-output requires held-out test arguments.")
    return args


def main() -> None:
    args = parse_args()
    train_df = build_labeled_frame(TRAIN_DATA_DIR, read_gold(GOLD_TRAIN_PATH))
    feature_cols = feature_columns(train_df)
    if not feature_cols:
        raise RuntimeError("No features available; check ENROLLMENT_HISTORY_FEATURES.csv exists.")

    if args.test_data_dir is None:
        _, threshold, val_macro_f1 = train_and_predict(train_df, None, feature_cols)
        if val_macro_f1 is not None:
            print(f"Val macro F1: {val_macro_f1:.4f}")
            print(f"Validation threshold: {threshold:.3f}")
        return

    test_gold = read_gold(args.gold_test_path)
    predict_df = test_gold[KEY_COLS].merge(
        load_course_offerings(args.test_data_dir),
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )

    test_pred, threshold, val_macro_f1 = train_and_predict(
        train_df, predict_df, feature_cols
    )
    assert test_pred is not None
    predictions = submission_frame(predict_df, test_pred)
    if val_macro_f1 is not None:
        print(f"Val macro F1: {val_macro_f1:.4f}")
        print(f"Validation threshold: {threshold:.3f}")

    test_macro_f1 = evaluate_macro_f1(test_gold, predictions)
    print(f"Test macro F1: {test_macro_f1:.4f}")
    if args.prediction_output is not None:
        predictions.to_csv(args.prediction_output, index=False)
        print(f"Predictions written to: {args.prediction_output}")


if __name__ == "__main__":
    main()

