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
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

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

HISTORY_COLUMNS = ["lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]
CATEGORICAL_FEATURES = [
    "term_season",
    "OFFER_DEPT_CODE",
    "OFFER_SCHOOL_NAME",
    "CLUSTER_TYPE",
    "HGN_CODE",
    "subject_prefix",
    "SUBJECT_ID_SORT",
    "MASTER_SUBJECT_ID_SORT",
    "SUBJECT_TITLE",
    "title_first_word",
]
NUMERIC_FEATURES = [
    *HISTORY_COLUMNS,
    *[f"log_{col}" for col in HISTORY_COLUMNS],
    *[f"{col}_dept_pct" for col in HISTORY_COLUMNS],
    *[f"{col}_dept_ratio" for col in HISTORY_COLUMNS],
    "history_max",
    "history_nonzero",
    "history_trend",
    "dept_term_size",
    "TOTAL_UNITS",
    "LECTURE_UNITS",
    "LAB_UNITS",
    "PREPARATION_UNITS",
    "subject_number",
    "subject_level",
    "term_year",
    "section_row_count",
    "lecture_section_count",
    "lab_section_count",
    "recitation_section_count",
    "design_section_count",
    "master_section_count",
    "evaluated_section_count",
    "repeatable_section_count",
    "instructor_count",
    "meet_time_count",
    "meet_place_count",
    "title_word_count",
    "title_length",
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
    if usecols is not None:
        available = set(pd.read_csv(path, nrows=0).columns)
        usecols = [col for col in usecols if col in available]
    return normalize_keys(pd.read_csv(path, usecols=usecols, low_memory=False))


def aggregate_sections(data_dir: Path) -> pd.DataFrame | None:
    """Aggregate section-level scheduling signals to one row per course offering."""
    path = data_dir / OFFERED_TABLE
    if not path.exists():
        return None

    flag_columns = {
        "IS_LECTURE_SECTION": "lecture_section_count",
        "IS_LAB_SECTION": "lab_section_count",
        "IS_RECITATION_SECTION": "recitation_section_count",
        "IS_DESIGN_SECTION": "design_section_count",
        "IS_MASTER_SECTION": "master_section_count",
        "EVALUATE_THIS_SUBJECT": "evaluated_section_count",
        "IS_REPEATABLE_SUBJECT": "repeatable_section_count",
    }
    count_columns = {
        "RESPONSIBLE_FACULTY_NAME": "instructor_count",
        "MEET_TIME": "meet_time_count",
        "MEET_PLACE": "meet_place_count",
    }
    offered = read_table(
        data_dir,
        OFFERED_TABLE,
        KEY_COLS + list(flag_columns) + list(count_columns),
    )
    grouped = offered.groupby(KEY_COLS, dropna=False, sort=False)
    frames = [grouped.size().rename("section_row_count")]

    for source, output in flag_columns.items():
        if source in offered.columns:
            offered[output] = (offered[source].astype(str).str.upper() == "Y").astype("int8")
            frames.append(offered.groupby(KEY_COLS, dropna=False, sort=False)[output].sum())
    for source, output in count_columns.items():
        if source in offered.columns:
            frames.append(grouped[source].nunique(dropna=True).rename(output))

    return pd.concat(frames, axis=1).reset_index()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe temporal, course, and within-department features."""
    out = df.copy()
    out["term_rank"] = term_to_rank(out["TERM_CODE"])
    out["term_year"] = pd.to_numeric(out["TERM_CODE"].astype(str).str[:4], errors="coerce")
    out["term_season"] = out["TERM_CODE"].astype(str).str[4:]
    out["subject_prefix"] = out["SUBJECT_ID_SORT"].astype(str).str.split(".").str[0]
    out["subject_number"] = pd.to_numeric(
        out["SUBJECT_ID_SORT"].astype(str).str.extract(r"\.(\d+)")[0],
        errors="coerce",
    )
    out["subject_level"] = pd.cut(
        out["subject_number"],
        [-np.inf, 99, 199, 499, 999, np.inf],
        labels=False,
    )

    title = out.get("SUBJECT_TITLE", pd.Series("", index=out.index)).fillna("").astype(str)
    out["title_word_count"] = title.str.count(r"\S+")
    out["title_length"] = title.str.len()
    out["title_first_word"] = title.str.extract(r"^\s*([A-Za-z]+)")[0]

    existing_history = [col for col in HISTORY_COLUMNS if col in out.columns]
    for col in existing_history:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))
        group = out.groupby(["TERM_CODE", "OFFER_DEPT_CODE"], dropna=False)[col]
        out[f"{col}_dept_pct"] = group.rank(pct=True, method="average")
        out[f"{col}_dept_ratio"] = out[col] / (group.transform("median") + 1.0)

    if existing_history:
        out["history_max"] = out[existing_history].max(axis=1)
        out["history_nonzero"] = (out[existing_history] > 0).sum(axis=1)
    if {"lag1_enrollment", "lag2_enrollment"}.issubset(out.columns):
        out["history_trend"] = out["lag1_enrollment"] - out["lag2_enrollment"]
    out["dept_term_size"] = out.groupby(
        ["TERM_CODE", "OFFER_DEPT_CODE"],
        dropna=False,
    )["TERM_CODE"].transform("size")

    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].fillna("__MISSING__").astype(str)
    for col in NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
    return out


def load_course_offerings(data_dir: Path) -> pd.DataFrame:
    """Load only features available before enrollment is observed."""
    summary_columns = KEY_COLS + [
        "SUBJECT_TITLE",
        "MASTER_SUBJECT_ID_SORT",
        "CLUSTER_TYPE",
        "HGN_CODE",
        "OFFER_DEPT_CODE",
        "OFFER_SCHOOL_NAME",
        "TOTAL_UNITS",
        "LECTURE_UNITS",
        "LAB_UNITS",
        "PREPARATION_UNITS",
    ]
    df = read_table(data_dir, SUMMARY_TABLE, summary_columns).drop_duplicates(KEY_COLS)

    history_path = data_dir / HISTORY_TABLE
    if history_path.exists():
        history = read_table(data_dir, HISTORY_TABLE, KEY_COLS + HISTORY_COLUMNS)
        df = df.merge(history.drop_duplicates(KEY_COLS), on=KEY_COLS, how="left", validate="one_to_one")

    sections = aggregate_sections(data_dir)
    if sections is not None:
        df = df.merge(sections, on=KEY_COLS, how="left", validate="one_to_one")

    return add_engineered_features(df).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if col in df.columns and col not in LEAKAGE_COLUMNS
    ]


def labels_to_y(series: pd.Series) -> pd.Series:
    return (series.astype(str) == "Y").astype(int)


def read_gold(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing gold labels: {path}")
    return normalize_keys(pd.read_csv(path))


def build_labeled_frame(data_dir: Path, gold: pd.DataFrame) -> pd.DataFrame:
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
    gold[TARGET_COL] = gold[TARGET_COL].astype(str).str.strip().str.upper()
    pred[TARGET_COL] = pred[TARGET_COL].astype(str).str.strip().str.upper()

    gold = gold[gold[TARGET_COL].isin(["Y", "N"])]
    if pred.duplicated(KEY_COLS).any():
        raise ValueError("Predictions contain duplicate keys.")
    merged = gold.merge(
        pred,
        on=KEY_COLS,
        how="left",
        suffixes=("_gold", "_pred"),
        validate="one_to_one",
    )
    valid_prediction = merged[f"{TARGET_COL}_pred"].isin(["Y", "N"])
    if not valid_prediction.all():
        raise ValueError(f"Missing or invalid predictions for {(~valid_prediction).sum()} gold rows.")

    y_true = (merged[f"{TARGET_COL}_gold"] == "Y").astype(int)
    y_pred = (merged[f"{TARGET_COL}_pred"] == "Y").astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# --- Model ---

def make_model(iterations: int = 500) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=8,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_seed=42,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


def model_categorical_features(feature_cols: list[str]) -> list[str]:
    return [col for col in CATEGORICAL_FEATURES if col in feature_cols]


def prepare_model_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Make train and prediction frames schema-compatible for CatBoost."""
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = "__MISSING__" if col in CATEGORICAL_FEATURES else np.nan
        elif col in CATEGORICAL_FEATURES:
            out[col] = out[col].fillna("__MISSING__").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
    return out[feature_cols]


def fit_threshold(y_val: pd.Series, prob_val: pd.Series) -> float:
    """Choose the exact validation threshold that maximizes macro F1."""
    if len(prob_val) == 0 or y_val.nunique() < 2:
        return 0.5

    probability = np.asarray(prob_val, dtype=float)
    truth = np.asarray(y_val, dtype=int)
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    sorted_truth = truth[order]

    predicted_positive = np.arange(1, len(truth) + 1)
    true_positive = np.cumsum(sorted_truth)
    false_positive = predicted_positive - true_positive
    false_negative = true_positive[-1] - true_positive
    true_negative = len(truth) - predicted_positive - false_negative

    positive_f1 = np.divide(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
        out=np.zeros_like(true_positive, dtype=float),
        where=(2 * true_positive + false_positive + false_negative) != 0,
    )
    negative_f1 = np.divide(
        2 * true_negative,
        2 * true_negative + false_positive + false_negative,
        out=np.zeros_like(true_negative, dtype=float),
        where=(2 * true_negative + false_positive + false_negative) != 0,
    )
    macro_f1 = (positive_f1 + negative_f1) / 2.0

    # A threshold cannot split rows with identical probabilities.
    valid_candidate = np.r_[sorted_probability[:-1] != sorted_probability[1:], True]
    candidate_indices = np.flatnonzero(valid_candidate)
    best_index = candidate_indices[np.argmax(macro_f1[candidate_indices])]
    return float(sorted_probability[best_index])


def validation_masks(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    fit_mask = train_df["term_rank"] < VAL_TERM_RANK_THRESHOLD
    val_mask = ~fit_mask
    if fit_mask.any() and val_mask.any():
        return fit_mask, val_mask

    ranks = np.sort(train_df["term_rank"].dropna().unique())
    if len(ranks) < 2:
        raise ValueError("At least two distinct terms are required for time validation.")
    fallback_threshold = ranks[max(1, int(len(ranks) * 0.8))]
    fit_mask = train_df["term_rank"] < fallback_threshold
    return fit_mask, ~fit_mask


def validate_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[float, float, int]:
    fit_mask, val_mask = validation_masks(train_df)
    x_fit = prepare_model_frame(train_df.loc[fit_mask], feature_cols)
    x_val = prepare_model_frame(train_df.loc[val_mask], feature_cols)
    y_fit = train_df.loc[fit_mask, "y"]
    y_val = train_df.loc[val_mask, "y"]

    model = make_model()
    model.fit(
        x_fit,
        y_fit,
        cat_features=model_categorical_features(feature_cols),
        eval_set=(x_val, y_val),
        early_stopping_rounds=50,
        use_best_model=True,
    )
    probability = pd.Series(model.predict_proba(x_val)[:, 1], index=x_val.index)
    threshold = fit_threshold(y_val, probability)
    prediction = (probability >= threshold).astype(int)
    score = float(f1_score(y_val, prediction, average="macro", zero_division=0))
    best_iteration = model.get_best_iteration()
    iterations = max(1, best_iteration + 1) if best_iteration >= 0 else model.get_param("iterations")
    return score, threshold, int(iterations)


def train_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.Series, float, float]:
    val_macro_f1, threshold, iterations = validate_model(train_df, feature_cols)
    x_train = prepare_model_frame(train_df, feature_cols)
    x_predict = prepare_model_frame(predict_df, feature_cols)

    model = make_model(iterations=iterations)
    model.fit(
        x_train,
        train_df["y"],
        cat_features=model_categorical_features(feature_cols),
    )
    prob_pred = pd.Series(model.predict_proba(x_predict)[:, 1], index=predict_df.index)
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
    parser.add_argument("--gold-train", type=Path, default=GOLD_TRAIN_PATH)
    parser.add_argument("--test-data-dir", type=Path, default=TEST_DATA_DIR)
    parser.add_argument("--gold-test", type=Path, default=GOLD_TEST_PATH)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=PROJECT_ROOT / "test_predictions.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_gold = read_gold(args.gold_train)
    train_df = build_labeled_frame(args.train_data_dir, train_gold)
    feature_cols = feature_columns(train_df)
    if not feature_cols:
        raise RuntimeError("No features available; check ENROLLMENT_HISTORY_FEATURES.csv exists.")

    val_macro_f1, threshold, iterations = validate_model(train_df, feature_cols)
    print(f"Val macro F1:  {val_macro_f1:.4f}")
    print(f"Threshold:       {threshold:.4f}")
    print(f"Model iterations: {iterations}")

    if args.test_data_dir is None and args.gold_test is None:
        print("Test data not configured; validation completed using training data only.")
        return
    if args.test_data_dir is None or args.gold_test is None:
        raise ValueError("--test-data-dir and --gold-test must be supplied together.")

    test_gold = read_gold(args.gold_test)
    predict_df = test_gold[KEY_COLS].merge(
        load_course_offerings(args.test_data_dir),
        on=KEY_COLS,
        how="left",
        validate="one_to_one",
    )

    x_train = prepare_model_frame(train_df, feature_cols)
    x_test = prepare_model_frame(predict_df, feature_cols)
    model = make_model(iterations=iterations)
    model.fit(
        x_train,
        train_df["y"],
        cat_features=model_categorical_features(feature_cols),
    )
    test_probability = model.predict_proba(x_test)[:, 1]
    test_pred = pd.Series((test_probability >= threshold).astype(int), index=predict_df.index)
    submission = submission_frame(predict_df, test_pred)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.predictions_out, index=False)
    print(f"Predictions:      {args.predictions_out}")

    if TARGET_COL in test_gold.columns:
        test_macro_f1 = evaluate_macro_f1(test_gold, submission)
        print(f"Test macro F1:   {test_macro_f1:.4f}")


if __name__ == "__main__":
    main()

