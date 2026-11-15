#!/usr/bin/env python3
"""Train and run a leakage-safe parking-violation count forecaster."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
except ImportError:  # The non-temporal fallback still works without LightGBM.
    LGBMRegressor = None


STREET_COL = "Street Name"
TYPE_COL = "Violation Description"
TARGET_COL = "violation_count"
KEY_COLS = ["street_norm", "type_norm"]

HISTORY_DATASETS = {
    2020: "p7t3-5i9s",
    2021: "kvfd-bves",
}

CODE_TO_TYPE = {
    "5": "BUS LANE VIOLATION",
    "7": "FAILURE TO STOP AT RED LIGHT",
    "14": "14-No Standing",
    "20": "20A-No Parking (Non-COM)",
    "21": "21-No Parking (street clean)",
    "36": "PHTO SCHOOL ZN SPEED VIOLATION",
    "38": "38-Failure to Dsplay Meter Rec",
    "40": "40-Fire Hydrant",
    "70": "70A-Reg. Sticker Expired (NYS)",
    "71": "71A-Insp Sticker Expired (NYS)",
}

CAMERA_TYPES = {
    "PHTO SCHOOL ZN SPEED VIOLATION",
    "FAILURE TO STOP AT RED LIGHT",
    "BUS LANE VIOLATION",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast NYC parking violations by (street, violation type)."
    )
    parser.add_argument(
        "--train-path",
        default="violations_per_street_2022.csv",
        help="2022 aggregate CSV used as the primary training signal.",
    )
    parser.add_argument(
        "--test-path",
        default=None,
        help="Optional keys-only or labeled test/eval CSV.",
    )
    parser.add_argument(
        "--output-path",
        default="submission.csv",
        help="Prediction CSV path (default: submission.csv).",
    )
    parser.add_argument(
        "--history-2021-path",
        default="corpus/violations_per_street_2021.csv",
        help="Cached 2021 historical aggregate.",
    )
    parser.add_argument(
        "--history-2020-path",
        default="corpus/violations_per_street_2020.csv",
        help="Cached 2020 historical aggregate.",
    )
    parser.add_argument(
        "--no-download-history",
        action="store_true",
        help="Do not download missing historical aggregates from NYC Open Data.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help="Fraction of 2022 rows reserved for development validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else base_dir / value


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().upper())


def canonicalize_street(value: object) -> str:
    text = normalize_text(value).replace(".", "")
    replacements = {
        "AVENUE": "AVE",
        "STREET": "ST",
        "BOULEVARD": "BLVD",
        "ROAD": "RD",
        "PLACE": "PL",
        "DRIVE": "DR",
        "LANE": "LN",
        "PARKWAY": "PKWY",
        "EXPRESSWAY": "EXPWY",
    }
    parts = [replacements.get(token, token) for token in text.split()]
    return " ".join(parts)


def read_core(path: Path, require_target: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    frame = pd.read_csv(path)
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    aliases = {
        STREET_COL: ["street name", "street_name"],
        TYPE_COL: ["violation description", "violation_type", "violation_description"],
        TARGET_COL: ["violation_count", "violation count"],
    }

    selected: dict[str, pd.Series] = {}
    for canonical, options in aliases.items():
        source = next((lookup[name] for name in options if name in lookup), None)
        if source is not None:
            selected[canonical] = frame[source]

    missing_keys = [col for col in (STREET_COL, TYPE_COL) if col not in selected]
    if missing_keys:
        raise ValueError(
            f"{path} is missing required key columns: {', '.join(missing_keys)}"
        )
    if require_target and TARGET_COL not in selected:
        raise ValueError(f"{path} is missing required target column: {TARGET_COL}")

    result = pd.DataFrame(selected)
    result[STREET_COL] = result[STREET_COL].fillna("").astype(str)
    result[TYPE_COL] = result[TYPE_COL].fillna("").astype(str)
    if (result[STREET_COL].str.strip() == "").any() or (
        result[TYPE_COL].str.strip() == ""
    ).any():
        raise ValueError(f"{path} contains empty street or violation-type keys")

    if TARGET_COL in result:
        result[TARGET_COL] = pd.to_numeric(result[TARGET_COL], errors="coerce")
        if require_target and result[TARGET_COL].isna().any():
            raise ValueError(f"{path} contains non-numeric or missing target values")
        if (result[TARGET_COL].dropna() < 0).any():
            raise ValueError(f"{path} contains negative target values")

    return result


def prepare_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[[STREET_COL, TYPE_COL]].copy()
    rows["street_norm"] = rows[STREET_COL].map(normalize_text)
    rows["street_canon"] = rows[STREET_COL].map(canonicalize_street)
    rows["type_norm"] = rows[TYPE_COL].map(normalize_text)
    if TARGET_COL in frame:
        rows[TARGET_COL] = frame[TARGET_COL].to_numpy()
    return rows


def map_historical_type(code: object, description: object) -> str | None:
    code_text = normalize_text(code)
    if code_text.endswith(".0"):
        code_text = code_text[:-2]
    if code_text in CODE_TO_TYPE:
        return CODE_TO_TYPE[code_text]

    description_text = normalize_text(description)
    if description_text in CAMERA_TYPES:
        return description_text
    return None


def standardize_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept either raw Socrata rows or a previously standardized cache."""
    columns = {str(col).strip().lower(): col for col in frame.columns}
    if {"street name", "violation description", "violation_count"} <= set(columns):
        standardized = frame[
            [
                columns["street name"],
                columns["violation description"],
                columns["violation_count"],
            ]
        ].copy()
        standardized.columns = [STREET_COL, TYPE_COL, TARGET_COL]
    elif {"street_name", "violation_code", "violation_count"} <= set(columns):
        descriptions = (
            frame[columns["violation_description"]]
            if "violation_description" in columns
            else pd.Series("", index=frame.index)
        )
        standardized = pd.DataFrame(
            {
                STREET_COL: frame[columns["street_name"]],
                TYPE_COL: [
                    map_historical_type(code, description)
                    for code, description in zip(
                        frame[columns["violation_code"]],
                        descriptions,
                    )
                ],
                TARGET_COL: frame[columns["violation_count"]],
            }
        )
        standardized = standardized[standardized[TYPE_COL].notna()]
    else:
        raise ValueError("Unrecognized historical aggregate schema")

    standardized[TARGET_COL] = pd.to_numeric(
        standardized[TARGET_COL], errors="coerce"
    ).fillna(0.0)
    standardized = prepare_rows(standardized)
    return (
        standardized.groupby(KEY_COLS, as_index=False, sort=False)[TARGET_COL]
        .sum()
        .astype({TARGET_COL: float})
    )


def download_history(year: int, output_path: Path) -> pd.DataFrame:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required to download history") from exc

    dataset_id = HISTORY_DATASETS[year]
    url = f"https://data.cityofnewyork.us/resource/{dataset_id}.json"
    selected_codes = ",".join(f"'{code}'" for code in CODE_TO_TYPE)
    where = (
        "street_name is not null AND "
        f"violation_code in ({selected_codes})"
    )
    select = "street_name, violation_code, count(*) as violation_count"
    group = "street_name, violation_code"

    page_size = 200_000
    offset = 0
    pages: list[pd.DataFrame] = []
    print(f"Downloading filtered FY{year} aggregate from NYC Open Data ...")
    while True:
        response = requests.get(
            url,
            params={
                "$select": select,
                "$where": where,
                "$group": group,
                "$order": group,
                "$limit": page_size,
                "$offset": offset,
            },
            timeout=240,
        )
        response.raise_for_status()
        records = response.json()
        if not records:
            break
        pages.append(pd.DataFrame.from_records(records))
        if len(records) < page_size:
            break
        offset += page_size

    if not pages:
        raise RuntimeError(f"NYC Open Data returned no FY{year} history rows")

    raw = pd.concat(pages, ignore_index=True)
    history = standardize_history(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = history.rename(
        columns={
            "street_norm": STREET_COL,
            "type_norm": TYPE_COL,
        }
    )[[STREET_COL, TYPE_COL, TARGET_COL]]
    cache.to_csv(output_path, index=False)
    print(f"Cached {len(cache):,} FY{year} pairs at {output_path}")
    return history


def load_history(
    year: int, path: Path, allow_download: bool
) -> pd.DataFrame | None:
    if path.exists():
        return standardize_history(pd.read_csv(path))
    if not allow_download:
        return None
    try:
        return download_history(year, path)
    except Exception as exc:
        print(f"Warning: could not load FY{year} history: {exc}", file=sys.stderr)
        return None


def empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=[*KEY_COLS, TARGET_COL])


def _mapping(
    frame: pd.DataFrame, group_cols: str | list[str], aggregation: str = "sum"
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.groupby(group_cols)[TARGET_COL].agg(aggregation)


def build_features(
    rows: pd.DataFrame, lag1: pd.DataFrame, lag2: pd.DataFrame
) -> pd.DataFrame:
    features = rows[["street_norm", "street_canon", "type_norm"]].copy()
    work = rows[KEY_COLS].copy()

    # Normalization can collapse spelling/case variants from a source file.
    # Averaging keeps the lookup unique without multiplying a forecast across
    # duplicate aliases.
    pair1 = _mapping(lag1, KEY_COLS, "mean")
    pair2 = _mapping(lag2, KEY_COLS, "mean")
    keys = pd.MultiIndex.from_frame(work)
    features["lag1"] = keys.map(pair1).fillna(0.0).to_numpy(dtype=float)
    features["lag2"] = keys.map(pair2).fillna(0.0).to_numpy(dtype=float)
    features["seen_lag1"] = keys.isin(pair1.index).astype(np.int8)
    features["seen_lag2"] = keys.isin(pair2.index).astype(np.int8)

    for label, history in (("lag1", lag1), ("lag2", lag2)):
        street_total = _mapping(history, "street_norm")
        type_total = _mapping(history, "type_norm")
        street_pairs = _mapping(history, "street_norm", "size")
        type_streets = _mapping(history, "type_norm", "size")
        features[f"street_{label}_total"] = (
            rows["street_norm"].map(street_total).fillna(0.0).to_numpy()
        )
        features[f"type_{label}_total"] = (
            rows["type_norm"].map(type_total).fillna(0.0).to_numpy()
        )
        features[f"street_{label}_pairs"] = (
            rows["street_norm"].map(street_pairs).fillna(0.0).to_numpy()
        )
        features[f"type_{label}_streets"] = (
            rows["type_norm"].map(type_streets).fillna(0.0).to_numpy()
        )

    numeric_bases = [
        "lag1",
        "lag2",
        "street_lag1_total",
        "street_lag2_total",
        "type_lag1_total",
        "type_lag2_total",
    ]
    for col in numeric_bases:
        features[f"log1p_{col}"] = np.log1p(features[col].clip(lower=0.0))

    features["pair_delta"] = features["lag1"] - features["lag2"]
    features["pair_ratio"] = (features["lag1"] + 1.0) / (
        features["lag2"] + 1.0
    )
    features["pair_ratio"] = features["pair_ratio"].clip(0.0, 20.0)
    features["street_ratio"] = (features["street_lag1_total"] + 1.0) / (
        features["street_lag2_total"] + 1.0
    )
    features["type_ratio"] = (features["type_lag1_total"] + 1.0) / (
        features["type_lag2_total"] + 1.0
    )
    features["street_share"] = features["lag1"] / (
        features["street_lag1_total"] + 1.0
    )
    features["type_share"] = features["lag1"] / (
        features["type_lag1_total"] + 1.0
    )

    street = rows[STREET_COL].astype(str)
    features["street_length"] = street.str.len().astype(float)
    features["street_tokens"] = street.str.split().str.len().astype(float)
    features["street_digits"] = street.str.count(r"\d").astype(float)
    features["has_at_sign"] = street.str.contains("@", regex=False).astype(np.int8)
    features["has_photo_token"] = (
        street.str.upper().str.contains("PHTO|PHOTO", regex=True).astype(np.int8)
    )
    return features


def encode_categories(
    train_features: pd.DataFrame, other_features: Iterable[pd.DataFrame]
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    train_encoded = train_features.copy()
    others = [frame.copy() for frame in other_features]
    categorical = ["street_norm", "street_canon", "type_norm"]
    for col in categorical:
        all_values = pd.concat(
            [train_encoded[col], *(frame[col] for frame in others)],
            ignore_index=True,
        ).astype(str)
        categories = pd.Index(all_values.unique())
        train_encoded[col] = pd.Categorical(
            train_encoded[col].astype(str), categories=categories
        )
        for frame in others:
            frame[col] = pd.Categorical(frame[col].astype(str), categories=categories)
    return train_encoded, others, categorical


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def best_blend(
    y_true: np.ndarray, model_prediction: np.ndarray, persistence: np.ndarray
) -> tuple[float, float]:
    best_weight = 1.0
    best_score = float("inf")
    for weight in np.linspace(0.0, 1.0, 101):
        prediction = weight * model_prediction + (1.0 - weight) * persistence
        score = rmse(y_true, np.clip(prediction, 0.0, None))
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight, best_score


def fit_temporal_model(
    train_rows: pd.DataFrame,
    history_2021: pd.DataFrame,
    history_2020: pd.DataFrame,
    validation_size: float,
    seed: int,
) -> tuple[object, int, float]:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed")

    all_features = build_features(train_rows, history_2021, history_2020)
    indices = np.arange(len(train_rows))
    fit_idx, val_idx = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        stratify=train_rows["type_norm"],
    )
    fit_features = all_features.iloc[fit_idx].reset_index(drop=True)
    val_features = all_features.iloc[val_idx].reset_index(drop=True)
    fit_features, [val_features], categorical = encode_categories(
        fit_features, [val_features]
    )
    y = train_rows[TARGET_COL].to_numpy(dtype=float)

    model = LGBMRegressor(
        objective="regression_l2",
        n_estimators=1_500,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        fit_features,
        y[fit_idx],
        categorical_feature=categorical,
        eval_set=[(val_features, y[val_idx])],
        eval_metric="rmse",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )

    model_prediction = np.clip(model.predict(val_features), 0.0, None)
    persistence = all_features.iloc[val_idx]["lag1"].to_numpy(dtype=float)
    model_score = rmse(y[val_idx], model_prediction)
    persistence_score = rmse(y[val_idx], persistence)
    blend_weight, blend_score = best_blend(
        y[val_idx], model_prediction, persistence
    )
    unseen = int((all_features.iloc[val_idx]["seen_lag1"] == 0).sum())
    print(
        f"Development validation ({len(val_idx):,} held-out 2022 rows; "
        f"{unseen:,} unseen in FY2021):"
    )
    print(f"  FY2021 persistence RMSE: {persistence_score:,.3f}")
    print(f"  LightGBM temporal RMSE:  {model_score:,.3f}")
    print(
        f"  Selected blend RMSE:     {blend_score:,.3f} "
        f"(model weight={blend_weight:.2f})"
    )

    best_iteration = max(1, int(model.best_iteration_ or model.n_estimators))
    final_features, _, categorical = encode_categories(all_features, [])
    final_model = LGBMRegressor(
        **{
            **model.get_params(),
            "n_estimators": best_iteration,
        }
    )
    final_model.fit(
        final_features,
        y,
        categorical_feature=categorical,
        callbacks=[log_evaluation(0)],
    )
    return final_model, best_iteration, blend_weight


def fallback_validation(
    train_rows: pd.DataFrame, validation_size: float, seed: int
) -> None:
    indices = np.arange(len(train_rows))
    fit_idx, val_idx = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        stratify=train_rows["type_norm"],
    )
    fit = train_rows.iloc[fit_idx]
    val = train_rows.iloc[val_idx]
    type_medians = fit.groupby("type_norm")[TARGET_COL].median()
    prediction = val["type_norm"].map(type_medians).fillna(fit[TARGET_COL].median())
    print(
        f"Development validation ({len(val_idx):,} held-out 2022 rows; "
        "history unavailable):"
    )
    print(f"  Violation-type median RMSE: {rmse(val[TARGET_COL], prediction):,.3f}")


def fallback_predict(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
    pair_values = train_rows.set_index(KEY_COLS)[TARGET_COL]
    keys = pd.MultiIndex.from_frame(test_rows[KEY_COLS])
    exact = keys.map(pair_values)
    type_medians = train_rows.groupby("type_norm")[TARGET_COL].median()
    fallback = test_rows["type_norm"].map(type_medians)
    fallback = fallback.fillna(train_rows[TARGET_COL].median()).to_numpy(dtype=float)
    return np.where(pd.isna(exact), fallback, np.asarray(exact, dtype=float))


def predict_temporal(
    model: object,
    blend_weight: float,
    train_rows: pd.DataFrame,
    history_2021: pd.DataFrame,
    history_2020: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> np.ndarray:
    lag1 = train_rows[KEY_COLS + [TARGET_COL]].copy()
    prediction_lag2 = history_2021 if not history_2020.empty else empty_history()
    features = build_features(test_rows, lag1, prediction_lag2)
    training_features = build_features(train_rows, history_2021, history_2020)
    _, [features], _ = encode_categories(training_features, [features])
    model_prediction = np.clip(model.predict(features), 0.0, None)
    persistence = features["lag1"].to_numpy(dtype=float)
    seen = features["seen_lag1"].to_numpy(dtype=bool)
    prediction = model_prediction.copy()
    prediction[seen] = (
        blend_weight * model_prediction[seen]
        + (1.0 - blend_weight) * persistence[seen]
    )
    return np.clip(prediction, 0.0, None)


def main() -> int:
    args = parse_args()
    if not 0.05 <= args.validation_size <= 0.5:
        raise ValueError("--validation-size must be between 0.05 and 0.5")

    script_dir = Path(__file__).resolve().parent
    train_path = resolve_path(args.train_path, script_dir)
    train_frame = read_core(train_path, require_target=True)
    train_rows = prepare_rows(train_frame)

    history_2021_path = resolve_path(args.history_2021_path, script_dir)
    history_2020_path = resolve_path(args.history_2020_path, script_dir)
    allow_download = not args.no_download_history
    history_2021 = load_history(2021, history_2021_path, allow_download)
    # FY2020 is useful as a second lag when supplied, but its public endpoint is
    # too slow to make it a required/default download.
    history_2020 = load_history(2020, history_2020_path, allow_download=False)
    if history_2020 is None:
        history_2020 = empty_history()

    temporal_ready = (
        history_2021 is not None
        and not history_2021.empty
        and LGBMRegressor is not None
    )
    if temporal_ready:
        model, best_iteration, blend_weight = fit_temporal_model(
            train_rows,
            history_2021,
            history_2020,
            args.validation_size,
            args.seed,
        )
        print(f"Final model trees: {best_iteration}")
    else:
        model = None
        blend_weight = 1.0
        fallback_validation(train_rows, args.validation_size, args.seed)
        print(
            "Using persistence plus violation-type medians for inference "
            "(historical data or LightGBM unavailable)."
        )

    if args.test_path is None:
        print("No --test-path supplied; training and development validation complete.")
        return 0

    test_path = resolve_path(args.test_path, script_dir)
    test_frame = read_core(test_path, require_target=False)
    test_rows = prepare_rows(test_frame)
    train_keys = pd.MultiIndex.from_frame(train_rows[KEY_COLS])
    test_keys = pd.MultiIndex.from_frame(test_rows[KEY_COLS])
    unseen_mask = ~test_keys.isin(train_keys)

    if temporal_ready:
        prediction = predict_temporal(
            model,
            blend_weight,
            train_rows,
            history_2021,
            history_2020,
            test_rows,
        )
        unseen_strategy = "temporal/marginal model"
    else:
        prediction = fallback_predict(train_rows, test_rows)
        unseen_strategy = "violation-type median"

    output = pd.DataFrame(
        {
            "street_name": test_frame[STREET_COL],
            "violation_type": test_frame[TYPE_COL],
            "predicted_count": np.round(prediction, 3),
        }
    )
    output_path = resolve_path(args.output_path, Path.cwd())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote {len(output):,} predictions to {output_path}")
    print(
        f"Unseen test/eval key pairs: {int(unseen_mask.sum()):,}/{len(test_rows):,}; "
        f"handled with {unseen_strategy}."
    )

    if TARGET_COL in test_frame and test_frame[TARGET_COL].notna().all():
        print(
            f"Test/eval RMSE: "
            f"{rmse(test_frame[TARGET_COL].to_numpy(dtype=float), prediction):,.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
