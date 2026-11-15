#!/usr/bin/env python3
"""Forecast NYC parking violations for an arbitrary future key space.

The model learns the year-over-year relationship from 2021 to 2022, then uses
2022 counts and the same feature construction to predict 2023. Test targets,
when present, are used only after prediction for RMSE reporting.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = HERE / "violations_per_street_2022.csv"
DEFAULT_HISTORY_PATH = HERE / "corpus" / "violations_per_street_2021.csv"

STREET_SUFFIXES = {
    "AVENUE": "AVE",
    "STREET": "ST",
    "PLACE": "PL",
    "ROAD": "RD",
    "BOULEVARD": "BLVD",
    "DRIVE": "DR",
    "LANE": "LN",
    "PARKWAY": "PKWY",
    "COURT": "CT",
    "TERRACE": "TER",
    "EXPRESSWAY": "EXPY",
    "HIGHWAY": "HWY",
    "EAST": "E",
    "WEST": "W",
    "NORTH": "N",
    "SOUTH": "S",
    "SAINT": "ST",
}
ROAD_SUFFIX_PATTERN = r"\b(?:AVE|ST|PL|RD|BLVD|DR|LN|PKWY|CT|TER|EXPY|HWY)\b"
GROUP_STATS = ("sum", "mean", "median", "max", "std", "size")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict future violation counts by street and violation type."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="2022 training CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Prior-year CSV used to learn the 2021->2022 transition.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help="Optional future keys/evaluation CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("submission.csv"),
        help="Prediction CSV path (default: submission.csv).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.25,
        help="Fraction of 2022 transition rows reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=350,
        help="Number of ExtraTrees estimators.",
    )
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.n_estimators < 1:
        parser.error("--n-estimators must be positive")
    return args


def _column_lookup(columns: pd.Index) -> dict[str, str]:
    return {
        re.sub(r"[\s_]+", " ", str(column).strip().lower()): column
        for column in columns
    }


def read_violations(path: Path, *, require_target: bool) -> tuple[pd.DataFrame, bool]:
    """Read core columns without ever treating the target as a feature."""
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    raw = pd.read_csv(path)
    names = _column_lookup(raw.columns)
    street_column = names.get("street name")
    type_column = names.get("violation description") or names.get("violation type")
    target_column = names.get("violation count")

    missing = []
    if street_column is None:
        missing.append("Street Name/street_name")
    if type_column is None:
        missing.append("Violation Description/violation_type")
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if require_target and target_column is None:
        raise ValueError(f"{path} is missing required column: violation_count")

    frame = pd.DataFrame(
        {
            "street": raw[street_column].fillna("").astype(str),
            "vtype": raw[type_column].fillna("").astype(str),
        }
    )
    has_target = target_column is not None
    if has_target:
        counts = pd.to_numeric(raw[target_column], errors="coerce")
        if counts.isna().any():
            bad = int(counts.isna().sum())
            raise ValueError(f"{path} has {bad} missing/non-numeric violation_count values")
        if (counts < 0).any():
            raise ValueError(f"{path} contains negative violation_count values")
        frame["count"] = counts.astype(float)
    return frame, has_target


def normalize_street(value: object) -> str:
    """Normalize common spelling, suffix, punctuation, and ordinal differences."""
    text = re.sub(r"\s+", " ", str(value).upper().strip())
    for source, replacement in STREET_SUFFIXES.items():
        text = re.sub(rf"\b{source}\b", replacement, text)
    text = re.sub(r"\b(\d+)(?:ST|ND|RD|TH)\b", r"\1", text)
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def roadway_key(value: object) -> str:
    """Create a broad roadway key for renamed/truncated camera locations."""
    text = str(value).upper().strip().split("@", maxsplit=1)[0]
    text = re.sub(r"^\s*(?:EB|WB|NB|SB)\s+", "", text)
    text = normalize_street(text)
    text = re.sub(r"^(?:N|S|E|W)\s+", "", text)
    text = re.sub(ROAD_SUFFIX_PATTERN, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach matching keys while retaining original output values and targets."""
    result = frame.copy()
    result["street"] = result["street"].str.strip().str.replace(r"\s+", " ", regex=True)
    result["vtype"] = result["vtype"].str.strip().str.replace(r"\s+", " ", regex=True)
    result["street_upper"] = result["street"].str.upper()
    result["street_norm"] = result["street"].map(normalize_street)
    result["roadway"] = result["street"].map(roadway_key)
    result["vtype_key"] = result["vtype"].str.upper()
    return result


def _lookup_stats(
    current: pd.DataFrame,
    past: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    grouped = past.groupby(keys, sort=False)["count"].agg(GROUP_STATS)
    index = pd.MultiIndex.from_frame(current[keys])
    return grouped.reindex(index).reset_index(drop=True)


def build_features(current: pd.DataFrame, past: pd.DataFrame) -> pd.DataFrame:
    """Build numeric lag, hierarchy, key-space, and name-shape features.

    `current` contributes keys only. All count-derived features come from
    `past`, which prevents an evaluation file's target from entering the model.
    """
    features = pd.DataFrame(index=current.index)
    pair_groups = (
        ("upper", ["street_upper", "vtype_key"]),
        ("norm", ["street_norm", "vtype_key"]),
        ("road", ["roadway", "vtype_key"]),
    )

    for label, keys in pair_groups:
        stats = _lookup_stats(current, past, keys)
        for stat in GROUP_STATS:
            features[f"past_{label}_{stat}"] = stats[stat].to_numpy()

        current_rows = current.groupby(keys, sort=False)[keys[0]].transform("size")
        features[f"current_{label}_rows"] = current_rows.astype(float)
        features[f"past_{label}_allocated"] = (
            features[f"past_{label}_sum"] / current_rows.clip(lower=1)
        )

    for label, key in (
        ("upper", "street_upper"),
        ("norm", "street_norm"),
        ("road", "roadway"),
    ):
        grouped = past.groupby(key, sort=False)["count"].agg(("sum", "mean", "max", "size"))
        for stat in grouped.columns:
            features[f"past_{label}_all_{stat}"] = current[key].map(grouped[stat])

    type_stats = past.groupby("vtype_key", sort=False)["count"].agg(
        ("sum", "mean", "median", "max", "size")
    )
    for stat in type_stats.columns:
        features[f"past_type_{stat}"] = current["vtype_key"].map(type_stats[stat])

    features["street_is_upper"] = current["street"].eq(current["street_upper"]).astype(float)
    features["street_has_at"] = current["street"].str.contains("@", regex=False).astype(float)
    features["street_length"] = current["street"].str.len().astype(float)
    features["street_tokens"] = current["street_norm"].str.split().str.len().astype(float)
    features["street_digits"] = current["street"].str.count(r"\d").astype(float)
    features["street_directional"] = (
        current["street_upper"].str.match(r"^(?:EB|WB|NB|SB)\b").astype(float)
    )

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def make_model(seed: int, n_estimators: int) -> ExtraTreesRegressor:
    """Return the tuned, dependency-light regression model."""
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=4,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual.astype(float) - predicted.astype(float)) ** 2)))


def validation_indices(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    type_counts = frame["vtype_key"].value_counts()
    can_stratify = (
        len(type_counts) > 1
        and type_counts.min() >= 2
        and round(len(frame) * fraction) >= len(type_counts)
    )
    stratify = frame["vtype_key"] if can_stratify else None
    return train_test_split(
        indices,
        test_size=fraction,
        random_state=seed,
        stratify=stratify,
    )


def normalized_seen_mask(test: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    train_pairs = pd.MultiIndex.from_frame(
        train[["street_norm", "vtype_key"]].drop_duplicates()
    )
    test_pairs = pd.MultiIndex.from_frame(test[["street_norm", "vtype_key"]])
    return test_pairs.isin(train_pairs)


def main() -> None:
    args = parse_args()
    history_raw, _ = read_violations(args.history_path, require_target=True)
    train_raw, _ = read_violations(args.train_path, require_target=True)
    history = prepare_keys(history_raw)
    train = prepare_keys(train_raw)

    print(
        f"[data] history={len(history):,} rows ({args.history_path})\n"
        f"[data] train={len(train):,} rows ({args.train_path})"
    )

    # Development validation learns on a subset of the 2021->2022 transition.
    # Every validation feature still uses only prior-year (2021) counts.
    transition_features = build_features(train, history)
    train_indices, validation_indices_ = validation_indices(
        train, args.validation_fraction, args.seed
    )
    validation_model = make_model(args.seed, args.n_estimators)
    validation_model.fit(
        transition_features.iloc[train_indices],
        train["count"].iloc[train_indices],
    )
    validation_predictions = np.clip(
        validation_model.predict(transition_features.iloc[validation_indices_]),
        0.0,
        None,
    )
    validation_score = rmse(
        train["count"].iloc[validation_indices_].to_numpy(),
        validation_predictions,
    )
    print(
        f"[validation] 2021->2022 holdout rows={len(validation_indices_):,} "
        f"RMSE={validation_score:.3f}"
    )

    # Refit on all known transitions. This model learns how 2021 lag/hierarchy
    # signals mapped to 2022; future features then use 2022 as their lag.
    model = make_model(args.seed, args.n_estimators)
    model.fit(transition_features, train["count"])

    if args.test_path is None:
        print("[done] no --test-path supplied; validation completed without inference.")
        return

    test_raw, has_target = read_violations(args.test_path, require_target=False)
    test = prepare_keys(test_raw)
    test_features = build_features(test, train)
    predictions = np.clip(model.predict(test_features), 0.0, None)

    seen = normalized_seen_mask(test, train)
    unseen_count = int((~seen).sum())
    print(
        f"[test] rows={len(test):,}; unseen normalized key pairs={unseen_count:,} "
        f"({100.0 * unseen_count / max(len(test), 1):.1f}%)"
    )
    if unseen_count:
        print(
            "[test] unseen pairs use learned roadway/street/type backoffs and "
            "key-shape features."
        )

    submission = pd.DataFrame(
        {
            "street_name": test["street"],
            "violation_type": test["vtype"],
            "predicted_count": predictions,
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)
    print(f"[done] wrote {args.output_path} ({len(submission):,} rows)")

    if has_target:
        score = rmse(test["count"].to_numpy(), predictions)
        print(f"[eval] RMSE={score:.3f}")


if __name__ == "__main__":
    main()
