#!/usr/bin/env python3
"""Leakage-safe baseline for forecasting NYC parking-violation counts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

STREET = "Street Name"
VIOLATION = "Violation Description"
TARGET = "violation_count"
REQUIRED_KEYS = (STREET, VIOLATION)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(__file__).with_name("violations_per_street_2022.csv"),
    )
    parser.add_argument("--test-path", type=Path)
    parser.add_argument(
        "--history-path",
        type=Path,
        help="Optional prior-year aggregate CSV with the same three-column schema.",
    )
    parser.add_argument("--output-path", type=Path, default=Path("submission.csv"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_data(path: Path, require_target: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_KEYS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if require_target and TARGET not in frame.columns:
        raise ValueError(f"{path} is missing required target column {TARGET!r}")
    if frame[list(REQUIRED_KEYS)].isna().any().any():
        raise ValueError(f"{path} contains missing key values")
    if require_target:
        frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")
        if (~np.isfinite(frame[TARGET]) | (frame[TARGET] < 0)).any():
            raise ValueError(f"{path} contains invalid target values")
    return frame


def normalize_street(value: object) -> str:
    text = str(value).upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    replacements = {
        "AVENUE": "AVE",
        "STREET": "ST",
        "ROAD": "RD",
        "BOULEVARD": "BLVD",
        "PLACE": "PL",
        "DRIVE": "DR",
        "PARKWAY": "PKWY",
        "COURT": "CT",
        "LANE": "LN",
        "TERRACE": "TER",
    }
    words = [replacements.get(word, word) for word in text.split()]
    return " ".join(words)


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_exact_street"] = result[STREET].astype(str)
    result["_street"] = result[STREET].map(normalize_street)
    result["_exact_violation"] = result[VIOLATION].astype(str)
    result["_violation"] = result["_exact_violation"].str.upper()
    result["_key"] = result["_exact_street"] + "\x1f" + result["_exact_violation"]
    return result


class HierarchicalForecaster:
    """Exact-key persistence with smoothed fallbacks for unseen key pairs."""

    def __init__(self, street_strength: float = 3.0) -> None:
        self.street_strength = street_strength

    def fit(
        self, train: pd.DataFrame, history: pd.DataFrame | None = None
    ) -> "HierarchicalForecaster":
        data = add_keys(train)
        data = data.groupby(
            ["_key", "_exact_street", "_exact_violation", "_street", "_violation"],
            as_index=False,
            sort=False,
        )[TARGET].sum()

        self.overall_mean = float(data[TARGET].mean())
        self.key_counts = data.set_index("_key")[TARGET]
        self.violation_mean = data.groupby("_violation")[TARGET].mean()
        self.violation_median = data.groupby("_violation")[TARGET].median()

        baseline = data["_violation"].map(self.violation_mean).clip(lower=1.0)
        activity = data[TARGET] / baseline
        street_stats = pd.DataFrame(
            {"_street": data["_street"], "activity": activity}
        ).groupby("_street")["activity"].agg(["sum", "count"])
        self.street_activity = (
            street_stats["sum"] + self.street_strength
        ) / (street_stats["count"] + self.street_strength)

        self.growth = pd.Series(1.0, index=self.violation_mean.index)
        if history is not None:
            old = add_keys(history)
            old = old.groupby(["_key", "_violation"], as_index=False)[TARGET].sum()
            joined = old.merge(
                data[["_key", "_violation", TARGET]],
                on=["_key", "_violation"],
                suffixes=("_old", "_new"),
            )
            for violation, group in joined.groupby("_violation"):
                x = group[f"{TARGET}_old"].to_numpy(dtype=float)
                y = group[f"{TARGET}_new"].to_numpy(dtype=float)
                denominator = float(x @ x)
                if len(group) >= 20 and denominator > 0:
                    self.growth.loc[violation] = np.clip(
                        float(x @ y) / denominator, 0.5, 1.5
                    )
        return self

    def predict(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        data = add_keys(rows)
        exact = data["_key"].map(self.key_counts)
        category_mean = (
            data["_violation"].map(self.violation_mean).fillna(self.overall_mean)
        )
        category_median = (
            data["_violation"].map(self.violation_median).fillna(self.overall_mean)
        )
        activity = data["_street"].map(self.street_activity)

        # For known streets, activity adjusts a conservative category baseline.
        # For wholly unseen streets, the category mean minimizes squared error.
        fallback = category_mean.copy()
        known_street = activity.notna()
        fallback.loc[known_street] = (
            0.65 * category_mean.loc[known_street]
            + 0.35
            * category_median.loc[known_street]
            * activity.loc[known_street]
        )

        growth = data["_violation"].map(self.growth).fillna(1.0)
        predictions = exact.fillna(fallback) * growth
        return predictions.clip(lower=0).to_numpy(dtype=float), exact.notna().to_numpy()


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual.astype(float) - predicted))))


def validate(train: pd.DataFrame, fraction: float, seed: int) -> None:
    if not 0 < fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    keyed = add_keys(train)
    streets = keyed["_street"].drop_duplicates().sample(frac=1, random_state=seed)
    validation_streets = set(streets.iloc[: max(1, round(len(streets) * fraction))])
    validation_mask = keyed["_street"].isin(validation_streets)

    fit_rows = train.loc[~validation_mask].copy()
    validation_rows = train.loc[validation_mask].copy()
    model = HierarchicalForecaster().fit(fit_rows)
    predictions, _ = model.predict(validation_rows)
    score = rmse(validation_rows[TARGET].to_numpy(), predictions)
    print(
        f"Development grouped-holdout RMSE: {score:,.3f} "
        f"({len(validation_rows):,} rows; {len(validation_streets):,} unseen streets)"
    )


def main() -> None:
    args = parse_args()
    train = read_data(args.train_path, require_target=True)
    validate(train, args.validation_fraction, args.seed)

    history = (
        read_data(args.history_path, require_target=True)
        if args.history_path is not None
        else None
    )
    model = HierarchicalForecaster().fit(train, history)

    if args.test_path is None:
        print("No --test-path supplied; validation completed without writing predictions.")
        return

    test = read_data(args.test_path, require_target=False)
    predictions, exact_match = model.predict(test)
    submission = pd.DataFrame(
        {
            "street_name": test[STREET],
            "violation_type": test[VIOLATION],
            "predicted_count": predictions,
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)

    unseen = int((~exact_match).sum())
    print(
        f"Wrote {len(submission):,} predictions to {args.output_path}. "
        f"Unseen key pairs: {unseen:,} ({unseen / max(len(test), 1):.1%}); "
        "handled with smoothed violation-type/street fallbacks."
    )
    if TARGET in test.columns:
        truth = pd.to_numeric(test[TARGET], errors="raise").to_numpy(dtype=float)
        if (~np.isfinite(truth) | (truth < 0)).any():
            raise ValueError(f"{args.test_path} contains invalid target values")
        print(f"Test/eval RMSE: {rmse(truth, predictions)}")


if __name__ == "__main__":
    main()
