#!/usr/bin/env python3
"""Forecast NYC parking-violation counts for an arbitrary evaluation key space.

The main forecast is normalized year-over-year persistence. A supervised
fallback model handles pairs with no prior count. When the 2021 corpus table is
available, that model learns a genuine 2021 -> 2022 transition and then applies
the learned relationship to 2022 -> 2023. Its features include activity from
the other violation types on the same street, stable intersection prefixes,
street structure, and the composition of the supplied key space.

The test target, when present, is read only after prediction for scoring.
"""
from __future__ import annotations

import argparse
import os
import re
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN = os.path.join(
    HERE, "train", "all_data", "violations_per_street_2022.csv"
)
DEFAULT_HISTORY = os.path.join(
    HERE, "train", "all_data", "corpus", "violations_per_street_2021.csv"
)
DEFAULT_OUTPUT = os.path.join(HERE, "submission.csv")

EXACT = "exact key"
ALIAS = "normalized alias"
PREFIX = "intersection prefix"
MODEL = "temporal fallback"

CAMERA_VTYPES = {
    "PHTO SCHOOL ZN SPEED VIOLATION",
    "BUS LANE VIOLATION",
    "FAILURE TO STOP AT RED LIGHT",
}

_SUFFIXES = {
    r"\bAVENUE\b": "AVE",
    r"\bSTREET\b": "ST",
    r"\bPLACE\b": "PL",
    r"\bROAD\b": "RD",
    r"\bBOULEVARD\b": "BLVD",
    r"\bDRIVE\b": "DR",
    r"\bLANE\b": "LN",
    r"\bPARKWAY\b": "PKWY",
    r"\bCOURT\b": "CT",
    r"\bTERRACE\b": "TER",
    r"\bEXPRESSWAY\b": "EXPY",
    r"\bHIGHWAY\b": "HWY",
    r"\bEAST\b": "E",
    r"\bWEST\b": "W",
    r"\bNORTH\b": "N",
    r"\bSOUTH\b": "S",
    r"\bSAINT\b": "ST",
}


def normalize_street(value: object) -> str:
    """Return a stable street join key."""
    text = "" if pd.isna(value) else str(value).upper().strip()
    for pattern, replacement in _SUFFIXES.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\b(\d+)(?:ST|ND|RD|TH)\b", r"\1", text)
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prefix_key(value: object) -> str:
    """Normalize the stable portion before '@' in intersection camera names."""
    text = "" if pd.isna(value) else str(value)
    return normalize_street(text.split("@", 1)[0])


def normalize_vtype(value: object) -> str:
    """Use the stable violation code where one exists, otherwise normalized text."""
    text = "" if pd.isna(value) else re.sub(r"\s+", " ", str(value).upper().strip())
    match = re.match(r"^(\d+[A-Z]?)\s*[-–]", text)
    return match.group(1) if match else text


def read_keyed_csv(path: str, require_target: bool = False) -> tuple[pd.DataFrame, bool]:
    """Read and canonicalize a train, history, or evaluation CSV."""
    raw = pd.read_csv(path)
    names = {str(column).lower().strip(): column for column in raw.columns}
    street_col = names.get("street name") or names.get("street_name")
    vtype_col = names.get("violation description") or names.get("violation_type")
    if street_col is None or vtype_col is None:
        raise ValueError(
            f"{path}: expected Street Name/street_name and "
            "Violation Description/violation_type columns"
        )

    target_col = names.get("violation_count") or names.get("violation count")
    has_target = target_col is not None
    if require_target and not has_target:
        raise ValueError(f"{path}: training data must contain violation_count")

    frame = pd.DataFrame(
        {
            "street": raw[street_col].fillna("").astype(str),
            "vtype": raw[vtype_col].fillna("").astype(str),
        }
    )
    frame["street_key"] = frame["street"].map(normalize_street)
    frame["raw_street_key"] = frame["street"].str.upper().str.strip()
    frame["prefix_key"] = frame["street"].map(prefix_key)
    frame["vtype_key"] = frame["vtype"].map(normalize_vtype)
    if has_target:
        frame["count"] = pd.to_numeric(raw[target_col], errors="coerce").fillna(0.0)
    return frame, has_target


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                (np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
                ** 2
            )
        )
    )


def _reindex_values(series: pd.Series, keys: pd.DataFrame) -> np.ndarray:
    index = pd.MultiIndex.from_frame(keys)
    return series.reindex(index).to_numpy(dtype=float)


def _structural_features(target: pd.DataFrame) -> pd.DataFrame:
    street = target["street_key"]
    features = pd.DataFrame(index=target.index)
    features["street_is_numbered"] = street.str.match(r"^\d+(?: |$)").astype(float)
    features["street_tokens"] = street.str.split().map(len).astype(float)
    features["street_length"] = street.str.len().astype(float)
    features["street_has_ave"] = street.str.contains(r"\bAVE\b").astype(float)
    features["street_has_st"] = street.str.contains(r"\bST\b").astype(float)
    features["street_has_highway"] = street.str.contains(
        r"\b(?:PKWY|EXPY|HWY)\b"
    ).astype(float)
    features["is_intersection"] = target["street"].str.contains(
        "@", regex=False
    ).astype(float)
    return features


def make_features(
    target: pd.DataFrame,
    lag: pd.DataFrame,
    vtypes: list[str],
    hide_pair: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build leakage-safe lag/context features aligned to ``target`` rows."""
    lag_raw_pairs = lag.groupby(
        ["raw_street_key", "vtype_key"], sort=False
    )["count"].sum()
    exact = _reindex_values(
        lag_raw_pairs, target[["raw_street_key", "vtype_key"]]
    )
    exact_siblings = (
        target.groupby(["raw_street_key", "vtype_key"])["street"]
        .transform("size")
        .to_numpy(dtype=float)
    )
    exact = exact / exact_siblings

    lag_alias = lag.groupby(["street_key", "vtype_key"], sort=False)["count"].sum()
    alias = _reindex_values(
        lag_alias, target[["street_key", "vtype_key"]]
    )
    alias_siblings = (
        target.groupby(["street_key", "vtype_key"])["street"]
        .transform("size")
        .to_numpy(dtype=float)
    )
    alias = alias / alias_siblings

    lag_prefix = lag.groupby(["prefix_key", "vtype_key"], sort=False)["count"].sum()
    prefix = _reindex_values(
        lag_prefix, target[["prefix_key", "vtype_key"]]
    )
    prefix_siblings = (
        target.groupby(["prefix_key", "vtype_key"])["street"]
        .transform("size")
        .to_numpy(dtype=float)
    )
    prefix = prefix / prefix_siblings

    prior = exact.copy()
    source = np.full(len(target), MODEL, dtype=object)
    source[~np.isnan(exact)] = EXACT
    use_alias = np.isnan(prior) & ~np.isnan(alias)
    prior[use_alias] = alias[use_alias]
    source[use_alias] = ALIAS
    use_prefix = (
        np.isnan(prior)
        & target["street"].str.contains("@", regex=False).to_numpy()
        & ~np.isnan(prefix)
    )
    prior[use_prefix] = prefix[use_prefix]
    source[use_prefix] = PREFIX
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)

    street_matrix = (
        lag.groupby(["street_key", "vtype_key"])["count"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=vtypes, fill_value=0.0)
    )
    prefix_matrix = (
        lag.groupby(["prefix_key", "vtype_key"])["count"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=vtypes, fill_value=0.0)
    )
    street_context = (
        street_matrix.reindex(target["street_key"]).fillna(0.0).to_numpy(dtype=float)
    )
    prefix_context = (
        prefix_matrix.reindex(target["prefix_key"]).fillna(0.0).to_numpy(dtype=float)
    )

    category = {name: position for position, name in enumerate(vtypes)}
    codes = target["vtype_key"].map(category).fillna(-1).to_numpy(dtype=int)
    rows = np.arange(len(target))
    known = codes >= 0

    # The target type has its own explicit lag feature. Excluding it from the
    # context makes all "other type" features safe and interpretable.
    street_other = street_context.copy()
    prefix_other = prefix_context.copy()
    street_other[rows[known], codes[known]] = 0.0
    prefix_other[rows[known], codes[known]] = 0.0

    features = _structural_features(target).reset_index(drop=True)
    features["vtype_code"] = codes.astype(float)
    features["lag_pair"] = 0.0 if hide_pair else prior
    features["log_lag_pair"] = np.log1p(features["lag_pair"])
    features["street_other_total"] = street_other.sum(axis=1)
    features["log_street_other_total"] = np.log1p(features["street_other_total"])
    features["street_other_active"] = (street_other > 0).sum(axis=1)
    features["street_other_max"] = street_other.max(axis=1)
    features["prefix_other_total"] = prefix_other.sum(axis=1)
    features["log_prefix_other_total"] = np.log1p(features["prefix_other_total"])
    features["keyspace_street_rows"] = (
        target.groupby("street_key")["street"].transform("size").to_numpy(dtype=float)
    )
    features["keyspace_vtype_rows"] = (
        target.groupby("vtype_key")["street"].transform("size").to_numpy(dtype=float)
    )
    for position, name in enumerate(vtypes):
        features[f"other_type_{position}"] = np.log1p(street_other[:, position])

    if hide_pair:
        prior = np.zeros(len(target), dtype=float)
        source[:] = MODEL
    return features.astype(float), prior, source


def fit_model(features: pd.DataFrame, target: np.ndarray) -> HistGradientBoostingRegressor:
    """Fit for squared error because RMSE, rather than relative error, is scored."""
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=350,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=20.0,
        random_state=17,
    )
    model.fit(features, np.asarray(target, dtype=float))
    return model


def model_predict(
    model: HistGradientBoostingRegressor, features: pd.DataFrame
) -> np.ndarray:
    return np.clip(model.predict(features), 0.0, None)


def validation_report(
    current: pd.DataFrame,
    history: pd.DataFrame,
    vtypes: list[str],
) -> None:
    """Run a deterministic street-grouped 2021 -> 2022 validation."""
    features, prior, source = make_features(current, history, vtypes)
    is_valid = current["street_key"].map(
        lambda value: zlib.crc32(value.encode("utf-8")) % 5 == 0
    ).to_numpy()
    if is_valid.sum() == 0 or (~is_valid).sum() == 0:
        print("[validation] skipped: not enough street groups")
        return

    model = fit_model(features.loc[~is_valid], current.loc[~is_valid, "count"].to_numpy())
    predicted = model_predict(model, features.loc[is_valid])
    valid_source = source[is_valid]
    y_valid = current.loc[is_valid, "count"].to_numpy(dtype=float)

    # Estimate persistence multipliers on the development fold only.
    train_rows = np.flatnonzero(~is_valid)
    for vtype in vtypes:
        fit_mask = (
            (current["vtype_key"].to_numpy()[train_rows] == vtype)
            & (prior[train_rows] > 0)
        )
        if not fit_mask.any():
            continue
        selected = train_rows[fit_mask]
        denominator = float(np.dot(prior[selected], prior[selected]))
        multiplier = (
            float(np.dot(prior[selected], current["count"].to_numpy()[selected]))
            / denominator
            if denominator > 0
            else 1.0
        )
        use = (
            (current.loc[is_valid, "vtype_key"].to_numpy() == vtype)
            & (valid_source != MODEL)
        )
        predicted[use] = prior[is_valid][use] * np.clip(multiplier, 0.0, 2.0)

    unseen = valid_source == MODEL
    print(
        f"[validation] grouped 2021->2022 rows={is_valid.sum()} "
        f"hybrid_RMSE={rmse(y_valid, predicted):.3f}"
    )
    if unseen.any():
        print(
            f"[validation] unseen-pair rows={unseen.sum()} "
            f"fallback_RMSE={rmse(y_valid[unseen], predicted[unseen]):.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN)
    parser.add_argument(
        "--test-path",
        default=None,
        help="Optional keys-only or labeled evaluation CSV",
    )
    parser.add_argument(
        "--history-path",
        default=DEFAULT_HISTORY,
        help="Optional prior-year corpus CSV used for temporal model training",
    )
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--organic-growth", type=float, default=1.35)
    parser.add_argument("--camera-growth", type=float, default=1.05)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the grouped development backtest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, _ = read_keyed_csv(args.train_path, require_target=True)
    vtypes = sorted(train["vtype_key"].unique())
    print(
        f"[data] train rows={len(train)} streets={train['street_key'].nunique()} "
        f"vtypes={len(vtypes)}"
    )

    history = None
    if args.history_path and os.path.isfile(args.history_path):
        history, history_has_target = read_keyed_csv(
            args.history_path, require_target=True
        )
        if not history_has_target:
            history = None
        else:
            print(
                f"[data] history rows={len(history)} "
                f"streets={history['street_key'].nunique()}"
            )
            if not args.skip_validation:
                validation_report(train, history, vtypes)
    elif args.history_path:
        print(f"[warn] history file not found: {args.history_path}")

    # With history, learn an actual annual transition. Without it, learn the
    # relationship between one violation type and activity in the other types.
    if history is not None:
        train_features, _, _ = make_features(train, history, vtypes)
    else:
        train_features, _, _ = make_features(
            train, train, vtypes, hide_pair=True
        )
    model = fit_model(train_features, train["count"].to_numpy())

    if not args.test_path:
        print("[done] model fitted; pass --test-path to generate submission.csv")
        return

    test, has_target = read_keyed_csv(args.test_path)
    # Never pass a test target into feature construction.
    test_keys = test.drop(columns=["count"], errors="ignore")
    test_features, prior, source = make_features(test_keys, train, vtypes)
    fallback = model_predict(model, test_features)

    growth = np.where(
        test["vtype_key"].isin(CAMERA_VTYPES).to_numpy(),
        args.camera_growth,
        args.organic_growth,
    )
    prediction = fallback
    persistence = source != MODEL
    prediction[persistence] = prior[persistence] * growth[persistence]

    # Unknown violation categories cannot be learned by the categorical code.
    # Use their training-wide median rather than an extrapolated tree value.
    unknown_type = ~test["vtype_key"].isin(vtypes).to_numpy()
    if unknown_type.any():
        prediction[unknown_type] = float(train["count"].median())

    prediction = np.round(np.clip(prediction, 0.0, None), 3)
    for label in (EXACT, ALIAS, PREFIX, MODEL):
        count = int((source == label).sum())
        print(f"  {label:20s}: {count:>7d} ({100 * count / len(test):5.1f}%)")
    unseen = source != EXACT
    print(
        f"[coverage] unseen literal training pairs={unseen.sum()} of {len(test)}; "
        f"alias={int((source == ALIAS).sum())}, "
        f"prefix={int((source == PREFIX).sum())}, "
        f"model={int((source == MODEL).sum())}"
    )

    submission = pd.DataFrame(
        {
            "street_name": test["street"],
            "violation_type": test["vtype"],
            "predicted_count": prediction,
        }
    )
    submission.to_csv(args.output_path, index=False)
    print(f"[done] wrote {args.output_path} ({len(submission)} rows)")

    if has_target:
        print(f"[eval] RMSE={rmse(test['count'].to_numpy(), prediction):.3f}")


if __name__ == "__main__":
    main()

