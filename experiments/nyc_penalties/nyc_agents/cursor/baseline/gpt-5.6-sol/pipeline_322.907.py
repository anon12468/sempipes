#!/usr/bin/env python3
"""NYC parking-violation forecasting.

Predicts next-year counts for each (street_name, violation_type) from the
previous year. 2023 is not used. Growth is estimated from the 2021 corpus
file to the 2022 training file, on a street-grouped holdout, then applied
to whatever key space a later test file defines.

Resolution order for each test row:
  exact   same street string and violation type in the history year
          -> history count x per-type growth
  head    camera / red-light / bus-lane rows whose text before "@" matches
          a history corridor -> that corridor's typical location count x growth
  core    same camera types, matched on direction + street-name tokens when
          the truncated "@" text differs (AVE vs BLVD, extra side token)
          -> typical location count x growth
  unseen  no history signal -> 0

Camera types dominate squared error, and their location strings are truncated
to 20 characters, so exact keys often miss a corridor that is still present
under a slightly different name.
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN = os.path.join(HERE, "train", "all_data", "violations_per_street_2022.csv")
DEFAULT_PRIOR = os.path.join(HERE, "train", "all_data", "corpus", "violations_per_street_2021.csv")
DEFAULT_OUT = os.path.join(HERE, "submission.csv")

# Automated enforcement. Counts live on a few corridors and the recorded
# location string is unstable across years, so these rows get the corridor
# fallback. Organic types do not: a loose name match would assign a large
# street total to a stray spelling.
CAMERA_VTYPES = {
    "PHTO SCHOOL ZN SPEED VIOLATION",
    "BUS LANE VIOLATION",
    "FAILURE TO STOP AT RED LIGHT",
}

_SUFFIX = {
    r"\bAVENUE\b": "AVE", r"\bSTREET\b": "ST", r"\bPLACE\b": "PL",
    r"\bROAD\b": "RD", r"\bBOULEVARD\b": "BLVD", r"\bDRIVE\b": "DR",
    r"\bLANE\b": "LN", r"\bPARKWAY\b": "PKWY", r"\bCOURT\b": "CT",
    r"\bTERRACE\b": "TER", r"\bEXPRESSWAY\b": "EXPY", r"\bHIGHWAY\b": "HWY",
    r"\bEAST\b": "E", r"\bWEST\b": "W", r"\bNORTH\b": "N", r"\bSOUTH\b": "S",
    r"\bSAINT\b": "ST",
}
_DIR = {"NB", "SB", "EB", "WB"}
_SIDE = {"N", "S", "E", "W"}
_SUF = {"AVE", "ST", "PL", "RD", "BLVD", "DR", "LN", "PKWY", "CT", "TER", "EXPY", "HWY"}

EXACT, HEAD, CORE, UNSEEN = "exact", "head", "core", "unseen"


def norm_street(s) -> str:
    """Canonical street key: case, suffixes, ordinals, punctuation."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"\s+", " ", s)
    for pattern, repl in _SUFFIX.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_vtype(s) -> str:
    return re.sub(r"\s+", " ", str(s).upper().strip())


def head_key(street) -> str:
    """Text before '@', normalized. Stable when only the cross street changes."""
    return norm_street(str(street).split("@")[0])


def corridor_key(street) -> tuple[str, str]:
    """(name tokens, direction) for a camera corridor.

    Drops leading NB/SB/EB/WB, side markers, and street suffixes so
    'WB N CONDUIT AVE @ ...' and 'WB CONDUIT BLVD @ ...' share a key.
    Requires an alphabetic token of length >= 4 so bare numbers ('3', '181')
    are not merged across unrelated streets.
    """
    dire = ""
    name: list[str] = []
    for tok in head_key(street).split():
        if tok in _DIR and not dire:
            dire = tok
        elif tok in _SIDE or tok in _SUF:
            continue
        else:
            name.append(tok)
    if not any(tok.isalpha() and len(tok) >= 4 for tok in name):
        return "", dire
    return " ".join(name), dire


def rmse(a, b) -> float:
    err = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(np.mean(err ** 2)))


def read_violations(path: str) -> tuple[pd.DataFrame, bool]:
    """Read a violations file.

    Returns a frame with original key strings plus normalized join columns.
    `count` is present only when the file has a ground-truth column. That
    column is never used as a feature.
    """
    df = pd.read_csv(path)
    lookup = {c.lower().strip(): c for c in df.columns}
    street_col = lookup.get("street name") or lookup.get("street_name") or df.columns[0]
    vtype_col = (
        lookup.get("violation description")
        or lookup.get("violation_type")
        or df.columns[1]
    )
    out = pd.DataFrame({
        "street_raw": df[street_col].astype(str),
        "vtype_raw": df[vtype_col].astype(str),
    })
    out["street"] = out["street_raw"].str.upper().str.strip()
    out["vtype"] = out["vtype_raw"].map(norm_vtype)
    out["k"] = out["street"].map(norm_street)
    out["hk"] = out["street"].map(head_key)
    cores = out["hk"].map(corridor_key)
    out["core"] = cores.map(lambda x: x[0])
    out["dire"] = cores.map(lambda x: x[1])

    target_col = lookup.get("violation_count") or lookup.get("violation count")
    has_target = target_col is not None
    if has_target:
        out["count"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    return out, has_target


def _sum_index(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    return df.groupby(keys, sort=False)["count"].sum()


def history_index(history: pd.DataFrame) -> dict:
    """Aggregates of a prior year, keyed for the four match levels."""
    core_rows = history[history["core"] != ""]
    return {
        "exact": _sum_index(history, ["street", "vtype"]),
        "head": _sum_index(history, ["hk", "vtype"]),
        "head_n": history.groupby(["hk", "vtype"], sort=False)["street"].nunique(),
        "core": _sum_index(core_rows, ["core", "dire", "vtype"]),
        "core_n": core_rows.groupby(["core", "dire", "vtype"], sort=False)["street"].nunique(),
    }


def attach_history(frame: pd.DataFrame, index: dict) -> pd.DataFrame:
    """Join prior-year counts. Does not read `count` on `frame`."""
    out = frame.copy()
    out["x_exact"] = out.set_index(["street", "vtype"]).index.map(index["exact"])
    out["x_head"] = out.set_index(["hk", "vtype"]).index.map(index["head"])
    out["n_head"] = out.set_index(["hk", "vtype"]).index.map(index["head_n"])
    out["x_core"] = out.set_index(["core", "dire", "vtype"]).index.map(index["core"])
    out["n_core"] = out.set_index(["core", "dire", "vtype"]).index.map(index["core_n"])
    out.loc[out["core"] == "", "x_core"] = np.nan
    return out


def fit_growth(paired: pd.DataFrame) -> dict[str, float]:
    """Per-type multiplier g minimizing sum (y - g x) in aggregate: g = sum(y)/sum(x).

    Fit only on rows with an exact history match. Types with no support fall
    back to the global ratio.
    """
    sub = paired[paired["x_exact"].notna()]
    if sub.empty or float(sub["x_exact"].sum()) <= 0:
        return {"_": 1.0}
    growth: dict[str, float] = {}
    for vtype, grp in sub.groupby("vtype", sort=False):
        denom = float(grp["x_exact"].sum())
        growth[str(vtype)] = float(grp["count"].sum()) / denom if denom > 0 else 1.0
    growth["_"] = float(sub["count"].sum()) / float(sub["x_exact"].sum())
    return growth


def _growth_array(vtypes: pd.Series, growth: dict[str, float]) -> np.ndarray:
    default = growth.get("_", 1.0)
    return vtypes.map(lambda v: growth.get(v, default)).astype(float).to_numpy()


def predict_counts(frame: pd.DataFrame, growth: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Predict non-negative counts and the source label for each row.

    `frame` must already carry x_exact / x_head / n_head / x_core / n_core.
    The ground-truth `count` column, if present, is ignored.
    """
    pred = np.full(len(frame), np.nan)
    source = np.empty(len(frame), dtype=object)
    g = _growth_array(frame["vtype"], growth)
    is_cam = frame["vtype"].isin(CAMERA_VTYPES).to_numpy()

    exact = frame["x_exact"].notna().to_numpy()
    pred[exact] = frame["x_exact"].to_numpy()[exact] * g[exact]
    source[exact] = EXACT

    head = np.isnan(pred) & is_cam & frame["x_head"].notna().to_numpy()
    n_head = frame["n_head"].fillna(1.0).to_numpy()
    pred[head] = frame["x_head"].to_numpy()[head] / n_head[head] * g[head]
    source[head] = HEAD

    core = np.isnan(pred) & is_cam & frame["x_core"].notna().to_numpy()
    n_core = frame["n_core"].fillna(1.0).to_numpy()
    pred[core] = frame["x_core"].to_numpy()[core] / n_core[core] * g[core]
    source[core] = CORE

    source[np.isnan(pred)] = UNSEEN
    pred = np.clip(np.nan_to_num(pred, nan=0.0), 0, None)
    return pred, source


def as_submission_counts(pred: np.ndarray) -> np.ndarray:
    """Nearest non-negative integer. Counts in the source files are integers."""
    return np.floor(np.clip(pred, 0, None) + 0.5).astype(np.int64)


def street_folds(keys: pd.Series, n_folds: int, seed: int = 1) -> list[set]:
    """Group holdout on the normalized street, so spelling variants stay together."""
    streets = np.array(sorted(set(keys)))
    rng = np.random.default_rng(seed)
    rng.shuffle(streets)
    return [set(part) for part in np.array_split(streets, n_folds)]


def _predict_with_growth(frame: pd.DataFrame, growth: dict[str, float], kind: str) -> np.ndarray:
    if kind == "model":
        pred, _ = predict_counts(frame, growth)
        return pred
    g = np.ones(len(frame))
    if kind == "baseline":
        g = np.where(frame["vtype"].isin(CAMERA_VTYPES), 1.05, 1.35)
    exact = frame["x_exact"].notna().to_numpy()
    pred = np.zeros(len(frame))
    pred[exact] = frame["x_exact"].to_numpy()[exact] * g[exact]
    return pred


def cross_validate(history: pd.DataFrame, target: pd.DataFrame, n_folds: int) -> dict:
    """Street-grouped pseudo-time CV: features from `history`, labels from `target`.

    Growth is refit on the complement of each fold. Also scores two references
    on the same folds: carrying the exact count forward, and the baseline
    recipe (exact count x 1.35, or x 1.05 for camera types, else 0).
    """
    indexed = attach_history(target, history_index(history))
    folds = street_folds(indexed["k"], n_folds)
    model_scores, persist_scores, base_scores = [], [], []
    source_counts = {EXACT: 0, HEAD: 0, CORE: 0, UNSEEN: 0}

    for hold in folds:
        train = indexed[~indexed["k"].isin(hold)]
        valid = indexed[indexed["k"].isin(hold)]
        growth = fit_growth(train)
        pred, source = predict_counts(valid, growth)
        y = valid["count"].to_numpy()
        model_scores.append(rmse(pred, y))
        persist_scores.append(rmse(_predict_with_growth(valid, growth, "persist"), y))
        base_scores.append(rmse(_predict_with_growth(valid, growth, "baseline"), y))
        for label in source_counts:
            source_counts[label] += int((source == label).sum())

    return {
        "model": model_scores,
        "persist": persist_scores,
        "baseline": base_scores,
        "sources": source_counts,
        "n": int(len(indexed)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict parking violations per street and type.")
    parser.add_argument("--train-path", default=os.environ.get("TRAIN_PATH", DEFAULT_TRAIN),
                        help="History year used as the feature source at prediction time.")
    parser.add_argument("--prior-path", default=os.environ.get("PRIOR_PATH", DEFAULT_PRIOR),
                        help="Earlier year used only to estimate per-type growth.")
    parser.add_argument("--test-path", default=os.environ.get("TEST_PATH") or None,
                        help="Optional key file. violation_count, if present, is scored and not used as a feature.")
    parser.add_argument("--out-path", default=os.environ.get("OUT_PATH", DEFAULT_OUT))
    parser.add_argument("--cv-folds", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.train_path):
        raise SystemExit(f"train file not found: {args.train_path}")

    train, train_has_y = read_violations(args.train_path)
    if not train_has_y:
        raise SystemExit(f"train file has no violation_count column: {args.train_path}")
    print(f"[data] train rows={len(train)} streets={train['street'].nunique()} "
          f"vtypes={train['vtype'].nunique()} path={args.train_path}")

    if args.prior_path and os.path.isfile(args.prior_path):
        prior, prior_has_y = read_violations(args.prior_path)
        if not prior_has_y:
            raise SystemExit(f"prior file has no violation_count column: {args.prior_path}")
        print(f"[data] prior rows={len(prior)} streets={prior['street'].nunique()} path={args.prior_path}")
        report = cross_validate(prior, train, args.cv_folds)
        model_mean = float(np.mean(report["model"]))
        print(f"[valid] street-grouped {args.cv_folds}-fold RMSE "
              f"mean={model_mean:.3f} folds={[round(s, 3) for s in report['model']]}")
        print(f"[valid] same folds, exact persistence RMSE mean={float(np.mean(report['persist'])):.3f}")
        print(f"[valid] same folds, baseline growth (1.35 / 1.05) RMSE "
              f"mean={float(np.mean(report['baseline'])):.3f}")
        src = report["sources"]
        total = sum(src.values())
        print("[valid] prediction sources over held-out rows "
              "(unseen rows are predicted as 0):")
        for label in (EXACT, HEAD, CORE, UNSEEN):
            n = src[label]
            print(f"  {label:8s} {n:7d} ({100.0 * n / total:5.1f}%)")
        growth = fit_growth(attach_history(train, history_index(prior)))
        print("[model] per-type growth fit on all exact  prior->train pairs:")
        for vtype in sorted(v for v in growth if v != "_"):
            print(f"  {vtype:42s} {growth[vtype]:.4f}")
        print(f"  {'(global)':42s} {growth['_']:.4f}")
        history = train
    else:
        print(f"[warn] prior file not found ({args.prior_path}); using growth 1.0 (persistence).")
        print("[valid] skipped: per-type growth needs the prior year.")
        growth = {"_": 1.0}
        history = train

    if not args.test_path:
        print("[done] no --test-path; submission.csv not written.")
        return

    if not os.path.isfile(args.test_path):
        raise SystemExit(f"test file not found: {args.test_path}")

    test, has_target = read_violations(args.test_path)
    # Ground truth stays in `count` for scoring only. Features come from history.
    featured = attach_history(test, history_index(history))
    raw_pred, source = predict_counts(featured, growth)
    pred = as_submission_counts(raw_pred)

    print(f"[test] rows={len(test)} path={args.test_path}")
    for label in (EXACT, HEAD, CORE, UNSEEN):
        n = int((source == label).sum())
        print(f"  {label:8s} {n:7d} ({100.0 * n / max(len(test), 1):5.1f}%)")
    n_exact = int((source == EXACT).sum())
    n_head = int((source == HEAD).sum())
    n_core = int((source == CORE).sum())
    n_unseen = int((source == UNSEEN).sum())
    print(f"[test] exact matches to training: {n_exact} of {len(test)}.")
    print(f"[test] no exact training key: {len(test) - n_exact}. "
          f"{n_head} used the camera corridor before '@', "
          f"{n_core} used a direction+name corridor match, "
          f"{n_unseen} had no history signal and were predicted as 0.")

    submission = pd.DataFrame({
        "street_name": test["street_raw"].to_numpy(),
        "violation_type": test["vtype_raw"].to_numpy(),
        "predicted_count": pred,
    })
    submission.to_csv(args.out_path, index=False)
    print(f"[done] wrote {args.out_path} ({len(submission)} rows)")

    if has_target:
        # Score the same integers that were written. `count` was not a feature.
        print(f"[eval] RMSE={rmse(pred, test['count'].to_numpy()):.3f}")


if __name__ == "__main__":
    main()
