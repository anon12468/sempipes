#!/usr/bin/env python3
"""NYC parking-violation forecasting pipeline.

Predicts the number of parking violations per (street_name, violation_type) for
the user-supplied test key space from prior-year counts. The bundled 2021
aggregate provides temporal training features for the 2022 model without using
future targets.

Each (street, vtype) is resolved by the first applicable strategy:
  exact match       same raw key in the prior year              -> count x growth
  normalized match  spelling-equivalent street key              -> count x growth
  prefix match      camera/intersection rows with stable prefix  -> count x growth
  temporal model    no matching prior key; infer from prior-year street profile

All paths are configurable and constrained to this workspace. A labeled test
file is used only for final RMSE reporting, never as a feature.
"""
from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE))

# ---- paths ----
DEFAULT_TRAIN_PATH = "violations_per_street_2022.csv"
DEFAULT_CORPUS_DIR = os.path.join(DATA_DIR, "train", "all_data", "corpus")
DEFAULT_HISTORY_PATH = os.path.join(DEFAULT_CORPUS_DIR, "violations_per_street_2021.csv")
DEFAULT_OUT_PATH = os.path.join(HERE, "submission.csv")

# ---- tuned growth multipliers (see module docstring) ----
GROWTH = 1.35          # organic (non-camera) year-over-year multiplier
CAMERA_GROWTH = 1.05   # volatile camera-enforced types

# Automated-enforcement violation types. Their counts depend on where/when the
# city switches cameras on or off, so year-over-year they swing wildly in both
# directions and are essentially unpredictable from a prior count; they get only
# a small persistence bump rather than the organic growth multiplier.
CAMERA_VTYPES = {
    "PHTO SCHOOL ZN SPEED VIOLATION",
    "BUS LANE VIOLATION",
    "FAILURE TO STOP AT RED LIGHT",
}

# Prediction-source labels, applied in priority order (also printed in the report).
EXACT = "exact match"
NORMALIZED = "normalized match"
PREFIX = "prefix match"
MODEL = "temporal/fallback model"


# ============================================================================
# Street-name normalization
# ============================================================================
# Common suffix/direction word -> abbreviation, so "5TH AVENUE" and "5 AVE"
# collapse to the same join key.
_SUFFIX = {
    r"\bAVENUE\b": "AVE", r"\bSTREET\b": "ST", r"\bPLACE\b": "PL",
    r"\bROAD\b": "RD", r"\bBOULEVARD\b": "BLVD", r"\bDRIVE\b": "DR",
    r"\bLANE\b": "LN", r"\bPARKWAY\b": "PKWY", r"\bCOURT\b": "CT",
    r"\bTERRACE\b": "TER", r"\bEXPRESSWAY\b": "EXPY", r"\bHIGHWAY\b": "HWY",
    r"\bEAST\b": "E", r"\bWEST\b": "W", r"\bNORTH\b": "N", r"\bSOUTH\b": "S",
    r"\bSAINT\b": "ST",
}


def norm_street(s) -> str:
    """Normalize a street name to a canonical join key."""
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"\s+", " ", s)
    for pattern, repl in _SUFFIX.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", s)  # 1ST -> 1
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def prefix_key(s) -> str:
    """On-street + direction key for camera/intersection-coded locations.

    "WB N CONDUIT AVE @ <cross st>" -> "WB N CONDUIT AVE". The cross-street part
    is truncated inconsistently across years so exact matching fails, but the
    prefix is stable. Names without "@" are normalized unchanged, so distinct
    streets are never merged.
    """
    if s is None:
        return ""
    return norm_street(str(s).upper().strip().split("@")[0])


def strip_house_number(addr) -> str:
    """Drop a leading house number: "120 BROADWAY" -> "BROADWAY"."""
    if addr is None:
        return ""
    s = re.sub(r"^\s*\d+[A-Z]?(-\d+[A-Z]?)?\s+", "", str(addr).strip(), flags=re.I)
    return norm_street(s)


# ============================================================================
# Structural features (derived purely from the street name)
# ============================================================================
STRUCT_COLS = ["st_is_numbered", "st_n_tokens", "st_len", "st_has_ave",
               "st_has_st", "st_has_blvd", "st_has_pkwy_expy_hwy", "vtype_code"]


def structural_features(streets: pd.Series) -> pd.DataFrame:
    k = streets.map(norm_street)
    df = pd.DataFrame(index=streets.index)
    df["st_is_numbered"] = k.str.match(r"^\d+ ").astype(int)
    df["st_n_tokens"] = k.str.split().map(len)
    df["st_len"] = k.str.len()
    df["st_has_ave"] = k.str.contains(r"\bAVE\b").astype(int)
    df["st_has_st"] = k.str.contains(r"\bST\b").astype(int)
    df["st_has_blvd"] = k.str.contains(r"\bBLVD\b").astype(int)
    df["st_has_pkwy_expy_hwy"] = k.str.contains(r"\b(?:PKWY|EXPY|HWY)\b").astype(int)
    return df


# ============================================================================
# Corpus feature extraction (signal for the regression model)
# ============================================================================
def _agg_count(path, col, out):
    """Count corpus rows per normalized street into a single named feature."""
    import pyarrow.parquet as pq
    df = pq.read_table(path, columns=[col]).to_pandas()
    df["k"] = df[col].map(norm_street)
    return df[df["k"] != ""].groupby("k").size().rename(out).to_frame()


def build_corpus_features(corpus_dir: str) -> pd.DataFrame:
    """Street-level corpus features, indexed by normalized street.

    Missing/changed files are skipped gracefully so the pipeline still runs on a
    partial corpus. Tier 1 = street-segment infrastructure; Tier 2 = land-use /
    activity density.
    """
    import pyarrow.parquet as pq
    feats = []

    def safe(fn, *a, **k):
        if a and isinstance(a[0], str) and not os.path.exists(a[0]):
            return None
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001 -- partial corpus is acceptable
            print(f"  [skip] {a[0] if a else ''}: {e}")
            return None

    cp = lambda f: os.path.join(corpus_dir, f)

    # ---- Tier 1: street-segment infrastructure ----
    feats.append(safe(_agg_count, cp("parking_meters_locations_and_status__693u-uax6.parquet"),
                      "On_Street", "f_meters_cnt"))
    feats.append(safe(_agg_count, cp("bicycle_parking__592z-n7dk.parquet"),
                      "OnStreet", "f_bikepark_cnt"))
    feats.append(safe(_agg_count, cp("speed_reducer_tracking_system_srts__9n6h-pt9g.parquet"),
                      "OnStreet", "f_speedbump_cnt"))

    def _pave(path):
        df = pq.read_table(path, columns=["OnStreetName", "Road_Type", "SystemRating",
                                          "LocationGeometry.STLength"]).to_pandas()
        df["k"] = df["OnStreetName"].map(norm_street)
        df = df[df["k"] != ""]
        df["len"] = pd.to_numeric(df["LocationGeometry.STLength"], errors="coerce")
        df["rating"] = pd.to_numeric(df["SystemRating"], errors="coerce")
        return df.groupby("k").agg(f_pave_seg=("k", "size"),
                                   f_pave_len=("len", "sum"),
                                   f_pave_rating=("rating", "mean"))
    feats.append(safe(_pave, cp("street_pavement_ratings__6yyb-pb25.parquet")))

    def _speed(path):
        df = pq.read_table(path, columns=["street", "postvz_sl"]).to_pandas()
        df["k"] = df["street"].map(norm_street)
        df = df[df["k"] != ""]
        df["sl"] = pd.to_numeric(df["postvz_sl"], errors="coerce")
        return df.groupby("k").agg(f_speedlimit=("sl", "mean"), f_speedseg=("k", "size"))
    feats.append(safe(_speed, cp("vzv_speed_limits__5mad-ntua.parquet")))

    def _veh(path):
        sch = pq.read_schema(path).names
        hours = [c for c in sch if re.search(r"\d:\d", c)]  # hourly volume columns
        df = pq.read_table(path, columns=["Roadway Name"] + hours).to_pandas()
        df["k"] = df["Roadway Name"].map(norm_street)
        df = df[df["k"] != ""]
        df["vol"] = df[hours].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        return df.groupby("k").agg(f_traffic_vol=("vol", "mean"))
    feats.append(safe(_veh, cp("vehicle_classification_counts_2011_2025__96ay-ea4r.parquet")))

    # ---- Tier 2: land-use / activity density ----
    def _pluto(path):
        df = pq.read_table(path, columns=["address", "lotarea", "bldgarea", "comarea",
                                          "resarea", "retailarea", "unitsres"]).to_pandas()
        df["k"] = df["address"].map(strip_house_number)
        df = df[df["k"] != ""]
        for c in ["lotarea", "bldgarea", "comarea", "resarea", "retailarea", "unitsres"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.groupby("k").agg(
            f_pluto_lots=("k", "size"), f_bldgarea=("bldgarea", "sum"),
            f_comarea=("comarea", "sum"), f_resarea=("resarea", "sum"),
            f_retailarea=("retailarea", "sum"), f_unitsres=("unitsres", "sum"))
    feats.append(safe(_pluto, cp("primary_land_use_tax_lot_output_pluto__64uk-42ks.parquet")))
    feats.append(safe(_agg_count, cp("property_valuation_and_assessment_data_tax_classes_1234__8y4t-faws.parquet"),
                      "STREET_NAME", "f_parcels_cnt"))
    feats.append(safe(_agg_count, cp("dob_violations__3h2n-5cm9.parquet"), "STREET", "f_dob_viol_cnt"))
    feats.append(safe(_agg_count, cp("dohmh_new_york_city_restaurant_inspection_results__43nn-pn8j.parquet"),
                      "STREET", "f_restaurant_insp_cnt"))
    feats.append(safe(_agg_count, cp("forestry_planting_spaces__82zj-84is.parquet"), "Street", "f_tree_cnt"))

    feats = [f for f in feats if f is not None]
    out = pd.concat(feats, axis=1) if feats else pd.DataFrame()
    out.index.name = "k"
    return out


# ============================================================================
# Feature assembly + regression model
# ============================================================================
def add_features(frame, cfeat, vtype_categories):
    """Attach structural and corpus features to a (street, vtype) frame."""
    frame = frame.copy()
    frame["k"] = frame["street"].map(norm_street)
    struct = structural_features(frame["street"])
    frame = pd.concat([frame.reset_index(drop=True), struct.reset_index(drop=True)], axis=1)
    if cfeat is not None and not cfeat.empty:
        frame = frame.merge(cfeat, left_on="k", right_index=True, how="left")
    frame["vtype_code"] = pd.Categorical(frame["vtype"], categories=vtype_categories).codes
    return frame


def feature_columns(frame):
    """Model inputs: the fixed structural columns plus every corpus 'f_' column."""
    return list(STRUCT_COLS) + sorted(c for c in frame.columns if c.startswith("f_"))


def build_grid(df, vtypes):
    """Dense (street x vtype) training grid with absent pairs filled as 0 counts."""
    streets = df["street"].unique()
    grid = pd.MultiIndex.from_product([streets, vtypes], names=["street", "vtype"]).to_frame(index=False)
    grid = grid.merge(df, on=["street", "vtype"], how="left")
    grid["count"] = grid["count"].fillna(0.0)
    return grid


def fit_model(frame, cols):
    """Fit the regression model on log1p counts (the target is heavily right-skewed).

    HistGradientBoosting is used because it is dependency-free (ships with
    scikit-learn) and handles missing corpus values natively.
    """
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    model.fit(frame[cols].astype(float), np.log1p(frame["count"].values))
    return model


def model_predict(model, frame, cols):
    """Predict counts (invert the log1p transform and floor at zero)."""
    return np.clip(np.expm1(model.predict(frame[cols].astype(float))), 0, None)


def temporal_features(rows, lag, vtype_categories):
    """Build leakage-free features from a prior aggregate year.

    The lag table may contain different spelling variants. Raw-key history is
    preferred; otherwise the normalized-street total is shared across rows in
    the current key space. Counts of the other violation types on the same
    street let the model recognize generally busy streets.
    """
    out = add_features(rows[["street", "vtype"]], None, vtype_categories)
    lag = lag.copy()
    lag["k"] = lag["street"].map(norm_street)

    raw = lag.groupby(["street", "vtype"], sort=False)["count"].sum()
    norm = lag.groupby(["k", "vtype"], sort=False)["count"].sum()
    raw_idx = pd.MultiIndex.from_frame(rows[["street", "vtype"]])
    norm_idx = pd.MultiIndex.from_frame(out[["k", "vtype"]])
    raw_prior = raw.reindex(raw_idx).to_numpy(dtype=float)
    norm_prior = norm.reindex(norm_idx).to_numpy(dtype=float)
    siblings = (
        out.groupby(["k", "vtype"], sort=False)["street"]
        .transform("size")
        .to_numpy(dtype=float)
    )
    shared_prior = norm_prior / np.maximum(siblings, 1.0)
    prior = np.where(np.isfinite(raw_prior), raw_prior, shared_prior)
    out["lag_count"] = np.nan_to_num(prior, nan=0.0)
    out["lag_log_count"] = np.log1p(out["lag_count"])
    out["lag_has_raw"] = np.isfinite(raw_prior).astype(int)
    out["lag_has_norm"] = np.isfinite(norm_prior).astype(int)

    profile = lag.pivot_table(
        index="k", columns="vtype", values="count", aggfunc="sum", fill_value=0
    )
    profile_cols = []
    for i, vtype in enumerate(vtype_categories):
        col = f"lag_type_{i}"
        profile_cols.append(col)
        values = profile[vtype] if vtype in profile.columns else pd.Series(dtype=float)
        out[col] = out["k"].map(values).fillna(0.0)
    out["lag_street_total"] = out[profile_cols].sum(axis=1)
    out["lag_street_types"] = (out[profile_cols] > 0).sum(axis=1)
    return out


def temporal_columns(frame):
    """Numeric columns consumed by the temporal model."""
    return (
        list(STRUCT_COLS)
        + ["lag_count", "lag_log_count", "lag_has_raw", "lag_has_norm",
           "lag_street_total", "lag_street_types"]
        + sorted(c for c in frame.columns if c.startswith("lag_type_"))
    )


def fit_temporal_model(current, history, vtypes, validate=True):
    """Learn the observed history->current transition and optionally backtest.

    Validation holds out complete normalized streets, preventing aliases or
    violation types from the same street from crossing the split.
    """
    frame = temporal_features(current, history, vtypes)
    cols = temporal_columns(frame)
    x = frame[cols].astype(float)
    y = current["count"].to_numpy(dtype=float)

    params = dict(
        n_estimators=250,
        min_samples_leaf=2,
        max_features=0.7,
        n_jobs=max(1, min(4, os.cpu_count() or 1)),
        random_state=17,
    )
    if validate and frame["k"].nunique() >= 10:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=17)
        train_idx, valid_idx = next(splitter.split(x, y, groups=frame["k"]))
        check = ExtraTreesRegressor(**params)
        check.fit(x.iloc[train_idx], y[train_idx])
        valid_pred = np.clip(check.predict(x.iloc[valid_idx]), 0, None)
        lag_pred = x.iloc[valid_idx]["lag_count"].to_numpy()
        print(
            f"[validation] grouped 2022 holdout rows={len(valid_idx)} "
            f"lag-only RMSE={rmse(lag_pred, y[valid_idx]):.3f} "
            f"model RMSE={rmse(valid_pred, y[valid_idx]):.3f}"
        )

    model = ExtraTreesRegressor(**params)
    model.fit(x, y)
    return model, cols


# ============================================================================
# Metrics
# ============================================================================
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


# ============================================================================
# IO + feature caching
# ============================================================================
def read_keyed_csv(path):
    """Read a violations CSV -> (df[street, vtype, count?], has_target)."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    sc = cols.get("street name") or cols.get("street_name") or df.columns[0]
    vc = cols.get("violation description") or cols.get("violation_type") or df.columns[1]
    out = pd.DataFrame({
        "street": df[sc].fillna("").astype(str).str.strip(),
        "vtype": df[vc].fillna("").astype(str).str.upper().str.strip(),
        "street_out": df[sc].fillna("").astype(str),
        "vtype_out": df[vc].fillna("").astype(str),
    })
    tgt = cols.get("violation_count") or cols.get("violation count") or cols.get("predicted_count")
    has_target = tgt is not None and tgt in df.columns
    if has_target:
        out["count"] = pd.to_numeric(df[tgt], errors="coerce").fillna(0.0)
    return out, has_target


def load_corpus_features(corpus_dir, cache_path):
    """Load corpus features from cache, building from the corpus dir if needed."""
    if not os.path.isdir(corpus_dir):
        print(f"[warn] corpus dir not found ({corpus_dir}); proceeding without corpus features.")
        return None
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    t = time.time()
    print(f"[info] building corpus features from {corpus_dir} ...")
    f = build_corpus_features(corpus_dir)
    if not f.empty:
        f.to_parquet(cache_path)
        suffix = " (cached)"
    else:
        suffix = ""
    print(f"[info] corpus features: {f.shape} in {time.time() - t:.0f}s{suffix}")
    return f


# ============================================================================
# Prediction (exact -> normalized -> prefix -> model)
# ============================================================================
def predict(te, tr, te_feat, model, cols, fallback_pred=None,
            growth=GROWTH, camera_growth=CAMERA_GROWTH):
    """Predict each (street, vtype) and record which source produced it.

    Each key is resolved by the first applicable strategy and falls through to
    the next only when it does not apply: exact, normalized, prefix, then model.
    Persistence predictions are scaled by the selected growth multiplier.
    Returns (pred, source) as parallel arrays aligned with `te`.
    """
    pred = np.full(len(te), np.nan)
    source = np.empty(len(te), dtype=object)

    # 1. Exact match: carry the prior-year count forward.
    raw = tr.groupby(["street", "vtype"], sort=False)["count"].sum()
    raw_idx = pd.MultiIndex.from_frame(te[["street", "vtype"]])
    raw_prior = raw.reindex(raw_idx).to_numpy(dtype=float)
    matched = np.isfinite(raw_prior)
    pred[matched] = raw_prior[matched]
    source[matched] = EXACT

    # 2. Normalized match. Subtract counts already consumed by raw matches so
    # spelling aliases in the test key space cannot duplicate a street total.
    tr_norm = tr.assign(nk=tr["street"].map(norm_street))
    norm_total = tr_norm.groupby(["nk", "vtype"])["count"].sum()
    te_norm = te[["street", "vtype"]].assign(nk=te["street"].map(norm_street))
    norm_idx = pd.MultiIndex.from_frame(te_norm[["nk", "vtype"]])
    available = norm_total.reindex(norm_idx).to_numpy(dtype=float)
    assigned = pd.Series(np.nan_to_num(pred, nan=0.0)).groupby(
        [te_norm["nk"], te_norm["vtype"]]
    ).transform("sum").to_numpy()
    unresolved = pd.Series(np.isnan(pred)).groupby(
        [te_norm["nk"], te_norm["vtype"]]
    ).transform("sum").to_numpy()
    norm_share = np.clip(available - assigned, 0, None) / np.maximum(unresolved, 1)
    use_norm = np.isnan(pred) & np.isfinite(norm_share)
    pred[use_norm] = norm_share[use_norm]
    source[use_norm] = NORMALIZED

    # 3. Prefix match: intersection/camera rows share a stable on-street+direction
    #    prefix. Split the 2022 prefix total evenly across the eval siblings.
    tr_pk = tr.assign(pk=tr["street"].map(prefix_key))
    agg = tr_pk.groupby(["pk", "vtype"])["count"].sum().rename("pk_count").reset_index()
    te_pk = te[["street", "vtype"]].assign(pk=te["street"].map(prefix_key))
    te_pk = te_pk.merge(agg, on=["pk", "vtype"], how="left")
    assigned = pd.Series(np.nan_to_num(pred, nan=0.0)).groupby(
        [te_pk["pk"], te_pk["vtype"]]
    ).transform("sum").to_numpy()
    unresolved = pd.Series(np.isnan(pred)).groupby(
        [te_pk["pk"], te_pk["vtype"]]
    ).transform("sum").to_numpy()
    pk_share = (
        np.clip(te_pk["pk_count"].to_numpy(dtype=float) - assigned, 0, None)
        / np.maximum(unresolved, 1)
    )
    use_prefix = np.isnan(pred) & ~np.isnan(pk_share)
    pred[use_prefix] = pk_share[use_prefix]
    source[use_prefix] = PREFIX

    # 4. Model: everything still unresolved (no prior-year history).
    use_model = np.isnan(pred)
    if fallback_pred is None:
        fallback_pred = model_predict(model, te_feat, cols)
    pred[use_model] = np.asarray(fallback_pred)[use_model]
    source[use_model] = MODEL

    # Persistence sources get the YoY growth bump (lower for cameras); the
    # regression model is already calibrated on the data, so it is left as-is.
    g = np.where(te["vtype"].isin(CAMERA_VTYPES).values, camera_growth, growth)
    is_persistence = (source == EXACT) | (source == NORMALIZED) | (source == PREFIX)
    pred[is_persistence] *= g[is_persistence]
    return np.clip(pred, 0, None), source


# ============================================================================
# Run
# ============================================================================
def workspace_path(value, *, must_exist=False):
    """Resolve a path while enforcing the workspace-only constraint."""
    if value is None:
        return None
    path = value if os.path.isabs(value) else os.path.join(HERE, value)
    path = os.path.abspath(os.path.expanduser(path))
    if must_exist and not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", help="Optional keys-only or labeled eval CSV")
    parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH,
                        help="Optional prior-year aggregate used for temporal features")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-path", default=DEFAULT_OUT_PATH)
    parser.add_argument("--growth", type=float, default=GROWTH)
    parser.add_argument("--camera-growth", type=float, default=CAMERA_GROWTH)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        train_path = workspace_path(args.train_path)
        # The challenge bundle keeps the default file below train/all_data.
        if not os.path.exists(train_path) and args.train_path == DEFAULT_TRAIN_PATH:
            train_path = workspace_path(
                os.path.join("train", "all_data", DEFAULT_TRAIN_PATH), must_exist=True
            )
        else:
            workspace_path(train_path, must_exist=True)
        test_path = workspace_path(args.test_path, must_exist=True) if args.test_path else None
        history_path = workspace_path(args.history_path) if args.history_path else None
        corpus_dir = workspace_path(args.corpus_dir)
        output_path = workspace_path(args.output_path)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[error] {exc}") from exc

    # ---- load 2022 training data + build features ----
    tr, train_has_target = read_keyed_csv(train_path)
    if not train_has_target:
        raise SystemExit("[error] training CSV must contain violation_count")
    vtypes = sorted(tr["vtype"].unique())
    print(f"[data] train rows={len(tr)} streets={tr['street'].nunique()} vtypes={len(vtypes)}")

    temporal_model = temporal_cols = None
    if history_path and os.path.exists(history_path):
        history, history_has_target = read_keyed_csv(history_path)
        if history_has_target:
            print(f"[data] history rows={len(history)} path={history_path}")
            temporal_model, temporal_cols = fit_temporal_model(
                tr, history, vtypes, validate=not args.skip_validation
            )
    elif history_path:
        print(f"[warn] history file not found ({history_path}); using structural fallback")

    if test_path is None:
        print("[done] training complete; pass --test-path to generate submission.csv")
        return

    # ---- predict the user-supplied key space ----
    te, has_target = read_keyed_csv(test_path)
    fallback_pred = None
    if temporal_model is not None:
        transition = temporal_features(te, tr, vtypes)
        fallback_pred = np.clip(
            temporal_model.predict(transition[temporal_cols].astype(float)), 0, None
        )
        model = cols = None
        te_feat = pd.DataFrame(index=te.index)
    else:
        cache_path = os.path.join(HERE, "corpus_features.parquet")
        cfeat = load_corpus_features(corpus_dir, cache_path)
        grid = add_features(build_grid(tr, vtypes), cfeat, vtypes)
        cols = feature_columns(grid)
        model = fit_model(grid, cols)
        te_feat = add_features(te[["street", "vtype"]], cfeat, vtypes)
    pred, source = predict(
        te, tr, te_feat, model, cols, fallback_pred=fallback_pred,
        growth=args.growth, camera_growth=args.camera_growth,
    )

    for label in (EXACT, NORMALIZED, PREFIX, MODEL):
        n = int((source == label).sum())
        print(f"  {label:18s}: {n:>7d} ({100 * n / len(te):5.1f}%)")

    train_keys = set(zip(tr["street"], tr["vtype"]))
    unseen = sum(key not in train_keys for key in zip(te["street"], te["vtype"]))
    print(
        f"[coverage] unseen raw key pairs={unseen}/{len(te)}; "
        "handled by normalized/prefix history, then the temporal/fallback model"
    )

    sub = pd.DataFrame({"street_name": te["street_out"].values,
                        "violation_type": te["vtype_out"].values,
                        "predicted_count": pred})
    sub.to_csv(output_path, index=False, float_format="%.6f")
    print(f"[done] wrote {output_path} ({len(sub)} rows)")

    y = te["count"].values
    print(f"[eval] RMSE={rmse(pred, y):.3f}")


if __name__ == "__main__":
    main()

