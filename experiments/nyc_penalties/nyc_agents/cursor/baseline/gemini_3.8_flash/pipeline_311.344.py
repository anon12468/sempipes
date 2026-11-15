#!/usr/bin/env python3
"""NYC parking-violation forecasting pipeline.

Predicts the number of parking violations per (street_name, violation_type)
using 2022 records as the primary training signal, enriched with 2021 history
and infrastructure / activity features from the corpus/ tables.

Prediction hierarchy:
  1. exact / casing match: 2022 count for (street, vtype) with smart multi-casing
     allocation across active variants in the evaluation set.
  2. normalized match: matches on canonical street name (suffix/ordinal normalized)
     and distributes historical counts across siblings.
  3. camera prefix match: on-street + direction prefix match for camera / intersection
     rows (e.g. "WB N CONDUIT AVE @ ..."), distributed across sibling locations.
  4. camera corridor match: matches high-volume camera corridors (e.g. Conduit, Horace
     Harding, Bruckner, Woodhaven) across active cameras.
  5. 2021 corpus match: for keys with no 2022 record, leverages the 2021 dataset in
     corpus/ scaled by the 2-year trend factor.
  6. regression model: LightGBM (with HistGradientBoosting fallback) trained on
     structural features, 2021 historical activity, and 19 corpus features.

Outputs submission.csv with columns:
  street_name, violation_type, predicted_count
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Default paths ----
DEFAULT_TRAIN = os.path.join(HERE, "train", "all_data", "violations_per_street_2022.csv")
if not os.path.exists(DEFAULT_TRAIN):
    alt_train = os.path.join(HERE, "violations_per_street_2022.csv")
    if os.path.exists(alt_train):
        DEFAULT_TRAIN = alt_train

DEFAULT_CORPUS = os.path.join(HERE, "train", "all_data", "corpus")
if not os.path.isdir(DEFAULT_CORPUS):
    alt_corpus = os.path.join(HERE, "corpus")
    if os.path.isdir(alt_corpus):
        DEFAULT_CORPUS = alt_corpus

DEFAULT_OUT = os.path.join(HERE, "submission.csv")

# ---- Growth multipliers ----
DEFAULT_GROWTH = 1.35          # Organic year-over-year multiplier (2022 -> 2023)
DEFAULT_CAMERA_GROWTH = 1.05   # Automated enforcement types multiplier
DEFAULT_GROWTH_2021 = 0.90     # 2-year multiplier for 2021 historical counts

# Automated-enforcement violation types
CAMERA_VTYPES = {
    "PHTO SCHOOL ZN SPEED VIOLATION",
    "BUS LANE VIOLATION",
    "FAILURE TO STOP AT RED LIGHT",
}

# Prediction source labels
EXACT = "exact / casing match"
NORM = "normalized street match"
PREFIX = "camera prefix match"
CORRIDOR = "camera corridor match"
HIST_2021 = "2021 corpus match"
MODEL = "regression model"


# ============================================================================
# String Normalization
# ============================================================================
_SUFFIX = {
    r"\bAVENUE\b": "AVE", r"\bAV\b": "AVE", r"\bSTREET\b": "ST", r"\bSTR\b": "ST",
    r"\bPLACE\b": "PL", r"\bROAD\b": "RD", r"\bBOULEVARD\b": "BLVD", r"\bBLV\b": "BLVD",
    r"\bDRIVE\b": "DR", r"\bLANE\b": "LN", r"\bPARKWAY\b": "PKWY", r"\bPKW\b": "PKWY",
    r"\bCOURT\b": "CT", r"\bTERRACE\b": "TER", r"\bEXPRESSWAY\b": "EXPY",
    r"\bEXPWY\b": "EXPY", r"\bEXP\b": "EXPY", r"\bHIGHWAY\b": "HWY", r"\bHW\b": "HWY",
    r"\bEAST\b": "E", r"\bWEST\b": "W", r"\bNORTH\b": "N", r"\bSOUTH\b": "S",
    r"\bSAINT\b": "ST", r"\bPLAZA\b": "PLZ", r"\bTURNPIKE\b": "TPKE",
}


def norm_vtype(v) -> str:
    """Canonical uppercase violation type for reliable cross-year matching."""
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v).upper().strip())


def norm_street(s) -> str:
    """Normalize a street name to a canonical join key."""
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"\s+", " ", s)
    for pattern, repl in _SUFFIX.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", s)  # 1ST -> 1
    s = re.sub(r"\b0+(\d+)\b", r"\1", s)              # 0103 -> 103
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def prefix_key(s) -> str:
    """On-street + direction key for camera and intersection-coded locations.

    Extracts base street before intersection markers (@, -, /, .).
    """
    if s is None:
        return ""
    s = str(s).upper().strip()
    part = re.split(r"[@\-(/\.]", s)[0].strip()
    return norm_street(part)


def corridor_key(s) -> str:
    """Corridor key stripping directional indicators (e.g. 'WB N CONDUIT' -> 'CONDUIT')."""
    if s is None:
        return ""
    k = norm_street(s)
    # Strip leading directions
    k = re.sub(r"^(EB|WB|NB|SB|E|W|N|S)\s+", "", k)
    # Strip trailing directions
    k = re.sub(r"\s+(EB|WB|NB|SB|E|W|N|S)$", "", k)
    return k.strip()


def strip_house_number(addr) -> str:
    """Drop leading house numbers for PLUTO/land-use address matching."""
    if addr is None:
        return ""
    s = re.sub(r"^\s*\d+[A-Z]?(-\d+[A-Z]?)?\s+", "", str(addr).strip(), flags=re.I)
    return norm_street(s)


# ============================================================================
# Structural Features
# ============================================================================
STRUCT_COLS = [
    "st_is_numbered", "st_n_tokens", "st_len", "st_has_ave", "st_has_st",
    "st_has_blvd", "st_has_pkwy_expy_hwy", "st_has_expy", "st_has_hwy",
    "st_has_conduit", "st_has_broadway", "is_camera_type", "vtype_code",
]


def structural_features(streets: pd.Series, vtypes: pd.Series, vtype_categories: list[str]) -> pd.DataFrame:
    k = streets.map(norm_street)
    df = pd.DataFrame(index=streets.index)
    df["st_is_numbered"] = k.str.match(r"^\d+ ").astype(int)
    df["st_n_tokens"] = k.str.split().map(len)
    df["st_len"] = k.str.len()
    df["st_has_ave"] = k.str.contains(r"\bAVE\b").astype(int)
    df["st_has_st"] = k.str.contains(r"\bST\b").astype(int)
    df["st_has_blvd"] = k.str.contains(r"\bBLVD\b").astype(int)
    df["st_has_pkwy_expy_hwy"] = k.str.contains(r"\b(?:PKWY|EXPY|HWY)\b").astype(int)
    df["st_has_expy"] = k.str.contains(r"\bEXPY\b").astype(int)
    df["st_has_hwy"] = k.str.contains(r"\bHWY\b").astype(int)
    df["st_has_conduit"] = k.str.contains(r"CONDUIT").astype(int)
    df["st_has_broadway"] = k.str.contains(r"BROADWAY").astype(int)
    vcan = vtypes.map(norm_vtype)
    df["is_camera_type"] = vcan.isin(CAMERA_VTYPES).astype(int)
    df["vtype_code"] = pd.Categorical(vcan, categories=vtype_categories).codes
    return df


# ============================================================================
# Corpus Feature Extraction (Parquet + 2021 Violations Table)
# ============================================================================
def _agg_count(path, col, out):
    import pyarrow.parquet as pq
    df = pq.read_table(path, columns=[col]).to_pandas()
    df["k"] = df[col].map(norm_street)
    return df[df["k"] != ""].groupby("k").size().rename(out).to_frame()


def build_corpus_features(corpus_dir: str) -> pd.DataFrame:
    """Extract street-level features from corpus parquet tables."""
    import pyarrow.parquet as pq
    feats = []

    def safe(fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            print(f"  [skip] {a[0] if a else ''}: {e}")
            return None

    cp = lambda f: os.path.join(corpus_dir, f)

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
        hours = [c for c in sch if re.search(r"\d:\d", c)]
        df = pq.read_table(path, columns=["Roadway Name"] + hours).to_pandas()
        df["k"] = df["Roadway Name"].map(norm_street)
        df = df[df["k"] != ""]
        df["vol"] = df[hours].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        return df.groupby("k").agg(f_traffic_vol=("vol", "mean"))
    feats.append(safe(_veh, cp("vehicle_classification_counts_2011_2025__96ay-ea4r.parquet")))

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


def load_corpus_features(corpus_dir: str) -> pd.DataFrame | None:
    """Load corpus features from parquet cache, building from corpus_dir if missing."""
    cache = os.path.join(HERE, "corpus_features.parquet")
    if os.path.exists(cache):
        try:
            return pd.read_parquet(cache)
        except Exception as e:
            print(f"[warn] cache read failed ({e}), rebuilding ...")
    if not os.path.isdir(corpus_dir):
        print(f"[warn] corpus dir not found ({corpus_dir}); proceeding without corpus features.")
        return None
    t = time.time()
    print(f"[info] building corpus features from {corpus_dir} ...")
    f = build_corpus_features(corpus_dir)
    try:
        f.to_parquet(cache)
        print(f"[info] corpus features: {f.shape} in {time.time() - t:.0f}s (cached)")
    except Exception as e:
        print(f"[warn] could not cache corpus features: {e}")
    return f


def load_2021_corpus(corpus_dir: str):
    """Load historical 2021 violations from corpus for feature augmentation and fallback."""
    p21 = os.path.join(corpus_dir, "violations_per_street_2021.csv")
    if not os.path.exists(p21):
        print(f"[info] 2021 corpus dataset not found at {p21}; skipping 2021 augmentation.")
        return None, None, None

    df21 = pd.read_csv(p21)
    df21["k"] = df21["Street Name"].map(norm_street)
    df21["vtype_can"] = df21["Violation Description"].map(norm_vtype)
    df21["upper_st"] = df21["Street Name"].str.upper().str.strip()

    # (k, vtype) pair count in 2021
    p21_pair = df21.groupby(["k", "vtype_can"])["violation_count"].sum().rename("f_2021_pair_c").reset_index()

    # Street-level aggregates in 2021
    st21 = df21.groupby("k").agg(
        f_2021_st_tot=("violation_count", "sum"),
        f_2021_st_max=("violation_count", "max"),
        f_2021_st_ntypes=("vtype_can", "nunique"),
    ).reset_index()

    return df21, p21_pair, st21


# ============================================================================
# Feature Assembly + Regression Model
# ============================================================================
def add_features(frame: pd.DataFrame, cfeat: pd.DataFrame | None, p21_pair: pd.DataFrame | None,
                 st21: pd.DataFrame | None, vtype_categories: list[str]) -> pd.DataFrame:
    """Attach structural, 2021 historical, and corpus features to a (street, vtype) frame."""
    frame = frame.copy()
    frame["k"] = frame["street"].map(norm_street)
    frame["vtype_can"] = frame["vtype"].map(norm_vtype)

    struct = structural_features(frame["street"], frame["vtype"], vtype_categories)
    out = pd.concat([frame.reset_index(drop=True), struct.reset_index(drop=True)], axis=1)

    # Attach 2021 historical features
    if p21_pair is not None and not p21_pair.empty:
        out = out.merge(p21_pair, on=["k", "vtype_can"], how="left")
    if st21 is not None and not st21.empty:
        out = out.merge(st21, on="k", how="left")

    # Attach infrastructure / land-use corpus features
    if cfeat is not None and not cfeat.empty:
        out = out.merge(cfeat, left_on="k", right_index=True, how="left")

    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model input columns: structural + 2021 + corpus f_ features."""
    cols = [c for c in STRUCT_COLS if c in frame.columns]
    f_cols = sorted(c for c in frame.columns if c.startswith("f_"))
    return cols + f_cols


def build_grid(df: pd.DataFrame, vtypes: list[str]) -> pd.DataFrame:
    """Dense (street x vtype) training grid with absent pairs filled as 0 counts."""
    sc = "street" if "street" in df.columns else df.columns[0]
    vc = "vtype" if "vtype" in df.columns else df.columns[1]
    cc = "count" if "count" in df.columns else (df.columns[2] if len(df.columns) > 2 else None)

    streets = df[sc].unique()
    grid = pd.MultiIndex.from_product([streets, vtypes], names=["street", "vtype"]).to_frame(index=False)
    grid["vtype_can"] = grid["vtype"].map(norm_vtype)

    df_clean = pd.DataFrame({
        "street": df[sc].astype(str),
        "vtype_can": df[vc].map(norm_vtype),
        "count": pd.to_numeric(df[cc], errors="coerce").fillna(0.0) if cc else 0.0,
    })
    grid = grid.merge(df_clean, on=["street", "vtype_can"], how="left")
    grid["count"] = grid["count"].fillna(0.0)
    return grid


def fit_model(frame: pd.DataFrame, cols: list[str]):
    """Fit a gradient-boosted regression model on log1p counts."""
    X = frame[cols].astype(float).fillna(0.0)
    y = np.log1p(frame["count"].values)

    if HAS_LIGHTGBM:
        train_data = lgb.Dataset(X, label=y)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "verbosity": -1,
            "random_state": 42,
        }
        model = lgb.train(params, train_data, num_boost_round=300)
        return ("lightgbm", model)

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=100, l2_regularization=1.0, random_state=42,
    )
    model.fit(X, y)
    return ("hist_gb", model)


def model_predict(model_tuple, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Predict violation counts using the fitted model (expm1 floored at 0)."""
    kind, model = model_tuple
    X = frame[cols].astype(float).fillna(0.0)
    if kind == "lightgbm":
        pred_log = model.predict(X)
    else:
        pred_log = model.predict(X)
    return np.clip(np.expm1(pred_log), 0, None)


# ============================================================================
# Metrics
# ============================================================================
def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


# ============================================================================
# IO Helper
# ============================================================================
def read_keyed_csv(path: str):
    """Read a violations CSV -> (df[street, vtype, count?], has_target)."""
    df = pd.read_csv(path)
    cols = {re.sub(r"[\s_]+", " ", str(c).lower().strip()): c for c in df.columns}
    sc = cols.get("street name") or cols.get("street") or df.columns[0]
    vc = cols.get("violation description") or cols.get("violation type") or cols.get("violation") or df.columns[1]
    out = pd.DataFrame({"street": df[sc].astype(str), "vtype": df[vc].astype(str)})

    tgt = (cols.get("violation count") or cols.get("violation_count")
           or cols.get("predicted count") or cols.get("predicted_count"))
    has_target = tgt is not None and tgt in df.columns
    if has_target:
        out["count"] = pd.to_numeric(df[tgt], errors="coerce").fillna(0.0)
    return out, has_target


# ============================================================================
# Hierarchical Predictor
# ============================================================================
def predict(te: pd.DataFrame, tr: pd.DataFrame, te_feat: pd.DataFrame,
            model_tuple, cols: list[str], df21: pd.DataFrame | None = None,
            growth: float = DEFAULT_GROWTH, camera_growth: float = DEFAULT_CAMERA_GROWTH,
            growth_2021: float = DEFAULT_GROWTH_2021):
    """Predict each (street, vtype) using the 6-stage hierarchy.

    Stages:
      1. exact / casing match: handles multi-casing street variants in 2022.
      2. normalized match: handles street suffix/ordinal variations.
      3. camera prefix match: handles on-street + direction prefix (@ and camera rows).
      4. camera corridor match: handles major camera corridors (Conduit, Horace Harding, etc.).
      5. 2021 corpus match: handles streets absent in 2022 that existed in 2021.
      6. regression model: handles remaining genuinely unseen streets.
    """
    te = te.copy()
    tr = tr.copy()

    te["vtype_can"] = te["vtype"].map(norm_vtype)
    tr["vtype_can"] = tr["vtype"].map(norm_vtype)

    te["upper_st"] = te["street"].str.upper().str.strip()
    tr["upper_st"] = tr["street"].str.upper().str.strip()

    te["norm_st"] = te["street"].map(norm_street)
    tr["norm_st"] = tr["street"].map(norm_street)

    te["pk"] = te["street"].map(prefix_key)
    tr["pk"] = tr["street"].map(prefix_key)

    te["corridor"] = te["street"].map(corridor_key)
    tr["corridor"] = tr["street"].map(corridor_key)

    pred = np.full(len(te), np.nan)
    source = np.empty(len(te), dtype=object)

    # ------------------------------------------------------------------------
    # Stage 1: Exact / Casing Allocation
    # ------------------------------------------------------------------------
    # Aggregate total count for the (upper_st, vtype_can) group in training
    tr_upper_tot = tr.groupby(["upper_st", "vtype_can"])["count"].sum().rename("tot_c").reset_index()
    tr_raw = tr.groupby(["street", "vtype_can"])["count"].sum().rename("raw_c").reset_index()

    m_up = te.merge(tr_upper_tot, on=["upper_st", "vtype_can"], how="left")
    m_up = m_up.merge(tr_raw, on=["street", "vtype_can"], how="left")

    te_raw_sum = m_up.groupby(["upper_st", "vtype_can"])["raw_c"].transform("sum")
    te_sibs = m_up.groupby(["upper_st", "vtype_can"])["street"].transform("size")

    val_up = np.where(
        te_raw_sum > 0,
        m_up["tot_c"] * (m_up["raw_c"].fillna(0.0) / te_raw_sum),
        m_up["tot_c"] / te_sibs,
    )
    mask_up = ~m_up["tot_c"].isna() & (val_up > 0)
    pred[mask_up] = val_up[mask_up]
    source[mask_up] = EXACT

    # ------------------------------------------------------------------------
    # Stage 2: Normalized Street Match
    # ------------------------------------------------------------------------
    need_norm = np.isnan(pred)
    if need_norm.any():
        tr_norm = tr.groupby(["norm_st", "vtype_can"])["count"].sum().rename("norm_c").reset_index()
        m_norm = te.merge(tr_norm, on=["norm_st", "vtype_can"], how="left")
        te_norm_sibs = te[need_norm].groupby(["norm_st", "vtype_can"])["street"].transform("size")
        val_norm = (m_norm["norm_c"] / te_norm_sibs).values
        mask_norm = need_norm & ~np.isnan(val_norm) & (val_norm > 0)
        pred[mask_norm] = val_norm[mask_norm]
        source[mask_norm] = NORM

    # ------------------------------------------------------------------------
    # Stage 3: Camera Prefix Match
    # ------------------------------------------------------------------------
    is_cam_target = te["vtype_can"].isin(CAMERA_VTYPES) | te["street"].str.contains("@")
    need_pk = np.isnan(pred) & is_cam_target
    if need_pk.any():
        tr_pk_rows = tr[tr["vtype_can"].isin(CAMERA_VTYPES) | tr["street"].str.contains("@")]
        tr_pk = tr_pk_rows.groupby(["pk", "vtype_can"]).agg(
            pk_sum=("count", "sum"),
            pk_mean=("count", "mean"),
        ).reset_index()
        m_pk = te.merge(tr_pk, on=["pk", "vtype_can"], how="left")
        te_pk_sibs = te[need_pk].groupby(["pk", "vtype_can"])["street"].transform("size")
        # Bounding allocation by pk_mean prevents dumping the entire corridor's total onto a single sibling
        val_pk = np.minimum(m_pk["pk_sum"] / te_pk_sibs, m_pk["pk_mean"]).values
        mask_pk = need_pk & ~np.isnan(val_pk) & (val_pk > 0)
        pred[mask_pk] = val_pk[mask_pk]
        source[mask_pk] = PREFIX

    # ------------------------------------------------------------------------
    # Stage 4: Camera Corridor Match
    # ------------------------------------------------------------------------
    need_cor = np.isnan(pred) & te["vtype_can"].isin(CAMERA_VTYPES)
    if need_cor.any():
        tr_cor_rows = tr[tr["vtype_can"].isin(CAMERA_VTYPES) & (tr["corridor"].str.len() > 3)]
        tr_cor = tr_cor_rows.groupby(["corridor", "vtype_can"]).agg(
            cor_sum=("count", "sum"),
            cor_mean=("count", "mean"),
        ).reset_index()
        m_cor = te.merge(tr_cor, on=["corridor", "vtype_can"], how="left")
        te_cor_sibs = te[need_cor].groupby(["corridor", "vtype_can"])["street"].transform("size")
        val_cor = np.minimum(m_cor["cor_sum"] / te_cor_sibs, m_cor["cor_mean"]).values
        mask_cor = need_cor & ~np.isnan(val_cor) & (val_cor > 0)
        pred[mask_cor] = val_cor[mask_cor]
        source[mask_cor] = CORRIDOR

    # ------------------------------------------------------------------------
    # Apply YoY growth to 2022 persistence sources
    # ------------------------------------------------------------------------
    g_2022 = np.where(te["vtype_can"].isin(CAMERA_VTYPES), camera_growth, growth)
    is_persist_2022 = ~np.isnan(pred)
    pred[is_persist_2022] *= g_2022[is_persist_2022]

    # ------------------------------------------------------------------------
    # Stage 5: 2021 Corpus History Fallback
    # ------------------------------------------------------------------------
    need_2021 = np.isnan(pred)
    if need_2021.any() and df21 is not None and not df21.empty:
        # Match normalized street and violation type in 2021
        p21_agg = df21.groupby(["k", "vtype_can"])["violation_count"].sum().rename("c21").reset_index()
        m_21 = te.merge(p21_agg, left_on=["norm_st", "vtype_can"], right_on=["k", "vtype_can"], how="left")
        val_21 = m_21["c21"].values * growth_2021
        mask_21 = need_2021 & ~np.isnan(val_21) & (val_21 > 0)
        pred[mask_21] = val_21[mask_21]
        source[mask_21] = HIST_2021

    # ------------------------------------------------------------------------
    # Stage 6: Regression Model (Remaining Unresolved)
    # ------------------------------------------------------------------------
    need_model = np.isnan(pred)
    if need_model.any():
        pred_m = model_predict(model_tuple, te_feat, cols)
        pred[need_model] = pred_m[need_model]
        source[need_model] = MODEL

    return np.clip(pred, 0, None), source


# ============================================================================
# Development Validation
# ============================================================================
def run_development_validation(df_all: pd.DataFrame, cfeat: pd.DataFrame | None,
                               df21: pd.DataFrame | None, p21_pair: pd.DataFrame | None,
                               st21: pd.DataFrame | None, vtypes: list[str],
                               val_split: float = 0.20, seed: int = 42):
    """Run grouped holdout validation on 2022 data and report development RMSE."""
    print("\n" + "=" * 70)
    print("DEVELOPMENT VALIDATION (Grouped Holdout on 2022 Data)")
    print("=" * 70)

    df = df_all.copy()
    df["k"] = df["street"].map(norm_street)
    df["vtype_can"] = df["vtype"].map(norm_vtype)

    n_splits = int(round(1.0 / val_split))
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, val_idx = next(gkf.split(df, groups=df["k"]))

    tr_val = df.iloc[train_idx].copy().reset_index(drop=True)
    eval_val = df.iloc[val_idx].copy().reset_index(drop=True)

    print(f"[val] train streets={tr_val['k'].nunique()} (rows={len(tr_val)})")
    print(f"[val] eval streets={eval_val['k'].nunique()} (rows={len(eval_val)}) [unseen holdout]")

    grid_val = add_features(build_grid(tr_val, vtypes), cfeat, p21_pair, st21, vtypes)
    cols = feature_columns(grid_val)
    model_val = fit_model(grid_val, cols)

    eval_feat = add_features(eval_val[["street", "vtype"]], cfeat, p21_pair, st21, vtypes)

    # In grouped holdout of the SAME year (2022), persistence growth within year is 1.0
    pred_val, src_val = predict(
        eval_val, tr_val, eval_feat, model_val, cols, df21=df21,
        growth=1.0, camera_growth=1.0, growth_2021=0.665,
    )
    pred_val_int = np.clip(np.round(pred_val), 0, None).astype(int)

    y_val = eval_val["count"].values
    val_rmse = rmse(pred_val_int, y_val)
    print(f"\n[val] Overall Development Validation RMSE = {val_rmse:.3f}")

    print("\n[val] Prediction breakdown by strategy:")
    for label in (EXACT, NORM, PREFIX, CORRIDOR, HIST_2021, MODEL):
        n = int((src_val == label).sum())
        if n > 0:
            sub_rmse = rmse(pred_val_int[src_val == label], y_val[src_val == label])
            pct = 100.0 * n / len(eval_val)
            print(f"  {label:28s}: {n:>7d} ({pct:5.1f}%) | Sub-RMSE = {sub_rmse:8.2f}")

    unseen_count = int((src_val != EXACT).sum())
    print(f"\n[val] Keys unseen in exact training: {unseen_count} / {len(eval_val)} "
          f"({100.0 * unseen_count / len(eval_val):.1f}%)")
    print("=" * 70 + "\n")
    return val_rmse


# ============================================================================
# Main CLI & Pipeline Runner
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="NYC Parking Violations Forecasting Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-path",
        default=os.environ.get("TRAIN_PATH", DEFAULT_TRAIN),
        help="Path to training CSV (violations_per_street_2022.csv)",
    )
    parser.add_argument(
        "--test-path", "--eval-path",
        default=os.environ.get("TEST_PATH", None),
        help="Path to test/eval CSV (optional; keys-only or keys + ground truth)",
    )
    parser.add_argument(
        "--corpus-dir",
        default=os.environ.get("CORPUS_DIR", DEFAULT_CORPUS),
        help="Path to corpus directory containing augmentation tables and 2021 CSV",
    )
    parser.add_argument(
        "--out-path", "--output-path",
        default=os.environ.get("OUT_PATH", DEFAULT_OUT),
        help="Output path for submission.csv",
    )
    parser.add_argument(
        "--growth",
        type=float,
        default=float(os.environ.get("GROWTH", DEFAULT_GROWTH)),
        help="Organic year-over-year growth multiplier",
    )
    parser.add_argument(
        "--camera-growth",
        type=float,
        default=float(os.environ.get("CAMERA_GROWTH", DEFAULT_CAMERA_GROWTH)),
        help="Camera violation year-over-year multiplier",
    )
    parser.add_argument(
        "--growth-2021",
        type=float,
        default=float(os.environ.get("GROWTH_2021", DEFAULT_GROWTH_2021)),
        help="2-year growth multiplier for 2021 corpus records",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.20,
        help="Holdout validation fraction for development testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[init] train_path  : {args.train_path}")
    print(f"[init] corpus_dir  : {args.corpus_dir}")
    print(f"[init] test_path   : {args.test_path or 'None (running development validation)'}")
    print(f"[init] out_path    : {args.out_path}")
    print(f"[init] growth      : {args.growth:.2f} (organic), {args.camera_growth:.2f} (camera)")
    print(f"[init] backend     : {'LightGBM' if HAS_LIGHTGBM else 'HistGradientBoostingRegressor'}")

    # 1. Load 2022 training data
    if not os.path.exists(args.train_path):
        sys.exit(f"[error] Training file not found at: {args.train_path}")

    tr, _ = read_keyed_csv(args.train_path)
    vtypes = sorted(tr["vtype"].unique())
    print(f"[data] train loaded: rows={len(tr)} unique_streets={tr['street'].nunique()} vtypes={len(vtypes)}")

    # 2. Load corpus features + 2021 historical data
    cfeat = load_corpus_features(args.corpus_dir)
    df21, p21_pair, st21 = load_2021_corpus(args.corpus_dir)

    # 3. Always run development validation on 2022 data
    run_development_validation(
        tr, cfeat, df21, p21_pair, st21, vtypes,
        val_split=args.val_split, seed=args.seed,
    )

    # 4. Fit full regression model on 100% of training data
    print("[train] fitting full regression model on all training data ...")
    t0 = time.time()
    full_grid = add_features(build_grid(tr, vtypes), cfeat, p21_pair, st21, vtypes)
    cols = feature_columns(full_grid)
    full_model = fit_model(full_grid, cols)
    print(f"[train] full model fitted in {time.time() - t0:.1f}s")

    # 5. Inference on test set or generate reference submission
    if args.test_path and os.path.exists(args.test_path):
        print(f"\n[eval] Reading test file: {args.test_path} ...")
        te, has_target = read_keyed_csv(args.test_path)
        print(f"[eval] test loaded: rows={len(te)} unique_streets={te['street'].nunique()}")

        te_feat = add_features(te[["street", "vtype"]], cfeat, p21_pair, st21, vtypes)
        pred, source = predict(
            te, tr, te_feat, full_model, cols, df21=df21,
            growth=args.growth, camera_growth=args.camera_growth,
            growth_2021=args.growth_2021,
        )
        pred_int = np.clip(np.round(pred), 0, None).astype(int)

        print("\n[eval] Prediction breakdown by strategy:")
        for label in (EXACT, NORM, PREFIX, CORRIDOR, HIST_2021, MODEL):
            n = int((source == label).sum())
            if n > 0:
                pct = 100.0 * n / len(te)
                print(f"  {label:28s}: {n:>7d} ({pct:5.1f}%)")

        tr_keys = set(zip(tr["street"].map(norm_street), tr["vtype"].map(norm_vtype)))
        te_keys = set(zip(te["street"].map(norm_street), te["vtype"].map(norm_vtype)))
        unseen_keys = te_keys - tr_keys
        print(f"\n[eval] (street, vtype) keys in test unseen in training: {len(unseen_keys)} / {len(te_keys)}")
        print("       Handled via normalized prefix, corridor, 2021 corpus fallback, and regression model.")

        sub = pd.DataFrame({
            "street_name": te["street"].values,
            "violation_type": te["vtype"].values,
            "predicted_count": pred_int,
        })
        sub.to_csv(args.out_path, index=False)
        print(f"\n[done] wrote {args.out_path} ({len(sub)} rows)")

        if has_target:
            y_test = te["count"].values
            test_rmse = rmse(pred_int, y_test)
            print(f"[eval] Evaluation RMSE on ground truth = {test_rmse:.3f}")
    else:
        if args.test_path:
            print(f"[warn] test-path not found: {args.test_path}")

        # When no test path is provided, generate submission.csv for 2022 key space as reference
        print("\n[info] No test-path provided. Generating reference submission for training keys ...")
        ref_te = tr[["street", "vtype"]].copy()
        ref_feat = add_features(ref_te, cfeat, p21_pair, st21, vtypes)
        pred_ref, _ = predict(
            ref_te, tr, ref_feat, full_model, cols, df21=df21,
            growth=args.growth, camera_growth=args.camera_growth,
            growth_2021=args.growth_2021,
        )
        pred_ref_int = np.clip(np.round(pred_ref), 0, None).astype(int)

        sub = pd.DataFrame({
            "street_name": ref_te["street"].values,
            "violation_type": ref_te["vtype"].values,
            "predicted_count": pred_ref_int,
        })
        sub.to_csv(args.out_path, index=False)
        print(f"[done] wrote reference {args.out_path} ({len(sub)} rows)")
        print("\n[info] Pipeline is ready. To run predictions on 2023 eval data:")
        print("       python3 pipeline.py --test-path <path_to_2023_file>")


if __name__ == "__main__":
    main()
