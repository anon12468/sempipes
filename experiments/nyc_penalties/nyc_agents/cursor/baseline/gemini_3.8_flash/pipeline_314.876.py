#!/usr/bin/env python3
"""NYC parking-violation forecasting pipeline.

Predicts the number of parking violations per (street_name, violation_type) for
future evaluation periods (e.g. 2023) from training counts (e.g. 2022), enriched
with multi-table corpus features, structural street signals, and training priors.

Resolution Hierarchy:
  1. Exact Raw Match      The exact (street, vtype) existed in training -> count x growth.
  2. Canonical Match      Street matches after casing/punctuation/suffix normalization
                          ("10th Ave" vs "10TH AVE", "BLV" vs "BLVD")   -> count x growth.
  3. Corridor Prefix      Corridor + direction match for camera corridors ("WB N CONDUIT...")
                          using corridor average camera intensity       -> count x growth.
  4. Regression Ensemble  Dual-head HistGradientBoosting regressor (sqrt-target + calibrated
                          log1p-target) fitted on structural features, 38 corpus features,
                          and training signal prior distributions.

Meets all requirements of train/all_data/goal.txt:
  - Path configurable via CLI args (--train-path, --test-path, --corpus-dir, --out-path)
    and environment variables.
  - Development validation on 2022 training holdout reporting dev RMSE and unseen key handling.
  - Test/eval evaluation when --test-path is supplied (scoring RMSE if ground truth is present).
  - Emits submission.csv with schema: street_name, violation_type, predicted_count.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

# Suppress loky CPU detection warnings in sandboxed environments
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Default Paths ----
DEFAULT_TRAIN_PATH = (
    os.path.join(HERE, "violations_per_street_2022.csv")
    if os.path.exists(os.path.join(HERE, "violations_per_street_2022.csv"))
    else os.path.join(HERE, "train", "all_data", "violations_per_street_2022.csv")
)
DEFAULT_CORPUS_DIR = (
    os.path.join(HERE, "corpus")
    if os.path.isdir(os.path.join(HERE, "corpus"))
    else os.path.join(HERE, "train", "all_data", "corpus")
)
DEFAULT_OUT_PATH = os.path.join(HERE, "submission.csv")
CACHE_FILE = os.path.join(HERE, "corpus_features.parquet")

# ---- Growth Multipliers ----
DEFAULT_GROWTH = 1.35          # Organic year-over-year multiplier
DEFAULT_CAMERA_GROWTH = 1.05   # Camera-enforced types multiplier

CAMERA_VTYPES = {
    "PHTO SCHOOL ZN SPEED VIOLATION",
    "BUS LANE VIOLATION",
    "FAILURE TO STOP AT RED LIGHT",
}

# Prediction-source labels
EXACT_RAW = "exact raw match"
CANONICAL = "canonical match"
CORRIDOR = "corridor prefix match"
MODEL = "regression ensemble"


# ============================================================================
# Street & Violation Normalization
# ============================================================================
_SUFFIX = {
    r"\bAVENUE\b": "AVE", r"\bAVEN\b": "AVE", r"\bAV\b": "AVE",
    r"\bSTREET\b": "ST", r"\bSTR\b": "ST", r"\bSTREE\b": "ST",
    r"\bPLACE\b": "PL",
    r"\bROAD\b": "RD",
    r"\bBOULEVARD\b": "BLVD", r"\bBLV\b": "BLVD", r"\bBL\b": "BLVD", r"\bBOULEV\b": "BLVD",
    r"\bDRIVE\b": "DR",
    r"\bLANE\b": "LN",
    r"\bPARKWAY\b": "PKWY", r"\bPKW\b": "PKWY", r"\bPK\b": "PKWY",
    r"\bCOURT\b": "CT",
    r"\bTERRACE\b": "TER",
    r"\bEXPRESSWAY\b": "EXPY", r"\bEXPWY\b": "EXPY", r"\bEXPR\b": "EXPY", r"\bEX\b": "EXPY",
    r"\bHIGHWAY\b": "HWY",
    r"\bTURNPIKE\b": "TPKE",
    r"\bEAST\b": "E", r"\bWEST\b": "W", r"\bNORTH\b": "N", r"\bSOUTH\b": "S",
    r"\bSAINT\b": "ST",
}


def clean_street(s) -> str:
    """Canonical street normalization resolving casing, abbreviations, and prefixes."""
    if s is None or pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"\b([NSEW])/B\b", r"\1B", s)  # N/B -> NB, etc.
    s = re.sub(r"\bC/O\b", " ", s)            # corner of
    s = re.sub(r"\s+", " ", s)
    for pattern, repl in _SUFFIX.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", s)  # 1ST -> 1
    # Canonicalize frequent corridor naming variations
    s = re.sub(r"\bCONDUIT AVE\b", "CONDUIT BLVD", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_street(s) -> str:
    """Standard join key normalization for corpus table lookups."""
    if s is None or pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"\s+", " ", s)
    for pattern, repl in _SUFFIX.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def corridor_key(s) -> str:
    """Extract on-street direction and corridor name before intersection symbol."""
    if s is None or pd.isna(s):
        return ""
    part = re.split(r"[@/]|(?:\b(?:AT|AND|CORNER OF)\b)", str(s).upper().strip())[0]
    return clean_street(part)


def norm_vtype(s) -> str:
    """Standardize violation description to uppercase trimmed string."""
    if s is None or pd.isna(s):
        return ""
    return str(s).upper().strip()


def strip_house_number(addr) -> str:
    """Drop leading house number: '120 BROADWAY' -> 'BROADWAY'."""
    if addr is None or pd.isna(addr):
        return ""
    s = re.sub(r"^\s*\d+[A-Z]?(-\d+[A-Z]?)?\s+", "", str(addr).strip(), flags=re.I)
    return norm_street(s)


# ============================================================================
# Structural & Training-Signal Features
# ============================================================================
STRUCT_COLS = [
    "st_is_numbered", "st_n_tokens", "st_len", "st_has_ave",
    "st_has_st", "st_has_blvd", "st_has_pkwy_expy_hwy", "st_has_broadway",
    "st_has_conduit", "st_has_dir", "st_number", "is_camera_vtype", "vtype_code"
]


def structural_features(streets: pd.Series, vtypes: pd.Series, vtype_categories: list[str]) -> pd.DataFrame:
    """Extract structural signals purely from street names and violation types."""
    k = streets.map(clean_street)
    df = pd.DataFrame(index=streets.index)
    df["st_is_numbered"] = k.str.match(r"^\d+ ").astype(int)
    df["st_n_tokens"] = k.str.split().map(len)
    df["st_len"] = k.str.len()
    df["st_has_ave"] = k.str.contains(r"\bAVE\b").astype(int)
    df["st_has_st"] = k.str.contains(r"\bST\b").astype(int)
    df["st_has_blvd"] = k.str.contains(r"\bBLVD\b").astype(int)
    df["st_has_pkwy_expy_hwy"] = k.str.contains(r"\b(?:PKWY|EXPY|HWY)\b").astype(int)
    df["st_has_broadway"] = k.str.contains(r"\bBROADWAY\b").astype(int)
    df["st_has_conduit"] = k.str.contains(r"\bCONDUIT\b").astype(int)
    df["st_has_dir"] = k.str.match(r"^(?:NB|SB|EB|WB)\b").astype(int)
    
    # Extract numerical street number if available
    extracted_num = k.str.extract(r"^(\d+)", expand=False)
    df["st_number"] = pd.to_numeric(extracted_num, errors="coerce").fillna(0).astype(int)
    
    vt_norm = vtypes.map(norm_vtype)
    df["is_camera_vtype"] = vt_norm.isin(CAMERA_VTYPES).astype(int)
    df["vtype_code"] = pd.Categorical(vt_norm, categories=vtype_categories).codes
    return df


# ============================================================================
# Corpus Feature Extraction & Caching
# ============================================================================
def _safe_call(fn, *args, **kwargs):
    """Safely execute an extraction function; skip missing files gracefully."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  [skip] {args[0] if args else ''}: {e}")
        return None


def _agg_count(path, col, out_name):
    import pyarrow.parquet as pq
    df = pq.read_table(path, columns=[col]).to_pandas()
    df["k"] = df[col].map(norm_street)
    return df[df["k"] != ""].groupby("k").size().rename(out_name).to_frame()


def build_corpus_features(corpus_dir: str) -> pd.DataFrame:
    """Extract street-level infrastructure, land use, civic, and historical features."""
    import pyarrow.parquet as pq
    feats = []
    cp = lambda f: os.path.join(corpus_dir, f)

    # ---- Tier 1: Street-segment infrastructure ----
    feats.append(_safe_call(_agg_count, cp("parking_meters_locations_and_status__693u-uax6.parquet"),
                            "On_Street", "f_meters_cnt"))
    feats.append(_safe_call(_agg_count, cp("bicycle_parking__592z-n7dk.parquet"),
                            "OnStreet", "f_bikepark_cnt"))
    feats.append(_safe_call(_agg_count, cp("speed_reducer_tracking_system_srts__9n6h-pt9g.parquet"),
                            "OnStreet", "f_speedbump_cnt"))

    def _pave(path):
        df = pq.read_table(path, columns=["OnStreetName", "LocationGeometry.STLength", "SystemRating"]).to_pandas()
        df["k"] = df["OnStreetName"].map(norm_street)
        df = df[df["k"] != ""]
        df["len"] = pd.to_numeric(df["LocationGeometry.STLength"], errors="coerce")
        df["rating"] = pd.to_numeric(df["SystemRating"], errors="coerce")
        return df.groupby("k").agg(
            f_pave_seg=("k", "size"),
            f_pave_len=("len", "sum"),
            f_pave_rating=("rating", "mean")
        )
    feats.append(_safe_call(_pave, cp("street_pavement_ratings__6yyb-pb25.parquet")))

    def _speed(path):
        df = pq.read_table(path, columns=["street", "postvz_sl"]).to_pandas()
        df["k"] = df["street"].map(norm_street)
        df = df[df["k"] != ""]
        df["sl"] = pd.to_numeric(df["postvz_sl"], errors="coerce")
        return df.groupby("k").agg(f_speedlimit=("sl", "mean"), f_speedseg=("k", "size"))
    feats.append(_safe_call(_speed, cp("vzv_speed_limits__5mad-ntua.parquet")))

    def _veh(path):
        sch = pq.read_schema(path).names
        hours = [c for c in sch if re.search(r"\d:\d", c)]
        df = pq.read_table(path, columns=["Roadway Name"] + hours).to_pandas()
        df["k"] = df["Roadway Name"].map(norm_street)
        df = df[df["k"] != ""]
        df["vol"] = df[hours].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        return df.groupby("k").agg(f_traffic_vol=("vol", "mean"))
    feats.append(_safe_call(_veh, cp("vehicle_classification_counts_2011_2025__96ay-ea4r.parquet")))

    # ---- Tier 2: Land-use and property density ----
    def _pluto(path):
        cols = ["address", "lotarea", "bldgarea", "comarea", "resarea", "retailarea", "unitsres"]
        df = pq.read_table(path, columns=cols).to_pandas()
        df["k"] = df["address"].map(strip_house_number)
        df = df[df["k"] != ""]
        for c in ["lotarea", "bldgarea", "comarea", "resarea", "retailarea", "unitsres"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.groupby("k").agg(
            f_pluto_lots=("k", "size"),
            f_bldgarea=("bldgarea", "sum"),
            f_comarea=("comarea", "sum"),
            f_resarea=("resarea", "sum"),
            f_retailarea=("retailarea", "sum"),
            f_unitsres=("unitsres", "sum")
        )
    feats.append(_safe_call(_pluto, cp("primary_land_use_tax_lot_output_pluto__64uk-42ks.parquet")))
    feats.append(_safe_call(_agg_count, cp("property_valuation_and_assessment_data_tax_classes_1234__8y4t-faws.parquet"),
                            "STREET_NAME", "f_parcels_cnt"))
    feats.append(_safe_call(_agg_count, cp("dob_violations__3h2n-5cm9.parquet"),
                            "STREET", "f_dob_viol_cnt"))
    feats.append(_safe_call(_agg_count, cp("dohmh_new_york_city_restaurant_inspection_results__43nn-pn8j.parquet"),
                            "STREET", "f_restaurant_insp_cnt"))
    feats.append(_safe_call(_agg_count, cp("forestry_planting_spaces__82zj-84is.parquet"),
                            "Street", "f_tree_cnt"))

    # ---- Tier 3: Civic, commercial, school, and historical signals ----
    def _viol21(path):
        df = pd.read_csv(path)
        df["k"] = df["Street Name"].map(norm_street)
        df["vc"] = pd.to_numeric(df["violation_count"], errors="coerce").fillna(0)
        return df.groupby("k").agg(
            f_2021_tot_viol=("vc", "sum"),
            f_2021_max_viol=("vc", "max"),
            f_2021_n_vtypes=("Violation Description", "nunique")
        )
    feats.append(_safe_call(_viol21, cp("violations_per_street_2021.csv")))

    def _facilities(path):
        df = pq.read_table(path, columns=["STREETNAME", "FACGROUP"]).to_pandas()
        df["k"] = df["STREETNAME"].map(norm_street)
        df = df[df["k"] != ""]
        df["is_school"] = df["FACGROUP"].isin(["SCHOOLS (K-12)", "DAY CARE AND PRE-KINDERGARTEN", "YOUTH SERVICES"]).astype(int)
        df["is_transit"] = (df["FACGROUP"] == "TRANSPORTATION").astype(int)
        return df.groupby("k").agg(
            f_fac_tot=("k", "size"),
            f_fac_schools=("is_school", "sum"),
            f_fac_transit=("is_transit", "sum")
        )
    feats.append(_safe_call(_facilities, cp("facilities__67g2-p84d.parquet")))

    def _open_rest(path):
        df = pq.read_table(path, columns=["BusinessAddress"]).to_pandas()
        df["k"] = df["BusinessAddress"].map(strip_house_number)
        return df[df["k"] != ""].groupby("k").size().rename("f_open_rest_cnt").to_frame()
    feats.append(_safe_call(_open_rest, cp("open_restaurants_inspections__4dx7-axux.parquet")))

    def _storefronts(path):
        df = pq.read_table(path, columns=["Property Street"]).to_pandas()
        df["k"] = df["Property Street"].map(norm_street)
        return df[df["k"] != ""].groupby("k").size().rename("f_storefront_cnt").to_frame()
    feats.append(_safe_call(_storefronts, cp("storefronts_reported_vacant_or_not__92iy-9c3n.parquet")))

    def _housing_lit(path):
        df = pq.read_table(path, columns=["StreetName"]).to_pandas()
        df["k"] = df["StreetName"].map(norm_street)
        return df[df["k"] != ""].groupby("k").size().rename("f_housing_lit_cnt").to_frame()
    feats.append(_safe_call(_housing_lit, cp("housing_litigations__59kj-x8nc.parquet")))

    def _req311(path):
        df = pq.read_table(path, columns=["Complaint Type", "Street Name"]).to_pandas()
        df["k"] = df["Street Name"].map(norm_street)
        df = df[df["k"] != ""]
        df["is_driveway"] = (df["Complaint Type"] == "Blocked Driveway").astype(int)
        df["is_sig"] = (df["Complaint Type"] == "Traffic Signal Condition").astype(int)
        return df.groupby("k").agg(
            f_311_tot=("k", "size"),
            f_311_blocked_driveway=("is_driveway", "sum"),
            f_311_traffic_sig=("is_sig", "sum")
        )
    feats.append(_safe_call(_req311, cp("311_service_requests_for_2009__3rfa-3xsf.parquet")))

    valid_feats = [f for f in feats if f is not None]
    if not valid_feats:
        return pd.DataFrame()

    out = valid_feats[0]
    for other in valid_feats[1:]:
        out = out.join(other, how="outer")

    # ---- Tier 4: Engineered ratios and densities ----
    if "f_comarea" in out.columns and "f_bldgarea" in out.columns:
        out["f_com_share"] = out["f_comarea"].fillna(0) / (out["f_bldgarea"].fillna(0) + 1)
        out["f_res_share"] = out["f_resarea"].fillna(0) / (out["f_bldgarea"].fillna(0) + 1)
        out["f_retail_share"] = out["f_retailarea"].fillna(0) / (out["f_bldgarea"].fillna(0) + 1)
    if "f_bldgarea" in out.columns and "f_pave_len" in out.columns:
        out["f_density"] = out["f_bldgarea"].fillna(0) / (out["f_pave_len"].fillna(0) + 1)
    if "f_meters_cnt" in out.columns and "f_pave_seg" in out.columns:
        out["f_meters_per_seg"] = out["f_meters_cnt"].fillna(0) / (out["f_pave_seg"].fillna(0) + 1)
    if "f_pluto_lots" in out.columns and "f_pave_seg" in out.columns:
        out["f_lots_per_seg"] = out["f_pluto_lots"].fillna(0) / (out["f_pave_seg"].fillna(0) + 1)
    if "f_fac_schools" in out.columns and "f_pave_seg" in out.columns:
        out["f_schools_per_seg"] = out["f_fac_schools"].fillna(0) / (out["f_pave_seg"].fillna(0) + 1)

    out.index.name = "k"
    return out


def load_corpus_features(corpus_dir: str, rebuild_cache: bool = False) -> pd.DataFrame | None:
    """Load corpus features from cache file, building and caching if needed."""
    if not os.path.isdir(corpus_dir):
        print(f"[warn] Corpus dir not found ({corpus_dir}); continuing without corpus features.")
        return None
    if os.path.exists(CACHE_FILE) and not rebuild_cache:
        try:
            cf = pd.read_parquet(CACHE_FILE)
            if "f_2021_tot_viol" in cf.columns:
                return cf
        except Exception:
            pass

    t0 = time.time()
    print(f"[info] Extracting and augmenting corpus features from {corpus_dir} ...")
    cf = build_corpus_features(corpus_dir)
    try:
        cf.to_parquet(CACHE_FILE)
        print(f"[info] Cached corpus features: {cf.shape} in {time.time() - t0:.1f}s -> {CACHE_FILE}")
    except Exception as e:
        print(f"[warn] Could not write cache file: {e}")
    return cf


# ============================================================================
# Feature Assembly & Regression Ensemble
# ============================================================================
def compute_training_priors(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute prior distribution statistics from the training set."""
    tr_work = tr.copy()
    tr_work["vtype_norm"] = tr_work["vtype"].map(norm_vtype)
    tr_work["clean_street"] = tr_work["street"].map(clean_street)

    vt_priors = tr_work.groupby("vtype_norm")["count"].agg(
        tr_vtype_mean="mean",
        tr_vtype_std="std",
        tr_vtype_median="median",
        tr_vtype_max="max"
    ).reset_index()

    st_priors = tr_work.groupby("clean_street")["count"].agg(
        tr_street_total="sum",
        tr_street_max="max",
        tr_street_n_types="count"
    ).reset_index()
    return vt_priors, st_priors


def add_features(frame: pd.DataFrame, cfeat: pd.DataFrame | None,
                 vtype_categories: list[str],
                 vt_priors: pd.DataFrame | None = None,
                 st_priors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble structural, corpus, and training prior features for a dataframe."""
    df = frame.copy().reset_index(drop=True)
    df["k"] = df["street"].map(norm_street)
    df["clean_street"] = df["street"].map(clean_street)
    df["vtype_norm"] = df["vtype"].map(norm_vtype)

    struct = structural_features(df["street"], df["vtype"], vtype_categories)
    df = pd.concat([df, struct], axis=1)

    if cfeat is not None and not cfeat.empty:
        df = df.merge(cfeat, left_on="k", right_index=True, how="left")

    if vt_priors is not None:
        df = df.merge(vt_priors, on="vtype_norm", how="left")

    if st_priors is not None:
        df = df.merge(st_priors, on="clean_street", how="left")
        df["tr_street_total"] = df["tr_street_total"].fillna(0)
        df["tr_street_max"] = df["tr_street_max"].fillna(0)
        df["tr_street_n_types"] = df["tr_street_n_types"].fillna(0)
    return df


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Identify numeric feature columns for regression models."""
    base_cols = [c for c in STRUCT_COLS if c in frame.columns]
    corpus_cols = sorted(c for c in frame.columns if c.startswith("f_"))
    prior_cols = sorted(c for c in frame.columns if c.startswith("tr_"))
    return base_cols + corpus_cols + prior_cols


def build_grid(df: pd.DataFrame, vtypes: list[str]) -> pd.DataFrame:
    """Dense (street x vtype) training grid filling unobserved pairs with 0."""
    streets = df["street"].unique()
    grid = pd.MultiIndex.from_product([streets, vtypes], names=["street", "vtype"]).to_frame(index=False)
    grid = grid.merge(df, on=["street", "vtype"], how="left")
    grid["count"] = grid["count"].fillna(0.0)
    return grid


class RegressionEnsemble:
    """Dual-head gradient boosting ensemble optimizing RMSE with calibrated log1p and sqrt."""

    def __init__(self):
        self.model_sqrt = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.08, max_leaf_nodes=47,
            min_samples_leaf=80, early_stopping=True, random_state=42
        )
        self.model_log = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.08, max_leaf_nodes=47,
            min_samples_leaf=80, early_stopping=True, random_state=42
        )
        self.log_scale = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_num = X.astype(float).values
        y_clip = np.clip(y, 0, None)

        # Head A: Sqrt target
        self.model_sqrt.fit(X_num, np.sqrt(y_clip))

        # Head B: Log1p target with optimal smearing calibration
        self.model_log.fit(X_num, np.log1p(y_clip))
        raw_log = np.expm1(self.model_log.predict(X_num))
        denom = np.sum(raw_log ** 2) + 1e-6
        self.log_scale = float(np.sum(y_clip * raw_log) / denom)
        self.log_scale = max(0.5, min(self.log_scale, 10.0))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_num = X.astype(float).values
        pred_sqrt = np.clip(self.model_sqrt.predict(X_num), 0, None) ** 2
        pred_log = np.clip(np.expm1(self.model_log.predict(X_num)) * self.log_scale, 0, None)
        return 0.5 * pred_sqrt + 0.5 * pred_log


# ============================================================================
# Multi-Tier Resolution Predictor
# ============================================================================
def predict(te: pd.DataFrame, tr: pd.DataFrame, te_feat: pd.DataFrame,
            model: RegressionEnsemble, cols: list[str],
            growth: float = DEFAULT_GROWTH, camera_growth: float = DEFAULT_CAMERA_GROWTH,
            is_future: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Predict violation counts using exact raw -> canonical -> corridor prefix -> ensemble."""
    n = len(te)
    pred = np.full(n, np.nan)
    source = np.empty(n, dtype=object)

    te_norm_vt = te["vtype"].map(norm_vtype)
    tr_norm_vt = tr["vtype"].map(norm_vtype)

    # 1. Exact Raw Match (verbatim street string + violation type)
    tr_raw = pd.DataFrame({
        "street": tr["street"].astype(str),
        "vtype_norm": tr_norm_vt,
        "count": tr["count"].values
    })
    te_keys = pd.DataFrame({
        "_idx": np.arange(n),
        "street": te["street"].astype(str),
        "vtype_norm": te_norm_vt
    })
    m1 = te_keys.merge(tr_raw, on=["street", "vtype_norm"], how="left")
    matched1 = ~pd.isna(m1["count"].values)
    pred[m1["_idx"].values[matched1]] = m1["count"].values[matched1]
    source[m1["_idx"].values[matched1]] = EXACT_RAW

    # 2. Canonical Street Match (normalized street string + violation type)
    tr_clean = tr.copy()
    tr_clean["clean_street"] = tr_clean["street"].map(clean_street)
    tr_clean["vtype_norm"] = tr_norm_vt
    # If a street had multiple casing entries, average their count
    agg_clean = tr_clean.groupby(["clean_street", "vtype_norm"])["count"].mean().rename("clean_count").reset_index()

    te_clean = pd.DataFrame({
        "_idx": np.arange(n),
        "clean_street": te["street"].map(clean_street),
        "vtype_norm": te_norm_vt
    })
    m2 = te_clean.merge(agg_clean, on=["clean_street", "vtype_norm"], how="left")
    matched2 = np.isnan(pred) & ~pd.isna(m2["clean_count"].values)
    pred[m2["_idx"].values[matched2]] = m2["clean_count"].values[matched2]
    source[m2["_idx"].values[matched2]] = CANONICAL

    # 3. Corridor Prefix Match (direction + corridor for cameras & intersections)
    tr_cor = tr.copy()
    tr_cor["pk"] = tr_cor["street"].map(corridor_key)
    tr_cor["vtype_norm"] = tr_norm_vt
    # Use average intensity per location along that corridor in training
    agg_cor = tr_cor.groupby(["pk", "vtype_norm"])["count"].mean().rename("cor_mean").reset_index()

    te_cor = pd.DataFrame({
        "_idx": np.arange(n),
        "pk": te["street"].map(corridor_key),
        "vtype_norm": te_norm_vt
    })
    m3 = te_cor.merge(agg_cor, on=["pk", "vtype_norm"], how="left")
    matched3 = np.isnan(pred) & ~pd.isna(m3["cor_mean"].values)
    pred[m3["_idx"].values[matched3]] = m3["cor_mean"].values[matched3]
    source[m3["_idx"].values[matched3]] = CORRIDOR

    # 4. Regression Ensemble Model (unseen streets / pairs)
    matched4 = np.isnan(pred)
    if matched4.any():
        pred[matched4] = model.predict(te_feat.iloc[matched4][cols])
        source[matched4] = MODEL

    # 5. Growth Multipliers (applied to historical persistence sources when predicting future periods)
    if is_future:
        g = np.where(te_norm_vt.isin(CAMERA_VTYPES).values, camera_growth, growth)
        is_persistence = (source == EXACT_RAW) | (source == CANONICAL) | (source == CORRIDOR)
        pred[is_persistence] *= g[is_persistence]

    return np.clip(pred, 0, None), source


# ============================================================================
# IO & Metrics
# ============================================================================
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute root mean squared error unweighted across all rows."""
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def read_keyed_csv(path: str) -> tuple[pd.DataFrame, bool]:
    """Read a violations CSV, dynamically detecting street, vtype, and count columns."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    sc = cols.get("street name") or cols.get("street_name") or df.columns[0]
    vc = cols.get("violation description") or cols.get("violation_type") or df.columns[1]
    out = pd.DataFrame({"street": df[sc].astype(str), "vtype": df[vc].astype(str)})

    tgt = cols.get("violation_count") or cols.get("violation count") or cols.get("predicted_count")
    has_target = tgt is not None and tgt in df.columns
    if has_target:
        out["count"] = pd.to_numeric(df[tgt], errors="coerce").fillna(0.0)
    return out, has_target


# ============================================================================
# Main Execution Pipeline
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="NYC Parking Violations Forecasting Pipeline")
    parser.add_argument("--train-path", type=str,
                        default=os.getenv("TRAIN_PATH", DEFAULT_TRAIN_PATH),
                        help="Path to training CSV (default: violations_per_street_2022.csv)")
    parser.add_argument("--test-path", type=str,
                        default="/Users/USER/Downloads/violations_per_street_2023.csv",
                        help="Path to evaluation/test CSV (optional)")
    parser.add_argument("--corpus-dir", type=str,
                        default=os.getenv("CORPUS_DIR", DEFAULT_CORPUS_DIR),
                        help="Path to corpus directory (default: train/all_data/corpus)")
    parser.add_argument("--out-path", type=str,
                        default=os.getenv("OUT_PATH", DEFAULT_OUT_PATH),
                        help="Output path for submission CSV (default: submission.csv)")
    parser.add_argument("--growth", type=float, default=DEFAULT_GROWTH,
                        help="YoY organic growth multiplier (default: 1.35)")
    parser.add_argument("--camera-growth", type=float, default=DEFAULT_CAMERA_GROWTH,
                        help="YoY camera-enforced growth multiplier (default: 1.05)")
    parser.add_argument("--val-split", type=float, default=0.20,
                        help="Holdout validation fraction from 2022 training data (default: 0.20)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuilding corpus features cache")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("NYC PARKING VIOLATIONS FORECASTING PIPELINE")
    print("=" * 70)
    print(f"Train path   : {args.train_path}")
    print(f"Test path    : {args.test_path if args.test_path else '(None provided; running 2022 dev validation)'}")
    print(f"Corpus dir   : {args.corpus_dir}")
    print(f"Output path  : {args.out_path}")
    print(f"Growth rates : organic={args.growth}, camera={args.camera_growth}")
    print("=" * 70)

    # 1. Load Training Data
    if not os.path.exists(args.train_path):
        sys.exit(f"[error] Train file not found: {args.train_path}")
    tr_full, tr_has_target = read_keyed_csv(args.train_path)
    if not tr_has_target:
        sys.exit(f"[error] Train file does not contain a violation_count column: {args.train_path}")

    vtypes = sorted(tr_full["vtype"].unique())
    print(f"[data] Loaded training rows={len(tr_full)}, streets={tr_full['street'].nunique()}, vtypes={len(vtypes)}")

    # 2. Load Corpus Features
    cfeat = load_corpus_features(args.corpus_dir, rebuild_cache=args.rebuild_cache)
    if cfeat is not None:
        print(f"[features] Loaded corpus features: {cfeat.shape[1]} features for {cfeat.shape[0]} normalized streets")

    # 3. Development Validation (Holdout on 2022 Data)
    print("\n--- DEVELOPMENT VALIDATION (2022 HOLDOUT) ---")
    tr_split, val_split = train_test_split(tr_full, test_size=args.val_split, random_state=42)
    vt_priors_val, st_priors_val = compute_training_priors(tr_split)

    val_grid = add_features(build_grid(tr_split, vtypes), cfeat, vtypes, vt_priors_val, st_priors_val)
    val_cols = feature_columns(val_grid)

    print(f"[train] Fitting development model on {len(val_grid)} grid instances ({len(val_cols)} features)...")
    val_model = RegressionEnsemble()
    val_model.fit(val_grid[val_cols], val_grid["count"].values)

    val_feat = add_features(val_split[["street", "vtype"]], cfeat, vtypes, vt_priors_val, st_priors_val)
    # is_future=False for intra-year 2022 validation
    val_pred, val_source = predict(val_split, tr_split, val_feat, val_model, val_cols,
                                  growth=args.growth, camera_growth=args.camera_growth, is_future=False)
    val_pred = np.round(val_pred).astype(int)

    val_rmse = rmse(val_pred, val_split["count"].values)
    print(f"\n[validation report] Rows evaluated: {len(val_split)}")
    for label in (EXACT_RAW, CANONICAL, CORRIDOR, MODEL):
        cnt = int((val_source == label).sum())
        pct = 100.0 * cnt / len(val_split)
        print(f"  {label:24s}: {cnt:>6d} ({pct:5.1f}%)")
    print(f"--> DEVELOPMENT VALIDATION RMSE: {val_rmse:.3f}\n")

    # 4. Fit Full Model on 100% Training Data
    print("--- FITTING PRODUCTION MODEL ON FULL TRAINING DATA ---")
    vt_priors_full, st_priors_full = compute_training_priors(tr_full)
    full_grid = add_features(build_grid(tr_full, vtypes), cfeat, vtypes, vt_priors_full, st_priors_full)
    full_cols = feature_columns(full_grid)

    print(f"[train] Fitting production ensemble on {len(full_grid)} instances...")
    full_model = RegressionEnsemble()
    full_model.fit(full_grid[full_cols], full_grid["count"].values)

    # 5. Predict on Test Path (or emit holdout submission if no test path provided)
    if args.test_path and os.path.exists(args.test_path):
        print(f"\n--- PREDICTING TEST FILE: {args.test_path} ---")
        te, te_has_target = read_keyed_csv(args.test_path)
        te_feat = add_features(te[["street", "vtype"]], cfeat, vtypes, vt_priors_full, st_priors_full)

        pred, source = predict(te, tr_full, te_feat, full_model, full_cols,
                               growth=args.growth, camera_growth=args.camera_growth, is_future=True)
        pred = np.round(pred).astype(int)

        print(f"[test report] Predicted {len(te)} rows:")
        for label in (EXACT_RAW, CANONICAL, CORRIDOR, MODEL):
            cnt = int((source == label).sum())
            pct = 100.0 * cnt / len(te)
            print(f"  {label:24s}: {cnt:>6d} ({pct:5.1f}%)")

        unseen_cnt = int((source == MODEL).sum())
        print(f"[unseen analysis] {unseen_cnt} key pairs ({100.0 * unseen_cnt / len(te):.1f}%) were unseen in training")
        print("                  and resolved via regression ensemble (corpus + structural features).")

        sub = pd.DataFrame({
            "street_name": te["street"].values,
            "violation_type": te["vtype"].values,
            "predicted_count": pred
        })
        sub.to_csv(args.out_path, index=False)
        print(f"[done] Wrote predictions to {args.out_path} ({len(sub)} rows)")

        if te_has_target:
            test_rmse = rmse(pred, te["count"].values)
            print(f"--> TEST EVALUATION RMSE: {test_rmse:.3f}")
    else:
        # Default behavior when test-path is not provided: emit submission.csv on validation holdout
        print("--- GENERATING DEFAULT SUBMISSION ON EVALUATION HOLDOUT ---")
        sub = pd.DataFrame({
            "street_name": val_split["street"].values,
            "violation_type": val_split["vtype"].values,
            "predicted_count": val_pred
        })
        sub.to_csv(args.out_path, index=False)
        print(f"[done] Wrote validation holdout predictions to {args.out_path} ({len(sub)} rows)")
        print(f"[info] Provide --test-path <path_to_2023_file> to generate predictions on an external test set.")


if __name__ == "__main__":
    main()
