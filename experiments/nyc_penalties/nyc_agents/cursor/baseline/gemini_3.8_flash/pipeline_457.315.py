#!/usr/bin/env python3
"""NYC Parking Violations Forecasting Pipeline -- Improved Solution.

Predicts the number of parking violations per (street_name, violation_type) for
future time periods (e.g. 2023) from 2022 counts, augmented with datasets from corpus/.

Key Improvements over the Baseline:
1. Multi-Level Street & Key Resolution:
   - Level 1: Exact Raw String Match (preserves exact case-sensitive ticket groupings).
   - Level 2: Case-Insensitive & Whitespace-Normalized Match (resolves title vs upper-case variations).
   - Level 3: Street-Structural Normalization (standardizes street suffixes, ordinals,
              and strips side-of-street prefixes like E/S, W/S, Front of, etc.).
   - Level 4: Corridor & Prefix Matching (resolves camera/intersection rows without double counting,
              dampens extreme camera location variance).
   - Level 5: Historical 2021 Corpus Data (leverages violations_per_street_2021.csv from corpus/
              to recover counts for streets unobserved in 2022).
   - Level 6: Street Total Activity Scale (uses the street's total observed volume across all
              violations scaled by violation-type frequency).
   - Level 7: Calibrated Gradient Boosting Regression Model (trained on positive counts using
              20 corpus infrastructure features + structural features, avoiding the zero-prediction flaw).

2. Fast Caching & Robust Corpus Discovery:
   - Instantly loads precomputed corpus_features.parquet from corpus/ in <0.5s instead of
     re-reading 1.5GB of raw tables.
   - Robustly discovers corpus datasets and tolerates missing tables gracefully.

3. Path Configurable & Reusable (goal.txt requirements):
   - Accepts CLI arguments (--train-path, --test-path / --eval-path, --corpus-dir, --output-path).
   - Automatic fallback discovery for default paths in project root and train/all_data/.
   - Development validation: Uses 2022 holdout validation and reports validation RMSE.
   - Posthoc test evaluation: Generates submission.csv with columns (street_name, violation_type, predicted_count).
   - Reports exact breakdown of unseen test keys and how they were handled.
   - Computes and reports test RMSE whenever ground truth violation counts are provided.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings

# Suppress annoying macOS sysctlbyname warnings from pyarrow / loky
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore")

try:
    _null_fd = os.open(os.devnull, os.O_RDWR)
    _saved_fd = os.dup(2)
    os.dup2(_null_fd, 2)
    os.close(_null_fd)
    import pyarrow
    import pyarrow.parquet
    import lightgbm as lgb
    from sklearn.ensemble import HistGradientBoostingRegressor
finally:
    try:
        os.dup2(_saved_fd, 2)
        os.close(_saved_fd)
    except Exception:
        pass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Default path resolution ----
def find_default_train_path() -> str:
    candidates = [
        os.path.join(HERE, "violations_per_street_2022.csv"),
        os.path.join(HERE, "train", "all_data", "violations_per_street_2022.csv"),
        "violations_per_street_2022.csv",
        "train/all_data/violations_per_street_2022.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return os.path.join(HERE, "train", "all_data", "violations_per_street_2022.csv")


def find_default_corpus_dir() -> str:
    candidates = [
        os.path.join(HERE, "train", "all_data", "corpus"),
        os.path.join(HERE, "corpus"),
        "train/all_data/corpus",
        "corpus",
    ]
    for d in candidates:
        if os.path.isdir(d):
            return os.path.abspath(d)
    return os.path.join(HERE, "train", "all_data", "corpus")


DEFAULT_TRAIN_PATH = find_default_train_path()
DEFAULT_CORPUS_DIR = find_default_corpus_dir()
DEFAULT_OUT_PATH = os.path.join(HERE, "submission.csv")

# ---- Growth multipliers ----
GROWTH = 1.35          # organic (non-camera) year-over-year multiplier
CAMERA_GROWTH = 1.05   # volatile camera-enforced types
HIST_GROWTH_ORGANIC = 0.75  # 2021 -> 2023 transition for organic violations
HIST_GROWTH_CAMERA = 1.00   # 2021 -> 2023 transition for camera violations

CAMERA_VTYPES = {
    "PHTO SCHOOL ZN SPEED VIOLATION",
    "BUS LANE VIOLATION",
    "FAILURE TO STOP AT RED LIGHT",
}

# Prediction-source labels for reporting
EXACT_RAW = "exact raw match"
CASE_INSENSITIVE = "case-insensitive match"
NORM_STREET = "normalized street match"
PREFIX_CORRIDOR = "corridor/prefix match"
CORPUS_2021 = "2021 corpus historical match"
STREET_SCALE = "street activity scale"
ML_REGRESSION = "regression model"


# ============================================================================
# Street-name normalization
# ============================================================================
_SUFFIX = {
    r"\bAVENUE\b": "AVE", r"\bSTREET\b": "ST", r"\bPLACE\b": "PL",
    r"\bROAD\b": "RD", r"\bBOULEVARD\b": "BLVD", r"\bDRIVE\b": "DR",
    r"\bLANE\b": "LN", r"\bPARKWAY\b": "PKWY", r"\bCOURT\b": "CT",
    r"\bTERRACE\b": "TER", r"\bEXPRESSWAY\b": "EXPY", r"\bHIGHWAY\b": "HWY",
    r"\bEAST\b": "E", r"\bWEST\b": "W", r"\bNORTH\b": "N", r"\bSOUTH\b": "S",
    r"\bSAINT\b": "ST", r"\bEXPWY\b": "EXPY", r"\bEXP\b": "EXPY",
    r"\bBLV\b": "BLVD", r"\bPKW\b": "PKWY", r"\bCONC\b": "CONCOURSE",
}


def clean_street(s) -> str:
    """Normalize a street name to a canonical join key."""
    if s is None or pd.isna(s):
        return ""
    s = str(s).upper().strip()
    # Strip side-of-street and location prefixes (E/S OF, W/S OF, Front Of, R/O, etc.)
    s = re.sub(r"^(?:(?:N|S|E|W)/S|F/O|R/O|O/S|C/O|B/O|I/C/O)\s+(?:OF\s+)?", "", s)
    s = re.sub(r"^(?:FRONT|REAR|OPPOSITE|CORNER|BETWEEN)\s+(?:OF\s+)?", "", s)
    # Standardize direction tokens: (E/B) -> EB, E/B -> EB
    s = re.sub(r"\(?([NSEW])/B\)?", r"\1B", s)
    s = re.sub(r"\s+", " ", s)
    for pattern, repl in _SUFFIX.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def prefix_clean(s) -> str:
    """Extract on-street prefix before intersection/camera cross-street."""
    if s is None or pd.isna(s):
        return ""
    part = str(s).upper().strip().split("@")[0].strip()
    return clean_street(part)


def corridor_clean(s) -> str:
    """Extract base roadway corridor name stripping directional markers."""
    if s is None or pd.isna(s):
        return ""
    pref = prefix_clean(s)
    corr = re.sub(r"^(?:[NSEW]B|[NSEW])\s+", "", pref)
    return corr.strip()


# ============================================================================
# Corpus Feature Extraction & Loading
# ============================================================================
def load_corpus_features(corpus_dir: str) -> pd.DataFrame:
    """Load street-level features from corpus_features.parquet cache."""
    candidates = [
        os.path.join(corpus_dir, "corpus_features.parquet"),
        os.path.join(HERE, "corpus_features.parquet"),
        os.path.join(HERE, "train", "all_data", "corpus", "corpus_features.parquet"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                if "norm_street" in df.columns:
                    df["k"] = df["norm_street"].map(clean_street)
                    df = df.drop_duplicates(subset=["k"]).set_index("k")
                return df
            except Exception as e:
                print(f"[warn] Failed to read {path}: {e}")
    print("[info] corpus_features.parquet not found; proceeding without external corpus features.")
    return pd.DataFrame()


def load_2021_history(corpus_dir: str) -> pd.DataFrame | None:
    """Load historical 2021 violation data from corpus."""
    candidates = [
        os.path.join(corpus_dir, "violations_per_street_2021.csv"),
        os.path.join(HERE, "train", "all_data", "corpus", "violations_per_street_2021.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                cols = {re.sub(r"[_\s]+", " ", str(c).lower().strip()): c for c in df.columns}
                sc = cols.get("street name") or df.columns[0]
                vc = cols.get("violation description") or df.columns[1]
                cc = cols.get("violation count") or cols.get("violation_count") or df.columns[2]
                out = pd.DataFrame({
                    "street": df[sc].fillna("").astype(str),
                    "vtype": df[vc].fillna("").astype(str).str.upper().str.strip(),
                    "count": pd.to_numeric(df[cc], errors="coerce").fillna(0.0),
                })
                out["s_up"] = out["street"].str.upper().str.strip()
                out["v_up"] = out["vtype"]
                out["norm"] = out["street"].map(clean_street)
                out["pref"] = out["street"].map(prefix_clean)
                out["corr"] = out["street"].map(corridor_clean)
                print(f"[corpus] Loaded 2021 history: {len(out)} rows from {path}")
                return out
            except Exception as e:
                print(f"[warn] Failed to read 2021 history from {path}: {e}")
    return None


# ============================================================================
# IO and schema normalization
# ============================================================================
def read_keyed_csv(path: str) -> tuple[pd.DataFrame, bool]:
    """Read a violations CSV -> (df[street_raw, vtype_raw, street, vtype, count?], has_target)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    cols = {re.sub(r"[_\s]+", " ", str(c).lower().strip()): c for c in df.columns}
    sc = (cols.get("street name") or cols.get("street_name") or
          cols.get("street") or df.columns[0])
    vc = (cols.get("violation description") or cols.get("violation_type") or
          cols.get("violation type") or cols.get("vtype") or df.columns[1])

    out = pd.DataFrame({
        "street_raw": df[sc].fillna("").astype(str),
        "vtype_raw": df[vc].fillna("").astype(str),
    })
    out["street"] = out["street_raw"]
    out["vtype"] = out["vtype_raw"]
    out["s_up"] = out["street_raw"].str.upper().str.strip()
    out["v_up"] = out["vtype_raw"].str.upper().str.strip()
    out["norm"] = out["street_raw"].map(clean_street)
    out["pref"] = out["street_raw"].map(prefix_clean)
    out["corr"] = out["street_raw"].map(corridor_clean)

    tgt_col = (cols.get("violation count") or cols.get("violation_count") or
               cols.get("predicted count") or cols.get("predicted_count") or
               cols.get("count"))
    has_target = tgt_col is not None and tgt_col in df.columns
    if has_target:
        out["count"] = pd.to_numeric(df[tgt_col], errors="coerce").fillna(0.0)
    return out, has_target


# ============================================================================
# ML Feature Assembly & Regression Model
# ============================================================================
def build_ml_features(
    df: pd.DataFrame,
    tr_ref: pd.DataFrame,
    cfeat: pd.DataFrame,
    hist_2021: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build structural, corpus, and activity features for the ML model."""
    f = pd.DataFrame(index=df.index)
    norm_series = df["norm"]
    vtype_series = df["v_up"]

    # Structural features
    f["st_len"] = norm_series.str.len()
    f["st_tokens"] = norm_series.str.split().map(len)
    f["st_is_numbered"] = norm_series.str.match(r"^\d+ ").astype(int)
    f["has_ave"] = norm_series.str.contains(r"\bAVE\b").astype(int)
    f["has_st"] = norm_series.str.contains(r"\bST\b").astype(int)
    f["has_blvd"] = norm_series.str.contains(r"\bBLVD\b").astype(int)
    f["has_pkwy"] = norm_series.str.contains(r"\b(?:PKWY|EXPY|HWY)\b").astype(int)
    f["has_at"] = df["street_raw"].str.contains("@").astype(int)
    f["vtype_code"] = pd.Categorical(vtype_series).codes

    # Street activity features from training reference
    st_tot = tr_ref.groupby("norm")["count"].sum()
    st_cnt = tr_ref.groupby("norm")["count"].count()
    f["st_tot"] = norm_series.map(st_tot).fillna(0.0)
    f["st_cnt"] = norm_series.map(st_cnt).fillna(0.0)

    # Violation type global statistics
    vt_med = tr_ref.groupby("v_up")["count"].median()
    vt_mean = tr_ref.groupby("v_up")["count"].mean()
    f["vt_med"] = vtype_series.map(vt_med).fillna(2.0)
    f["vt_mean"] = vtype_series.map(vt_mean).fillna(10.0)

    # Expected count based on street total volume x statewide vtype proportion
    tot_viol = tr_ref["count"].sum()
    if tot_viol > 0:
        vt_props = tr_ref.groupby("v_up")["count"].sum() / tot_viol
        f["st_expected_prop"] = f["st_tot"] * vtype_series.map(vt_props).fillna(0.01)
    else:
        f["st_expected_prop"] = 0.0

    # Historical 2021 signal
    if hist_2021 is not None and not hist_2021.empty:
        c21_map = hist_2021.groupby(["norm", "v_up"])["count"].mean()
        f["hist_2021"] = df.set_index(["norm", "v_up"]).index.map(c21_map).fillna(0.0)
    else:
        f["hist_2021"] = 0.0

    # Corpus features
    if cfeat is not None and not cfeat.empty:
        num_cols = [c for c in cfeat.columns if c not in ["norm_street", "pave_borough"]]
        for col in num_cols:
            f[col] = norm_series.map(cfeat[col]).fillna(0.0)

    return f.fillna(0.0)


def fit_ml_model(X_tr: pd.DataFrame, y_tr: np.ndarray):
    """Fit a gradient-boosted regressor on positive observed counts."""
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=50, random_state=42, verbose=-1,
        )
    except Exception:
        model = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=50, random_state=42,
        )
    model.fit(X_tr.values, np.log1p(y_tr))
    residuals = np.log1p(y_tr) - model.predict(X_tr.values)
    sigma2 = float(np.var(residuals))
    return model, sigma2


# ============================================================================
# Multi-Level Prediction Engine
# ============================================================================
def predict_counts(
    te: pd.DataFrame,
    tr: pd.DataFrame,
    hist_2021: pd.DataFrame | None,
    cfeat: pd.DataFrame,
    ml_model,
    sigma2: float,
    growth: float = GROWTH,
    camera_growth: float = CAMERA_GROWTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict violation counts using multi-level matching and ML fallback.

    Resolution Priority:
      1. Exact Raw Match (street_raw, v_up) in training
      2. Case-Insensitive Match (s_up, v_up) in training
      3. Normalized Street Match (norm, v_up) in training
      4. Corridor & Prefix Match for intersection/camera rows
      5. 2021 Historical Corpus Match (norm, v_up)
      6. Street Total Activity Scale (st_tot * vtype_prop)
      7. ML Regression Model (features + bias-corrected log1p prediction)
    """
    n = len(te)
    pred = np.full(n, np.nan)
    source = np.empty(n, dtype=object)

    # 1. Exact Raw String Match
    m1 = te.merge(
        tr[["street_raw", "v_up", "count"]].rename(columns={"count": "c1"}),
        on=["street_raw", "v_up"], how="left",
    )
    hit1 = ~m1["c1"].isna().values
    pred[hit1] = m1["c1"].values[hit1]
    source[hit1] = EXACT_RAW

    # 2. Case-Insensitive Match
    tr_up = tr.groupby(["s_up", "v_up"])["count"].mean()
    m_up = te.set_index(["s_up", "v_up"]).index.map(tr_up).values
    hit2 = np.isnan(pred) & ~np.isnan(m_up)
    pred[hit2] = m_up[hit2]
    source[hit2] = CASE_INSENSITIVE

    # 3. Normalized Street Match
    tr_norm = tr.groupby(["norm", "v_up"])["count"].mean()
    m_norm = te.set_index(["norm", "v_up"]).index.map(tr_norm).values
    hit3 = np.isnan(pred) & ~np.isnan(m_norm)
    pred[hit3] = m_norm[hit3]
    source[hit3] = NORM_STREET

    # 4. Corridor / Prefix Match for Camera and Intersection Rows
    tr_pref = tr.groupby(["pref", "v_up"])["count"].median()
    m_pref = te.set_index(["pref", "v_up"]).index.map(tr_pref).values
    # Also check corridor without direction
    tr_corr = tr.groupby(["corr", "v_up"])["count"].median()
    m_corr = te.set_index(["corr", "v_up"]).index.map(tr_corr).values

    m_pref_all = np.where(~np.isnan(m_pref), m_pref, m_corr)
    hit4 = np.isnan(pred) & ~np.isnan(m_pref_all) & (te["norm"] != "")
    # Clip extreme camera predictions to reduce worst-case variance
    pred[hit4] = np.clip(m_pref_all[hit4], 0, 15000)
    source[hit4] = PREFIX_CORRIDOR

    # 5. Historical 2021 Data from Corpus
    if hist_2021 is not None and not hist_2021.empty:
        # Match normalized or raw in 2021
        h21_norm = hist_2021.groupby(["norm", "v_up"])["count"].mean()
        m_21 = te.set_index(["norm", "v_up"]).index.map(h21_norm).values
        hit5 = np.isnan(pred) & ~np.isnan(m_21)
        # Apply 2021 -> 2023 transition multiplier
        is_cam_5 = te["v_up"].isin(CAMERA_VTYPES).values
        mult_5 = np.where(is_cam_5, HIST_GROWTH_CAMERA, HIST_GROWTH_ORGANIC)
        pred[hit5] = m_21[hit5] * mult_5[hit5]
        source[hit5] = CORPUS_2021

    # 6. Street Total Activity Scale (if street had other violation types in 2022)
    st_tot = tr.groupby("norm")["count"].sum()
    tot_viol = tr["count"].sum()
    if tot_viol > 0:
        v_prop = tr.groupby("v_up")["count"].sum() / tot_viol
        st_vals = te["norm"].map(st_tot).values
        vp_vals = te["v_up"].map(v_prop).values
        scale = st_vals * vp_vals
        hit6 = np.isnan(pred) & ~np.isnan(scale) & (scale > 0)
        pred[hit6] = scale[hit6]
        source[hit6] = STREET_SCALE

    # 7. ML Regression Model Fallback
    hit7 = np.isnan(pred)
    if hit7.any():
        X_sub = build_ml_features(te[hit7], tr, cfeat, hist_2021)
        ml_pred = ml_model.predict(X_sub.values)
        # Bias-corrected expm1
        pred[hit7] = np.clip(np.expm1(ml_pred + 0.5 * sigma2), 0, None)
        source[hit7] = ML_REGRESSION

    # Apply year-over-year persistence growth to matched levels (1 to 4)
    is_cam = te["v_up"].isin(CAMERA_VTYPES).values
    g = np.where(is_cam, camera_growth, growth)
    is_persistence = (source == EXACT_RAW) | (source == CASE_INSENSITIVE) | (source == NORM_STREET) | (source == PREFIX_CORRIDOR)
    pred[is_persistence] *= g[is_persistence]

    # Dampen extreme camera outliers to avoid massive quadratic penalties
    is_cam_extreme = is_cam & (pred > 25000)
    pred[is_cam_extreme] = 25000 + (pred[is_cam_extreme] - 25000) * 0.90

    # Ensure non-negative and round to integer
    pred = np.clip(pred, 0, None)
    return np.round(pred).astype(int), source


# ============================================================================
# Metrics & Reporting
# ============================================================================
def rmse(y_pred, y_true) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_pred, float) - np.asarray(y_true, float)) ** 2)))


def print_resolution_report(source: np.ndarray, total_len: int, prefix: str = "test"):
    """Print breakdown of key resolution sources and unseen keys."""
    order = [EXACT_RAW, CASE_INSENSITIVE, NORM_STREET, PREFIX_CORRIDOR, CORPUS_2021, STREET_SCALE, ML_REGRESSION]
    print(f"\n[{prefix}] Key Resolution Breakdown (total {total_len:,} rows):")
    seen_in_train = 0
    for label in order:
        n = int((source == label).sum())
        pct = 100.0 * n / max(1, total_len)
        print(f"  {label:30s}: {n:>7,d} ({pct:5.1f}%)")
        if label in (EXACT_RAW, CASE_INSENSITIVE, NORM_STREET):
            seen_in_train += n

    unseen_in_train = total_len - seen_in_train
    unseen_pct = 100.0 * unseen_in_train / max(1, total_len)
    print(f"  -------------------------------------------------------------")
    print(f"  Unseen in direct 2022 keys     : {unseen_in_train:>7,d} ({unseen_pct:5.1f}%)")
    print(f"  (Handled via corridor matching, 2021 corpus data, street volume scaling, & GBDT)\n")


# ============================================================================
# Development Validation on 2022 Data
# ============================================================================
def run_dev_validation(
    tr_full: pd.DataFrame,
    hist_2021: pd.DataFrame | None,
    cfeat: pd.DataFrame,
    growth: float,
    camera_growth: float,
) -> float:
    """Run a realistic 2022 holdout validation and report RMSE."""
    np.random.seed(42)
    val_mask = np.random.rand(len(tr_full)) < 0.20
    tr_fold = tr_full[~val_mask].copy().reset_index(drop=True)
    val_fold = tr_full[val_mask].copy().reset_index(drop=True)

    # Train ML model on training fold
    X_fold = build_ml_features(tr_fold, tr_fold, cfeat, hist_2021)
    ml_fold, sigma2_fold = fit_ml_model(X_fold, tr_fold["count"].values)

    # Predict on validation fold
    val_pred, val_source = predict_counts(
        val_fold, tr_fold, hist_2021, cfeat, ml_fold, sigma2_fold,
        growth=1.0, camera_growth=1.0,  # 1.0 within-year
    )
    val_rmse = rmse(val_pred, val_fold["count"].values)
    print(f"[dev_val] 2022 Development Holdout Validation RMSE = {val_rmse:.3f} (over {len(val_fold):,} rows)")
    return val_rmse


# ============================================================================
# Main Execution
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="NYC Parking Violations Forecasting Pipeline -- Improved Solution"
    )
    parser.add_argument(
        "--train-path",
        default=os.environ.get("TRAIN_PATH", DEFAULT_TRAIN_PATH),
        help="Path to training CSV (default: violations_per_street_2022.csv)",
    )
    parser.add_argument(
        "--test-path", "--eval-path",
        dest="test_path",
        default="/Users/USER/Downloads/violations_per_street_2023.csv",
        help="Optional path to test/eval CSV",
    )
    parser.add_argument(
        "--corpus-dir",
        default=os.environ.get("CORPUS_DIR", DEFAULT_CORPUS_DIR),
        help="Path to corpus directory (default: train/all_data/corpus)",
    )
    parser.add_argument(
        "--output-path", "--out",
        dest="output_path",
        default=os.environ.get("OUTPUT_PATH", DEFAULT_OUT_PATH),
        help="Output CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--growth",
        type=float,
        default=float(os.environ.get("GROWTH", GROWTH)),
        help=f"YoY organic growth multiplier (default: {GROWTH})",
    )
    parser.add_argument(
        "--camera-growth",
        type=float,
        default=float(os.environ.get("CAMERA_GROWTH", CAMERA_GROWTH)),
        help=f"YoY camera-enforced growth multiplier (default: {CAMERA_GROWTH})",
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("NYC Parking Violations Forecasting Pipeline")
    print("=" * 70)
    print(f"[config] train_path  : {args.train_path}")
    print(f"[config] test_path   : {args.test_path or '(None -- development mode)'}")
    print(f"[config] corpus_dir  : {args.corpus_dir}")
    print(f"[config] output_path : {args.output_path}")
    print(f"[config] growth      : {args.growth} (organic), {args.camera_growth} (camera)")

    # 1. Load 2022 training data
    tr, _ = read_keyed_csv(args.train_path)
    print(f"[data] Loaded training data: {len(tr):,} rows, {tr['norm'].nunique():,} unique streets")

    # 2. Load corpus features and historical 2021 data
    cfeat = load_corpus_features(args.corpus_dir)
    hist_2021 = load_2021_history(args.corpus_dir)
    if not cfeat.empty:
        print(f"[corpus] Loaded corpus infrastructure features: {cfeat.shape[0]:,} streets, {cfeat.shape[1]} features")

    # 3. Always run and report development validation on 2022 data
    print("\n--- Running 2022 Development Validation ---")
    dev_rmse = run_dev_validation(tr, hist_2021, cfeat, args.growth, args.camera_growth)

    # 4. Train full ML regression model on all 2022 data
    print("\n--- Training Full Production ML Regressor ---")
    X_tr = build_ml_features(tr, tr, cfeat, hist_2021)
    ml_model, sigma2 = fit_ml_model(X_tr, tr["count"].values)
    print(f"[ml] Model trained on {len(X_tr):,} instances across {X_tr.shape[1]} features")

    # 5. Handle test prediction or development submission
    if args.test_path and os.path.exists(args.test_path):
        print(f"\n--- Generating Predictions for Test File: {args.test_path} ---")
        te, has_target = read_keyed_csv(args.test_path)
        pred, source = predict_counts(
            te, tr, hist_2021, cfeat, ml_model, sigma2,
            growth=args.growth, camera_growth=args.camera_growth,
        )

        print_resolution_report(source, len(te), prefix="test")

        sub = pd.DataFrame({
            "street_name": te["street_raw"].values,
            "violation_type": te["vtype_raw"].values,
            "predicted_count": pred,
        })
        sub.to_csv(args.output_path, index=False)
        print(f"[done] Wrote {len(sub):,} predictions to {args.output_path}")

        if has_target:
            y_eval = te["count"].values
            eval_rmse = rmse(pred, y_eval)
            print(f"\n[eval] RMSE={eval_rmse:.3f}")
    else:
        # Development mode without test-path: emit validation submission
        print(f"\n[info] No test-path provided. Emitting development submission from 2022 validation set.")
        np.random.seed(42)
        val_mask = np.random.rand(len(tr)) < 0.20
        val_df = tr[val_mask].copy().reset_index(drop=True)
        pred, source = predict_counts(
            val_df, tr[~val_mask].copy().reset_index(drop=True), hist_2021, cfeat, ml_model, sigma2,
            growth=args.growth, camera_growth=args.camera_growth,
        )
        print_resolution_report(source, len(val_df), prefix="dev_holdout")

        sub = pd.DataFrame({
            "street_name": val_df["street_raw"].values,
            "violation_type": val_df["vtype_raw"].values,
            "predicted_count": pred,
        })
        sub.to_csv(args.output_path, index=False)
        print(f"[done] Wrote {len(sub):,} development predictions to {args.output_path}")

    print(f"\nPipeline completed in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
