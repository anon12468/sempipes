from __future__ import annotations

import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "NYC_DATA_DIR",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", "projects", "nyc_data", "nyc_data")),
)

# ---- paths ----
TRAIN_PATH = os.path.join(DATA_DIR, "train", "all_data", "violations_per_street_2022.csv")
TEST_PATH = os.path.join(DATA_DIR, "violations_per_street_2023.csv")
CORPUS_DIR = os.path.join(DATA_DIR, "train", "all_data", "corpus")
OUT_PATH = os.path.join(HERE, "submission.csv")

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
EXACT, PREFIX, MODEL = "exact match", "prefix match", "regression model"


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
    scikit-learn), handles missing corpus values natively, and -- as
    benchmarked against CatBoost/LightGBM/XGBoost and their ensemble -- gives the
    best end-to-end score here; the alternatives differ by well under 0.1 RMSE.
    """
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    model.fit(frame[cols].astype(float), np.log1p(frame["count"].values))
    return model


def model_predict(model, frame, cols):
    """Predict counts (invert the log1p transform and floor at zero)."""
    return np.clip(np.expm1(model.predict(frame[cols].astype(float))), 0, None)


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
    out = pd.DataFrame({"street": df[sc].astype(str), "vtype": df[vc].astype(str)})
    tgt = cols.get("violation_count") or cols.get("violation count") or cols.get("predicted_count")
    has_target = tgt is not None and tgt in df.columns
    if has_target:
        out["count"] = pd.to_numeric(df[tgt], errors="coerce").fillna(0.0)
    return out, has_target


def load_corpus_features():
    """Load corpus features from cache, building from the corpus dir if needed."""
    if not os.path.isdir(CORPUS_DIR):
        print(f"[warn] corpus dir not found ({CORPUS_DIR}); proceeding without corpus features.")
        return None
    cache = os.path.join(HERE, "corpus_features.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    t = time.time()
    print(f"[info] building corpus features from {CORPUS_DIR} ...")
    f = build_corpus_features(CORPUS_DIR)
    f.to_parquet(cache)
    print(f"[info] corpus features: {f.shape} in {time.time() - t:.0f}s (cached)")
    return f


# ============================================================================
# Prediction (exact match -> prefix match -> regression model)
# ============================================================================
def predict(te, tr, te_feat, model, cols):
    """Predict each (street, vtype) and record which source produced it.

    Each key is resolved by the first applicable strategy and falls through to
    the next only when it does not apply: exact match, then prefix match, then
    the regression model. Persistence predictions (exact/prefix) are scaled by
    GROWTH, except camera-enforced types which use CAMERA_GROWTH. Returns
    (pred, source) as parallel arrays aligned with `te`.
    """
    pred = np.full(len(te), np.nan)
    source = np.empty(len(te), dtype=object)

    # 1. Exact match: carry the 2022 count forward.
    merged = te.merge(tr.rename(columns={"count": "_prior"}), on=["street", "vtype"], how="left")
    matched = ~pd.isna(merged["_prior"].values)
    pred[matched] = np.nan_to_num(merged["_prior"].values)[matched]
    source[matched] = EXACT

    # 2. Prefix match: intersection/camera rows share a stable on-street+direction
    #    prefix. Split the 2022 prefix total evenly across the eval siblings.
    tr_pk = tr.assign(pk=tr["street"].map(prefix_key))
    agg = tr_pk.groupby(["pk", "vtype"])["count"].sum().rename("pk_count").reset_index()
    te_pk = te[["street", "vtype"]].assign(pk=te["street"].map(prefix_key))
    te_pk = te_pk.merge(agg, on=["pk", "vtype"], how="left")
    siblings = te_pk.groupby(["pk", "vtype"])["street"].transform("size")
    pk_share = (te_pk["pk_count"] / siblings).values
    use_prefix = np.isnan(pred) & ~np.isnan(pk_share)
    pred[use_prefix] = pk_share[use_prefix]
    source[use_prefix] = PREFIX

    # 3. Regression model: everything still unresolved (no 2022 history).
    use_model = np.isnan(pred)
    pred[use_model] = model_predict(model, te_feat, cols)[use_model]
    source[use_model] = MODEL

    # Persistence sources get the YoY growth bump (lower for cameras); the
    # regression model is already calibrated on the data, so it is left as-is.
    g = np.where(te["vtype"].isin(CAMERA_VTYPES).values, CAMERA_GROWTH, GROWTH)
    is_persistence = (source == EXACT) | (source == PREFIX)
    pred[is_persistence] *= g[is_persistence]
    return np.clip(pred, 0, None), source


# ============================================================================
# Run
# ============================================================================
def main():
    # ---- load 2022 training data + build features ----
    tr, _ = read_keyed_csv(TRAIN_PATH)
    vtypes = sorted(tr["vtype"].unique())
    print(f"[data] train rows={len(tr)} streets={tr['street'].nunique()} vtypes={len(vtypes)}")

    cfeat = load_corpus_features()
    grid = add_features(build_grid(tr, vtypes), cfeat, vtypes)
    cols = feature_columns(grid)
    model = fit_model(grid, cols)

    # ---- predict the 2023 key space ----
    te, has_target = read_keyed_csv(TEST_PATH)
    te_feat = add_features(te[["street", "vtype"]], cfeat, vtypes)
    pred, source = predict(te, tr, te_feat, model, cols)
    pred = np.round(pred).astype(int)

    for label in (EXACT, PREFIX, MODEL):
        n = int((source == label).sum())
        print(f"  {label:18s}: {n:>7d} ({100 * n / len(te):5.1f}%)")

    sub = pd.DataFrame({"street_name": te["street"].values,
                        "violation_type": te["vtype"].values,
                        "predicted_count": pred})
    sub.to_csv(OUT_PATH, index=False)
    print(f"[done] wrote {OUT_PATH} ({len(sub)} rows)")

    if has_target:
        y = te["count"].values
        print(f"[eval] RMSE={rmse(pred, y):.3f}")


if __name__ == "__main__":
    main()
