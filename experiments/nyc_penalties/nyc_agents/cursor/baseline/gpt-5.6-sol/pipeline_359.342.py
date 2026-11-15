#!/usr/bin/env python3
"""NYC parking-violation forecasting pipeline.

Predict counts for a supplied test key space from prior-year counts. Test targets,
when present, are used only after prediction to report RMSE. Predictions use, in
order, exact, normalized-street, intersection-prefix, and regression fallbacks.

Without a test file the script reports a deterministic grouped-holdout score and
trains the final model without writing a submission.
"""
from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_PATH = os.path.join(
    HERE, "train", "all_data", "violations_per_street_2022.csv"
)
DEFAULT_CORPUS_DIR = os.path.join(HERE, "train", "all_data", "corpus")
DEFAULT_OUT_PATH = os.path.join(HERE, "submission.csv")

# ---- baseline year-over-year priors (never fitted on test/eval targets) ----
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
MODEL = "regression model"


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


def fit_model(frame, cols):
    """Fit the regression model on log1p counts (the target is heavily right-skewed).

    The model is trained on observed positive pairs, matching the distribution of
    a test key-space file. HistGradientBoosting handles missing corpus values.
    """
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=100, l2_regularization=10.0, random_state=0)
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
    original_vtype = df[vc].astype(str)
    out = pd.DataFrame({
        "street": df[sc].astype(str),
        "vtype": original_vtype.str.upper().str.strip(),
        "vtype_original": original_vtype,
    })
    tgt = cols.get("violation_count") or cols.get("violation count")
    has_target = tgt is not None and tgt in df.columns
    if has_target:
        out["count"] = pd.to_numeric(df[tgt], errors="coerce").fillna(0.0)
    return out, has_target


def _prepare_cached_features(frame):
    """Convert either supported cache schema to numeric, normalized features."""
    frame = frame.copy()
    if "norm_street" in frame.columns:
        frame["k"] = frame.pop("norm_street").map(norm_street)
        frame = frame.set_index("k")
    elif frame.index.name != "k":
        frame.index = pd.Index(frame.index.map(norm_street), name="k")
    frame = frame.select_dtypes(include=[np.number])
    frame = frame.groupby(level=0).sum(min_count=1)
    return frame.rename(columns=lambda c: c if str(c).startswith("f_") else f"f_{c}")


def load_corpus_features(corpus_dir):
    """Load corpus features from cache, building from the corpus dir if needed."""
    if not corpus_dir or not os.path.isdir(corpus_dir):
        print(f"[warn] corpus dir not found ({corpus_dir}); proceeding without corpus features.")
        return None
    caches = [
        os.path.join(corpus_dir, "corpus_features.parquet"),
        os.path.join(HERE, "corpus_features.parquet"),
    ]
    for cache in caches:
        if os.path.exists(cache):
            features = _prepare_cached_features(pd.read_parquet(cache))
            print(f"[data] corpus features={features.shape} cache={cache}")
            return features
    t = time.time()
    print(f"[info] building corpus features from {corpus_dir} ...")
    f = build_corpus_features(corpus_dir)
    print(f"[info] corpus features: {f.shape} in {time.time() - t:.0f}s")
    return f


# ============================================================================
# Prediction (exact -> normalized -> prefix -> regression model)
# ============================================================================
def predict(te, tr, te_feat, model, cols, apply_growth=True):
    """Predict each (street, vtype) and record which source produced it.

    Each key is resolved by the first applicable strategy and falls through to
    the next only when it does not apply. Persistence predictions are scaled by
    type-specific growth during future forecasting, but not during same-year
    grouped-holdout validation.
    """
    pred = np.full(len(te), np.nan)
    source = np.full(len(te), None, dtype=object)

    # 1. Exact match: carry the prior count forward. Group first so malformed
    # duplicate keys cannot expand the test rows during the merge.
    exact = tr.groupby(["street", "vtype"], as_index=False)["count"].sum()
    exact = exact.rename(columns={"count": "_prior"})
    merged = te[["street", "vtype"]].merge(
        exact, on=["street", "vtype"], how="left", validate="many_to_one"
    )
    matched = ~pd.isna(merged["_prior"].values)
    pred[matched] = np.nan_to_num(merged["_prior"].values)[matched]
    source[matched] = EXACT

    # 2. Normalized match: reconcile suffixes, ordinals, punctuation, and case.
    tr_norm = tr.assign(k=tr["street"].map(norm_street))
    norm_agg = tr_norm.groupby(["k", "vtype"])["count"].sum().rename("norm_count").reset_index()
    te_norm = te[["street", "vtype"]].assign(k=te["street"].map(norm_street))
    te_norm = te_norm.merge(norm_agg, on=["k", "vtype"], how="left")
    norm_siblings = te_norm.groupby(["k", "vtype"])["street"].transform("size")
    norm_share = (te_norm["norm_count"] / norm_siblings).values
    use_norm = np.isnan(pred) & ~np.isnan(norm_share)
    pred[use_norm] = norm_share[use_norm]
    source[use_norm] = NORMALIZED

    # 3. Prefix match: camera rows share a stable on-street+direction prefix.
    tr_pk = tr.assign(pk=tr["street"].map(prefix_key))
    agg = tr_pk.groupby(["pk", "vtype"])["count"].sum().rename("pk_count").reset_index()
    te_pk = te[["street", "vtype"]].assign(pk=te["street"].map(prefix_key))
    te_pk = te_pk.merge(agg, on=["pk", "vtype"], how="left")
    siblings = te_pk.groupby(["pk", "vtype"])["street"].transform("size")
    pk_share = (te_pk["pk_count"] / siblings).values
    use_prefix = np.isnan(pred) & ~np.isnan(pk_share)
    pred[use_prefix] = pk_share[use_prefix]
    source[use_prefix] = PREFIX

    # 4. Regression model: everything still unresolved.
    use_model = np.isnan(pred)
    pred[use_model] = model_predict(model, te_feat, cols)[use_model]
    source[use_model] = MODEL

    if apply_growth:
        g = np.where(te["vtype"].isin(CAMERA_VTYPES).values, CAMERA_GROWTH, GROWTH)
        is_persistence = np.isin(source, [EXACT, NORMALIZED, PREFIX])
        pred[is_persistence] *= g[is_persistence]
    return np.clip(pred, 0, None), source


# ============================================================================
# Run
# ============================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        default=os.environ.get("TRAIN_PATH", DEFAULT_TRAIN_PATH),
        help="Prior-year aggregate CSV (default: bundled 2022 CSV).",
    )
    parser.add_argument(
        "--test-path",
        default=os.environ.get("TEST_PATH"),
        help="Optional test/eval CSV. May contain violation_count for scoring.",
    )
    parser.add_argument(
        "--corpus-dir",
        default=os.environ.get("CORPUS_DIR", DEFAULT_CORPUS_DIR),
        help="Directory containing augmentation tables.",
    )
    parser.add_argument(
        "--output-path",
        default=os.environ.get("OUTPUT_PATH", DEFAULT_OUT_PATH),
        help="Prediction CSV path (default: submission.csv beside this script).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the deterministic 2022 grouped-holdout check.",
    )
    return parser.parse_args(argv)


def validate_grouped_holdout(tr, cfeat, vtypes):
    """Evaluate cold-start behavior on a deterministic 20% street holdout."""
    street_keys = tr["street"].map(norm_street)
    fold = pd.util.hash_pandas_object(street_keys, index=False).to_numpy() % 5
    fit = tr.loc[fold != 0].reset_index(drop=True)
    valid = tr.loc[fold == 0].reset_index(drop=True)
    if fit.empty or valid.empty:
        print("[validation] skipped: dataset is too small for a grouped holdout")
        return

    fit_feat = add_features(fit, cfeat, vtypes)
    cols = feature_columns(fit_feat)
    model = fit_model(fit_feat, cols)
    valid_feat = add_features(valid[["street", "vtype"]], cfeat, vtypes)
    pred, source = predict(
        valid, fit, valid_feat, model, cols, apply_growth=False
    )
    pred = np.rint(pred)
    print(
        f"[validation] grouped street holdout rows={len(valid)} "
        f"RMSE={rmse(pred, valid['count']):.3f}"
    )
    for label in (NORMALIZED, PREFIX, MODEL):
        n = int((source == label).sum())
        if n:
            print(f"  {label:18s}: {n:>7d} ({100 * n / len(valid):5.1f}%)")


def main(argv=None):
    args = parse_args(argv)

    tr, has_train_target = read_keyed_csv(args.train_path)
    if not has_train_target:
        raise ValueError("Training CSV must contain violation_count.")
    tr = tr[["street", "vtype", "count"]]
    if tr.empty:
        raise ValueError("Training CSV contains no rows.")
    vtypes = sorted(tr["vtype"].unique())
    print(f"[data] train rows={len(tr)} streets={tr['street'].nunique()} vtypes={len(vtypes)}")

    cfeat = load_corpus_features(args.corpus_dir)
    if not args.skip_validation:
        validate_grouped_holdout(tr, cfeat, vtypes)

    train_feat = add_features(tr, cfeat, vtypes)
    cols = feature_columns(train_feat)
    model = fit_model(train_feat, cols)

    if not args.test_path:
        print("[done] final model trained; no --test-path supplied, so no submission was written.")
        return

    te, has_target = read_keyed_csv(args.test_path)
    if te.empty:
        raise ValueError("Test CSV contains no rows.")
    te_feat = add_features(te[["street", "vtype"]], cfeat, vtypes)
    pred, source = predict(te, tr, te_feat, model, cols)
    pred = np.rint(pred).astype(np.int64)

    for label in (EXACT, NORMALIZED, PREFIX, MODEL):
        n = int((source == label).sum())
        print(f"  {label:18s}: {n:>7d} ({100 * n / len(te):5.1f}%)")
    unseen = int((source != EXACT).sum())
    print(
        f"[coverage] unseen exact key pairs={unseen}/{len(te)}; "
        "handled by normalized, prefix, or regression fallbacks as shown above"
    )

    sub = pd.DataFrame({"street_name": te["street"].values,
                        "violation_type": te["vtype_original"].values,
                        "predicted_count": pred})
    sub.to_csv(args.output_path, index=False)
    print(f"[done] wrote {args.output_path} ({len(sub)} rows)")

    if has_target:
        y = te["count"].values
        print(f"[eval] RMSE={rmse(pred, y):.3f}")


if __name__ == "__main__":
    main()
