"""NYC Parking Violations Prediction Pipeline
===========================================
Predicts future (2023) parking violations for each (street_name, violation_type) pair
using 2022 aggregate data and corpus feature augmentations.

Conforms to train/all_data/goal.txt requirements:
- Path-configurable via CLI arguments or environment variables.
- Pure 2022 development validation when no test path is provided.
- Generates submission.csv with columns: street_name, violation_type, predicted_count.
- Evaluates unweighted RMSE if violation_count is provided in test file.
- Multi-tier matching engine + calibrated HistGradientBoosting regression model.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONSTANTS & CANONICAL MAPPINGS
# =============================================================================

CANONICAL_VTYPES: dict[str, str] = {
    # 14 - No Standing
    "14-No Standing": "14-No Standing",
    "NO STANDING": "14-No Standing",
    "NO STANDING-DAY/TIME LIMITS": "14-No Standing",
    "NO STANDING-BUS STOP": "14-No Standing",
    "NO STANDING-TAXI STAND": "14-No Standing",
    "NO STANDING-EXC. TRUCK MTR": "14-No Standing",
    "NO STANDING-FOR HIRE VEH ST": "14-No Standing",
    "NO STANDING-COMM MTR ZN": "14-No Standing",
    "NO STANDING-HOTEL LOADING": "14-No Standing",
    # 20A - No Parking Non-COM
    "20A-No Parking (Non-COM)": "20A-No Parking (Non-COM)",
    "20-No Parking (Day/Time)": "20A-No Parking (Non-COM)",
    "NO PARKING-DAY/TIME LIMITS": "20A-No Parking (Non-COM)",
    "NO PARKING-STREET CLEANING": "21-No Parking (Street Clean)",
    # 21 - Street Cleaning
    "21-No Parking (Street Clean)": "21-No Parking (Street Clean)",
    # 36 - School Zone Speed
    "36-PHTO SCHOOL ZN SPEED DETECT": "36-PHTO SCHOOL ZN SPEED DETECT",
    "PHTO SCHOOL ZN SPEED DETECT": "36-PHTO SCHOOL ZN SPEED DETECT",
    "SCHOOL ZONE SPEED": "36-PHTO SCHOOL ZN SPEED DETECT",
    # 38 - Failure to Stop at Red Light
    "38-Failure to Stop at Red Light": "38-Failure to Stop at Red Light",
    "FAIL TO STOP AT RED LIGHT": "38-Failure to Stop at Red Light",
    "RED LIGHT": "38-Failure to Stop at Red Light",
    # 40 - Fire Hydrant
    "40-Fire Hydrant": "40-Fire Hydrant",
    "FIRE HYDRANT": "40-Fire Hydrant",
    # 46 - Double Parking
    "46-Double Parking": "46-Double Parking",
    "DOUBLE PARKING": "46-Double Parking",
    # 70A - Reg Sticker Expired
    "70A-Reg. Sticker Expired (NYS)": "70A-Reg. Sticker Expired (NYS)",
    "REG. STICKER-EXPIRED/MISSING": "70A-Reg. Sticker Expired (NYS)",
    "EXPIRED REG": "70A-Reg. Sticker Expired (NYS)",
}

CAMERA_VTYPES = {
    "36-PHTO SCHOOL ZN SPEED DETECT",
    "38-Failure to Stop at Red Light",
}

DEFAULT_GROWTH = 1.35
DEFAULT_CAMERA_GROWTH = 1.05
RANDOM_SEED = 42

# Suffix abbreviations for standardizing NYC street names
_SUFFIX_MAP = {
    "AVENUE": "AVE", "AV": "AVE",
    "STREET": "ST", "STR": "ST",
    "BOULEVARD": "BLVD", "BOUL": "BLVD",
    "ROAD": "RD",
    "DRIVE": "DR",
    "PLACE": "PL",
    "LANE": "LN",
    "PARKWAY": "PKWY",
    "COURT": "CT",
    "TERRACE": "TER", "TERR": "TER",
    "EXPRESSWAY": "EXPY", "EXPWY": "EXPY",
    "HIGHWAY": "HWY",
    "TURNPIKE": "TPKE",
    "CIRCLE": "CIR",
    "BROADWAY": "BROADWAY",
}

_ORDINAL_MAP = {
    "FIRST": "1ST", "SECOND": "2ND", "THIRD": "3RD", "FOURTH": "4TH",
    "FIFTH": "5TH", "SIXTH": "6TH", "SEVENTH": "7TH", "EIGHTH": "8TH",
    "NINTH": "9TH", "TENTH": "10TH", "ELEVENTH": "11TH", "TWELFTH": "12TH",
}

# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def canonicalize_vtype(raw_val: str | None) -> str:
    """Map raw violation description to canonical category."""
    if raw_val is None:
        return "14-No Standing"
    raw_str = str(raw_val).strip()
    if raw_str in CANONICAL_VTYPES:
        return CANONICAL_VTYPES[raw_str]
    for key, val in CANONICAL_VTYPES.items():
        if key.lower() in raw_str.lower() or raw_str.lower() in key.lower():
            return val
    return raw_str


def norm_street(s: str | None) -> str:
    """Normalize NYC street strings into a standard uppercase form."""
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"^(?:[NSEW]B?|NORTHBOUND|SOUTHBOUND|EASTBOUND|WESTBOUND|[NSEW]/B)\s+", "", s)
    s = re.sub(r"\s*\([NSEW]/B\)", "", s)
    s = re.sub(r"\s*@\s*", " @ ", s)

    tokens = s.split()
    norm_tokens = []
    for t in tokens:
        clean = re.sub(r"[^\w@]", "", t)
        if not clean:
            continue
        clean = _ORDINAL_MAP.get(clean, clean)
        clean = _SUFFIX_MAP.get(clean, clean)
        m = re.match(r"^0*(\d+)(ST|ND|RD|TH)?$", clean)
        if m:
            clean = m.group(1)
        norm_tokens.append(clean)

    s = " ".join(norm_tokens)
    s = re.sub(r"\bWEST\b", "W", s)
    s = re.sub(r"\bEAST\b", "E", s)
    s = re.sub(r"\bNORTH\b", "N", s)
    s = re.sub(r"\bSOUTH\b", "S", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def corridor_key(s: str | None) -> str:
    """
    Extract the main road corridor name, especially useful for camera violations
    recorded at various intersections or mileposts along an arterial corridor.
    """
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"^(?:[NSEW]B?|NORTHBOUND|SOUTHBOUND|EASTBOUND|WESTBOUND|[NSEW]/B)\s+", "", s)
    s = s.split("@")[0].strip()
    s = re.sub(r"\s*\([NSEW]/B\)", "", s)
    s = norm_street(s)
    s = re.sub(r"^(?:N|S|E|W|NORTH|SOUTH|EAST|WEST|SO|NO)\s+", "", s)
    s = re.sub(r"\s+(?:N|S|E|W|NORTH|SOUTH|EAST|WEST)$", "", s)
    s = re.sub(r"\b(?:AVE|ST|PL|RD|BLVD|DR|LN|PKWY|CT|TER|EXPY|HWY)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =============================================================================
# DATA READING & PATH RESOLUTION
# =============================================================================

def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path, Path]:
    """Resolve train, test, corpus, and output paths with fallback discovery."""
    train_path = None
    if args.train_path:
        train_path = Path(args.train_path)
    elif os.environ.get("TRAIN_PATH"):
        train_path = Path(os.environ["TRAIN_PATH"])
    else:
        candidates = [
            Path("violations_per_street_2022.csv"),
            Path("train/all_data/violations_per_street_2022.csv"),
        ]
        for c in candidates:
            if c.exists():
                train_path = c
                break
        if train_path is None:
            train_path = candidates[0]

    test_path = Path(args.test_path)

    corpus_dir = None
    if args.corpus_dir:
        corpus_dir = Path(args.corpus_dir)
    elif os.environ.get("CORPUS_DIR"):
        corpus_dir = Path(os.environ["CORPUS_DIR"])
    else:
        candidates = [
            Path("train/all_data/corpus"),
            Path("corpus"),
        ]
        for c in candidates:
            if c.exists():
                corpus_dir = c
                break
        if corpus_dir is None:
            corpus_dir = Path("corpus")

    output_path = Path(args.output_path or os.environ.get("OUTPUT_PATH", "submission.csv"))
    return train_path, test_path, corpus_dir, output_path


def read_keyed_csv(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, bool]:
    """
    Read CSV adhering to the core schema:
    (Street Name, Violation Description, violation_count [optional]).
    Returns clean dataframe, raw street series, raw violation type series, has_target boolean.
    """
    df = pd.read_csv(path)
    col_map = {}
    for col in df.columns:
        norm = col.strip().lower().replace(" ", "_")
        if norm in ("street_name", "street"):
            col_map[col] = "street"
        elif norm in ("violation_description", "violation_type", "vtype"):
            col_map[col] = "vtype"
        elif norm in ("violation_count", "count"):
            col_map[col] = "count"

    df = df.rename(columns=col_map)
    if "street" not in df.columns or "vtype" not in df.columns:
        raise ValueError(f"File {path} must contain 'Street Name' and 'Violation Description' columns.")

    raw_street = df["street"].astype(str)
    raw_vtype = df["vtype"].astype(str)

    has_target = "count" in df.columns
    clean_df = pd.DataFrame()
    clean_df["street"] = raw_street.str.strip().str.upper()
    clean_df["vtype"] = raw_vtype.map(canonicalize_vtype)
    if has_target:
        clean_df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).values

    return clean_df, raw_street, raw_vtype, has_target


# =============================================================================
# FEATURE GENERATION & CORPUS ENRICHMENT
# =============================================================================

def load_corpus_features(corpus_dir: Path) -> pd.DataFrame:
    """
    Load precomputed street features or build enriched features from corpus tables.
    Caches an enriched table to disk for instant subsequent loading.
    """
    precomputed_file = corpus_dir / "corpus_features.parquet"
    enriched_cache = corpus_dir / "corpus_features_enriched.parquet"

    if enriched_cache.exists():
        try:
            return pd.read_parquet(enriched_cache)
        except Exception:
            pass

    if precomputed_file.exists():
        try:
            df = pd.read_parquet(precomputed_file)
            if "norm_street" in df.columns:
                df["norm_st"] = df["norm_street"].astype(str)
            else:
                df["norm_st"] = df["street_name"].map(norm_street)
            df = df.drop_duplicates(subset=["norm_st"])

            # Enrich with PLUTO tax-lot features if available
            pluto_file = corpus_dir / "primary_land_use_tax_lot_output_pluto__64uk-42ks.parquet"
            if pluto_file.exists():
                try:
                    pldf = pd.read_parquet(
                        pluto_file,
                        columns=["address", "bldgclass", "landuse", "numfloors", "unitsres", "unitstotal"]
                    )
                    unique_addrs = pldf["address"].dropna().unique()
                    addr_norm_map = {
                        addr: norm_street(re.sub(r"^\d+\s*", "", str(addr)))
                        for addr in unique_addrs
                    }
                    pldf["norm_st"] = pldf["address"].map(addr_norm_map)
                    pluto_aggs = pldf.groupby("norm_st").agg(
                        pluto_lot_count=("landuse", "count"),
                        pluto_total_units=("unitstotal", "sum"),
                        pluto_res_units=("unitsres", "sum"),
                        pluto_avg_floors=("numfloors", "mean"),
                        pluto_comm_lots=("landuse", lambda s: ((s == "05") | (s == 5)).sum()),
                    ).reset_index()

                    df = df.merge(pluto_aggs, on="norm_st", how="left")
                    for col in ["pluto_lot_count", "pluto_total_units", "pluto_res_units", "pluto_comm_lots"]:
                        df[col] = df[col].fillna(0)
                    df["pluto_avg_floors"] = df["pluto_avg_floors"].fillna(1.0)
                except Exception:
                    pass

            try:
                df.to_parquet(enriched_cache, index=False)
            except Exception:
                pass

            return df
        except Exception as e:
            print(f"[warning] Failed to read precomputed corpus features: {e}", file=sys.stderr)

    return pd.DataFrame({"norm_st": []})


def load_historical_2021(corpus_dir: Path) -> pd.DataFrame:
    """Load 2021 violations dataset if present in corpus as prior signal."""
    hist_path = corpus_dir / "violations_per_street_2021.csv"
    if not hist_path.exists():
        return pd.DataFrame(columns=["street", "vtype", "count_2021", "norm_st"])

    try:
        df, _, _, _ = read_keyed_csv(hist_path)
        df["norm_st"] = df["street"].map(norm_street)
        df = df.rename(columns={"count": "count_2021"})
        return df[["street", "vtype", "norm_st", "count_2021"]]
    except Exception as e:
        print(f"[warning] Could not load 2021 historical data: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["street", "vtype", "count_2021", "norm_st"])


def compute_training_aggregates(train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute street-level, corridor-level, and vtype priors from training signal."""
    if "norm_st" not in train_df.columns:
        train_df["norm_st"] = train_df["street"].map(norm_street)
    if "corridor" not in train_df.columns:
        train_df["corridor"] = train_df["street"].map(corridor_key)

    st_aggs = train_df.groupby("norm_st").agg(
        st_total_count=("count", "sum"),
        st_vtype_count=("vtype", "nunique"),
        st_max_count=("count", "max"),
        st_camera_count=("count", lambda s: s[train_df.loc[s.index, "vtype"].isin(CAMERA_VTYPES)].sum()),
    ).reset_index()

    corr_aggs = train_df.groupby(["corridor", "vtype"]).agg(
        corr_vtype_count=("count", "sum"),
        corr_st_count=("norm_st", "nunique"),
    ).reset_index()

    vtype_aggs = train_df.groupby("vtype").agg(
        vt_prior_mean=("count", "mean"),
        vt_prior_p50=("count", lambda s: np.percentile(s, 50)),
        vt_prior_p75=("count", lambda s: np.percentile(s, 75)),
        vt_prior_sum=("count", "sum"),
    ).reset_index()

    return {
        "street_aggs": st_aggs,
        "corridor_aggs": corr_aggs,
        "vtype_aggs": vtype_aggs,
    }


def add_features(
    df: pd.DataFrame,
    corpus_feats: pd.DataFrame,
    hist_2021: pd.DataFrame,
    train_aggs: dict[str, pd.DataFrame],
    vtypes: list[str],
) -> pd.DataFrame:
    """
    Enrich key pairs with corpus features, training aggregations, historical 2021 priors,
    and engineered interaction features.
    """
    out = df.copy()
    if "norm_st" not in out.columns:
        out["norm_st"] = out["street"].map(norm_street)
    if "corridor" not in out.columns:
        out["corridor"] = out["street"].map(corridor_key)

    if not corpus_feats.empty and "norm_st" in corpus_feats.columns:
        c_cols = [c for c in corpus_feats.columns if c not in ("street_name", "norm_street")]
        out = out.merge(corpus_feats[c_cols], on="norm_st", how="left")

    st_aggs = train_aggs.get("street_aggs")
    if st_aggs is not None and not st_aggs.empty:
        out = out.merge(st_aggs, on="norm_st", how="left")

    corr_aggs = train_aggs.get("corridor_aggs")
    if corr_aggs is not None and not corr_aggs.empty:
        out = out.merge(corr_aggs, on=["corridor", "vtype"], how="left")

    vt_aggs = train_aggs.get("vtype_aggs")
    if vt_aggs is not None and not vt_aggs.empty:
        out = out.merge(vt_aggs, on="vtype", how="left")

    if not hist_2021.empty:
        hist_exact = hist_2021.groupby(["street", "vtype"])["count_2021"].sum().reset_index()
        hist_exact = hist_exact.rename(columns={"count_2021": "hist_2021_exact"})
        out = out.merge(hist_exact, on=["street", "vtype"], how="left")

        hist_norm = hist_2021.groupby(["norm_st", "vtype"])["count_2021"].sum().reset_index()
        hist_norm = hist_norm.rename(columns={"count_2021": "hist_2021_norm"})
        out = out.merge(hist_norm, on=["norm_st", "vtype"], how="left")
    else:
        out["hist_2021_exact"] = np.nan
        out["hist_2021_norm"] = np.nan

    # One-hot encode violation types
    for vt in vtypes:
        col_name = f"vt_{re.sub(r'[^a-zA-Z0-9]', '_', vt)[:20]}"
        out[col_name] = (out["vtype"] == vt).astype(float)

    # Engineered interaction features
    if "meter_count" in out.columns and "st_total_count" in out.columns:
        out["meter_density"] = out["meter_count"].fillna(0) / (out["st_total_count"].fillna(0) + 1.0)
    if "storefront_count" in out.columns:
        vacant = out.get("storefront_vacant_count", 0)
        out["storefront_density"] = out["storefront_count"].fillna(0) / (out.get("pluto_lot_count", pd.Series(1, index=out.index)).fillna(0) + 1.0)
        out["storefront_occupancy"] = 1.0 - (pd.Series(vacant, index=out.index).fillna(0) / (out["storefront_count"].fillna(0) + 1e-4))

    # Cross interaction between meters and meter/parking violation types
    meter_vt_mask = out["vtype"].isin(["14-No Standing", "20A-No Parking (Non-COM)"]).astype(float)
    out["meter_x_meter_vtype"] = out.get("meter_count", pd.Series(0, index=out.index)).fillna(0) * meter_vt_mask

    # Cross interaction between speed limits and speed camera violations
    speed_vt_mask = (out["vtype"] == "36-PHTO SCHOOL ZN SPEED DETECT").astype(float)
    out["speed_x_camera"] = out.get("speed_limit_mean", pd.Series(25.0, index=out.index)).fillna(25.0) * speed_vt_mask

    for col in out.columns:
        if col not in ("street", "vtype", "norm_st", "corridor", "count"):
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].fillna(0.0)

    return out


def build_grid(train_df: pd.DataFrame, vtypes: list[str]) -> pd.DataFrame:
    """Build full Cartesian product of streets x vtypes for GBDT training."""
    unique_streets = train_df[["street", "norm_st", "corridor"]].drop_duplicates(subset=["street"])
    grid = unique_streets.merge(pd.DataFrame({"vtype": vtypes}), how="cross")
    grid = grid.merge(train_df[["street", "vtype", "count"]], on=["street", "vtype"], how="left")
    grid["count"] = grid["count"].fillna(0.0)
    return grid


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Extract numeric feature columns for model training."""
    exclude = {"street", "vtype", "norm_st", "corridor", "count", "pk"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


# =============================================================================
# MODEL TRAINING & CALIBRATION
# =============================================================================

def fit_model(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    seed: int = RANDOM_SEED,
) -> tuple[HistGradientBoostingRegressor, dict[str, float]]:
    """
    Fit HistGradientBoostingRegressor on log1p(count) and compute per-vtype
    mean-preservation calibration factors (Duan's smearing adaptation).
    """
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=63,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=seed,
    )
    X = train_frame[feature_cols].astype(float)
    y_log = np.log1p(train_frame["count"].values)
    model.fit(X, y_log)

    preds_raw = np.expm1(model.predict(X))
    calib_df = pd.DataFrame({
        "vtype": train_frame["vtype"],
        "y": train_frame["count"].values,
        "p": preds_raw,
    })
    calib_ratios = calib_df.groupby("vtype").apply(
        lambda g: float(g["y"].mean() / (g["p"].mean() + 1e-6)),
        include_groups=False,
    ).to_dict()

    return model, calib_ratios


# =============================================================================
# HIERARCHICAL MATCH RESOLUTION ENGINE
# =============================================================================

def predict(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_feats: pd.DataFrame,
    model: HistGradientBoostingRegressor,
    feature_cols: list[str],
    calib_ratios: dict[str, float],
    hist_2021: pd.DataFrame,
    growth: float = DEFAULT_GROWTH,
    camera_growth: float = DEFAULT_CAMERA_GROWTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical resolution engine for future counts:
      Tier 1: Exact raw match (street, vtype) carried forward with growth multiplier.
      Tier 2: Normalized street match (norm_st, vtype) with sibling count sharing.
      Tier 3: Prefix match on on-street + direction before '@' for intersections.
      Tier 4: Corridor matching for camera violations along arterial highways/boulevards.
      Tier 5: Historical 2021 match with 2-year growth adjustment.
      Tier 6: Calibrated GBDT regression model predictions.
      Tier 7: Empirical Bayes fallback floors (10.0 for camera, 1.5 for organic).
    """
    n = len(test_df)
    preds = np.zeros(n, dtype=float)
    tier_assigned = np.full(n, "", dtype=object)

    te = test_df.copy()
    if "norm_st" not in te.columns:
        te["norm_st"] = te["street"].map(norm_street)
    if "corridor" not in te.columns:
        te["corridor"] = te["street"].map(corridor_key)
    te["orig_idx"] = np.arange(n)

    te_keys = pd.DataFrame({
        "street": te["street"],
        "vtype": te["vtype"],
        "norm_st": te["norm_st"],
        "corridor": te["corridor"],
        "orig_idx": te["orig_idx"],
    })

    tr = train_df.copy()
    if "norm_st" not in tr.columns:
        tr["norm_st"] = tr["street"].map(norm_street)
    if "corridor" not in tr.columns:
        tr["corridor"] = tr["street"].map(corridor_key)

    growth_mult = np.where(te["vtype"].isin(CAMERA_VTYPES), camera_growth, growth)

    # -------------------------------------------------------------------------
    # Tier 1: Exact raw match
    # -------------------------------------------------------------------------
    tr_exact = tr.groupby(["street", "vtype"])["count"].sum().reset_index().rename(columns={"count": "exact_count"})
    m1 = te_keys.merge(tr_exact, on=["street", "vtype"], how="left")
    has_exact = m1["exact_count"].notna() & (m1["exact_count"] > 0)
    exact_indices = te_keys.loc[has_exact, "orig_idx"].values
    preds[exact_indices] = m1.loc[has_exact, "exact_count"].values * growth_mult[exact_indices]
    tier_assigned[exact_indices] = "exact match"

    # -------------------------------------------------------------------------
    # Tier 2: Normalized street match
    # -------------------------------------------------------------------------
    unresolved_mask = tier_assigned == ""
    if unresolved_mask.any():
        non_camera_unres = unresolved_mask & (~te_keys["vtype"].isin(CAMERA_VTYPES)) & (~te_keys["street"].str.contains("@"))
        if non_camera_unres.any():
            tr_norm = tr.groupby(["norm_st", "vtype"])["count"].sum().reset_index().rename(columns={"count": "norm_count"})
            test_norm_counts = te_keys.groupby(["norm_st", "vtype"])["orig_idx"].transform("count")
            m2 = te_keys.merge(tr_norm, on=["norm_st", "vtype"], how="left")
            has_norm = non_camera_unres & m2["norm_count"].notna() & (m2["norm_count"] > 0)
            norm_indices = te_keys.loc[has_norm, "orig_idx"].values
            alloc_factor = 1.0 / np.maximum(test_norm_counts.values[norm_indices], 1)
            preds[norm_indices] = m2.loc[has_norm, "norm_count"].values * growth_mult[norm_indices] * alloc_factor
            tier_assigned[norm_indices] = "normalized match"

    # -------------------------------------------------------------------------
    # Tier 3: Prefix match before '@' for intersections
    # -------------------------------------------------------------------------
    unresolved_mask = tier_assigned == ""
    if unresolved_mask.any():
        at_unres = unresolved_mask & te_keys["street"].str.contains("@")
        if at_unres.any():
            te_prefix = te_keys.copy()
            te_prefix["prefix_key"] = te_prefix["street"].apply(
                lambda s: norm_street(str(s).split("@")[0].strip())
            )
            tr_prefix = tr.copy()
            tr_prefix["prefix_key"] = tr_prefix["street"].apply(
                lambda s: norm_street(str(s).split("@")[0].strip())
            )
            tr_prefix_agg = tr_prefix.groupby(["prefix_key", "vtype"])["count"].mean().reset_index().rename(columns={"count": "pfx_count"})

            m3 = te_prefix.merge(tr_prefix_agg, on=["prefix_key", "vtype"], how="left")
            has_pfx = at_unres & m3["pfx_count"].notna() & (m3["pfx_count"] > 0)
            pfx_indices = te_keys.loc[has_pfx, "orig_idx"].values
            preds[pfx_indices] = m3.loc[has_pfx, "pfx_count"].values * growth_mult[pfx_indices]
            tier_assigned[pfx_indices] = "prefix match"

    # -------------------------------------------------------------------------
    # Tier 4: Corridor matching for camera violations
    # -------------------------------------------------------------------------
    unresolved_mask = tier_assigned == ""
    if unresolved_mask.any():
        cam_unres = unresolved_mask & te_keys["vtype"].isin(CAMERA_VTYPES)
        if cam_unres.any():
            tr_corr = tr[tr["corridor"] != ""].groupby(["corridor", "vtype"])["count"].median().reset_index().rename(columns={"count": "corr_count"})
            m4 = te_keys.merge(tr_corr, on=["corridor", "vtype"], how="left")
            has_corr = cam_unres & m4["corr_count"].notna() & (m4["corr_count"] > 0)
            corr_indices = te_keys.loc[has_corr, "orig_idx"].values
            preds[corr_indices] = m4.loc[has_corr, "corr_count"].values * camera_growth
            tier_assigned[corr_indices] = "corridor match"

    # -------------------------------------------------------------------------
    # Tier 5: Historical 2021 match
    # -------------------------------------------------------------------------
    unresolved_mask = tier_assigned == ""
    if unresolved_mask.any() and not hist_2021.empty:
        h21 = hist_2021.copy()
        if "norm_st" not in h21.columns:
            h21["norm_st"] = h21["street"].map(norm_street)
        h21_agg = h21.groupby(["norm_st", "vtype"])["count_2021"].sum().reset_index()

        m5 = te_keys.merge(h21_agg, on=["norm_st", "vtype"], how="left")
        has_hist = unresolved_mask & m5["count_2021"].notna() & (m5["count_2021"] > 0)
        hist_indices = te_keys.loc[has_hist, "orig_idx"].values
        hist_growth = np.where(te_keys.loc[has_hist, "vtype"].isin(CAMERA_VTYPES), 1.10, 1.82)
        preds[hist_indices] = m5.loc[has_hist, "count_2021"].values * hist_growth
        tier_assigned[hist_indices] = "historical 2021"

    # -------------------------------------------------------------------------
    # Tier 6: Calibrated GBDT regression model
    # -------------------------------------------------------------------------
    unresolved_mask = tier_assigned == ""
    if unresolved_mask.any():
        unres_indices = np.where(unresolved_mask)[0]
        X_unres = test_feats.iloc[unres_indices][feature_cols].astype(float)
        gbdt_preds_raw = np.expm1(model.predict(X_unres))

        calib_factors = np.array([
            calib_ratios.get(vt, 1.0)
            for vt in te.iloc[unres_indices]["vtype"]
        ])
        calibrated_preds = gbdt_preds_raw * calib_factors

        # Tier 7: Apply empirical floors to prevent extreme under-prediction
        floor_vals = np.where(te.iloc[unres_indices]["vtype"].isin(CAMERA_VTYPES), 10.0, 1.5)
        preds[unres_indices] = np.maximum(calibrated_preds, floor_vals)
        tier_assigned[unres_indices] = "regression model"

    return np.maximum(np.round(preds), 0).astype(int), tier_assigned


# =============================================================================
# EVALUATION & VALIDATION HARNESS
# =============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute unweighted Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def run_development_validation(
    train_df: pd.DataFrame,
    corpus_feats: pd.DataFrame,
    hist_2021: pd.DataFrame,
    vtypes: list[str],
    growth: float = DEFAULT_GROWTH,
    camera_growth: float = DEFAULT_CAMERA_GROWTH,
    run_cv: bool = False,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Development validation using strictly 2022 data.
    1. Pseudo-time split (Binomial arrival split simulating 1-year volume).
    2. Grouped holdout on unseen streets (20% holdout).
    3. Optional 5-Fold Grouped Cross-Validation.
    """
    print("=" * 60)
    print("DEVELOPMENT VALIDATION REPORT (2022 Data)")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Pseudo-Time Split (Binomial subsampling)
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    counts = train_df["count"].values.astype(int)
    pseudo_h1 = rng.binomial(n=counts, p=0.5)
    pseudo_h2 = counts - pseudo_h1

    tr_pseudo = train_df.copy()
    tr_pseudo["count"] = pseudo_h1
    tr_pseudo = tr_pseudo[tr_pseudo["count"] > 0].reset_index(drop=True)

    val_pseudo = train_df.copy()
    val_pseudo["count"] = pseudo_h2
    val_pseudo = val_pseudo[val_pseudo["count"] > 0].reset_index(drop=True)

    tr_aggs_pseudo = compute_training_aggregates(tr_pseudo)
    grid_pseudo = add_features(build_grid(tr_pseudo, vtypes), corpus_feats, hist_2021, tr_aggs_pseudo, vtypes)
    f_cols = get_feature_columns(grid_pseudo)
    model_pseudo, calib_pseudo = fit_model(grid_pseudo, f_cols, seed=seed)

    val_feats_pseudo = add_features(val_pseudo[["street", "vtype"]], corpus_feats, hist_2021, tr_aggs_pseudo, vtypes)
    val_preds_pseudo, tiers_pseudo = predict(
        val_pseudo, tr_pseudo, val_feats_pseudo,
        model_pseudo, f_cols, calib_pseudo, hist_2021,
        growth=1.0, camera_growth=1.0,
    )

    score_pseudo = rmse(val_pseudo["count"].values, val_preds_pseudo)
    print(f"\n[val-pseudo-time] Rows: {len(val_pseudo)} | RMSE: {score_pseudo:.3f}")
    tier_series = pd.Series(tiers_pseudo)
    for tier, count in tier_series.value_counts().items():
        mask = tier_series == tier
        tier_rmse = rmse(val_pseudo.loc[mask, "count"].values, val_preds_pseudo[mask])
        pct = 100.0 * count / len(tier_series)
        print(f"  {tier:<18}: {count:>6} ({pct:>5.1f}%) | sub-RMSE: {tier_rmse:.2f}")

    # -------------------------------------------------------------------------
    # 2. Grouped Street Holdout (20% unseen streets)
    # -------------------------------------------------------------------------
    unique_streets = train_df["norm_st"].unique()
    rng.shuffle(unique_streets)
    n_val_st = int(len(unique_streets) * 0.20)
    val_st_set = set(unique_streets[:n_val_st])

    is_val_st = train_df["norm_st"].isin(val_st_set)
    tr_st = train_df[~is_val_st].reset_index(drop=True)
    val_st = train_df[is_val_st].reset_index(drop=True)

    tr_aggs_st = compute_training_aggregates(tr_st)
    grid_st = add_features(build_grid(tr_st, vtypes), corpus_feats, hist_2021, tr_aggs_st, vtypes)
    model_st, calib_st = fit_model(grid_st, f_cols, seed=seed)

    val_feats_st = add_features(val_st[["street", "vtype"]], corpus_feats, hist_2021, tr_aggs_st, vtypes)
    val_preds_st, tiers_st = predict(
        val_st, tr_st, val_feats_st,
        model_st, f_cols, calib_st, hist_2021,
        growth=1.0, camera_growth=1.0,
    )
    score_unseen = rmse(val_st["count"].values, val_preds_st)
    print(f"\n[val-grouped-holdout] Evaluating unseen street generalization (20% holdout):")
    print(f"  Unseen streets: {len(val_st_set):,} | Rows: {len(val_st):,} | RMSE: {score_unseen:.2f}")

    # -------------------------------------------------------------------------
    # 3. Optional 5-Fold Grouped Cross-Validation
    # -------------------------------------------------------------------------
    if run_cv:
        print(f"\n[val-grouped-cv] Evaluating unseen street generalization across 5 folds:")
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        cv_scores = []
        for fold, (_, val_idx) in enumerate(kf.split(unique_streets), start=1):
            val_st_fold = set(unique_streets[val_idx])
            is_val_fold = train_df["norm_st"].isin(val_st_fold)
            tr_fold = train_df[~is_val_fold].reset_index(drop=True)
            val_fold = train_df[is_val_fold].reset_index(drop=True)

            tr_aggs_fold = compute_training_aggregates(tr_fold)
            grid_fold = add_features(build_grid(tr_fold, vtypes), corpus_feats, hist_2021, tr_aggs_fold, vtypes)
            model_fold, calib_fold = fit_model(grid_fold, f_cols, seed=seed)

            val_feats_fold = add_features(val_fold[["street", "vtype"]], corpus_feats, hist_2021, tr_aggs_fold, vtypes)
            preds_fold, _ = predict(
                val_fold, tr_fold, val_feats_fold,
                model_fold, f_cols, calib_fold, hist_2021,
                growth=1.0, camera_growth=1.0,
            )
            fold_score = rmse(val_fold["count"].values, preds_fold)
            cv_scores.append(fold_score)
            print(f"  Fold {fold}: val_rows={len(val_fold)} | RMSE={fold_score:.2f}")

        print(f"[val-grouped-cv] Mean Unseen Street RMSE: {np.mean(cv_scores):.2f} +/- {np.std(cv_scores):.2f}")

    print("=" * 60)
    return score_pseudo


# =============================================================================
# CLI & MAIN ENTRYPOINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYC Parking Violations Prediction Pipeline")
    parser.add_argument("--train-path", type=str, default=None,
                        help="Path to training data (default: violations_per_street_2022.csv)")
    parser.add_argument("--test-path", type=str, default="/Users/USER/Downloads/violations_per_street_2023.csv",
                        help="Path to test/eval data (posthoc test file)")
    parser.add_argument("--eval-path", type=str, default=None,
                        help="Alias for --test-path")
    parser.add_argument("--corpus-dir", type=str, default=None,
                        help="Path to corpus folder (default: train/all_data/corpus or corpus)")
    parser.add_argument("--output-path", type=str, default="submission.csv",
                        help="Path to write submission.csv (default: submission.csv)")
    parser.add_argument("--growth", type=float, default=DEFAULT_GROWTH,
                        help=f"Annual organic growth multiplier (default: {DEFAULT_GROWTH})")
    parser.add_argument("--camera-growth", type=float, default=DEFAULT_CAMERA_GROWTH,
                        help=f"Annual camera violation growth multiplier (default: {DEFAULT_CAMERA_GROWTH})")
    parser.add_argument("--run-cv", action="store_true",
                        help="Run full 5-fold grouped street cross-validation during dev mode")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    return parser.parse_args()


def main() -> None:
    start_time = time.time()
    args = parse_args()
    if args.eval_path and not args.test_path:
        args.test_path = args.eval_path

    train_path, test_path, corpus_dir, output_path = resolve_paths(args)

    print(f"[{time.strftime('%X')}] Loading training data from: {train_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    train_df, tr_raw_st, tr_raw_vt, _ = read_keyed_csv(train_path)
    train_df["norm_st"] = train_df["street"].map(norm_street)
    train_df["corridor"] = train_df["street"].map(corridor_key)

    vtypes = sorted(list(set(train_df["vtype"].unique()) | set(CANONICAL_VTYPES.values())))
    print(f"[{time.strftime('%X')}] Loaded {len(train_df):,} training records across {train_df['street'].nunique():,} unique streets.")

    print(f"[{time.strftime('%X')}] Loading corpus features from: {corpus_dir}")
    corpus_feats = load_corpus_features(corpus_dir)
    hist_2021 = load_historical_2021(corpus_dir)
    if not corpus_feats.empty:
        print(f"[{time.strftime('%X')}] Loaded {len(corpus_feats):,} corpus street features.")
    if not hist_2021.empty:
        print(f"[{time.strftime('%X')}] Loaded {len(hist_2021):,} historical 2021 records.")

    # -------------------------------------------------------------------------
    # Development Validation Mode (when no test path is provided)
    # -------------------------------------------------------------------------
    if test_path is None or not test_path.exists():
        print(f"\n[{time.strftime('%X')}] No test path provided. Running 2022 Development Validation...")
        dev_score = run_development_validation(
            train_df=train_df,
            corpus_feats=corpus_feats,
            hist_2021=hist_2021,
            vtypes=vtypes,
            growth=args.growth,
            camera_growth=args.camera_growth,
            run_cv=args.run_cv,
            seed=args.seed,
        )
        print(f"\n[{time.strftime('%X')}] Development validation completed successfully in {time.time() - start_time:.1f}s.")
        print(f"Final Development Validation RMSE: {dev_score:.3f}")
        return

    # -------------------------------------------------------------------------
    # Test Prediction & Submission Mode (when test path is provided)
    # -------------------------------------------------------------------------
    print(f"\n[{time.strftime('%X')}] Reading test/eval data from: {test_path}")
    test_df, te_raw_st, te_raw_vt, has_target = read_keyed_csv(test_path)
    print(f"[{time.strftime('%X')}] Loaded {len(test_df):,} test key pairs.")

    print(f"[{time.strftime('%X')}] Computing training aggregates & building feature matrix...")
    train_aggs = compute_training_aggregates(train_df)
    train_grid = build_grid(train_df, vtypes)
    train_feats = add_features(train_grid, corpus_feats, hist_2021, train_aggs, vtypes)
    feature_cols = get_feature_columns(train_feats)

    print(f"[{time.strftime('%X')}] Training HistGradientBoostingRegressor on {len(train_grid):,} grid pairs ({len(feature_cols)} features)...")
    model, calib_ratios = fit_model(train_feats, feature_cols, seed=args.seed)

    print(f"[{time.strftime('%X')}] Building test feature matrix and generating predictions...")
    test_feats = add_features(test_df[["street", "vtype"]], corpus_feats, hist_2021, train_aggs, vtypes)
    preds, tier_assigned = predict(
        test_df=test_df,
        train_df=train_df,
        test_feats=test_feats,
        model=model,
        feature_cols=feature_cols,
        calib_ratios=calib_ratios,
        hist_2021=hist_2021,
        growth=args.growth,
        camera_growth=args.camera_growth,
    )

    # Key pair unseen analysis
    tr_pairs = set(zip(train_df["street"], train_df["vtype"]))
    te_pairs = list(zip(test_df["street"], test_df["vtype"]))
    unseen_count = sum(pair not in tr_pairs for pair in te_pairs)
    unseen_pct = 100.0 * unseen_count / len(te_pairs)

    print("\n" + "=" * 60)
    print("TEST EVALUATION / PREDICTION REPORT")
    print("=" * 60)
    print(f"Total test key pairs: {len(test_df):,}")
    print(f"Unseen in training:   {unseen_count:,} ({unseen_pct:.1f}%)")
    print("\nMatch Resolution Breakdown:")
    tier_counts = pd.Series(tier_assigned).value_counts()
    for tier, count in tier_counts.items():
        pct = 100.0 * count / len(tier_assigned)
        print(f"  {tier:<18}: {count:>6} ({pct:>5.1f}%)")

    # Write submission.csv
    submission = pd.DataFrame({
        "street_name": te_raw_st,
        "violation_type": te_raw_vt,
        "predicted_count": preds,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"\n[{time.strftime('%X')}] Saved predictions to: {output_path} ({len(submission):,} rows)")

    # Score if ground truth violation_count was present
    if has_target:
        y_true = test_df["count"].values
        eval_rmse = rmse(y_true, preds)
        print(f"\nEvaluation Target Detected:")
        print(f"Evaluated Test RMSE: {eval_rmse:.3f}")
        for tier in tier_counts.index:
            mask = np.array(tier_assigned) == tier
            if mask.sum() > 0:
                sub_score = rmse(y_true[mask], preds[mask])
                print(f"  {tier:<18} sub-RMSE: {sub_score:.2f}")

    print("=" * 60)
    print(f"Pipeline executed in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
