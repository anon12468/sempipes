from __future__ import annotations

import os
import re
import time
import argparse
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer
from skrub import DataOp

import sempipes  # noqa: F401 - importing registers SemPipes methods on skrub.DataOp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "NYC_DATA_DIR",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", "projects", "nyc_data", "nyc_data")),
)

# ---- paths ----
TRAIN_PATH = os.path.join(DATA_DIR, "train", "all_data", "violations_per_street_2022.csv")
TEST_PATH = os.path.join(DATA_DIR, "violations_per_street_2023.csv")
CORPUS_DIR = os.path.join(DATA_DIR, "train", "all_data", "corpus")
OUT_PATH = os.path.join(HERE, "submission_sempipes.csv")
CORPUS_CACHE_PATH = os.path.join(HERE, "corpus_features_sempipes.parquet")
ROW_ORDER_COL = "__sempipes_row_order"
RIGHT_JOIN_KEY = "corpus_k"
OP_AGG_PARKING = "semagg_parking_meters"
OP_AGG_PAVEMENT = "semagg_pavement"
OP_AGG_PLUTO = "semagg_pluto"
OP_GEN = "semgen_nyc_penalty_features"
AGG_OPERATOR_NAMES = [OP_AGG_PARKING, OP_AGG_PAVEMENT, OP_AGG_PLUTO]
ENABLE_OPTIMIZED_SEMGEN = os.environ.get("NYC_ENABLE_SEMGEN", "0") == "1"
OPERATOR_NAMES = AGG_OPERATOR_NAMES + ([OP_GEN] if ENABLE_OPTIMIZED_SEMGEN else [])

# ---- tuned growth multipliers (see baseline.py) ----
GROWTH = 1.35
CAMERA_GROWTH = 1.05

CAMERA_VTYPES = {
    "PHTO SCHOOL ZN SPEED VIOLATION",
    "BUS LANE VIOLATION",
    "FAILURE TO STOP AT RED LIGHT",
}

EXACT, PREFIX, MODEL = "exact match", "prefix match", "regression model"

_SUFFIX = {
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

STRUCT_COLS = [
    "st_is_numbered",
    "st_n_tokens",
    "st_len",
    "st_has_ave",
    "st_has_st",
    "st_has_blvd",
    "st_has_pkwy_expy_hwy",
    "vtype_code",
]

AGG_LEFT_DESCRIPTION = "One row per normalized NYC street."
AGG_RIGHT_DESCRIPTION = "NYC open-data corpus records with a normalized street join key named corpus_k."
GEN_DESCRIPTION = (
    "Rows for NYC parking violation count prediction, with one row per street and violation type. "
    "The table already contains deterministic street-name features, stable corpus features, and SemPipes "
    "corpus aggregations."
)

PARKING_METERS_PROMPT = """
    The right table describes parking meter assets and operating context around NYC streets. The downstream
    task is to predict parking violation counts for each street and violation type. Meter presence is a proxy
    for curb regulation intensity, parking turnover, and enforcement opportunity.

    The left table may already contain stable baseline columns such as f_meters_cnt. Do not overwrite,
    duplicate, or re-create any column already present in the left table. Add complementary numeric features
    with names like sem_meter_count, sem_meter_active_share, sem_meter_rule_diversity, or
    sem_meter_facility_mix. Preserve all left rows. Missing evidence should produce 0, not NaN or errors.
"""

PAVEMENT_PROMPT = """
    The right table describes street pavement and roadway characteristics. The downstream task is to predict
    parking violation counts for each street and violation type. Road scale and condition are important
    proxies for traffic exposure, camera exposure, and enforcement visibility.

    The left table may already contain stable baseline columns such as f_pave_seg, f_pave_len, and
    f_pave_rating. Do not overwrite, duplicate, or re-create any column already present in the left table.
    Add complementary numeric features with names like sem_pave_segment_count, sem_pave_total_length,
    sem_pave_avg_rating, sem_pave_avg_segment_len, sem_pave_main_road_share, sem_pave_rating_spread, or
    sem_pave_unrated_share. Preserve all left rows. Missing evidence should produce 0, not NaN or errors.
"""

PLUTO_PROMPT = """
    The right table describes tax lots, buildings, and land-use context along NYC streets. The downstream
    task is to predict parking violation counts for each street and violation type. Land-use context is a
    proxy for curb demand, residential density, commercial activity, and neighborhood intensity.

    The left table may already contain stable baseline columns such as f_pluto_lots, f_bldgarea, f_comarea,
    f_resarea, f_retailarea, and f_unitsres. These columns are not present in the raw right table; the raw
    right table has fields such as lotarea, bldgarea, comarea, resarea, retailarea, and unitsres. Do not
    overwrite, duplicate, or re-create any column already present in the left table. Add complementary
    numeric features with names like sem_pluto_lot_count, sem_pluto_total_building_area,
    sem_pluto_commercial_share, sem_pluto_residential_density, sem_pluto_avg_units_per_lot, or
    sem_pluto_building_area_spread. Preserve all left rows. Missing evidence should produce 0, not NaN or
    errors.
"""

ROW_FEATURE_PROMPT = """
    The downstream task is to predict parking violation counts for each NYC street and violation type.
    The table contains street identity, violation type, deterministic street-name features, stable corpus
    columns with names like f_meters_cnt, f_pave_seg, f_pave_len, f_pave_rating, f_pluto_lots,
    f_bldgarea, f_comarea, f_resarea, f_retailarea, f_unitsres, and possibly additional semantic
    aggregations from NYC open-data tables. It also contains stable engineered columns with a sem_det_
    prefix; these encode violation groups, log-transformed corpus signals, density ratios, and core
    interactions.

    Generate row-level numeric features that help the fallback regression model distinguish streets with
    true enforcement/activity signal from the many near-zero street/type pairs. Useful families:
    - camera-enforced violation type flags for PHTO SCHOOL ZN SPEED VIOLATION, BUS LANE VIOLATION,
      and FAILURE TO STOP AT RED LIGHT;
    - parking/standing/inspection/sticker/no-parking violation type groups;
    - interactions between violation type groups and sem_det_ meter, road length, traffic, pavement,
      commercial/residential/retail activity, and numbered/avenue/street/blvd/pkwy street shape;
    - normalized ratios such as activity per road length, meters per segment, commercial share of total
      activity, and log1p transforms of highly skewed corpus counts.

    Prefer combining existing sem_det_ features over recomputing them from scratch. Use only columns already
    present in the table and do not use target counts.

    Upstream semantic aggregations are optimised independently, but the input schema is stabilised before
    this operator runs. Treat missing or all-zero corpus-derived columns as unavailable evidence, not as
    a failure. Do not assume that a specific semantic aggregation column is informative just because it is
    present; derive robust numeric features that tolerate neutral defaults.
"""


@dataclass(frozen=True)
class CorpusSource:
    label: str
    variable_name: str
    operator_name: str
    filename: str
    columns: list[str]
    key_column: str
    key_fn: Callable[[object], str]
    prompt: str
    how_many: int


def norm_street(s) -> str:
    """Normalize a street name to a canonical join key."""
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"\s+", " ", s)
    for pattern, repl in _SUFFIX.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def prefix_key(s) -> str:
    """On-street + direction key for camera/intersection-coded locations."""
    if s is None:
        return ""
    return norm_street(str(s).upper().strip().split("@")[0])


def strip_house_number(addr) -> str:
    """Drop a leading house number: "120 BROADWAY" -> "BROADWAY"."""
    if addr is None:
        return ""
    s = re.sub(r"^\s*\d+[A-Z]?(-\d+[A-Z]?)?\s+", "", str(addr).strip(), flags=re.I)
    return norm_street(s)


CORPUS_SOURCES = [
    CorpusSource(
        label="parking meters",
        variable_name="parking_meters",
        operator_name=OP_AGG_PARKING,
        filename="parking_meters_locations_and_status__693u-uax6.parquet",
        columns=["On_Street", "Meter_Hours", "Status", "Borough"],
        key_column="On_Street",
        key_fn=norm_street,
        prompt=PARKING_METERS_PROMPT,
        how_many=5,
    ),
    CorpusSource(
        label="pavement",
        variable_name="pavement",
        operator_name=OP_AGG_PAVEMENT,
        filename="street_pavement_ratings__6yyb-pb25.parquet",
        columns=["OnStreetName", "Road_Type", "SystemRating", "LocationGeometry.STLength"],
        key_column="OnStreetName",
        key_fn=norm_street,
        prompt=PAVEMENT_PROMPT,
        how_many=5,
    ),
    CorpusSource(
        label="pluto",
        variable_name="pluto",
        operator_name=OP_AGG_PLUTO,
        filename="primary_land_use_tax_lot_output_pluto__64uk-42ks.parquet",
        columns=["address", "lotarea", "bldgarea", "comarea", "resarea", "retailarea", "unitsres"],
        key_column="address",
        key_fn=strip_house_number,
        prompt=PLUTO_PROMPT,
        how_many=6,
    ),
]


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


def read_keyed_csv(path: str) -> tuple[pd.DataFrame, bool]:
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


def build_grid(df: pd.DataFrame, vtypes: list[str]) -> pd.DataFrame:
    """Dense (street x vtype) training grid with absent pairs filled as 0 counts."""
    streets = df["street"].unique()
    grid = pd.MultiIndex.from_product([streets, vtypes], names=["street", "vtype"]).to_frame(index=False)
    grid = grid.merge(df, on=["street", "vtype"], how="left")
    grid["count"] = grid["count"].fillna(0.0)
    return grid


def add_base_features(frame: pd.DataFrame, vtype_categories: list[str]) -> pd.DataFrame:
    """Attach deterministic structural features and the normalized street join key."""
    frame = frame.copy()
    frame["k"] = frame["street"].map(norm_street)
    struct = structural_features(frame["street"])
    frame = pd.concat([frame.reset_index(drop=True), struct.reset_index(drop=True)], axis=1)
    frame["vtype_code"] = pd.Categorical(frame["vtype"], categories=vtype_categories).codes
    return frame


def _sem_agg_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    prompt: str,
    name: str,
    how_many: int,
) -> pd.DataFrame:
    """Run one SemPipes aggregation op and return the augmented left frame."""
    left_op = skrub.var("left").skb.mark_as_X().skb.set_description(AGG_LEFT_DESCRIPTION)
    right_op = skrub.var("right").skb.set_description(AGG_RIGHT_DESCRIPTION)

    aggregated = left_op.sem_agg_features(
        right_op,
        left_on="k",
        right_on=RIGHT_JOIN_KEY,
        nl_prompt=prompt,
        name=name,
        how_many=how_many,
    )
    env = {
        "left": left,
        "right": right,
        f"sempipes_pipeline_summary__{name}": None,
        f"sempipes_prefitted_state__{name}": None,
        f"sempipes_memory__{name}": [],
        f"sempipes_inspirations__{name}": [],
    }
    return aggregated.skb.eval(env)


def _new_sem_columns(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    return [col for col in after.columns if col not in before.columns]


def _read_corpus_table(
    corpus_dir: str,
    source: CorpusSource,
) -> pd.DataFrame:
    """Read a corpus table and add the normalized street join key expected by SemPipes."""
    df = pq.read_table(os.path.join(corpus_dir, source.filename), columns=source.columns).to_pandas()
    df[RIGHT_JOIN_KEY] = df[source.key_column].map(source.key_fn)
    return df[df[RIGHT_JOIN_KEY] != ""].reset_index(drop=True)


def load_corpus_tables(corpus_dir: str = CORPUS_DIR) -> dict[str, pd.DataFrame]:
    """Load the raw-ish corpus tables used by the optimizable SemPipes DAG."""
    return {source.variable_name: _read_corpus_table(corpus_dir, source) for source in CORPUS_SOURCES}


def _apply_sem_agg_source(left: pd.DataFrame, corpus_dir: str, source: CorpusSource) -> pd.DataFrame:
    """Apply one declared corpus aggregation source to the left street table."""
    right = _read_corpus_table(corpus_dir, source)
    before = left
    out = _sem_agg_one(
        left,
        right,
        name=source.operator_name,
        how_many=source.how_many,
        prompt=source.prompt,
    )
    print(f"  [sempipes] {source.label}: {_new_sem_columns(before, out)}")
    return out


def _sem_agg_data_op(left: DataOp, source: CorpusSource) -> DataOp:
    """Attach one corpus source to a DataOp through sem_agg_features."""
    right = skrub.var(source.variable_name).skb.set_description(AGG_RIGHT_DESCRIPTION)
    return left.sem_agg_features(
        right,
        left_on="k",
        right_on=RIGHT_JOIN_KEY,
        nl_prompt=source.prompt,
        name=source.operator_name,
        how_many=source.how_many,
    )


class SchemaStabilizer(BaseEstimator, TransformerMixin):
    """Keep DataFrame columns stable across fit/transform for generated feature code.

    Semantic aggregation code can produce columns conditionally. Downstream generated
    code is allowed to use columns seen during fit, so transform-time frames must
    provide the same names even when a fold/search node lacks the underlying signal.
    """

    def __init__(self, numeric_fill_value: float = 0.0, text_fill_value: str = "") -> None:
        self.numeric_fill_value = numeric_fill_value
        self.text_fill_value = text_fill_value

    def fit(self, frame: pd.DataFrame, y=None):  # pylint: disable=unused-argument
        self.columns_ = list(frame.columns)
        self.fill_values_ = {
            column: self.numeric_fill_value
            if pd.api.types.is_numeric_dtype(frame[column])
            else self.text_fill_value
            for column in self.columns_
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self.columns_:
            if column not in out.columns:
                out[column] = self.fill_values_[column]
        extra_columns = [column for column in out.columns if column not in self.columns_]
        return out[self.columns_ + extra_columns]


def add_baseline_style_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable row-level features before LLM-generated feature synthesis."""
    out = frame.copy()

    def numeric(column: str) -> pd.Series:
        if column not in out.columns:
            return pd.Series(0.0, index=out.index)
        return pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype(float)

    def first_available(*columns: str) -> pd.Series:
        values = pd.Series(0.0, index=out.index)
        for column in columns:
            current = numeric(column)
            values = values.where(values != 0, current)
        return values

    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vtype = out["vtype"].fillna("").astype(str).str.upper()
    meters = first_available("f_meters_cnt", "sem_meter_count")
    pave_segments = first_available("f_pave_seg", "sem_pave_segment_count")
    pave_length = first_available("f_pave_len", "sem_pave_total_length")
    pave_rating = first_available("f_pave_rating", "sem_pave_avg_rating")
    traffic = numeric("f_traffic_vol")
    pluto_lots = first_available("f_pluto_lots", "sem_pluto_lot_count")
    bldg_area = first_available("f_bldgarea", "sem_pluto_total_building_area")
    commercial_area = numeric("f_comarea")
    residential_area = numeric("f_resarea")
    retail_area = numeric("f_retailarea")
    residential_units = numeric("f_unitsres")
    activity_area = bldg_area + commercial_area + residential_area + retail_area

    out["sem_det_is_camera_vtype"] = vtype.isin(CAMERA_VTYPES).astype(float)
    out["sem_det_is_school_zone_speed"] = vtype.str.contains("SCHOOL|SPEED", regex=True).astype(float)
    out["sem_det_is_bus_lane"] = vtype.str.contains("BUS LANE", regex=False).astype(float)
    out["sem_det_is_red_light"] = vtype.str.contains("RED LIGHT", regex=False).astype(float)
    out["sem_det_is_parking_vtype"] = vtype.str.contains("PARK|PARKING", regex=True).astype(float)
    out["sem_det_is_standing_vtype"] = vtype.str.contains("STAND|STANDING", regex=True).astype(float)
    out["sem_det_is_meter_vtype"] = vtype.str.contains("METER|MUNI", regex=True).astype(float)
    out["sem_det_is_inspection_or_sticker_vtype"] = vtype.str.contains(
        "INSPECTION|STICKER|REGISTRATION", regex=True
    ).astype(float)
    out["sem_det_is_no_parking_vtype"] = vtype.str.contains("NO PARK|NO STAND|NO STOP", regex=True).astype(float)

    out["sem_det_log_meters"] = np.log1p(meters.clip(lower=0))
    out["sem_det_log_pave_segments"] = np.log1p(pave_segments.clip(lower=0))
    out["sem_det_log_pave_length"] = np.log1p(pave_length.clip(lower=0))
    out["sem_det_log_traffic"] = np.log1p(traffic.clip(lower=0))
    out["sem_det_log_activity_area"] = np.log1p(activity_area.clip(lower=0))
    out["sem_det_log_residential_units"] = np.log1p(residential_units.clip(lower=0))

    out["sem_det_meters_per_segment"] = safe_div(meters, pave_segments)
    out["sem_det_meters_per_road_length"] = safe_div(meters, pave_length)
    out["sem_det_traffic_per_road_length"] = safe_div(traffic, pave_length)
    out["sem_det_activity_per_road_length"] = safe_div(activity_area, pave_length)
    out["sem_det_building_area_per_lot"] = safe_div(bldg_area, pluto_lots)
    out["sem_det_units_per_lot"] = safe_div(residential_units, pluto_lots)
    out["sem_det_commercial_share"] = safe_div(commercial_area + retail_area, activity_area)
    out["sem_det_residential_share"] = safe_div(residential_area, activity_area)
    out["sem_det_pavement_rating_x_length"] = pave_rating * np.log1p(pave_length.clip(lower=0))

    out["sem_det_camera_x_traffic"] = out["sem_det_is_camera_vtype"] * out["sem_det_log_traffic"]
    out["sem_det_camera_x_road_length"] = out["sem_det_is_camera_vtype"] * out["sem_det_log_pave_length"]
    out["sem_det_parking_x_meters"] = out["sem_det_is_parking_vtype"] * out["sem_det_log_meters"]
    out["sem_det_meter_vtype_x_meters"] = out["sem_det_is_meter_vtype"] * out["sem_det_log_meters"]
    out["sem_det_standing_x_commercial_share"] = out["sem_det_is_standing_vtype"] * out["sem_det_commercial_share"]
    out["sem_det_no_parking_x_road_scale"] = out["sem_det_is_no_parking_vtype"] * out["sem_det_log_pave_length"]
    out["sem_det_inspection_x_residential_units"] = (
        out["sem_det_is_inspection_or_sticker_vtype"] * out["sem_det_log_residential_units"]
    )

    return out


def coerce_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare model inputs from deterministic and generated feature columns."""
    non_features = {"street", "vtype", "k", RIGHT_JOIN_KEY, ROW_ORDER_COL}
    features = frame.drop(columns=[c for c in non_features if c in frame.columns], errors="ignore")
    return features.apply(pd.to_numeric, errors="coerce").astype(float)


def build_pipeline(seed: int = 42, enable_semgen: bool | None = None) -> DataOp:
    """Build the optimizable SemPipes regression fallback pipeline."""
    if enable_semgen is None:
        enable_semgen = ENABLE_OPTIMIZED_SEMGEN

    data = skrub.var("data").skb.mark_as_X().skb.set_description(GEN_DESCRIPTION)
    labels = skrub.var("labels").skb.mark_as_y().skb.set_description("Observed NYC parking violation counts.")

    features = data
    for source in CORPUS_SOURCES:
        features = _sem_agg_data_op(features, source)

    features = features.skb.apply(SchemaStabilizer(), how="no_wrap")
    features = features.skb.apply(FunctionTransformer(add_baseline_style_row_features), how="no_wrap")
    if enable_semgen:
        features = features.sem_gen_features(
            nl_prompt=ROW_FEATURE_PROMPT,
            name=OP_GEN,
            how_many=8,
        )
    numeric_features = features.skb.apply(FunctionTransformer(coerce_numeric_features), how="no_wrap")
    log_labels = labels.skb.apply_func(np.log1p)
    log_predictions = numeric_features.skb.apply(
        HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=200,
            l2_regularization=1.0,
            random_state=seed,
        ),
        y=log_labels,
    )
    mode = skrub.eval_mode()
    return log_predictions.skb.apply_func(
        lambda pred, m: 0 if m == "fit" else np.clip(np.expm1(pred), 0, None),
        m=mode,
    )


def build_corpus_features_with_sempipes(corpus_dir: str, streets: pd.Series) -> pd.DataFrame:
    """Street-level corpus features generated by ``sem_agg_features``.

    The left side is one row per normalized street. For each corpus table we do
    only deterministic key normalization, then SemPipes decides how to aggregate
    the right-side records into useful street-level features.
    """
    left = pd.DataFrame({"k": sorted(set(streets.map(norm_street)) - {""})})

    for source in CORPUS_SOURCES:
        left = _apply_sem_agg_source(left, corpus_dir, source)

    return left.set_index("k")


def load_corpus_features_with_sempipes(streets: pd.Series) -> pd.DataFrame | None:
    """Load cached SemPipes corpus features or build them from corpus tables."""
    if not os.path.isdir(CORPUS_DIR):
        print(f"[warn] corpus dir not found ({CORPUS_DIR}); proceeding without corpus features.")
        return None

    rebuild = os.environ.get("NYC_REBUILD_SEMPIPES_FEATURES") == "1"
    if os.path.exists(CORPUS_CACHE_PATH) and not rebuild:
        return pd.read_parquet(CORPUS_CACHE_PATH)

    t = time.time()
    print(f"[info] building SemPipes corpus features from {CORPUS_DIR} ...")
    features = build_corpus_features_with_sempipes(CORPUS_DIR, streets)
    features.to_parquet(CORPUS_CACHE_PATH)
    print(f"[info] SemPipes corpus features: {features.shape} in {time.time() - t:.0f}s (cached)")
    return features


def add_features(frame: pd.DataFrame, cfeat: pd.DataFrame | None, vtype_categories: list[str]) -> pd.DataFrame:
    """Attach deterministic structural features plus SemPipes corpus aggregates."""
    frame = add_base_features(frame, vtype_categories)
    if cfeat is not None and not cfeat.empty:
        frame = frame.merge(cfeat, left_on="k", right_index=True, how="left")
    return frame


def add_sem_generated_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate row-level SemPipes features once, then split train/test back apart."""
    train_count = train_frame["count"].reset_index(drop=True)
    train_x = train_frame.drop(columns=["count"]).reset_index(drop=True)
    test_x = test_frame.reset_index(drop=True)
    combined = pd.concat([train_x, test_x], ignore_index=True)
    combined = SchemaStabilizer().fit_transform(combined)
    combined = add_baseline_style_row_features(combined)
    combined[ROW_ORDER_COL] = np.arange(len(combined))

    feature_op = skrub.var("rows").skb.mark_as_X().skb.set_description(GEN_DESCRIPTION)
    generated = feature_op.sem_gen_features(
        nl_prompt=ROW_FEATURE_PROMPT,
        name="semgen_nyc_penalty_features",
        how_many=8,
    )
    env = {
        "rows": combined,
        "sempipes_pipeline_summary__semgen_nyc_penalty_features": None,
        "sempipes_prefitted_state__semgen_nyc_penalty_features": None,
        "sempipes_memory__semgen_nyc_penalty_features": [],
        "sempipes_inspirations__semgen_nyc_penalty_features": [],
    }
    combined = generated.skb.eval(env)
    combined = combined.sort_values(ROW_ORDER_COL).drop(columns=[ROW_ORDER_COL]).reset_index(drop=True)

    train_generated = combined.iloc[: len(train_x)].reset_index(drop=True)
    test_generated = combined.iloc[len(train_x) :].reset_index(drop=True)
    train_generated["count"] = train_count
    return train_generated, test_generated


def build_model_frames(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    vtypes: list[str],
    corpus_features: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test feature frames with deterministic, aggregated, and generated features."""
    grid = add_features(build_grid(tr, vtypes), corpus_features, vtypes)
    te_feat = add_features(te[["street", "vtype"]], corpus_features, vtypes)
    return add_sem_generated_features(grid, te_feat)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model inputs: fixed structural columns plus SemPipes/generated feature columns."""
    non_features = {"street", "vtype", "count", "k", ROW_ORDER_COL}
    generated_cols = [c for c in frame.columns if c not in non_features]
    return sorted(set(STRUCT_COLS).union(generated_cols))


def model_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert deterministic and generated features to the numeric matrix expected by HGBR."""
    return frame[cols].apply(pd.to_numeric, errors="coerce").astype(float)


def fit_model(frame: pd.DataFrame, cols: list[str], seed: int = 0) -> HistGradientBoostingRegressor:
    """Fit the regression model on log1p counts."""
    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(model_matrix(frame, cols), np.log1p(frame["count"].values))
    return model


def model_predict(model: HistGradientBoostingRegressor, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Predict counts (invert the log1p transform and floor at zero)."""
    return np.clip(np.expm1(model.predict(model_matrix(frame, cols))), 0, None)


def predict(
    te: pd.DataFrame,
    tr: pd.DataFrame,
    te_feat: pd.DataFrame,
    model: HistGradientBoostingRegressor,
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict each key with exact match, prefix match, then regression fallback."""
    pred = np.full(len(te), np.nan)
    source = np.empty(len(te), dtype=object)

    merged = te.merge(tr.rename(columns={"count": "_prior"}), on=["street", "vtype"], how="left")
    matched = ~pd.isna(merged["_prior"].values)
    pred[matched] = np.nan_to_num(merged["_prior"].values)[matched]
    source[matched] = EXACT

    tr_pk = tr.assign(pk=tr["street"].map(prefix_key))
    agg = tr_pk.groupby(["pk", "vtype"])["count"].sum().rename("pk_count").reset_index()
    te_pk = te[["street", "vtype"]].assign(pk=te["street"].map(prefix_key))
    te_pk = te_pk.merge(agg, on=["pk", "vtype"], how="left")
    siblings = te_pk.groupby(["pk", "vtype"])["street"].transform("size")
    pk_share = (te_pk["pk_count"] / siblings).values
    use_prefix = np.isnan(pred) & ~np.isnan(pk_share)
    pred[use_prefix] = pk_share[use_prefix]
    source[use_prefix] = PREFIX

    use_model = np.isnan(pred)
    pred[use_model] = model_predict(model, te_feat, cols)[use_model]
    source[use_model] = MODEL

    g = np.where(te["vtype"].isin(CAMERA_VTYPES).values, CAMERA_GROWTH, GROWTH)
    is_persistence = (source == EXACT) | (source == PREFIX)
    pred[is_persistence] *= g[is_persistence]
    return np.clip(pred, 0, None), source


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def resolve_llm_name(alias_or_name: str) -> str:
    """Map short experiment aliases to LiteLLM model names."""
    aliases = {
        "gemini": "gemini/gemini-2.5-flash",
        "gemini_flash": "gemini/gemini-2.5-flash",
        "gemini_pro": "gemini/gemini-2.5-pro",
        "gemini-pro": "gemini/gemini-2.5-pro",
        "pro": "gemini/gemini-2.5-pro",
    }
    return aliases.get(alias_or_name, alias_or_name)


def configure_llm(alias_or_name: str, temperature: float) -> sempipes.LLM:
    """Configure SemPipes code generation and return the selected LLM."""
    llm = sempipes.LLM(name=resolve_llm_name(alias_or_name), parameters={"temperature": temperature})
    sempipes.update_config(llm_for_code_generation=llm)
    return llm


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC penalties SemPipes pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the fallback regressor")
    parser.add_argument("--llm", default="gemini", help="LLM alias or LiteLLM model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature for code generation")
    args = parser.parse_args()

    llm = configure_llm(args.llm, args.temperature)
    np.random.seed(args.seed)
    print(f"[config] seed={args.seed} llm={llm.name} temperature={args.temperature}")

    tr, _ = read_keyed_csv(TRAIN_PATH)
    vtypes = sorted(tr["vtype"].unique())
    print(f"[data] train rows={len(tr)} streets={tr['street'].nunique()} vtypes={len(vtypes)}")

    te, has_target = read_keyed_csv(TEST_PATH)
    all_streets = pd.concat([tr["street"], te["street"]], ignore_index=True)
    corpus_features = load_corpus_features_with_sempipes(all_streets)

    grid, te_feat = build_model_frames(tr, te, vtypes, corpus_features)
    cols = feature_columns(grid)
    model = fit_model(grid, cols, seed=args.seed)

    pred, source = predict(te, tr, te_feat, model, cols)
    pred = np.round(pred).astype(int)

    for label in (EXACT, PREFIX, MODEL):
        n = int((source == label).sum())
        print(f"  {label:18s}: {n:>7d} ({100 * n / len(te):5.1f}%)")

    sub = pd.DataFrame(
        {
            "street_name": te["street"].values,
            "violation_type": te["vtype"].values,
            "predicted_count": pred,
        }
    )
    sub.to_csv(OUT_PATH, index=False)
    print(f"[done] wrote {OUT_PATH} ({len(sub)} rows)")

    if has_target:
        y = te["count"].values
        print(f"[eval] RMSE={rmse(pred, y):.3f}")


if __name__ == "__main__":
    main()
