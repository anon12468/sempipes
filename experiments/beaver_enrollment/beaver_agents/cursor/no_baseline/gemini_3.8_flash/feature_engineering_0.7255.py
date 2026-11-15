"""Feature engineering pipeline for high-enrollment prediction.

Strictly leakage-free: current-term enrollment and targets are never used as features.
All historical features are lagged/shifted so each row only sees past terms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SEASON_ORD = {"JA": 0, "SP": 1, "SU": 2, "FA": 3}

# Columns that are identifiers, targets, or intermediate fields — not model inputs.
META_COLUMNS = {
    "TERM_CODE",
    "SUBJECT_ID_SORT",
    "y",
    "split",
    "tk",
    "year",
    "sid",
    "enr",
    "dept",
    "faculty",
    "sid_season",
    "MASTER_SUBJECT_ID_SORT",
}

CATEGORICAL_FEATURES = [
    "offer_dept",
    "offer_school",
    "hgn_code",
    "cluster_type",
    "season",
    "prefix",
    "cat_hass",
    "cat_gir",
]


def term_key(term: pd.Series) -> pd.Series:
    """Order terms within a calendar year as JA, SP, SU, FA."""
    year = pd.to_numeric(term.astype(str).str.slice(0, 4), errors="coerce").fillna(0).astype(int)
    season = term.astype(str).str.slice(4).map(SEASON_ORD).fillna(9).astype(int)
    return year * 10 + season


def clean_cat(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.fillna("MISSING")
    out = out.replace({"": "MISSING", "nan": "MISSING", "None": "MISSING", "<NA>": "MISSING"})
    return out.astype(str)


def group_prior_mean(df: pd.DataFrame, key: str, value_col: str) -> Tuple[pd.Series, pd.Series]:
    """Mean and count of past values within `key`, strictly excluding the current row.

    Sorted by (key, tk, _idx) to maintain strict temporal ordering without looking ahead.
    """
    d = df[[key, "tk", value_col]].copy()
    d["_idx"] = df.index.to_numpy()
    d = d.sort_values([key, "tk", "_idx"], kind="mergesort")
    values = pd.to_numeric(d[value_col], errors="coerce").to_numpy(dtype=float)
    observed = np.isfinite(values).astype(np.float64)
    filled = np.where(np.isfinite(values), values, 0.0)
    d["_v"] = filled
    d["_o"] = observed
    grouped = d.groupby(key, sort=False)
    csum = grouped["_v"].cumsum()
    ccnt = grouped["_o"].cumsum()
    prior_sum = csum.groupby(d[key], sort=False).shift(1)
    prior_cnt = ccnt.groupby(d[key], sort=False).shift(1)
    rate = prior_sum.to_numpy() / np.where(prior_cnt.to_numpy() > 0, prior_cnt.to_numpy(), np.nan)
    idx = d["_idx"].to_numpy()
    rate_s = pd.Series(rate, index=idx)
    cnt_s = pd.Series(prior_cnt.to_numpy(), index=idx)
    return rate_s.reindex(df.index), cnt_s.reindex(df.index)


def _read_csv(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    kwargs = {"dtype": str, "keep_default_na": True, "low_memory": False}
    if usecols is not None:
        kwargs["usecols"] = usecols
    return pd.read_csv(path, **kwargs)


def _yn(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.upper() == "Y").astype(np.int8)


def _attach_section_features(frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    path = data_dir / "SUBJECT_OFFERED.csv"
    section_cols = [
        "n_sections",
        "n_lecture",
        "n_lab",
        "n_recitation",
        "n_design",
        "n_evening",
        "n_with_time",
        "n_ose",
        "n_repeatable",
        "n_evaluate",
        "n_faculty_sec",
        "n_places",
    ]
    if not path.exists():
        for col in section_cols:
            frame[col] = 0
        return frame

    sections = _read_csv(
        path,
        [
            "TERM_CODE",
            "SUBJECT_ID_SORT",
            "IS_LECTURE_SECTION",
            "IS_LAB_SECTION",
            "IS_RECITATION_SECTION",
            "IS_DESIGN_SECTION",
            "MEET_TIME",
            "MEET_PLACE",
            "RESPONSIBLE_FACULTY_NAME",
            "EVALUATE_THIS_SUBJECT",
            "IS_OSE_SUBJECT",
            "IS_REPEATABLE_SUBJECT",
        ],
    )
    sections["TERM_CODE"] = sections["TERM_CODE"].astype(str).str.strip()
    sections["SUBJECT_ID_SORT"] = sections["SUBJECT_ID_SORT"].astype(str).str.strip()
    meet = sections["MEET_TIME"].fillna("").str.upper()
    sections["lec"] = _yn(sections["IS_LECTURE_SECTION"])
    sections["lab"] = _yn(sections["IS_LAB_SECTION"])
    sections["rec"] = _yn(sections["IS_RECITATION_SECTION"])
    sections["des"] = _yn(sections["IS_DESIGN_SECTION"])
    sections["ose"] = _yn(sections["IS_OSE_SUBJECT"])
    sections["rep"] = _yn(sections["IS_REPEATABLE_SUBJECT"])
    sections["ev"] = _yn(sections["EVALUATE_THIS_SUBJECT"])
    sections["eve"] = meet.str.contains("EVE", regex=False).astype(np.int8)
    sections["has_time"] = (meet.str.strip() != "").astype(np.int8)
    agg = sections.groupby(["TERM_CODE", "SUBJECT_ID_SORT"], sort=False).agg(
        n_sections=("lec", "size"),
        n_lecture=("lec", "sum"),
        n_lab=("lab", "sum"),
        n_recitation=("rec", "sum"),
        n_design=("des", "sum"),
        n_evening=("eve", "sum"),
        n_with_time=("has_time", "sum"),
        n_ose=("ose", "sum"),
        n_repeatable=("rep", "sum"),
        n_evaluate=("ev", "sum"),
        n_faculty_sec=("RESPONSIBLE_FACULTY_NAME", "nunique"),
        n_places=("MEET_PLACE", "nunique"),
    )
    return frame.merge(agg, on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left")


def _attach_catalog_features(frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    path = data_dir / "COURSE_CATALOG_SUBJECT_OFFERED.csv"
    if not path.exists():
        frame["has_catalog"] = 0
        frame["cat_variable_units"] = np.nan
        frame["cat_prereq_len"] = np.nan
        frame["cat_hass"] = "MISSING"
        frame["cat_gir"] = "MISSING"
        return frame

    catalog = _read_csv(
        path,
        [
            "TERM_CODE",
            "SUBJECT_CODE",
            "SUBJECT_NUMBER",
            "IS_VARIABLE_UNITS",
            "PREREQUISITES",
            "HASS_ATTRIBUTE",
            "GIR_ATTRIBUTE",
        ],
    )
    catalog["TERM_CODE"] = catalog["TERM_CODE"].astype(str).str.strip()
    catalog["sid"] = (
        catalog["SUBJECT_CODE"].astype(str).str.strip()
        + "."
        + catalog["SUBJECT_NUMBER"].astype(str).str.strip()
    )
    catalog["cat_variable_units"] = _yn(catalog["IS_VARIABLE_UNITS"]).astype(float)
    prereq = catalog["PREREQUISITES"].fillna("").astype(str).str.strip()
    catalog["cat_prereq_len"] = np.where(prereq.isin(["", "nan", "None"]), 0, prereq.str.len())
    catalog["cat_hass"] = clean_cat(catalog["HASS_ATTRIBUTE"])
    catalog["cat_gir"] = clean_cat(catalog["GIR_ATTRIBUTE"])
    catalog = catalog.drop_duplicates(["TERM_CODE", "sid"], keep="first")
    catalog = catalog[["TERM_CODE", "sid", "cat_variable_units", "cat_prereq_len", "cat_hass", "cat_gir"]]
    frame = frame.merge(
        catalog,
        left_on=["TERM_CODE", "SUBJECT_ID_SORT"],
        right_on=["TERM_CODE", "sid"],
        how="left",
    )
    frame["has_catalog"] = frame["sid"].notna().astype(np.int8)
    frame = frame.drop(columns=["sid"])
    frame["cat_hass"] = clean_cat(frame["cat_hass"])
    frame["cat_gir"] = clean_cat(frame["cat_gir"])
    return frame


def load_offering_frame(data_dir: Path, gold_path: Optional[Path], split: str) -> pd.DataFrame:
    """One row per (TERM_CODE, SUBJECT_ID_SORT) offering with all schedule & catalog metadata."""
    summary = _read_csv(
        data_dir / "SUBJECT_OFFERED_SUMMARY.csv",
        [
            "TERM_CODE",
            "SUBJECT_ID_SORT",
            "SUBJECT_TITLE",
            "MASTER_SUBJECT_ID_SORT",
            "CLUSTER_TYPE",
            "CLUSTER_LIST",
            "HGN_CODE",
            "OFFER_DEPT_CODE",
            "OFFER_SCHOOL_NAME",
            "RESPONSIBLE_FACULTY_NAME",
            "TOTAL_UNITS",
            "LECTURE_UNITS",
            "LAB_UNITS",
            "PREPARATION_UNITS",
        ],
    )
    summary["TERM_CODE"] = summary["TERM_CODE"].astype(str).str.strip()
    summary["SUBJECT_ID_SORT"] = summary["SUBJECT_ID_SORT"].astype(str).str.strip()
    summary = summary.drop_duplicates(["TERM_CODE", "SUBJECT_ID_SORT"], keep="first")

    if gold_path is not None and gold_path.exists():
        gold = _read_csv(gold_path)
        gold["TERM_CODE"] = gold["TERM_CODE"].astype(str).str.strip()
        gold["SUBJECT_ID_SORT"] = gold["SUBJECT_ID_SORT"].astype(str).str.strip()
        keys = gold[["TERM_CODE", "SUBJECT_ID_SORT"]].drop_duplicates()
        if "HIGH_ENROLLMENT" in gold.columns:
            label = gold[["TERM_CODE", "SUBJECT_ID_SORT", "HIGH_ENROLLMENT"]].drop_duplicates(
                ["TERM_CODE", "SUBJECT_ID_SORT"]
            )
            keys = keys.merge(label, on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left")
            raw_target = keys["HIGH_ENROLLMENT"].astype(str).str.strip().str.upper()
            keys["y"] = np.where(raw_target.isin(["Y", "1", "TRUE"]), 1.0, 0.0)
            keys.loc[keys["HIGH_ENROLLMENT"].isna(), "y"] = np.nan
            keys = keys.drop(columns=["HIGH_ENROLLMENT"])
        else:
            keys["y"] = np.nan
        frame = keys.merge(summary, on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left")
    else:
        frame = summary.copy()
        frame["y"] = np.nan

    # Load SUBJECT_SUMMARY for training only to compute historical lag enrollments
    # In test split, actual current-term enrollments are never read.
    enroll_path = data_dir / "SUBJECT_SUMMARY.csv"
    if enroll_path.exists() and split == "train":
        enroll = _read_csv(
            enroll_path,
            ["TERM_CODE", "SUBJECT_ID_SORT", "SUBJECT_ENROLLMENT_FIFTH_WEEK", "DEPARTMENT_CODE"],
        )
        enroll["TERM_CODE"] = enroll["TERM_CODE"].astype(str).str.strip()
        enroll["SUBJECT_ID_SORT"] = enroll["SUBJECT_ID_SORT"].astype(str).str.strip()
        enroll = enroll.drop_duplicates(["TERM_CODE", "SUBJECT_ID_SORT"], keep="first")
        frame = frame.merge(enroll, on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left")
    else:
        frame["SUBJECT_ENROLLMENT_FIFTH_WEEK"] = np.nan
        frame["DEPARTMENT_CODE"] = np.nan

    hist_path = data_dir / "ENROLLMENT_HISTORY_FEATURES.csv"
    if hist_path.exists():
        hist = _read_csv(hist_path)
        hist["TERM_CODE"] = hist["TERM_CODE"].astype(str).str.strip()
        hist["SUBJECT_ID_SORT"] = hist["SUBJECT_ID_SORT"].astype(str).str.strip()
        for col in ("lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"):
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors="coerce")
        keep = ["TERM_CODE", "SUBJECT_ID_SORT", "lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"]
        hist = hist[[c for c in keep if c in hist.columns]].drop_duplicates(
            ["TERM_CODE", "SUBJECT_ID_SORT"], keep="first"
        )
        frame = frame.merge(hist, on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left")
    else:
        frame["lag1_enrollment"] = np.nan
        frame["lag2_enrollment"] = np.nan
        frame["roll3_enrollment_mean"] = np.nan

    frame = _attach_section_features(frame, data_dir)
    frame = _attach_catalog_features(frame, data_dir)
    frame["split"] = split
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build complete feature set with strict anti-leakage shifts."""
    df = frame.copy()
    df["TERM_CODE"] = df["TERM_CODE"].astype(str).str.strip()
    df["SUBJECT_ID_SORT"] = df["SUBJECT_ID_SORT"].astype(str).str.strip()
    df = df.reset_index(drop=True)

    df["tk"] = term_key(df["TERM_CODE"])
    df["year"] = pd.to_numeric(df["TERM_CODE"].str.slice(0, 4), errors="coerce").fillna(0).astype(int)
    df["season"] = df["TERM_CODE"].str.slice(4)
    df.loc[~df["season"].isin(SEASON_ORD), "season"] = "NA"
    df["sid"] = df["SUBJECT_ID_SORT"]
    df["enr"] = pd.to_numeric(df.get("SUBJECT_ENROLLMENT_FIFTH_WEEK"), errors="coerce")

    dept = clean_cat(df.get("OFFER_DEPT_CODE", pd.Series("MISSING", index=df.index)))
    if "DEPARTMENT_CODE" in df.columns:
        backup = clean_cat(df["DEPARTMENT_CODE"])
        dept = np.where(dept == "MISSING", backup, dept)
    df["dept"] = pd.Series(dept, index=df.index).astype(str)
    df["offer_dept"] = df["dept"]
    df["offer_school"] = clean_cat(df.get("OFFER_SCHOOL_NAME", pd.Series("MISSING", index=df.index)))
    df["hgn_code"] = clean_cat(df.get("HGN_CODE", pd.Series("MISSING", index=df.index)))
    df["cluster_type"] = clean_cat(df.get("CLUSTER_TYPE", pd.Series("MISSING", index=df.index)))
    df["faculty"] = clean_cat(df.get("RESPONSIBLE_FACULTY_NAME", pd.Series("MISSING", index=df.index)))
    df.loc[df["faculty"] == "MISSING", "faculty"] = np.nan

    sid = df["sid"]
    prefix = sid.str.split(".").str[0].fillna("MISSING")
    suffix = sid.str.split(".").str[1].fillna("")
    df["prefix"] = prefix
    level = suffix.str.extract(r"(\d+\.?\d*)", expand=False)
    df["course_level"] = pd.to_numeric(level, errors="coerce")
    df["is_grad_level"] = (df["course_level"] >= 500).astype(float)
    df["is_thesis"] = suffix.str.contains(r"TH", case=False, regex=True).astype(np.int8)
    df["is_urop"] = suffix.str.contains(r"UR", case=False, regex=True).astype(np.int8)
    df["is_special"] = suffix.str.match(r"S\d", case=False).fillna(False).astype(np.int8)
    df["is_numeric_prefix"] = prefix.str.fullmatch(r"\d+").fillna(False).astype(np.int8)
    df["is_harvard"] = prefix.str.startswith("HA").fillna(False).astype(np.int8)

    for col, default in (
        ("TOTAL_UNITS", np.nan),
        ("LECTURE_UNITS", np.nan),
        ("LAB_UNITS", np.nan),
        ("PREPARATION_UNITS", np.nan),
    ):
        df[col.lower()] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else default
    df = df.rename(
        columns={
            "total_units": "total_units",
            "lecture_units": "lecture_units",
            "lab_units": "lab_units",
            "preparation_units": "prep_units",
        }
    )

    title = df["SUBJECT_TITLE"].fillna("") if "SUBJECT_TITLE" in df.columns else pd.Series("", index=df.index)
    title_l = title.astype(str).str.lower()
    df["title_len"] = title_l.str.len().astype(np.int16)
    for name, token in (
        ("title_research", "research"),
        ("title_thesis", "thesis"),
        ("title_special", "special"),
        ("title_seminar", "seminar"),
        ("title_intro", "intro"),
        ("title_independent", "independent"),
        ("title_reading", "reading"),
        ("title_laboratory", "laboratory"),
        ("title_workshop", "workshop"),
        ("title_project", "project"),
    ):
        df[name] = title_l.str.contains(token, regex=False).astype(np.int8)

    cluster_list = df["CLUSTER_LIST"].fillna("").astype(str) if "CLUSTER_LIST" in df.columns else ""
    cluster_list = pd.Series(cluster_list, index=df.index)
    stripped = cluster_list.str.strip()
    df["cluster_n"] = np.where(stripped.eq("") | stripped.str.lower().isin(["nan", "none"]), 0, stripped.str.count(",") + 1)
    df["cluster_n"] = df["cluster_n"].astype(np.int16)

    for col in (
        "n_sections",
        "n_lecture",
        "n_lab",
        "n_recitation",
        "n_design",
        "n_evening",
        "n_with_time",
        "n_ose",
        "n_repeatable",
        "n_evaluate",
        "n_faculty_sec",
        "n_places",
    ):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["frac_lecture"] = df["n_lecture"] / df["n_sections"].clip(lower=1)
    df["frac_recitation"] = df["n_recitation"] / df["n_sections"].clip(lower=1)
    df["frac_evening"] = df["n_evening"] / df["n_sections"].clip(lower=1)
    df["frac_ose"] = df["n_ose"] / df["n_sections"].clip(lower=1)
    df["frac_repeatable"] = df["n_repeatable"] / df["n_sections"].clip(lower=1)
    df["frac_evaluate"] = df["n_evaluate"] / df["n_sections"].clip(lower=1)

    if "has_catalog" not in df.columns:
        df["has_catalog"] = 0
    df["has_catalog"] = pd.to_numeric(df["has_catalog"], errors="coerce").fillna(0)
    df["cat_variable_units"] = pd.to_numeric(df.get("cat_variable_units"), errors="coerce")
    df["cat_prereq_len"] = pd.to_numeric(df.get("cat_prereq_len"), errors="coerce")
    df["cat_hass"] = clean_cat(df["cat_hass"]) if "cat_hass" in df.columns else "MISSING"
    df["cat_gir"] = clean_cat(df["cat_gir"]) if "cat_gir" in df.columns else "MISSING"
    for col in ("lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    # Subject history: sort chronologically, shift so current term is never observed
    ordered = df.sort_values(["sid", "tk"], kind="mergesort")
    grouped = ordered.groupby("sid", sort=False)
    ordered["enr_lag1"] = grouped["enr"].shift(1)
    ordered["enr_lag2"] = grouped["enr"].shift(2)
    ordered["enr_lag3"] = grouped["enr"].shift(3)
    ordered["y_lag1"] = grouped["y"].shift(1)
    ordered["n_prior"] = grouped.cumcount()
    roll3 = grouped["enr_lag1"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    roll5 = grouped["enr_lag1"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    exp_max = grouped["enr_lag1"].expanding(min_periods=1).max().reset_index(level=0, drop=True)
    ordered["enr_roll3"] = roll3
    ordered["enr_roll5"] = roll5
    ordered["enr_exp_max"] = exp_max
    lag_cols = [
        "enr_lag1",
        "enr_lag2",
        "enr_lag3",
        "y_lag1",
        "n_prior",
        "enr_roll3",
        "enr_roll5",
        "enr_exp_max",
    ]
    df[lag_cols] = ordered[lag_cols]

    season_ordered = df.sort_values(["sid", "season", "tk"], kind="mergesort")
    season_grouped = season_ordered.groupby(["sid", "season"], sort=False)
    season_ordered["enr_lag_season"] = season_grouped["enr"].shift(1)
    season_ordered["y_lag_season"] = season_grouped["y"].shift(1)
    season_ordered["n_prior_season"] = season_grouped.cumcount()
    df[["enr_lag_season", "y_lag_season", "n_prior_season"]] = season_ordered[
        ["enr_lag_season", "y_lag_season", "n_prior_season"]
    ]

    df["sid_season"] = df["sid"] + "|" + df["season"]
    df["subj_prior_y_rate"], df["subj_prior_y_cnt"] = group_prior_mean(df, "sid", "y")
    df["subj_season_y_rate"], df["subj_season_y_cnt"] = group_prior_mean(df, "sid_season", "y")
    df["enr_exp_mean"], df["n_prior_enr"] = group_prior_mean(df, "sid", "enr")
    df["enr_season_mean"], _ = group_prior_mean(df, "sid_season", "enr")
    df["dept_prior_y_rate"], df["dept_prior_y_cnt"] = group_prior_mean(df, "dept", "y")
    df["prefix_prior_y_rate"], df["prefix_prior_y_cnt"] = group_prior_mean(df, "prefix", "y")
    faculty_key = df["faculty"].fillna("MISSING_FACULTY")
    df["_faculty_key"] = faculty_key
    fac_rate, fac_cnt = group_prior_mean(df, "_faculty_key", "y")
    df["fac_prior_y_rate"] = fac_rate.where(df["faculty"].notna())
    df["fac_prior_y_cnt"] = fac_cnt.where(df["faculty"].notna())

    # Department enrollment bar from strictly earlier terms only
    observed = df.loc[df["enr"].notna(), ["dept", "TERM_CODE", "tk", "season", "enr"]].copy()
    observed["pos"] = (observed["enr"] > 0).astype(np.int8)
    term_stats = (
        observed.groupby(["dept", "TERM_CODE"], sort=False)
        .agg(
            n=("enr", "size"),
            n_pos=("pos", "sum"),
            p75=("enr", lambda s: float(np.quantile(s[s > 0], 0.75)) if (s > 0).any() else np.nan),
            tk=("tk", "first"),
            season=("season", "first"),
        )
        .reset_index()
    )
    pieces = []
    for _, dept_rows in term_stats.groupby("dept", sort=False):
        dept_rows = dept_rows.sort_values("tk", kind="mergesort")
        shifted = dept_rows["p75"].shift(1)
        dept_rows["dept_prev_p75"] = shifted
        dept_rows["dept_exp_p75"] = shifted.expanding(min_periods=1).median()
        dept_rows["dept_prev_season_p75"] = dept_rows.groupby("season", sort=False)["p75"].shift(1)
        dept_rows["past_n"] = dept_rows.groupby("season", sort=False)["n"].cumsum() - dept_rows["n"]
        dept_rows["past_n_pos"] = dept_rows.groupby("season", sort=False)["n_pos"].cumsum() - dept_rows["n_pos"]
        pieces.append(dept_rows)
    term_stats = pd.concat(pieces, ignore_index=True) if pieces else term_stats
    term_stats["past_pos_rate"] = term_stats["past_n_pos"] / term_stats["past_n"].replace(0, np.nan)
    df = df.merge(
        term_stats[
            [
                "dept",
                "TERM_CODE",
                "dept_prev_p75",
                "dept_exp_p75",
                "dept_prev_season_p75",
                "past_pos_rate",
            ]
        ],
        on=["dept", "TERM_CODE"],
        how="left",
    )
    df["n_dept_term"] = df.groupby(["dept", "TERM_CODE"], sort=False)["sid"].transform("size")

    df["ratio_season_enr"] = df["enr_lag_season"] / df["dept_prev_season_p75"].replace(0, np.nan)
    df["ratio_exp_enr"] = df["enr_exp_mean"] / df["dept_exp_p75"].replace(0, np.nan)
    df["ratio_roll3"] = df["roll3_enrollment_mean"] / df["dept_exp_p75"].replace(0, np.nan)
    df["above_season_bar"] = np.where(
        df["enr_lag_season"].notna() & df["dept_prev_season_p75"].notna(),
        (df["enr_lag_season"] >= df["dept_prev_season_p75"]).astype(float),
        np.nan,
    )

    master = df["MASTER_SUBJECT_ID_SORT"].astype(str).str.strip() if "MASTER_SUBJECT_ID_SORT" in df.columns else df["sid"]
    df["is_own_master"] = (master == df["sid"]).astype(np.int8)
    master_lookup = df[["TERM_CODE", "sid", "enr_lag_season", "subj_prior_y_rate", "enr_exp_mean"]].rename(
        columns={
            "sid": "_master_sid",
            "enr_lag_season": "master_enr_lag_season",
            "subj_prior_y_rate": "master_prior_y_rate",
            "enr_exp_mean": "master_enr_exp_mean",
        }
    )
    df["_master_sid"] = master
    df = df.merge(master_lookup, on=["TERM_CODE", "_master_sid"], how="left")

    # Current-term enrollment is the target and MUST NOT remain as a model input.
    drop_cols = [
        c
        for c in (
            "SUBJECT_ENROLLMENT_FIFTH_WEEK",
            "SUBJECT_TITLE",
            "CLUSTER_LIST",
            "OFFER_DEPT_CODE",
            "OFFER_SCHOOL_NAME",
            "RESPONSIBLE_FACULTY_NAME",
            "HGN_CODE",
            "CLUSTER_TYPE",
            "TOTAL_UNITS",
            "LECTURE_UNITS",
            "LAB_UNITS",
            "PREPARATION_UNITS",
            "DEPARTMENT_CODE",
            "COURSE_NUMBER",
            "_faculty_key",
            "_master_sid",
        )
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)
    leaked = [
        c
        for c in df.columns
        if "ENROLLMENT" in c.upper() and c not in ("lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean")
    ]
    if leaked:
        raise RuntimeError(f"Current-enrollment columns left in feature frame: {leaked}")
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if col in META_COLUMNS or col.startswith("_"):
            continue
        cols.append(col)
    missing_cats = [c for c in CATEGORICAL_FEATURES if c not in cols]
    if missing_cats:
        raise RuntimeError(f"Missing categorical features: {missing_cats}")
    return cols


def prepare_matrix(
    df: pd.DataFrame, features: List[str], cat_maps: Optional[Dict] = None, fit: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    x = df[features].copy()
    maps = {} if fit else dict(cat_maps or {})
    for col in CATEGORICAL_FEATURES:
        values = clean_cat(x[col])
        if fit:
            categories = pd.Index(values.unique()).astype(str)
            if "MISSING" not in categories:
                categories = categories.append(pd.Index(["MISSING"]))
            maps[col] = categories
        known = pd.Index(maps[col])
        values = values.where(values.isin(known), "MISSING")
        x[col] = pd.Categorical(values, categories=list(known))
    for col in features:
        if col in CATEGORICAL_FEATURES:
            continue
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x, maps
