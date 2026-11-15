#!/usr/bin/env python3
"""Predict whether a course offering is high-enrollment.

High enrollment means the offering is in the top quartile of positive
enrollment among offerings in the same department and term.

Current-term enrollment is the quantity that defines the label, so it is
never used as a feature. Historical enrollment and past labels are shifted
so each row only sees earlier terms.

Development metric: macro F1 on a time holdout (latest train years).

Examples
--------
Train and report the time-holdout macro F1::

    python predict_high_enrollment.py

After the held-out tables and gold file are available, score them::

    python predict_high_enrollment.py \\
        --test-dir table_splits/test \\
        --gold-test table_splits/test/gold_enrollment_test.csv \\
        --output artifacts/test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_TRAIN_DIR = WORKSPACE / "table_splits" / "train"
ARTIFACT_DIR = WORKSPACE / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
FEATURE_CACHE = ARTIFACT_DIR / "train_features.parquet"
FEATURE_VERSION = 3

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
    year = term.str.slice(0, 4).astype(int)
    season = term.str.slice(4).map(SEASON_ORD).fillna(9).astype(int)
    return year * 10 + season


def clean_cat(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.fillna("MISSING")
    out = out.replace({"": "MISSING", "nan": "MISSING", "None": "MISSING", "<NA>": "MISSING"})
    return out.astype(str)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=["N", "Y"]))


def report_scores(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = f1_score(y_true, y_pred, average=None, labels=["N", "Y"])
    macro = float(np.mean(scores))
    print(
        f"{name:28s}  macro={macro:.4f}  "
        f"F1_N={scores[0]:.4f}  F1_Y={scores[1]:.4f}  "
        f"pred_Y={(y_pred == 'Y').mean():.3f}  true_Y={(y_true == 'Y').mean():.3f}"
    )
    return macro


def group_prior_mean(df: pd.DataFrame, key: str, value_col: str) -> tuple[pd.Series, pd.Series]:
    """Mean of past values within `key`, excluding the current row. NaNs are ignored."""
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


def _read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    kwargs = {"dtype": str, "keep_default_na": True}
    if usecols is not None:
        kwargs["usecols"] = usecols
    return pd.read_csv(path, **kwargs)


def _yn(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.upper() == "Y").astype(np.int8)


def load_offering_frame(data_dir: Path, gold_path: Path | None, split: str) -> pd.DataFrame:
    """One row per (TERM_CODE, SUBJECT_ID_SORT) with schedule attributes and enrollment."""
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
            keys["y"] = np.where(keys["HIGH_ENROLLMENT"].astype(str).str.upper() == "Y", 1.0, 0.0)
            keys.loc[keys["HIGH_ENROLLMENT"].isna(), "y"] = np.nan
            keys = keys.drop(columns=["HIGH_ENROLLMENT"])
        else:
            keys["y"] = np.nan
        frame = keys.merge(summary, on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left")
    else:
        frame = summary.copy()
        frame["y"] = np.nan

    enroll_path = data_dir / "SUBJECT_SUMMARY.csv"
    if enroll_path.exists():
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


def _attach_section_features(frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    path = data_dir / "SUBJECT_OFFERED.csv"
    if not path.exists():
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


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add schedule features and time-shifted history. Drops raw current enrollment."""
    df = frame.copy()
    df["TERM_CODE"] = df["TERM_CODE"].astype(str).str.strip()
    df["SUBJECT_ID_SORT"] = df["SUBJECT_ID_SORT"].astype(str).str.strip()
    df = df.reset_index(drop=True)

    df["tk"] = term_key(df["TERM_CODE"])
    df["year"] = df["TERM_CODE"].str.slice(0, 4).astype(int)
    df["season"] = df["TERM_CODE"].str.slice(4)
    df.loc[~df["season"].isin(SEASON_ORD), "season"] = "NA"
    df["sid"] = df["SUBJECT_ID_SORT"]
    df["enr"] = pd.to_numeric(df["SUBJECT_ENROLLMENT_FIFTH_WEEK"], errors="coerce")

    dept = clean_cat(df["OFFER_DEPT_CODE"])
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

    # Subject history. Sort once and write lagged enrollment back by index.
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

    # Department enrollment bar from earlier terms only (same term's enrollments are excluded).
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
    term_stats = pd.concat(pieces, ignore_index=True)
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

    # Current-term enrollment must not remain as a model input.
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
    leaked = [c for c in df.columns if "ENROLLMENT" in c.upper() and c not in ("lag1_enrollment", "lag2_enrollment", "roll3_enrollment_mean")]
    if leaked:
        raise RuntimeError(f"Current-enrollment columns left in the frame: {leaked}")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_COLUMNS or col.startswith("_"):
            continue
        cols.append(col)
    missing_cats = [c for c in CATEGORICAL_FEATURES if c not in cols]
    if missing_cats:
        raise RuntimeError(f"Missing categorical features: {missing_cats}")
    return cols


def recency_weights(year: pd.Series, year_max: int) -> np.ndarray:
    """Half-life of six years so recent offerings count more than early history."""
    return np.exp(-np.log(2.0) * (year_max - year.to_numpy()) / 6.0)


def prepare_matrix(df: pd.DataFrame, features: list[str], cat_maps: dict | None, fit: bool) -> tuple[pd.DataFrame, dict]:
    x = df[features].copy()
    maps = {} if fit else dict(cat_maps)
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


# Fixed-capacity booster. A short early-stopping run underfit this task;
# 500 trees of 31 leaves was the best 2017 setting among the configurations tried.
N_TREES = 500
NUM_LEAVES = 31


def fit_classifier(x_train, y_train, w_train) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=N_TREES,
        learning_rate=0.05,
        num_leaves=NUM_LEAVES,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    return model


def top_fraction_predict(score: np.ndarray, positive_rate: float) -> np.ndarray:
    """Label the highest-scoring fraction of rows as high enrollment.

    Using a fraction instead of a raw probability cutoff keeps the decision
    stable when the model is refit and its score scale shifts.
    """
    score = np.asarray(score, dtype=float)
    pred = np.full(len(score), "N", dtype=object)
    if len(score) == 0 or positive_rate <= 0:
        return pred
    k = int(round(positive_rate * len(score)))
    k = min(max(k, 1), len(score))
    chosen = np.argpartition(-score, k - 1)[:k]
    pred[chosen] = "Y"
    return pred


def labels_yn(y: pd.Series) -> np.ndarray:
    return np.where(y.to_numpy() == 1, "Y", "N")


def best_positive_rate(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    best_rate, best_score = 0.07, -1.0
    for rate in np.linspace(0.035, 0.12, 86):
        macro = macro_f1(y_true, top_fraction_predict(score, float(rate)))
        if macro > best_score:
            best_rate, best_score = float(rate), macro
    return best_rate, best_score


def build_training_frame(train_dir: Path, use_cache: bool) -> pd.DataFrame:
    if use_cache and FEATURE_CACHE.exists():
        cached = pd.read_parquet(FEATURE_CACHE)
        version = cached.attrs.get("feature_version") if hasattr(cached, "attrs") else None
        # Parquet metadata is stored beside the cache because pandas attrs are not always kept.
        version_path = FEATURE_CACHE.with_suffix(".version")
        cached_version = version_path.read_text().strip() if version_path.exists() else ""
        if cached_version == str(FEATURE_VERSION):
            print(f"Loaded cached features {FEATURE_CACHE}")
            return cached
    print("Building training features (current-term enrollment is not a feature)...")
    raw = load_offering_frame(train_dir, train_dir / "gold_enrollment_train.csv", split="train")
    featured = add_features(raw)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(FEATURE_CACHE, index=False)
    FEATURE_CACHE.with_suffix(".version").write_text(str(FEATURE_VERSION))
    print(f"Cached features to {FEATURE_CACHE}")
    return featured


def train_and_validate(train_dir: Path, use_cache: bool) -> dict:
    df = build_training_frame(train_dir, use_cache=use_cache)
    features = feature_columns(df)
    labeled = df[df["y"].notna()].copy()
    print(f"Rows={len(labeled):,}  features={len(features)}  positive rate={labeled['y'].mean():.3f}")

    # Sanity: same-season label stickiness should be a strong baseline if shifts are correct.
    hold_mask = labeled["year"] >= 2017
    season_pred = np.where(labeled.loc[hold_mask, "y_lag_season"].to_numpy() == 1, "Y", "N")
    either_pred = np.where(
        labeled.loc[hold_mask, "y_lag_season"].to_numpy() == 1,
        "Y",
        np.where(labeled.loc[hold_mask, "y_lag1"].to_numpy() == 1, "Y", "N"),
    )
    print("Baselines on 2017-2018 (rules use only earlier labels):")
    report_scores("same-season previous Y", labels_yn(labeled.loc[hold_mask, "y"]), season_pred)
    report_scores("previous or same-season Y", labels_yn(labeled.loc[hold_mask, "y"]), either_pred)
    corr = labeled[["roll3_enrollment_mean", "enr_roll3"]].corr().iloc[0, 1]
    print(f"Correlation of provided roll3 with shifted 3-term mean: {corr:.3f}")

    tune_mask = labeled["year"] == 2017
    dev_mask = labeled["year"] >= 2018
    fit_mask = labeled["year"] <= 2016

    x_all, cat_maps = prepare_matrix(labeled, features, cat_maps=None, fit=True)
    y_all = labeled["y"].to_numpy()
    print(f"Fitting classifier on years <= 2016 ({N_TREES} trees, {NUM_LEAVES} leaves)...")
    cls = fit_classifier(
        x_all.loc[fit_mask],
        y_all[fit_mask.to_numpy()],
        recency_weights(labeled.loc[fit_mask, "year"], 2016),
    )
    p_tune = cls.predict_proba(x_all.loc[tune_mask])[:, 1]
    p_dev = cls.predict_proba(x_all.loc[dev_mask])[:, 1]
    p_hold = cls.predict_proba(x_all.loc[hold_mask])[:, 1]
    y_tune = labels_yn(labeled.loc[tune_mask, "y"])
    y_dev = labels_yn(labeled.loc[dev_mask, "y"])
    positive_rate, tune_macro = best_positive_rate(y_tune, p_tune)
    print(f"Positive rate chosen on 2017: {positive_rate:.3f}  (tune macro F1 {tune_macro:.4f})")
    tune_pred = top_fraction_predict(p_tune, positive_rate)
    dev_pred = top_fraction_predict(p_dev, positive_rate)
    hold_pred = top_fraction_predict(p_hold, positive_rate)
    print("Holdout from the <=2016 model. The positive rate was chosen on 2017 only:")
    report_scores("2017 (tune)", y_tune, tune_pred)
    dev_macro = report_scores("2018", y_dev, dev_pred)
    hold_macro = report_scores("2017-2018", labels_yn(labeled.loc[hold_mask, "y"]), hold_pred)

    print("Refitting on all training years for the saved model...")
    year_max = int(labeled["year"].max())
    cls_final = fit_classifier(
        x_all,
        y_all,
        recency_weights(labeled["year"], year_max),
    )
    importance = pd.Series(cls_final.feature_importances_, index=features).sort_values(ascending=False)
    print("Top classifier features:")
    print(importance.head(15).to_string())

    bundle = {
        "classifier": cls_final,
        "positive_rate": positive_rate,
        "features": features,
        "categorical": CATEGORICAL_FEATURES,
        "cat_maps": {col: [str(v) for v in list(cat_maps[col])] for col in CATEGORICAL_FEATURES},
        "feature_version": FEATURE_VERSION,
        "n_estimators": N_TREES,
        "num_leaves": NUM_LEAVES,
        "metrics": {
            "dev_2018_macro_f1": dev_macro,
            "tune_2017_macro_f1": tune_macro,
            "holdout_2017_2018_macro_f1": hold_macro,
            "positive_rate": positive_rate,
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    report_path = ARTIFACT_DIR / "validation_report.json"
    report_path.write_text(json.dumps(bundle["metrics"], indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Development macro F1 on 2018: {dev_macro:.4f}")
    return bundle


def build_test_frame(train_dir: Path, test_dir: Path, gold_test: Path | None) -> pd.DataFrame:
    """Features for test keys. History includes train rows; test labels are never read into history."""
    train_raw = load_offering_frame(train_dir, train_dir / "gold_enrollment_train.csv", split="train")
    test_gold = gold_test if gold_test is not None and gold_test.exists() else None
    # Labels in the test gold file are for evaluation only. Keep them off the frame
    # until scoring so shifted label history cannot see them.
    test_raw = load_offering_frame(test_dir, None, split="test")
    test_keys = test_raw[["TERM_CODE", "SUBJECT_ID_SORT"]].drop_duplicates().assign(_drop=1)
    train_raw = train_raw.merge(test_keys, on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left")
    train_raw = train_raw[train_raw["_drop"].isna()].drop(columns=["_drop"])
    if test_gold is not None:
        gold = _read_csv(test_gold)
        gold["TERM_CODE"] = gold["TERM_CODE"].astype(str).str.strip()
        gold["SUBJECT_ID_SORT"] = gold["SUBJECT_ID_SORT"].astype(str).str.strip()
        keys = gold[["TERM_CODE", "SUBJECT_ID_SORT"]].drop_duplicates()
        test_raw = keys.merge(test_raw.drop(columns=["y"], errors="ignore"), on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left")
        test_raw["y"] = np.nan
        test_raw["split"] = "test"
    combined = pd.concat([train_raw, test_raw], ignore_index=True)
    featured = add_features(combined)
    return featured


def score_test(bundle: dict, train_dir: Path, test_dir: Path, gold_test: Path, output: Path) -> None:
    print(f"Scoring {gold_test} using tables in {test_dir}")
    featured = build_test_frame(train_dir, test_dir, gold_test)
    test = featured[featured["split"] == "test"].copy()
    if test.empty:
        raise RuntimeError(f"No test rows found in {test_dir}")
    x, _ = prepare_matrix(test, bundle["features"], bundle["cat_maps"], fit=False)
    scores = bundle["classifier"].predict_proba(x)[:, 1]
    pred = top_fraction_predict(scores, bundle["positive_rate"])
    result = pd.DataFrame(
        {
            "TERM_CODE": test["TERM_CODE"].to_numpy(),
            "SUBJECT_ID_SORT": test["SUBJECT_ID_SORT"].to_numpy(),
            "HIGH_ENROLLMENT": pred,
        }
    )
    gold = _read_csv(gold_test)
    gold["TERM_CODE"] = gold["TERM_CODE"].astype(str).str.strip()
    gold["SUBJECT_ID_SORT"] = gold["SUBJECT_ID_SORT"].astype(str).str.strip()
    # One prediction per gold key, in gold order.
    merged = gold.merge(result, on=["TERM_CODE", "SUBJECT_ID_SORT"], how="left", suffixes=("_true", "_pred"))
    if "HIGH_ENROLLMENT_pred" in merged.columns:
        prediction_col = "HIGH_ENROLLMENT_pred"
        truth_col = "HIGH_ENROLLMENT_true"
    else:
        prediction_col = "HIGH_ENROLLMENT"
        truth_col = None
    missing = merged[prediction_col].isna().sum() if prediction_col in merged.columns else len(merged)
    if missing:
        print(f"Warning: {missing} gold keys had no feature row; predicting N")
        merged[prediction_col] = merged[prediction_col].fillna("N")
    out = merged[["TERM_CODE", "SUBJECT_ID_SORT"]].copy()
    out["HIGH_ENROLLMENT"] = merged[prediction_col].to_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Wrote {len(out):,} predictions to {output}")
    if truth_col and truth_col in merged.columns and merged[truth_col].notna().any():
        y_true = merged[truth_col].astype(str).str.upper().to_numpy()
        y_pred = out["HIGH_ENROLLMENT"].to_numpy()
        report_scores("test gold", y_true, y_pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict high course enrollment (top quartile within department and term).")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=None, help="Held-out tables. Not read during training.")
    parser.add_argument("--gold-test", type=Path, default=None, help="gold_enrollment_test.csv with TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "test_predictions.csv")
    parser.add_argument("--use-cache", action="store_true", help="Reuse artifacts/train_features.parquet when the feature version matches.")
    parser.add_argument("--skip-train", action="store_true", help="Load artifacts/model.joblib instead of refitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_dir.exists():
        raise SystemExit(f"Train directory not found: {args.train_dir}")
    if args.skip_train:
        if not MODEL_PATH.exists():
            raise SystemExit(f"No saved model at {MODEL_PATH}")
        bundle = joblib.load(MODEL_PATH)
    else:
        bundle = train_and_validate(args.train_dir, use_cache=args.use_cache)
    if args.test_dir is not None or args.gold_test is not None:
        if args.test_dir is None or args.gold_test is None:
            raise SystemExit("Provide both --test-dir and --gold-test to score the held-out split.")
        if not args.test_dir.exists():
            raise SystemExit(f"Test directory not found: {args.test_dir}")
        if not args.gold_test.exists():
            raise SystemExit(f"Gold test file not found: {args.gold_test}")
        score_test(bundle, args.train_dir, args.test_dir, args.gold_test, args.output)


if __name__ == "__main__":
    main()
