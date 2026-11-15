#!/usr/bin/env python3
"""
NYC Open Data Parking Violation Count Predictor
------------------------------------------------
Predicts future parking violations per (street_name, violation_type) pair
using 2022 aggregate data and multi-dataset feature augmentation from corpus/.

Metric: RMSE
Author: Agent
"""

import os
import sys
import re
import argparse
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split


def extract_base_street(s):
    """Extract clean base street name by stripping directions, intersections, etc."""
    if not isinstance(s, str):
        return ""
    # Strip intersection delimiter @, &, /, CROSS and trailing tokens
    s = re.split(r'[@&/]|(?:\bCROSS\b)', s, flags=re.IGNORECASE)[0].strip()
    # Strip directional prefixes e.g. 'WB N CONDUIT AVE' -> 'N CONDUIT AVE'
    s = re.sub(r'^(?:WB|EB|NB|SB|W/B|E/B|N/B|S/B)\s+', '', s, flags=re.IGNORECASE)
    # Strip directional suffixes e.g. 'ASTORIA BLVD N' -> 'ASTORIA BLVD'
    s = re.sub(r'\s+(?:WB|EB|NB|SB|W/B|E/B|N/B|S/B)$', '', s, flags=re.IGNORECASE)
    return s.strip()


def normalize_street(s):
    """Normalize street name into canonical form for cross-table joins."""
    if not isinstance(s, str):
        return ""
    s = extract_base_street(s)
    s = s.upper().strip()
    s = re.sub(r'\s+', ' ', s)

    # Word numbers to digits
    word_to_num = {
        r'\bFIRST\b': '1', r'\bSECOND\b': '2', r'\bTHIRD\b': '3',
        r'\bFOURTH\b': '4', r'\bFIFTH\b': '5', r'\bSIXTH\b': '6',
        r'\bSEVENTH\b': '7', r'\bEIGHTH\b': '8', r'\bNINTH\b': '9',
        r'\bTENTH\b': '10', r'\bELEVENTH\b': '11', r'\bTWELFTH\b': '12'
    }
    for pat, rep in word_to_num.items():
        s = re.sub(pat, rep, s)

    # Strip ordinals: 14TH -> 14, 1ST -> 1
    s = re.sub(r'(\d+)(?:ST|ND|RD|TH)\b', r'\1', s)

    # Standardize street suffixes
    suffix_map = {
        r'\bAVE\b|\bAV\b': 'AVENUE',
        r'\bST\b|\bSTR\b': 'STREET',
        r'\bBLVD\b|\bBLV\b': 'BOULEVARD',
        r'\bPKWY\b|\bPWKY\b': 'PARKWAY',
        r'\bEXPY\b|\bEXPWY\b|\bEXP\b|\bEX\b': 'EXPRESSWAY',
        r'\bRD\b': 'ROAD',
        r'\bDR\b': 'DRIVE',
        r'\bPL\b': 'PLACE',
        r'\bCT\b': 'COURT',
        r'\bLN\b': 'LANE',
        r'\bTER\b|\bTERR\b': 'TERRACE',
        r'\bCIR\b': 'CIRCLE',
        r'\bN\b|\bN\.\b': 'NORTH',
        r'\bS\b|\bS\.\b': 'SOUTH',
        r'\bE\b|\bE\.\b': 'EAST',
        r'\bW\b|\bW\.\b': 'WEST',
    }
    for pat, rep in suffix_map.items():
        s = re.sub(pat, rep, s)

    # Strip special punctuation
    s = re.sub(r'[^A-Z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_violation(v):
    """Normalize violation description for robust key lookup."""
    if not isinstance(v, str):
        return ""
    v = v.upper().strip()
    v = re.sub(r'\s+', ' ', v)
    return v


def build_corpus_features(corpus_dir, cache_path=None):
    """
    Extract and aggregate external street-level features from NYC Open Data corpus.
    Caches aggregated dataframe to accelerate repeated pipeline executions.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data
        except Exception as e:
            print(f"[Warning] Could not load corpus cache ({e}); recomputing...")

    print(f"[*] Discovering and aggregating external street features from {corpus_dir}...")
    t0 = time.time()
    corpus_dfs = []

    # 1. Parking Meters
    meters_file = os.path.join(corpus_dir, 'parking_meters_locations_and_status__693u-uax6.parquet')
    if os.path.exists(meters_file):
        try:
            m_df = pd.read_parquet(meters_file, columns=['On_Street', 'Status'])
            m_df['norm_st'] = m_df['On_Street'].apply(normalize_street)
            m_agg = m_df.groupby('norm_st').agg(
                meter_count=('On_Street', 'count'),
                active_meter_count=('Status', lambda x: int((x == 'Active').sum()))
            ).reset_index()
            corpus_dfs.append(m_agg)
        except Exception as e:
            print(f"  [!] Failed loading parking meters: {e}")

    # 2. Street Pavement Ratings
    pave_file = os.path.join(corpus_dir, 'street_pavement_ratings__6yyb-pb25.parquet')
    if os.path.exists(pave_file):
        try:
            p_df = pd.read_parquet(pave_file, columns=['OnStreetName', 'BoroughName', 'Road_Type', 
                                                       'LocationGeometry.STLength', 'SystemRating'])
            p_df['norm_st'] = p_df['OnStreetName'].apply(normalize_street)
            p_agg = p_df.groupby('norm_st').agg(
                pave_segments=('OnStreetName', 'count'),
                pave_length_sum=('LocationGeometry.STLength', 'sum'),
                pave_length_mean=('LocationGeometry.STLength', 'mean'),
                pave_rating_mean=('SystemRating', 'mean'),
                pave_is_main=('Road_Type', lambda x: int((x == 'Main').any())),
                pave_is_service=('Road_Type', lambda x: int((x == 'Service').any())),
                pave_borough=('BoroughName', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown')
            ).reset_index()
            corpus_dfs.append(p_agg)
        except Exception as e:
            print(f"  [!] Failed loading street pavement: {e}")

    # 3. Speed Limits
    speed_file = os.path.join(corpus_dir, 'vzv_speed_limits__5mad-ntua.parquet')
    if os.path.exists(speed_file):
        try:
            s_df = pd.read_parquet(speed_file, columns=['street', 'postvz_sl', 'Shape_Leng'])
            s_df['norm_st'] = s_df['street'].apply(normalize_street)
            s_agg = s_df.groupby('norm_st').agg(
                speed_segments=('street', 'count'),
                speed_length_sum=('Shape_Leng', 'sum'),
                speed_limit_mean=('postvz_sl', 'mean'),
                speed_limit_max=('postvz_sl', 'max')
            ).reset_index()
            corpus_dfs.append(s_agg)
        except Exception as e:
            print(f"  [!] Failed loading speed limits: {e}")

    # 4. Facilities (Schools, Police, Emergency)
    fac_file = os.path.join(corpus_dir, 'facilities__67g2-p84d.parquet')
    if os.path.exists(fac_file):
        try:
            f_df = pd.read_parquet(fac_file, columns=['STREETNAME', 'FACGROUP'])
            f_df['norm_st'] = f_df['STREETNAME'].apply(normalize_street)
            f_agg = f_df.groupby('norm_st').agg(
                fac_count=('STREETNAME', 'count'),
                fac_schools=('FACGROUP', lambda x: int(x.str.contains('EDUCATION|SCHOOL|YOUTH', case=False, na=False).sum())),
                fac_safety=('FACGROUP', lambda x: int(x.str.contains('SAFETY|POLICE|FIRE|EMERGENCY', case=False, na=False).sum()))
            ).reset_index()
            corpus_dfs.append(f_agg)
        except Exception as e:
            print(f"  [!] Failed loading facilities: {e}")

    # 5. Speed Reducers (Speed humps / cushions)
    srts_file = os.path.join(corpus_dir, 'speed_reducer_tracking_system_srts__9n6h-pt9g.parquet')
    if os.path.exists(srts_file):
        try:
            sr_df = pd.read_parquet(srts_file, columns=['OnStreet'])
            sr_df['norm_st'] = sr_df['OnStreet'].apply(normalize_street)
            sr_agg = sr_df.groupby('norm_st').agg(speed_humps=('OnStreet', 'count')).reset_index()
            corpus_dfs.append(sr_agg)
        except Exception as e:
            print(f"  [!] Failed loading speed reducers: {e}")

    # 6. Bicycle Parking
    bike_file = os.path.join(corpus_dir, 'bicycle_parking__592z-n7dk.parquet')
    if os.path.exists(bike_file):
        try:
            b_df = pd.read_parquet(bike_file, columns=['OnStreet'])
            b_df['norm_st'] = b_df['OnStreet'].apply(normalize_street)
            b_agg = b_df.groupby('norm_st').agg(bike_racks=('OnStreet', 'count')).reset_index()
            corpus_dfs.append(b_agg)
        except Exception as e:
            print(f"  [!] Failed loading bicycle parking: {e}")

    # 7. DOB Violations
    dob_file = os.path.join(corpus_dir, 'dob_violations__3h2n-5cm9.parquet')
    if os.path.exists(dob_file):
        try:
            dob_df = pd.read_parquet(dob_file, columns=['STREET'])
            dob_df['norm_st'] = dob_df['STREET'].apply(normalize_street)
            dob_agg = dob_df.groupby('norm_st').agg(dob_viols=('STREET', 'count')).reset_index()
            corpus_dfs.append(dob_agg)
        except Exception as e:
            print(f"  [!] Failed loading DOB violations: {e}")

    # 8. Sidewalk Violations
    sw_file = os.path.join(corpus_dir, 'sidewalk_management_database_violations__6kbp-uz6m.parquet')
    if os.path.exists(sw_file):
        try:
            sw_df = pd.read_parquet(sw_file, columns=['ONSTNAME'])
            sw_df['norm_st'] = sw_df['ONSTNAME'].apply(normalize_street)
            sw_agg = sw_df.groupby('norm_st').agg(sidewalk_viols=('ONSTNAME', 'count')).reset_index()
            corpus_dfs.append(sw_agg)
        except Exception as e:
            print(f"  [!] Failed loading sidewalk violations: {e}")

    # Merge all corpus tables on norm_st
    if corpus_dfs:
        merged_corpus = corpus_dfs[0]
        for c_df in corpus_dfs[1:]:
            merged_corpus = merged_corpus.merge(c_df, on='norm_st', how='outer')
    else:
        merged_corpus = pd.DataFrame({'norm_st': []})

    print(f"[*] Corpus processing completed in {time.time()-t0:.2f}s ({len(merged_corpus)} unique streets augmented).")

    if cache_path:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(merged_corpus, f)
        except Exception as e:
            print(f"  [!] Failed caching corpus dataframe: {e}")

    return merged_corpus


def extract_features(df, corpus_features, st_history_map=None, is_train=False):
    """
    Engineer comprehensive tabular features combining street text, violation semantics,
    external corpus enrichments, and historical street-level context.
    """
    d = df.reset_index(drop=True).copy()

    # Column name flexibility
    street_col = [c for c in d.columns if c.lower() in ['street name', 'street_name', 'street']][0]
    viol_col = [c for c in d.columns if c.lower() in ['violation description', 'violation_type', 'violation']][0]

    # 1. Street Name Normalized for joins
    d['norm_st'] = d[street_col].astype(str).apply(normalize_street)

    # 2. Join Corpus external features
    if corpus_features is not None and len(corpus_features) > 0:
        d = d.merge(corpus_features, on='norm_st', how='left')

    # Re-extract series after merge to guarantee 100% index alignment
    st_raw = d[street_col].astype(str)
    vi_raw = d[viol_col].astype(str)

    # 3. Street Name Text Features
    d['is_intersection'] = st_raw.str.contains(r'[@&/]|(?:\bCROSS\b)', regex=True, case=False).fillna(False).astype(int)
    d['dir_wb'] = st_raw.str.contains(r'\b(?:WB|W/B)\b', regex=True, case=False).fillna(False).astype(int)
    d['dir_eb'] = st_raw.str.contains(r'\b(?:EB|E/B)\b', regex=True, case=False).fillna(False).astype(int)
    d['dir_nb'] = st_raw.str.contains(r'\b(?:NB|N/B)\b', regex=True, case=False).fillna(False).astype(int)
    d['dir_sb'] = st_raw.str.contains(r'\b(?:SB|S/B)\b', regex=True, case=False).fillna(False).astype(int)
    d['dir_bracket_nb'] = st_raw.str.contains(r'\([NESW]/B\)', regex=True, case=False).fillna(False).astype(int)
    d['has_camera_dir'] = ((d['dir_wb'] + d['dir_eb'] + d['dir_nb'] + d['dir_sb'] + d['dir_bracket_nb']) > 0).astype(int)

    d['has_digits'] = st_raw.str.contains(r'\d', regex=True).fillna(False).astype(int)
    d['name_len'] = st_raw.str.len().fillna(0).astype(int)
    d['word_count'] = st_raw.str.split().apply(len).fillna(0).astype(int)

    # Street Suffix Indicators
    d['is_ave'] = st_raw.str.contains(r'\b(?:AVE|AVENUE)\b', regex=True, case=False).fillna(False).astype(int)
    d['is_st'] = st_raw.str.contains(r'\b(?:ST|STREET)\b', regex=True, case=False).fillna(False).astype(int)
    d['is_blvd'] = st_raw.str.contains(r'\b(?:BLVD|BOULEVARD)\b', regex=True, case=False).fillna(False).astype(int)
    d['is_pkwy'] = st_raw.str.contains(r'\b(?:PKWY|PARKWAY)\b', regex=True, case=False).fillna(False).astype(int)
    d['is_expy'] = st_raw.str.contains(r'\b(?:EXP|EXPWY|EXPRESSWAY)\b', regex=True, case=False).fillna(False).astype(int)
    d['is_rd'] = st_raw.str.contains(r'\b(?:RD|ROAD)\b', regex=True, case=False).fillna(False).astype(int)

    # Prominent NYC corridors
    d['corridor_conduit'] = st_raw.str.contains('CONDUIT', case=False).fillna(False).astype(int)
    d['corridor_broadway'] = st_raw.str.contains('BROADWAY', case=False).fillna(False).astype(int)
    d['corridor_cross_bay'] = st_raw.str.contains('CROSS BAY', case=False).fillna(False).astype(int)
    d['corridor_bruckner'] = st_raw.str.contains('BRUCKNER', case=False).fillna(False).astype(int)
    d['corridor_horace'] = st_raw.str.contains('HORACE HARDING', case=False).fillna(False).astype(int)
    d['corridor_woodhaven'] = st_raw.str.contains('WOODHAVEN', case=False).fillna(False).astype(int)
    d['corridor_northern'] = st_raw.str.contains('NORTHERN', case=False).fillna(False).astype(int)
    d['corridor_atlantic'] = st_raw.str.contains('ATLANTIC', case=False).fillna(False).astype(int)
    d['corridor_14th'] = st_raw.str.contains(r'14(?:TH)?\s+(?:ST|STREET)', regex=True, case=False).fillna(False).astype(int)

    # 4. Violation Description Semantic Features
    d['v_is_speed'] = vi_raw.str.contains(r'SPEED|SCHOOL\s+ZN', regex=True, case=False).fillna(False).astype(int)
    d['v_is_bus'] = vi_raw.str.contains(r'BUS\s+LANE', regex=True, case=False).fillna(False).astype(int)
    d['v_is_red_light'] = vi_raw.str.contains(r'RED\s+LIGHT', regex=True, case=False).fillna(False).astype(int)
    d['v_is_meter'] = vi_raw.str.contains(r'METER', regex=True, case=False).fillna(False).astype(int)
    d['v_is_hydrant'] = vi_raw.str.contains(r'HYDRANT', regex=True, case=False).fillna(False).astype(int)
    d['v_is_clean'] = vi_raw.str.contains(r'STREET\s+CLEAN|CLEAN', regex=True, case=False).fillna(False).astype(int)
    d['v_is_standing'] = vi_raw.str.contains(r'STANDING', regex=True, case=False).fillna(False).astype(int)
    d['v_is_parking'] = vi_raw.str.contains(r'PARKING', regex=True, case=False).fillna(False).astype(int)
    d['v_is_sticker'] = vi_raw.str.contains(r'STICKER|EXPIRED|INSP|REG', regex=True, case=False).fillna(False).astype(int)
    d['v_is_camera'] = ((d['v_is_speed'] + d['v_is_bus'] + d['v_is_red_light']) > 0).astype(int)

    # 5. Semantic Interaction Features
    d['speed_cam_at_intersection'] = (d['v_is_speed'] * d['is_intersection']).fillna(0).astype(int)
    d['speed_cam_with_dir'] = (d['v_is_speed'] * d['has_camera_dir']).fillna(0).astype(int)
    d['bus_lane_at_intersection'] = (d['v_is_bus'] * d['is_intersection']).fillna(0).astype(int)
    d['red_light_at_intersection'] = (d['v_is_red_light'] * d['is_intersection']).fillna(0).astype(int)

    # 6. Historical Street-Level Context (Leave-One-Out for training, full for inference)
    target_col = [c for c in d.columns if c.lower() in ['violation_count', 'target']]
    has_target = len(target_col) > 0 and is_train

    if is_train and has_target:
        # In training: calculate out-of-fold / leave-one-out statistics
        y_vals = d[target_col[0]].values
        st_sum = d.groupby(street_col)[target_col[0]].transform('sum')
        st_cnt = d.groupby(street_col)[target_col[0]].transform('count')

        d['st_other_viols_sum'] = np.maximum(0, st_sum - y_vals)
        d['st_other_viols_cnt'] = np.maximum(0, st_cnt - 1)
        d['st_other_viols_mean'] = np.where(d['st_other_viols_cnt'] > 0, 
                                            d['st_other_viols_sum'] / d['st_other_viols_cnt'], 0.0)
        d['st_has_history'] = (d['st_other_viols_cnt'] > 0).astype(int)
    else:
        # In validation/test: look up street stats from training history map
        if st_history_map is not None:
            hist_df = st_history_map
            d = d.merge(hist_df, on=street_col, how='left')
            d['st_other_viols_sum'] = d['hist_st_sum'].fillna(0.0)
            d['st_other_viols_cnt'] = d['hist_st_cnt'].fillna(0.0)
            d['st_other_viols_mean'] = d['hist_st_mean'].fillna(0.0)
            d['st_has_history'] = d['hist_st_cnt'].notnull().astype(int)
        else:
            d['st_other_viols_sum'] = 0.0
            d['st_other_viols_cnt'] = 0.0
            d['st_other_viols_mean'] = 0.0
            d['st_has_history'] = 0

    # Categorical and missing value handling
    d['violation_category'] = vi_raw.astype('category')
    if 'pave_borough' in d.columns:
        d['pave_borough'] = d['pave_borough'].fillna('Missing').astype('category')
    else:
        d['pave_borough'] = pd.Series(['Missing'] * len(d)).astype('category')

    # Interaction with corpus
    if 'meter_count' in d.columns:
        d['meter_viol_meters'] = d['v_is_meter'] * d['meter_count'].fillna(0)
    else:
        d['meter_viol_meters'] = 0.0

    if 'pave_length_sum' in d.columns:
        d['clean_viol_pave_len'] = d['v_is_clean'] * d['pave_length_sum'].fillna(0)
        d['hydrant_viol_pave_len'] = d['v_is_hydrant'] * d['pave_length_sum'].fillna(0)
    else:
        d['clean_viol_pave_len'] = 0.0
        d['hydrant_viol_pave_len'] = 0.0

    return d


FEATURE_COLS = [
    'violation_category', 'pave_borough', 'is_intersection',
    'dir_wb', 'dir_eb', 'dir_nb', 'dir_sb', 'dir_bracket_nb', 'has_camera_dir',
    'has_digits', 'name_len', 'word_count',
    'is_ave', 'is_st', 'is_blvd', 'is_pkwy', 'is_expy', 'is_rd',
    'corridor_conduit', 'corridor_broadway', 'corridor_cross_bay', 'corridor_bruckner',
    'corridor_horace', 'corridor_woodhaven', 'corridor_northern', 'corridor_atlantic', 'corridor_14th',
    'v_is_speed', 'v_is_bus', 'v_is_red_light', 'v_is_meter', 'v_is_hydrant',
    'v_is_clean', 'v_is_standing', 'v_is_parking', 'v_is_sticker', 'v_is_camera',
    'speed_cam_at_intersection', 'speed_cam_with_dir', 'bus_lane_at_intersection', 'red_light_at_intersection',
    'meter_count', 'active_meter_count', 'pave_segments', 'pave_length_sum',
    'pave_length_mean', 'pave_rating_mean', 'pave_is_main', 'pave_is_service',
    'speed_segments', 'speed_length_sum', 'speed_limit_mean', 'speed_limit_max',
    'fac_count', 'fac_schools', 'fac_safety', 'speed_humps', 'bike_racks',
    'dob_viols', 'sidewalk_viols',
    'st_other_viols_sum', 'st_other_viols_cnt', 'st_other_viols_mean', 'st_has_history',
    'meter_viol_meters', 'clean_viol_pave_len', 'hydrant_viol_pave_len'
]


def train_lgb_model(X_train, y_train, seed=42):
    """Train a gradient boosted regressor optimizing RMSE directly."""
    # Ensure all feature columns exist
    X = X_train.copy()
    for col in FEATURE_COLS:
        if col not in X.columns:
            if col in ['violation_category', 'pave_borough']:
                X[col] = pd.Series(['Missing'] * len(X)).astype('category')
            else:
                X[col] = 0.0

    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=350,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X[FEATURE_COLS], y_train)
    return model


def compute_rmse(y_true, y_pred):
    """Compute un-weighted Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def run_development_validation(train_df, corpus_features, seed=42):
    """
    Run 2022-based development validation (grouped / holdout split) as specified in goal.txt.
    Reports development validation RMSE against baseline.
    """
    print("\n" + "=" * 70)
    print("DEVELOPMENT VALIDATION (2022 Data)")
    print("=" * 70)

    # 80/20 holdout split
    tr_subset, val_subset = train_test_split(train_df, test_size=0.20, random_state=seed)

    # Build historical street map from training split
    st_hist = tr_subset.groupby('Street Name')['violation_count'].agg(
        hist_st_sum='sum',
        hist_st_cnt='count',
        hist_st_mean='mean'
    ).reset_index()

    # Build features
    X_tr = extract_features(tr_subset, corpus_features, is_train=True)
    X_val = extract_features(val_subset, corpus_features, st_history_map=st_hist, is_train=False)

    y_tr = tr_subset['violation_count'].values
    y_val = val_subset['violation_count'].values

    # Train validation model
    val_model = train_lgb_model(X_tr, y_tr, seed=seed)

    # Predict on validation holdout
    val_preds_model = np.clip(val_model.predict(X_val[FEATURE_COLS]), 0.0, None)

    # Validation key-pair resolution
    # Check if exact key pair exists in training split
    tr_exact_lookup = tr_subset.set_index(['Street Name', 'Violation Description'])['violation_count'].to_dict()
    
    val_preds_final = []
    seen_in_tr_cnt = 0
    for i, (_, row) in enumerate(val_subset.iterrows()):
        key = (row['Street Name'], row['Violation Description'])
        if key in tr_exact_lookup:
            seen_in_tr_cnt += 1
            # Blended historical + model
            val_preds_final.append(0.95 * tr_exact_lookup[key] + 0.05 * val_preds_model[i])
        else:
            val_preds_final.append(val_preds_model[i])

    val_preds_final = np.asarray(val_preds_final)
    val_rmse = compute_rmse(y_val, val_preds_final)

    # Baseline comparison (violation type mean)
    v_means = tr_subset.groupby('Violation Description')['violation_count'].mean().to_dict()
    base_preds = val_subset['Violation Description'].map(v_means).fillna(tr_subset['violation_count'].mean()).values
    base_rmse = compute_rmse(y_val, base_preds)

    print(f"Validation Strategy         : 80/20 Holdout Split (seed={seed})")
    print(f"Training Subset Rows        : {len(tr_subset):,}")
    print(f"Validation Holdout Rows     : {len(val_subset):,}")
    print(f"Key Pairs Unseen in Train   : {len(val_subset) - seen_in_tr_cnt:,} ({(len(val_subset) - seen_in_tr_cnt)/len(val_subset)*100:.1f}%)")
    print(f"Baseline (Violation Mean)   : {base_rmse:.2f} RMSE")
    print(f"Development Model RMSE      : {val_rmse:.2f} RMSE")
    print(f"Relative Improvement        : {((base_rmse - val_rmse) / base_rmse)*100:+.2f}%")
    print("=" * 70 + "\n")
    return val_rmse


def resolve_path(provided_path, default_relative, fallback_locations):
    """Resolve file path across working directory, defaults, and common fallbacks."""
    if provided_path and os.path.exists(provided_path):
        return provided_path

    path_to_try = provided_path if provided_path else default_relative
    if os.path.exists(path_to_try):
        return path_to_try

    for fallback in fallback_locations:
        if os.path.exists(fallback):
            return fallback

    return path_to_try


def main():
    parser = argparse.ArgumentParser(
        description="Predict NYC parking violations per (street_name, violation_type) pair using 2022 aggregate data and corpus features."
    )
    parser.add_argument(
        "--train-path",
        default=os.environ.get("TRAIN_PATH", "violations_per_street_2022.csv"),
        help="Path to training CSV file (default: violations_per_street_2022.csv)"
    )
    parser.add_argument(
        "--test-path", "--eval-path",
        dest="test_path",
        default="/Users/USER/Downloads/violations_per_street_2023.csv",
        help="Path to test/eval CSV file for predictions / posthoc scoring (optional)"
    )
    parser.add_argument(
        "--corpus-dir",
        default=os.environ.get("CORPUS_DIR", "corpus"),
        help="Path to corpus folder containing parquet tables (default: corpus)"
    )
    parser.add_argument(
        "--output-path", "--submission-path",
        dest="output_path",
        default=os.environ.get("OUTPUT_PATH", "submission.csv"),
        help="Path to output submission CSV (default: submission.csv)"
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=0.95,
        help="Weight assigned to historical count for seen key pairs (default: 0.95)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip development validation step to speed up inference"
    )

    args = parser.parse_args()

    # 1. Resolve train path
    train_path = resolve_path(
        args.train_path,
        "violations_per_street_2022.csv",
        ["train/all_data/violations_per_street_2022.csv", "violations_per_street_2022.csv"]
    )
    if not os.path.exists(train_path):
        print(f"[Error] Training file not found at: {train_path}", file=sys.stderr)
        sys.exit(1)

    # 2. Resolve corpus directory
    corpus_dir = resolve_path(
        args.corpus_dir,
        "corpus",
        ["train/all_data/corpus", "corpus"]
    )

    print(f"[*] Training Path : {train_path}")
    print(f"[*] Corpus Dir    : {corpus_dir}")
    if args.test_path:
        print(f"[*] Test Path     : {args.test_path}")
    print(f"[*] Output Path   : {args.output_path}")

    # 3. Load Training Data
    train_df = pd.read_csv(train_path)
    street_col = [c for c in train_df.columns if c.lower() in ['street name', 'street_name', 'street']][0]
    viol_col = [c for c in train_df.columns if c.lower() in ['violation description', 'violation_type', 'violation']][0]
    target_col = [c for c in train_df.columns if c.lower() in ['violation_count', 'target']][0]

    # Standardize column names internally
    train_df = train_df.rename(columns={
        street_col: 'Street Name',
        viol_col: 'Violation Description',
        target_col: 'violation_count'
    })
    print(f"[*] Loaded training records: {len(train_df):,} rows ({train_df['Street Name'].nunique():,} unique streets).")

    # 4. Extract / Load Corpus features
    cache_file = ".corpus_features_cache.pkl"
    corpus_features = build_corpus_features(corpus_dir, cache_path=cache_file)

    # 5. Development Validation
    if not args.skip_validation:
        run_development_validation(train_df, corpus_features, seed=args.seed)

    # 6. Train Production Model on 100% Training Data
    print("[*] Training production LightGBM model on full 2022 dataset...")
    full_st_hist = train_df.groupby('Street Name')['violation_count'].agg(
        hist_st_sum='sum',
        hist_st_cnt='count',
        hist_st_mean='mean'
    ).reset_index()

    X_train_full = extract_features(train_df, corpus_features, is_train=True)
    y_train_full = train_df['violation_count'].values
    prod_model = train_lgb_model(X_train_full, y_train_full, seed=args.seed)
    print("[*] Production model trained successfully.")

    # Build historical lookup indices (both exact and normalized)
    exact_lookup = train_df.set_index(['Street Name', 'Violation Description'])['violation_count'].to_dict()

    train_df['norm_st'] = train_df['Street Name'].apply(normalize_street)
    train_df['norm_vi'] = train_df['Violation Description'].apply(normalize_violation)
    norm_lookup = train_df.groupby(['norm_st', 'norm_vi'])['violation_count'].mean().to_dict()

    # 7. Posthoc Test/Eval Prediction (if test-path provided)
    if args.test_path:
        test_path = args.test_path
        if not os.path.exists(test_path):
            print(f"[Error] Test file not found at: {test_path}", file=sys.stderr)
            sys.exit(1)

        test_df = pd.read_csv(test_path)
        test_st_col = [c for c in test_df.columns if c.lower() in ['street name', 'street_name', 'street']][0]
        test_vi_col = [c for c in test_df.columns if c.lower() in ['violation description', 'violation_type', 'violation']][0]
        test_has_target = any(c.lower() in ['violation_count', 'target'] for c in test_df.columns)

        print(f"\n[*] Processing test/evaluation file: {test_path} ({len(test_df):,} rows)...")

        # Extract features for test set (pass historical training map)
        X_test = extract_features(test_df, corpus_features, st_history_map=full_st_hist, is_train=False)

        # Generate ML model base predictions
        ml_test_preds = np.clip(prod_model.predict(X_test[FEATURE_COLS]), 0.0, None)

        # Resolve predictions with historical knowledge
        final_preds = []
        seen_exact = 0
        seen_norm = 0
        unseen_count = 0

        for i, (_, row) in enumerate(test_df.iterrows()):
            st = row[test_st_col]
            vi = row[test_vi_col]
            ml_pred = ml_test_preds[i]

            # 1. Exact match in training data
            exact_key = (st, vi)
            if exact_key in exact_lookup:
                seen_exact += 1
                y_hist = exact_lookup[exact_key]
                pred = args.blend_weight * y_hist + (1.0 - args.blend_weight) * ml_pred
            else:
                # 2. Normalized match in training data
                norm_key = (normalize_street(st), normalize_violation(vi))
                if norm_key in norm_lookup:
                    seen_norm += 1
                    y_hist = norm_lookup[norm_key]
                    pred = args.blend_weight * y_hist + (1.0 - args.blend_weight) * ml_pred
                else:
                    # 3. Unseen key pair: pure feature-augmented ML model prediction
                    unseen_count += 1
                    pred = ml_pred

            final_preds.append(max(0.0, round(float(pred), 2)))

        # Build output dataframe
        sub_df = pd.DataFrame({
            'street_name': test_df[test_st_col],
            'violation_type': test_df[test_vi_col],
            'predicted_count': final_preds
        })

        # Save submission.csv
        sub_df.to_csv(args.output_path, index=False)
        print(f"[*] Submission successfully written to: {args.output_path} ({len(sub_df):,} rows).")

        # Report seen/unseen statistics as required by goal.txt
        total_test = len(test_df)
        print("\n" + "=" * 70)
        print("TEST EVALUATION REPORT")
        print("=" * 70)
        print(f"Total Rows Evaluated        : {total_test:,}")
        print(f"Key Pairs Seen (Exact)      : {seen_exact:,} ({seen_exact/total_test*100:.1f}%)")
        print(f"Key Pairs Seen (Normalized) : {seen_norm:,} ({seen_norm/total_test*100:.1f}%)")
        print(f"Key Pairs Unseen in Train   : {unseen_count:,} ({unseen_count/total_test*100:.1f}%)")
        print(f"Handling of Unseen Pairs    : Unseen pairs were predicted dynamically by the")
        print(f"                              feature-augmented LightGBM model utilizing")
        print(f"                              street text characteristics (camera keywords,")
        print(f"                              directional prefixes, intersection markers),")
        print(f"                              NYC Open Data corpus enrichments (meters,")
        print(f"                              pavement length/condition, speed limits,")
        print(f"                              facilities, speed reducers, bike racks, DOB),")
        print(f"                              and violation category semantics.")

        # If test set contains ground truth violation_count, compute and report RMSE
        if test_has_target:
            test_target_col = [c for c in test_df.columns if c.lower() in ['violation_count', 'target']][0]
            y_test_true = test_df[test_target_col].values
            test_rmse = compute_rmse(y_test_true, final_preds)
            print("-" * 70)
            print(f"Posthoc Evaluation RMSE     : {test_rmse:.2f}")
        print("=" * 70 + "\n")
    else:
        print("\n[i] No --test-path provided. To generate predictions for test/eval data, run:")
        print(f"    python {sys.argv[0]} --test-path <path_to_test_file>.csv\n")


if __name__ == '__main__':
    main()
