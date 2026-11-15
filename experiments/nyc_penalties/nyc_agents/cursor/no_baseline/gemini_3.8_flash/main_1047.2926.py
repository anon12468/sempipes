#!/usr/bin/env python3
"""
NYC Parking Violations Prediction Pipeline
Predicts the number of parking violations for (street_name, violation_type) pairs
using 2022 data and feature augmentations from the NYC Open Data corpus.
"""

import os
import sys
import argparse
import re
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


def resolve_file(path, default_candidates):
    """Resolve file path by checking user path, then fallback candidate locations."""
    if path and os.path.exists(path):
        return path
    for cand in default_candidates:
        if os.path.exists(cand):
            return cand
    return path


def normalize_street(s):
    """
    Standardize street names for reliable cross-dataset joining.
    Strips camera direction prefixes (NB/SB/EB/WB) and intersection suffixes (@...).
    Normalizes abbreviations (ST->STREET, AVE->AVENUE, etc.) and ordinal numbers.
    """
    if not isinstance(s, str):
        return ''
    s = s.upper().strip()
    # Strip camera directional prefix (e.g., 'WB N CONDUIT AVE @ S' -> 'N CONDUIT AVE @ S')
    s = re.sub(r'^(NB|SB|EB|WB)\s+', '', s)
    # Strip intersection marker and cross-street
    if '@' in s:
        s = s.split('@')[0].strip()
    # Remove punctuation
    s = re.sub(r'[^\w\s]', ' ', s)
    tokens = s.split()
    replacements = {
        'ST': 'STREET', 'AVE': 'AVENUE', 'RD': 'ROAD', 'BLVD': 'BOULEVARD',
        'DR': 'DRIVE', 'CT': 'COURT', 'PL': 'PLACE', 'PKWY': 'PARKWAY',
        'EXPWY': 'EXPRESSWAY', 'EXPY': 'EXPRESSWAY', 'LN': 'LANE', 'TER': 'TERRACE',
        'HWY': 'HIGHWAY', 'CIR': 'CIRCLE', 'WAY': 'WAY', 'AL': 'ALLEY', 'ALY': 'ALLEY',
        'N': 'NORTH', 'S': 'SOUTH', 'E': 'EAST', 'W': 'WEST',
    }
    norm_tokens = []
    for t in tokens:
        m = re.match(r'^(\d+)(ST|ND|RD|TH)$', t)
        if m:
            norm_tokens.append(m.group(1))
        elif t in replacements:
            norm_tokens.append(replacements[t])
        else:
            norm_tokens.append(t)
    return ' '.join(norm_tokens)


def extract_lexical_features(df, street_col):
    """Extract syntactic and pattern features from street name strings."""
    s = df[street_col].astype(str)
    has_at = s.str.contains('@').astype(int)
    has_dir = s.str.match(r'^(NB|SB|EB|WB)\s').astype(int)
    name_len = s.str.len()
    word_count = s.str.split().str.len()
    is_arterial = s.str.contains(r'BLVD|BOULEVARD|EXPY|EXPRESSWAY|PKWY|PARKWAY|CONDUIT').astype(int)
    is_numbered = s.str.contains(r'\d').astype(int)
    
    return pd.DataFrame({
        'has_at': has_at,
        'has_dir': has_dir,
        'name_len': name_len,
        'word_count': word_count,
        'is_arterial': is_arterial,
        'is_numbered': is_numbered,
    }, index=df.index)


def load_corpus_features(corpus_dir):
    """
    Discover and aggregate features from corpus/ parquet datasets:
    1. Street pavement ratings: condition score, total street length, segment count, borough
    2. VZV speed limits: posted speed limit, shape length, segment count
    3. Parking meters: meter count per street
    4. Speed reducer tracking system (SRTS): speed cushion / hump count
    5. Bicycle parking: bike rack count
    6. DOB violations: building violation activity
    7. Facilities: public & city facilities count
    8. 311 service requests: complaint volume / urban activity indicator
    """
    if not corpus_dir or not os.path.exists(corpus_dir):
        print(f"[Warning] Corpus directory '{corpus_dir}' not found. External enrichments skipped.")
        return pd.DataFrame({'norm_street': []})

    t0 = time.time()
    corpus_dfs = []

    # 1. Pavement ratings
    pave_path = os.path.join(corpus_dir, 'street_pavement_ratings__6yyb-pb25.parquet')
    if os.path.exists(pave_path):
        try:
            pave = pd.read_parquet(pave_path, columns=['OnStreetName', 'BoroughName', 'SystemRating', 'LocationGeometry.STLength'])
            pave['norm_street'] = pave['OnStreetName'].dropna().apply(normalize_street)
            pave_agg = pave.groupby('norm_street').agg(
                pave_rating_mean=('SystemRating', 'mean'),
                pave_length_sum=('LocationGeometry.STLength', 'sum'),
                pave_segments_count=('SystemRating', 'count'),
                boro_mode=('BoroughName', lambda x: x.mode()[0] if not x.empty else 'Unknown')
            ).reset_index()
            corpus_dfs.append(pave_agg)
        except Exception as e:
            print(f"[Warning] Failed to read pavement ratings: {e}")

    # 2. Speed limits
    vzv_path = os.path.join(corpus_dir, 'vzv_speed_limits__5mad-ntua.parquet')
    if os.path.exists(vzv_path):
        try:
            vzv = pd.read_parquet(vzv_path, columns=['street', 'postvz_sl', 'Shape_Leng'])
            vzv['norm_street'] = vzv['street'].dropna().apply(normalize_street)
            vzv['postvz_sl'] = pd.to_numeric(vzv['postvz_sl'], errors='coerce')
            vzv_agg = vzv.groupby('norm_street').agg(
                speed_limit_mean=('postvz_sl', 'mean'),
                speed_limit_max=('postvz_sl', 'max'),
                vzv_length_sum=('Shape_Leng', 'sum'),
                vzv_segments_count=('postvz_sl', 'count')
            ).reset_index()
            corpus_dfs.append(vzv_agg)
        except Exception as e:
            print(f"[Warning] Failed to read speed limits: {e}")

    # 3. Parking meters
    meter_path = os.path.join(corpus_dir, 'parking_meters_locations_and_status__693u-uax6.parquet')
    if os.path.exists(meter_path):
        try:
            meters = pd.read_parquet(meter_path, columns=['On_Street'])
            meters['norm_street'] = meters['On_Street'].dropna().apply(normalize_street)
            meter_agg = meters.groupby('norm_street').size().reset_index(name='meter_count')
            corpus_dfs.append(meter_agg)
        except Exception as e:
            print(f"[Warning] Failed to read parking meters: {e}")

    # 4. Speed reducers
    srts_path = os.path.join(corpus_dir, 'speed_reducer_tracking_system_srts__9n6h-pt9g.parquet')
    if os.path.exists(srts_path):
        try:
            srts = pd.read_parquet(srts_path, columns=['OnStreet'])
            srts['norm_street'] = srts['OnStreet'].dropna().apply(normalize_street)
            srts_agg = srts.groupby('norm_street').size().reset_index(name='speed_reducer_count')
            corpus_dfs.append(srts_agg)
        except Exception as e:
            print(f"[Warning] Failed to read speed reducers: {e}")

    # 5. Bicycle parking
    bike_path = os.path.join(corpus_dir, 'bicycle_parking__592z-n7dk.parquet')
    if os.path.exists(bike_path):
        try:
            bike = pd.read_parquet(bike_path, columns=['OnStreet'])
            bike['norm_street'] = bike['OnStreet'].dropna().apply(normalize_street)
            bike_agg = bike.groupby('norm_street').size().reset_index(name='bike_parking_count')
            corpus_dfs.append(bike_agg)
        except Exception as e:
            print(f"[Warning] Failed to read bicycle parking: {e}")

    # 6. DOB Violations
    dob_path = os.path.join(corpus_dir, 'dob_violations__3h2n-5cm9.parquet')
    if os.path.exists(dob_path):
        try:
            dob = pd.read_parquet(dob_path, columns=['STREET'])
            dob['norm_street'] = dob['STREET'].dropna().apply(normalize_street)
            dob_agg = dob.groupby('norm_street').size().reset_index(name='dob_violation_count')
            corpus_dfs.append(dob_agg)
        except Exception as e:
            print(f"[Warning] Failed to read DOB violations: {e}")

    # 7. City facilities
    fac_path = os.path.join(corpus_dir, 'facilities__67g2-p84d.parquet')
    if os.path.exists(fac_path):
        try:
            fac = pd.read_parquet(fac_path, columns=['STREETNAME'])
            fac['norm_street'] = fac['STREETNAME'].dropna().apply(normalize_street)
            fac_agg = fac.groupby('norm_street').size().reset_index(name='facilities_count')
            corpus_dfs.append(fac_agg)
        except Exception as e:
            print(f"[Warning] Failed to read facilities: {e}")

    # 8. 311 Service requests
    s311_path = os.path.join(corpus_dir, '311_service_requests_for_2009__3rfa-3xsf.parquet')
    if os.path.exists(s311_path):
        try:
            s311 = pd.read_parquet(s311_path, columns=['Street Name'])
            s311['norm_street'] = s311['Street Name'].dropna().apply(normalize_street)
            s311_agg = s311.groupby('norm_street').size().reset_index(name='s311_count')
            corpus_dfs.append(s311_agg)
        except Exception as e:
            print(f"[Warning] Failed to read 311 requests: {e}")

    if not corpus_dfs:
        return pd.DataFrame({'norm_street': []})

    # Merge all corpus features into a unified lookup table
    merged = corpus_dfs[0]
    for cdf in corpus_dfs[1:]:
        merged = merged.merge(cdf, on='norm_street', how='outer')

    print(f"Loaded and aggregated {len(corpus_dfs)} corpus tables for {len(merged)} streets in {time.time()-t0:.2f}s.")
    return merged


def identify_columns(df):
    """Flexible column mapper for core schema variants."""
    street_col = None
    viol_col = None
    target_col = None

    for col in df.columns:
        norm_c = col.lower().strip().replace(' ', '_').replace('\t', '')
        if norm_c in ['street_name', 'streetname', 'street']:
            street_col = col
        elif norm_c in ['violation_description', 'violation_type', 'violationdescription', 'violationtype', 'violation']:
            viol_col = col
        elif norm_c in ['violation_count', 'violationcount', 'count', 'target']:
            target_col = col

    if not street_col or not viol_col:
        raise ValueError(f"Could not identify required columns (street_name, violation_type) in {df.columns.tolist()}")

    return street_col, viol_col, target_col


def build_feature_matrix(df, street_col, viol_col, corpus_lookup, street_stats_lookup, is_training=False, target_col=None):
    """Construct full feature matrix combining lexical, corpus, and cross-violation signals."""
    features = pd.DataFrame(index=df.index)

    # 1. Lexical features
    lex = extract_lexical_features(df, street_col)
    for col in lex.columns:
        features[col] = lex[col]

    # 2. Normalized street for joining corpus
    norm_streets = df[street_col].apply(normalize_street)
    temp_df = pd.DataFrame({'norm_street': norm_streets}, index=df.index)
    temp_merged = temp_df.merge(corpus_lookup, on='norm_street', how='left')

    corpus_cols = [c for c in corpus_lookup.columns if c != 'norm_street']
    for c in corpus_cols:
        if c == 'boro_mode':
            features['boro_mode'] = temp_merged['boro_mode'].fillna('Unknown').astype('category')
        else:
            features[c] = temp_merged[c].values

    # 3. Cross-violation street activity features
    # If training: subtract own count to prevent target leakage in street totals
    street_names = df[street_col].astype(str)
    stats_df = pd.DataFrame({'street': street_names}, index=df.index).merge(
        street_stats_lookup, left_on='street', right_index=True, how='left'
    )
    
    total_viols = stats_df['street_total'].fillna(0).values.astype(float)
    num_viols = stats_df['street_num'].fillna(0).values.astype(float)

    if is_training and target_col in df.columns:
        own_count = df[target_col].fillna(0).values.astype(float)
        other_total = np.maximum(0.0, total_viols - own_count)
        other_num = np.maximum(0.0, num_viols - 1.0)
    else:
        other_total = total_viols
        other_num = num_viols

    other_mean = np.zeros(len(other_total), dtype=float)
    nonzero = other_num > 0
    other_mean[nonzero] = other_total[nonzero] / other_num[nonzero]

    features['street_other_total'] = other_total
    features['street_other_mean'] = other_mean
    features['street_other_num'] = other_num

    # 4. Categorical violation description
    features['violation_cat'] = df[viol_col].astype('category')

    return features


def fit_lgbm(X, y):
    """Fit LightGBM model optimizing RMSE."""
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def build_history_lookups(df, street_col, viol_col, target_col):
    """Build exact and case-insensitive historical count lookup tables."""
    exact_history = {}
    ci_history_list = {}
    
    for _, row in df.iterrows():
        s_val = str(row[street_col])
        v_val = str(row[viol_col])
        c_val = float(row[target_col])
        
        exact_history[(s_val, v_val)] = c_val
        ci_key = (s_val.strip().upper(), v_val.strip().upper())
        if ci_key not in ci_history_list:
            ci_history_list[ci_key] = []
        ci_history_list[ci_key].append(c_val)
        
    ci_history_mean = {k: float(np.mean(v)) for k, v in ci_history_list.items()}
    return exact_history, ci_history_mean


def main():
    parser = argparse.ArgumentParser(description="Predict NYC Parking Violations for (street, violation_type) pairs.")
    parser.add_argument('--train-path', type=str, default=os.getenv('TRAIN_PATH', 'violations_per_street_2022.csv'),
                        help="Path to training data (default: violations_per_street_2022.csv)")
    parser.add_argument('--test-path', type=str, default=os.getenv('TEST_PATH', os.getenv('EVAL_PATH', None)),
                        help="Path to test/eval data (optional)")
    parser.add_argument('--eval-path', type=str, default=None,
                        help="Alias for --test-path")
    parser.add_argument('--output-path', type=str, default=os.getenv('OUTPUT_PATH', 'submission.csv'),
                        help="Path for output predictions (default: submission.csv)")
    parser.add_argument('--corpus-dir', type=str, default=os.getenv('CORPUS_DIR', 'train/all_data/corpus'),
                        help="Directory containing corpus parquet files")

    args = parser.parse_args()

    test_path = args.test_path or args.eval_path

    # Resolve paths
    train_path = resolve_file(args.train_path, [
        'violations_per_street_2022.csv',
        'train/all_data/violations_per_street_2022.csv'
    ])
    corpus_dir = resolve_file(args.corpus_dir, [
        'train/all_data/corpus',
        'corpus',
        '../corpus'
    ])

    if not os.path.exists(train_path):
        print(f"Error: Training file not found at {train_path}")
        sys.exit(1)

    print(f"=== NYC Parking Violations Prediction Pipeline ===")
    print(f"Train path : {train_path}")
    print(f"Test path  : {test_path}")
    print(f"Corpus dir : {corpus_dir}")
    print(f"Output path: {args.output_path}\n")

    # Load 2022 training data
    train_df = pd.read_csv(train_path)
    train_s_col, train_v_col, train_target_col = identify_columns(train_df)
    print(f"Loaded training data: {len(train_df):,} rows.")

    # Load corpus augmentations
    corpus_lookup = load_corpus_features(corpus_dir)

    # Compute street-level historical activity dictionary and pair dictionary
    street_stats = train_df.groupby(train_s_col)[train_target_col].agg(['sum', 'count'])
    street_stats.columns = ['street_total', 'street_num']

    exact_history, ci_history = build_history_lookups(train_df, train_s_col, train_v_col, train_target_col)

    # Mode 1: Development Validation Mode (when test-path is NOT provided)
    if not test_path:
        print("\n--- Running Development Validation Mode (80/20 Holdout) ---")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(kf.split(train_df))
        
        dev_train = train_df.iloc[train_idx].copy().reset_index(drop=True)
        dev_val = train_df.iloc[val_idx].copy().reset_index(drop=True)

        dev_stats = dev_train.groupby(train_s_col)[train_target_col].agg(['sum', 'count'])
        dev_stats.columns = ['street_total', 'street_num']

        dev_exact_history, dev_ci_history = build_history_lookups(dev_train, train_s_col, train_v_col, train_target_col)

        # Build features for dev train
        X_dev_train = build_feature_matrix(
            dev_train, train_s_col, train_v_col, corpus_lookup, dev_stats,
            is_training=True, target_col=train_target_col
        )
        y_dev_train = dev_train[train_target_col].values

        # Build features for dev val (target_col is NOT passed to build_feature_matrix)
        X_dev_val = build_feature_matrix(
            dev_val, train_s_col, train_v_col, corpus_lookup, dev_stats,
            is_training=False
        )
        y_dev_val = dev_val[train_target_col].values

        print(f"Training LightGBM model on {len(dev_train):,} rows...")
        model = fit_lgbm(X_dev_train, y_dev_train)

        raw_preds = model.predict(X_dev_val)
        val_preds = []
        unseen_count = 0
        seen_count = 0

        for i, row in dev_val.iterrows():
            s_val = str(row[train_s_col])
            v_val = str(row[train_v_col])
            exact_k = (s_val, v_val)
            ci_k = (s_val.strip().upper(), v_val.strip().upper())
            
            if exact_k in dev_exact_history:
                seen_count += 1
                pred = 0.95 * dev_exact_history[exact_k] + 0.05 * raw_preds[i]
            else:
                unseen_count += 1
                pred = raw_preds[i]
            val_preds.append(max(0.0, float(pred)))

        val_preds = np.array(val_preds)
        val_rmse = root_mean_squared_error(y_dev_val, val_preds)

        print("\n================ Development Validation Results ================")
        print(f"Validation Holdout Size  : {len(dev_val):,} rows")
        print(f"Seen Key Pairs in Train  : {seen_count:,} ({seen_count/len(dev_val)*100:.1f}%)")
        print(f"Unseen Key Pairs in Train: {unseen_count:,} ({unseen_count/len(dev_val)*100:.1f}%)")
        print(f"Handling Strategy        : Unseen pairs predict from LightGBM model using street lexical signals,")
        print(f"                           cross-violation street activity, and 8 merged open-data corpus tables.")
        print(f"Holdout Validation RMSE  : {val_rmse:.4f}")
        print("===============================================================\n")

        # Emit submission.csv for the validation holdout
        sub = pd.DataFrame({
            'street_name': dev_val[train_s_col],
            'violation_type': dev_val[train_v_col],
            'predicted_count': val_preds
        })
        sub.to_csv(args.output_path, index=False)
        print(f"Generated submission for validation holdout: {args.output_path} ({len(sub):,} rows)")

        # Train full model on 100% data
        print("\nTraining production model on 100% of 2022 data...")
        X_full = build_feature_matrix(
            train_df, train_s_col, train_v_col, corpus_lookup, street_stats,
            is_training=True, target_col=train_target_col
        )
        y_full = train_df[train_target_col].values
        fit_lgbm(X_full, y_full)
        print("Full model training complete.")

    # Mode 2: Test / Evaluation Mode (when test-path IS provided)
    else:
        print(f"\n--- Running Test / Evaluation Mode on: {test_path} ---")
        if not os.path.exists(test_path):
            print(f"Error: Test file not found at {test_path}")
            sys.exit(1)

        test_df = pd.read_csv(test_path)
        test_s_col, test_v_col, test_target_col = identify_columns(test_df)
        print(f"Loaded test file: {len(test_df):,} rows.")

        has_ground_truth = test_target_col is not None
        if has_ground_truth:
            y_test_true = test_df[test_target_col].values
            print("Ground truth violation_count detected in test file. Scoring will be performed.")
        else:
            y_test_true = None
            print("Pure prediction mode (no ground truth column).")

        # Train full model on 100% of 2022 training data
        print(f"Training LightGBM model on full training data ({len(train_df):,} rows)...")
        X_train = build_feature_matrix(
            train_df, train_s_col, train_v_col, corpus_lookup, street_stats,
            is_training=True, target_col=train_target_col
        )
        y_train = train_df[train_target_col].values
        model = fit_lgbm(X_train, y_train)

        # Build feature matrix for test data (DO NOT pass test ground truth as feature!)
        X_test = build_feature_matrix(
            test_df, test_s_col, test_v_col, corpus_lookup, street_stats,
            is_training=False
        )

        raw_test_preds = model.predict(X_test)
        final_preds = []
        seen_count = 0
        unseen_count = 0

        for i, row in test_df.iterrows():
            s_val = str(row[test_s_col])
            v_val = str(row[test_v_col])
            exact_k = (s_val, v_val)
            ci_k = (s_val.strip().upper(), v_val.strip().upper())
            
            if exact_k in exact_history:
                seen_count += 1
                pred = 0.95 * exact_history[exact_k] + 0.05 * raw_test_preds[i]
            elif ci_k in ci_history:
                seen_count += 1
                pred = 0.95 * ci_history[ci_k] + 0.05 * raw_test_preds[i]
            else:
                unseen_count += 1
                pred = raw_test_preds[i]
            final_preds.append(max(0.0, float(pred)))

        final_preds = np.array(final_preds)

        # Emit submission.csv
        sub = pd.DataFrame({
            'street_name': test_df[test_s_col],
            'violation_type': test_df[test_v_col],
            'predicted_count': final_preds
        })
        sub.to_csv(args.output_path, index=False)
        print(f"Generated predictions: {args.output_path} ({len(sub):,} rows)")

        print("\n================ Test / Evaluation Summary ================")
        print(f"Total Test Rows          : {len(test_df):,}")
        print(f"Seen Key Pairs in Train  : {seen_count:,} ({seen_count/len(test_df)*100:.1f}%)")
        print(f"Unseen Key Pairs in Train: {unseen_count:,} ({unseen_count/len(test_df)*100:.1f}%)")
        print(f"Handling Strategy        : Seen pairs leverage 2022 historical count anchors blended with GBDT model;")
        print(f"                           Unseen pairs predict from LightGBM model enriched by NYC Open Data corpus")
        print(f"                           (pavement ratings, speed limits, parking meters, speed humps, bike parking,")
        print(f"                           DOB violations, city facilities, and 311 complaints).")
        
        if has_ground_truth:
            test_rmse = root_mean_squared_error(y_test_true, final_preds)
            print(f"Test Set RMSE            : {test_rmse:.4f}")
        print("===========================================================\n")


if __name__ == '__main__':
    main()
