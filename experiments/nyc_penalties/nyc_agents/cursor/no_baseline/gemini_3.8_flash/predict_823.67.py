#!/usr/bin/env python3
"""
NYC Parking Violations Predictor
Predicts the number of parking violations in 2023 for each (street_name, violation_type) pair,
using 2022 data and augmented features discovered from corpus/ tables.
"""

import os
import sys
import re
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")

# Street normalization mapping
STREET_MAPPING = {
    'ST': 'STREET', 'STR': 'STREET',
    'AVE': 'AVENUE', 'AV': 'AVENUE',
    'RD': 'ROAD',
    'BLVD': 'BOULEVARD', 'BLV': 'BOULEVARD',
    'PKWY': 'PARKWAY', 'PKW': 'PARKWAY',
    'PL': 'PLACE', 'PLC': 'PLACE',
    'DR': 'DRIVE',
    'LN': 'LANE',
    'CT': 'COURT',
    'TER': 'TERRACE', 'TERR': 'TERRACE',
    'HWY': 'HIGHWAY',
    'EXPWY': 'EXPRESSWAY', 'EXP': 'EXPRESSWAY',
    'TPKE': 'TURNPIKE',
    'E': 'EAST', 'W': 'WEST', 'N': 'NORTH', 'S': 'SOUTH',
}

def normalize_street(s):
    """Normalize street names for robust fuzzy matching across datasets."""
    if not isinstance(s, str):
        return ''
    s = s.upper().strip()
    s = re.sub(r'[^A-Z0-9\s]', ' ', s)
    tokens = s.split()
    new_tokens = []
    for t in tokens:
        t = STREET_MAPPING.get(t, t)
        m = re.match(r'^(\d+)(ST|ND|RD|TH)$', t)
        if m:
            t = m.group(1)
        new_tokens.append(t)
    return ' '.join(new_tokens)


def resolve_path(path, candidates):
    """Find the existing file or directory path among candidate locations."""
    if path and os.path.exists(path):
        return os.path.abspath(path)
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return path


def load_or_extract_corpus_features(corpus_dir):
    """Load cached corpus features or extract and aggregate features from corpus/ parquet tables."""
    cache_path = os.path.join(corpus_dir, 'corpus_features.parquet')
    if os.path.exists(cache_path):
        try:
            df_cache = pq.read_table(cache_path).to_pandas()
            print(f"[Corpus] Loaded cached corpus features from {cache_path} ({len(df_cache)} streets)")
            return df_cache
        except Exception as e:
            print(f"[Corpus] Cache read failed ({e}), re-extracting from raw parquet files...")

    print(f"[Corpus] Discovering and aggregating features from {corpus_dir}...")
    start_t = time.time()
    
    extracted_dfs = []

    # 1. Parking Meters
    p = os.path.join(corpus_dir, 'parking_meters_locations_and_status__693u-uax6.parquet')
    if os.path.exists(p):
        df = pq.read_table(p, columns=['On_Street', 'Status']).to_pandas()
        df['norm_street'] = df['On_Street'].apply(normalize_street)
        f_meters = df.groupby('norm_street').agg(
            meter_count=('On_Street', 'count'),
            active_meter_count=('Status', lambda s: (s == 'Active').sum())
        ).reset_index()
        extracted_dfs.append(f_meters)

    # 2. VZV Speed Limits
    p = os.path.join(corpus_dir, 'vzv_speed_limits__5mad-ntua.parquet')
    if os.path.exists(p):
        df = pq.read_table(p, columns=['street', 'postvz_sl', 'Shape_Leng']).to_pandas()
        df['norm_street'] = df['street'].apply(normalize_street)
        df['postvz_sl'] = pd.to_numeric(df['postvz_sl'], errors='coerce')
        f_speed = df.groupby('norm_street').agg(
            speed_limit_mean=('postvz_sl', 'mean'),
            speed_limit_max=('postvz_sl', 'max'),
            speed_total_length=('Shape_Leng', 'sum'),
            speed_segment_count=('street', 'count')
        ).reset_index()
        extracted_dfs.append(f_speed)

    # 3. Street Pavement Ratings
    p = os.path.join(corpus_dir, 'street_pavement_ratings__6yyb-pb25.parquet')
    if os.path.exists(p):
        df = pq.read_table(p, columns=['OnStreetName', 'BoroughName', 'SystemRating', 'LocationGeometry.STLength']).to_pandas()
        df['norm_street'] = df['OnStreetName'].apply(normalize_street)
        f_pave = df.groupby('norm_street').agg(
            pave_rating_mean=('SystemRating', 'mean'),
            pave_rating_min=('SystemRating', 'min'),
            pave_length_sum=('LocationGeometry.STLength', 'sum'),
            pave_segment_count=('OnStreetName', 'count'),
            pave_borough=('BoroughName', lambda s: s.mode()[0] if not s.empty else 'Unknown')
        ).reset_index()
        extracted_dfs.append(f_pave)

    # 4. Storefronts (Commercial Activity)
    p = os.path.join(corpus_dir, 'storefronts_reported_vacant_or_not__92iy-9c3n.parquet')
    if os.path.exists(p):
        df = pq.read_table(p, columns=['Property Street', 'Vacant on 12/31']).to_pandas()
        df['norm_street'] = df['Property Street'].apply(normalize_street)
        f_store = df.groupby('norm_street').agg(
            storefront_count=('Property Street', 'count'),
            storefront_vacant_count=('Vacant on 12/31', lambda s: (s == 'Y').sum())
        ).reset_index()
        extracted_dfs.append(f_store)

    # 5. Restaurant Inspections
    p = os.path.join(corpus_dir, 'dohmh_new_york_city_restaurant_inspection_results__43nn-pn8j.parquet')
    if os.path.exists(p):
        df = pq.read_table(p, columns=['STREET', 'SCORE']).to_pandas()
        df['norm_street'] = df['STREET'].apply(normalize_street)
        df['SCORE'] = pd.to_numeric(df['SCORE'], errors='coerce')
        f_rest = df.groupby('norm_street').agg(
            restaurant_count=('STREET', 'count'),
            restaurant_score_mean=('SCORE', 'mean')
        ).reset_index()
        extracted_dfs.append(f_rest)

    # 6. Speed Reducers (SRTS)
    p = os.path.join(corpus_dir, 'speed_reducer_tracking_system_srts__9n6h-pt9g.parquet')
    if os.path.exists(p):
        df = pq.read_table(p, columns=['OnStreet']).to_pandas()
        df['norm_street'] = df['OnStreet'].apply(normalize_street)
        f_srts = df.groupby('norm_street').size().rename('speed_reducer_count').reset_index()
        extracted_dfs.append(f_srts)

    # 7. Forestry Street Planting Spaces
    p = os.path.join(corpus_dir, 'forestry_planting_spaces__82zj-84is.parquet')
    if os.path.exists(p):
        df = pq.read_table(p, columns=['Street']).to_pandas()
        df['norm_street'] = df['Street'].apply(normalize_street)
        f_tree = df.groupby('norm_street').size().rename('tree_count').reset_index()
        extracted_dfs.append(f_tree)

    # 8. Bicycle Parking
    p = os.path.join(corpus_dir, 'bicycle_parking__592z-n7dk.parquet')
    if os.path.exists(p):
        df = pq.read_table(p, columns=['OnStreet']).to_pandas()
        df['norm_street'] = df['OnStreet'].apply(normalize_street)
        f_bike = df.groupby('norm_street').size().rename('bike_rack_count').reset_index()
        extracted_dfs.append(f_bike)

    # 9. DOB Building Violations
    p = os.path.join(corpus_dir, 'dob_violations__3h2n-5cm9.parquet')
    if os.path.exists(p):
        df = pq.read_table(p, columns=['STREET']).to_pandas()
        df['norm_street'] = df['STREET'].apply(normalize_street)
        f_dob = df.groupby('norm_street').size().rename('dob_violation_count').reset_index()
        extracted_dfs.append(f_dob)

    # 10. 311 Service Requests
    p = os.path.join(corpus_dir, '311_service_requests_for_2009__3rfa-3xsf.parquet')
    if os.path.exists(p):
        df = pq.read_table(p, columns=['Street Name']).to_pandas()
        df['norm_street'] = df['Street Name'].apply(normalize_street)
        f_311 = df.groupby('norm_street').size().rename('service_311_count').reset_index()
        extracted_dfs.append(f_311)

    all_streets = set()
    for f in extracted_dfs:
        all_streets.update(f['norm_street'])
    all_streets.discard('')

    unified = pd.DataFrame({'norm_street': list(all_streets)})
    for f in extracted_dfs:
        unified = unified.merge(f, on='norm_street', how='left')

    try:
        unified.to_parquet(cache_path, index=False)
        print(f"[Corpus] Cached unified corpus features to {cache_path} ({len(unified)} streets, elapsed: {time.time() - start_t:.2f}s)")
    except Exception as e:
        print(f"[Corpus] Could not cache to file ({e}), continuing with in-memory features...")

    return unified


def detect_columns(df):
    """Identify street, violation, and optional target columns regardless of naming conventions."""
    cols = {c.lower().strip().replace(' ', '_'): c for c in df.columns}
    
    street_col = None
    for cand in ['street_name', 'street', 'streetname', 'st_name']:
        if cand in cols:
            street_col = cols[cand]
            break
            
    violation_col = None
    for cand in ['violation_description', 'violation_type', 'violation', 'violation_desc']:
        if cand in cols:
            violation_col = cols[cand]
            break
            
    target_col = None
    for cand in ['violation_count', 'count', 'target', 'predicted_count']:
        if cand in cols:
            target_col = cols[cand]
            break
            
    return street_col, violation_col, target_col


def engineer_features(df, corpus_features, known_violation_types=None):
    """Generate all street, violation, and corpus-augmented features for model training and inference."""
    street_col, violation_col, _ = detect_columns(df)
    
    feat_df = pd.DataFrame()
    feat_df['street_raw'] = df[street_col].astype(str)
    feat_df['violation_raw'] = df[violation_col].astype(str)
    feat_df['norm_street'] = feat_df['street_raw'].apply(normalize_street)
    
    # Merge corpus features
    feat_df = feat_df.merge(corpus_features, on='norm_street', how='left')
    
    # Numeric corpus features to fill with 0
    num_corpus_cols = [
        'meter_count', 'active_meter_count', 'speed_limit_mean', 'speed_limit_max', 
        'speed_total_length', 'speed_segment_count', 'pave_rating_mean', 'pave_rating_min',
        'pave_length_sum', 'pave_segment_count', 'storefront_count', 'storefront_vacant_count',
        'restaurant_count', 'restaurant_score_mean', 'speed_reducer_count', 'tree_count', 
        'bike_rack_count', 'dob_violation_count', 'service_311_count'
    ]
    for c in num_corpus_cols:
        if c in feat_df.columns:
            feat_df[c] = feat_df[c].fillna(0)
        else:
            feat_df[c] = 0.0

    # Categorical Borough
    if 'pave_borough' in feat_df.columns:
        feat_df['pave_borough'] = feat_df['pave_borough'].fillna('Unknown').astype('category')
    else:
        feat_df['pave_borough'] = pd.Series(['Unknown'] * len(feat_df), dtype='category')
        
    # Text features from Street Name
    feat_df['street_len'] = feat_df['street_raw'].str.len()
    feat_df['street_words'] = feat_df['street_raw'].str.split().str.len()
    feat_df['is_camera_street'] = feat_df['street_raw'].str.contains(r'^(NB|SB|EB|WB)\s|@', regex=True, na=False).astype(int)
    feat_df['has_at'] = feat_df['street_raw'].str.contains('@', regex=False).astype(int)
    feat_df['has_digits'] = feat_df['street_raw'].str.contains(r'\d', regex=True).astype(int)
    
    # Road type indicators
    feat_df['is_ave'] = feat_df['norm_street'].str.contains(r'\bAVENUE\b', regex=True).astype(int)
    feat_df['is_st'] = feat_df['norm_street'].str.contains(r'\bSTREET\b', regex=True).astype(int)
    feat_df['is_blvd'] = feat_df['norm_street'].str.contains(r'\bBOULEVARD\b', regex=True).astype(int)
    feat_df['is_expwy'] = feat_df['norm_street'].str.contains(r'\b(EXPRESSWAY|HIGHWAY|PARKWAY)\b', regex=True).astype(int)

    # Violation type features
    feat_df['is_speed_violation'] = (feat_df['violation_raw'] == 'PHTO SCHOOL ZN SPEED VIOLATION').astype(int)
    feat_df['is_bus_violation'] = (feat_df['violation_raw'] == 'BUS LANE VIOLATION').astype(int)
    feat_df['is_red_light'] = (feat_df['violation_raw'] == 'FAILURE TO STOP AT RED LIGHT').astype(int)
    feat_df['is_meter_violation'] = (feat_df['violation_raw'] == '38-Failure to Dsplay Meter Rec').astype(int)
    feat_df['is_hydrant'] = (feat_df['violation_raw'] == '40-Fire Hydrant').astype(int)
    feat_df['is_street_clean'] = (feat_df['violation_raw'] == '21-No Parking (street clean)').astype(int)
    feat_df['is_standing'] = (feat_df['violation_raw'] == '14-No Standing').astype(int)
    feat_df['is_camera_viol'] = feat_df['is_speed_violation'] | feat_df['is_bus_violation'] | feat_df['is_red_light']

    # Cross Interactions
    feat_df['camera_match'] = feat_df['is_camera_street'] * feat_df['is_camera_viol']
    feat_df['camera_mismatch'] = feat_df['is_camera_street'] * (1 - feat_df['is_camera_viol'])
    feat_df['meter_cross'] = feat_df['meter_count'] * feat_df['is_meter_violation']
    feat_df['speed_cross'] = feat_df['speed_limit_max'] * feat_df['is_speed_violation']
    feat_df['store_cross'] = feat_df['storefront_count'] * feat_df['is_standing']
    feat_df['restaurant_cross'] = feat_df['restaurant_count'] * feat_df['is_standing']
    feat_df['tree_cross'] = feat_df['tree_count'] * feat_df['is_street_clean']

    if known_violation_types is not None:
        feat_df['violation_cat'] = pd.Categorical(feat_df['violation_raw'], categories=known_violation_types)
    else:
        feat_df['violation_cat'] = feat_df['violation_raw'].astype('category')

    feature_columns = num_corpus_cols + [
        'pave_borough', 'street_len', 'street_words', 'is_camera_street', 'has_at', 'has_digits',
        'is_ave', 'is_st', 'is_blvd', 'is_expwy',
        'violation_cat', 'is_speed_violation', 'is_bus_violation', 'is_red_light', 'is_meter_violation',
        'is_hydrant', 'is_street_clean', 'is_standing', 'is_camera_viol',
        'camera_match', 'camera_mismatch', 'meter_cross', 'speed_cross', 'store_cross', 'restaurant_cross', 'tree_cross'
    ]

    return feat_df[feature_columns], feat_df['norm_street'], feat_df['street_raw'], feat_df['violation_raw']


def train_and_validate(train_df, corpus_features, n_splits=5, seed=42):
    """
    Train 5-Fold Grouped LightGBM ensemble and report development validation RMSE.
    Validation strictly evaluates out-of-fold holdout streets (unseen streets) to prevent leakage.
    """
    street_col, viol_col, target_col = detect_columns(train_df)
    print(f"\n[Validation] Running {n_splits}-Fold Grouped Cross-Validation by Street Name...")
    print(f"[Validation] Using 2022 dataset: {len(train_df)} rows across {train_df[street_col].nunique()} unique streets.")

    known_violations = sorted(train_df[viol_col].unique().tolist())
    X, norm_streets, raw_streets, raw_viols = engineer_features(train_df, corpus_features, known_violation_types=known_violations)
    y = train_df[target_col].values

    gkf = GroupKFold(n_splits=n_splits)
    oof_predictions = np.zeros(len(train_df))
    models = []
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=train_df[street_col])):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed + fold,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_tr, y_tr)
        val_pred = np.clip(model.predict(X_va), 0, None)
        oof_predictions[val_idx] = val_pred
        
        fold_rmse = root_mean_squared_error(y_va, val_pred)
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold + 1} (Holdout Streets) RMSE: {fold_rmse:8.2f}")
        models.append(model)

    overall_dev_rmse = root_mean_squared_error(y, oof_predictions)
    print(f"\n" + "="*50)
    print(f"DEVELOPMENT VALIDATION RMSE (Grouped Holdout): {overall_dev_rmse:8.2f}")
    print(f"Mean Fold RMSE: {np.mean(fold_rmses):8.2f} (+/- {np.std(fold_rmses):.2f})")
    print(f"="*50 + "\n")

    return models, known_violations


def run_inference(test_path, train_df, models, corpus_features, known_violations, output_path="submission.csv"):
    """
    Generate predictions for test/eval dataset.
    Handles seen and unseen keys, reports unseen statistics, outputs submission.csv,
    and reports Test RMSE if ground truth is present.
    """
    print(f"[Inference] Loading test/eval file from {test_path}...")
    test_df = pd.read_csv(test_path)
    street_col, viol_col, target_col = detect_columns(test_df)
    
    if street_col is None or viol_col is None:
        raise ValueError(f"Could not identify street or violation columns in {test_path}. Columns found: {list(test_df.columns)}")

    train_street_col, train_viol_col, train_target_col = detect_columns(train_df)

    # Build historical lookup tables from 2022 training data
    train_exact_dict = {}
    train_norm_dict = {}
    street_total_dict = {}

    for _, row in train_df.iterrows():
        s_raw = str(row[train_street_col]).strip()
        v_raw = str(row[train_viol_col]).strip()
        c = float(row[train_target_col])
        train_exact_dict[(s_raw.upper(), v_raw.upper())] = c
        
        s_norm = normalize_street(s_raw)
        train_norm_dict[(s_norm, v_raw.upper())] = c
        street_total_dict[s_raw.upper()] = street_total_dict.get(s_raw.upper(), 0.0) + c
        street_total_dict[s_norm] = street_total_dict.get(s_norm, 0.0) + c

    # Feature matrix for test data
    X_test, norm_streets, raw_streets, raw_viols = engineer_features(
        test_df, corpus_features, known_violation_types=known_violations
    )

    # Ensemble model predictions
    fold_preds = np.zeros((len(test_df), len(models)))
    for i, model in enumerate(models):
        fold_preds[:, i] = np.clip(model.predict(X_test), 0, None)
    model_pred = np.mean(fold_preds, axis=1)

    final_predictions = np.zeros(len(test_df))
    seen_count = 0
    unseen_violation_on_seen_street = 0
    unseen_street_count = 0

    for i in range(len(test_df)):
        s_raw = str(raw_streets.iloc[i]).strip().upper()
        v_raw = str(raw_viols.iloc[i]).strip().upper()
        s_norm = norm_streets.iloc[i]
        
        # 1. Exact historical key match
        if (s_raw, v_raw) in train_exact_dict:
            hist_count = train_exact_dict[(s_raw, v_raw)]
            # Anchor primarily on historical training signal
            final_predictions[i] = 0.98 * hist_count + 0.02 * model_pred[i]
            seen_count += 1
            
        # 2. Normalized street match
        elif (s_norm, v_raw) in train_norm_dict:
            hist_count = train_norm_dict[(s_norm, v_raw)]
            final_predictions[i] = 0.95 * hist_count + 0.05 * model_pred[i]
            seen_count += 1
            
        # 3. Unseen key
        else:
            if s_raw in street_total_dict or s_norm in street_total_dict:
                # Street was seen in 2022 with other violations, but not this violation
                tot = street_total_dict.get(s_raw, street_total_dict.get(s_norm, 1.0))
                # For low-activity streets, scale down model prediction; for high activity, use model
                if tot <= 5:
                    final_predictions[i] = min(model_pred[i], tot * 0.5)
                else:
                    final_predictions[i] = model_pred[i]
                unseen_violation_on_seen_street += 1
            else:
                # Completely unseen street: rely on corpus features and violation prior
                final_predictions[i] = model_pred[i]
                unseen_street_count += 1

    final_predictions = np.clip(np.round(final_predictions, 2), 0, None)

    # Report unseen keys statistics
    total_keys = len(test_df)
    unseen_total = unseen_violation_on_seen_street + unseen_street_count
    
    print("\n" + "="*55)
    print("TEST / EVALUATION KEY-SPACE ANALYSIS")
    print("="*55)
    print(f"Total test/eval rows: {total_keys:,}")
    print(f"Keys seen in 2022 training: {seen_count:,} ({seen_count / total_keys * 100:.1f}%)")
    print(f"Keys unseen in 2022 training: {unseen_total:,} ({unseen_total / total_keys * 100:.1f}%)")
    print(f"  - Unseen violation on seen street: {unseen_violation_on_seen_street:,} ({unseen_violation_on_seen_street / total_keys * 100:.1f}%)")
    print(f"  - Completely unseen street:        {unseen_street_count:,} ({unseen_street_count / total_keys * 100:.1f}%)")
    print("\nHow unseen keys were handled:")
    print("  1. Unseen violation on seen street: Predictions use the 5-fold LightGBM ensemble,")
    print("     regularized and bounded by the street's historical activity profile in 2022.")
    print("  2. Completely unseen streets: Predictions are synthesized by the LightGBM ensemble")
    print("     using external physical signals from 10 corpus datasets (parking meters, speed")
    print("     limits, pavement ratings, storefront density, restaurants, speed reducers,")
    print("     forestry planting spaces, bike racks, DOB violations, and borough indicators).")
    print("="*55 + "\n")

    # Generate submission file
    submission = pd.DataFrame({
        'street_name': test_df[street_col],
        'violation_type': test_df[viol_col],
        'predicted_count': final_predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"[Output] Successfully generated submission file: {output_path} ({len(submission)} rows)")

    # Score against ground truth if present
    if target_col is not None:
        y_true = test_df[target_col].values
        test_rmse = root_mean_squared_error(y_true, final_predictions)
        print("\n" + "*"*55)
        print(f"TEST / EVALUATION GROUND-TRUTH RMSE: {test_rmse:8.2f}")
        print("*"*55 + "\n")
        return test_rmse
    else:
        print("[Info] No ground truth violation_count column detected in test file (keys-only prediction).")
        return None


def main():
    parser = argparse.ArgumentParser(description="NYC Parking Violations 2023 Predictor")
    parser.add_argument(
        "--train-path",
        default=os.environ.get("TRAIN_PATH", "violations_per_street_2022.csv"),
        help="Path to 2022 training CSV file"
    )
    parser.add_argument(
        "--test-path", "--eval-path",
        dest="test_path",
        default=os.environ.get("TEST_PATH", os.environ.get("EVAL_PATH", None)),
        help="Path to test/evaluation CSV file"
    )
    parser.add_argument(
        "--corpus-dir",
        default=os.environ.get("CORPUS_DIR", "corpus"),
        help="Path to corpus directory containing supplementary parquet tables"
    )
    parser.add_argument(
        "--output-path",
        default=os.environ.get("OUTPUT_PATH", "submission.csv"),
        help="Output CSV path for predictions"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for cross-validation"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for grouped cross-validation"
    )
    
    args = parser.parse_args()

    # Resolve paths
    train_path = resolve_path(args.train_path, [
        args.train_path,
        os.path.join("train", "all_data", "violations_per_street_2022.csv"),
        "violations_per_street_2022.csv"
    ])
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path} or default candidate locations.")

    corpus_dir = resolve_path(args.corpus_dir, [
        args.corpus_dir,
        os.path.join("train", "all_data", "corpus"),
        "corpus"
    ])
    if not os.path.exists(corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found at {corpus_dir} or default candidate locations.")

    print("="*55)
    print("NYC PARKING VIOLATIONS FUTURE COUNT PREDICTOR")
    print("="*55)
    print(f"Training data path: {train_path}")
    print(f"Corpus directory:   {corpus_dir}")
    print(f"Test / eval path:   {args.test_path if args.test_path else 'None (Development Mode)'}")
    print(f"Output path:        {args.output_path}")

    # Load training data
    train_df = pd.read_csv(train_path)
    
    # Load corpus features
    corpus_features = load_or_extract_corpus_features(corpus_dir)

    # Train and evaluate on 2022 development validation
    models, known_violations = train_and_validate(
        train_df, corpus_features, n_splits=args.n_splits, seed=args.seed
    )

    # If test file is supplied, predict and generate submission
    if args.test_path:
        test_path = resolve_path(args.test_path, [args.test_path])
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test/evaluation file not found at {test_path}")
            
        run_inference(
            test_path=test_path,
            train_df=train_df,
            models=models,
            corpus_features=corpus_features,
            known_violations=known_violations,
            output_path=args.output_path
        )
    else:
        print("[Status] Development validation complete. No test file provided.")
        print("To generate predictions for a posthoc test file, run:")
        print(f"  python {sys.argv[0]} --test-path <path_to_test.csv>")


if __name__ == "__main__":
    main()
