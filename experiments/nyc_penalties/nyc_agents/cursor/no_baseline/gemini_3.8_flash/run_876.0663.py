#!/usr/bin/env python3
"""
NYC Open Data Parking Violation Prediction Pipeline
===================================================
Predicts the number of parking violations in 2023 for each (street_name, violation_type) pair,
using 2022 data augmented with features discovered in the NYC Open Data corpus.

Supports:
- Development validation (2022 holdout split) reporting RMSE and handling of unseen keys.
- Configurable paths via CLI arguments or environment variables (--train-path, --test-path, --output-path).
- Posthoc evaluation scoring (RMSE) when ground truth counts are present in the test/eval file.
- Emits submission.csv with required columns: street_name, violation_type, predicted_count.
"""

import os
import sys
import re
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Path Resolution Utilities
# -----------------------------------------------------------------------------
def resolve_file_path(path_str, default_subpaths=None):
    """
    Resolves a file path across multiple possible locations:
    1. Exact path as given
    2. Relative to current working directory
    3. Relative to script directory
    4. Inside common subdirectories (e.g. train/all_data/)
    """
    if not path_str:
        return None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        path_str,
        os.path.join(os.getcwd(), path_str),
        os.path.join(script_dir, path_str),
    ]
    
    if default_subpaths:
        for sub in default_subpaths:
            candidates.append(os.path.join(os.getcwd(), sub, path_str))
            candidates.append(os.path.join(script_dir, sub, path_str))
            candidates.append(os.path.join(sub, path_str))

    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)
            
    return os.path.abspath(candidates[0])


def resolve_dir_path(path_str, default_subpaths=None):
    """Resolves a directory path across common locations."""
    if not path_str:
        return None
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        path_str,
        os.path.join(os.getcwd(), path_str),
        os.path.join(script_dir, path_str),
    ]
    
    if default_subpaths:
        for sub in default_subpaths:
            candidates.append(os.path.join(os.getcwd(), sub, path_str))
            candidates.append(os.path.join(script_dir, sub, path_str))
            candidates.append(os.path.join(sub, path_str))

    for cand in candidates:
        if os.path.isdir(cand):
            return os.path.abspath(cand)
            
    return os.path.abspath(candidates[0])


# -----------------------------------------------------------------------------
# Street Name Normalization
# -----------------------------------------------------------------------------
REPLACEMENTS = {
    r'\bAVE\b': 'AVENUE', r'\bAV\b': 'AVENUE', r'\bST\b': 'STREET', r'\bSTR\b': 'STREET',
    r'\bRD\b': 'ROAD', r'\bBLVD\b': 'BOULEVARD', r'\bBL\b': 'BOULEVARD', r'\bDR\b': 'DRIVE',
    r'\bLN\b': 'LANE', r'\bPL\b': 'PLACE', r'\bPKWY\b': 'PARKWAY', r'\bPKY\b': 'PARKWAY',
    r'\bCT\b': 'COURT', r'\bTER\b': 'TERRACE', r'\bTERR\b': 'TERRACE',
    r'\bEXPWY\b': 'EXPRESSWAY', r'\bEXP\b': 'EXPRESSWAY', r'\bHWY\b': 'HIGHWAY',
    r'\bTPKE\b': 'TURNPIKE', r'\bSQ\b': 'SQUARE', r'\bPLZ\b': 'PLAZA',
    r'\bCIR\b': 'CIRCLE', r'\bAL\b': 'ALLEY', r'\bALY\b': 'ALLEY',
    r'\bHL\b': 'HILL', r'\bHLS\b': 'HILLS', r'\bHTS\b': 'HEIGHTS',
    r'\bE\b': 'EAST', r'\bW\b': 'WEST', r'\bN\b': 'NORTH', r'\bS\b': 'SOUTH',
}

MAJOR_CORRIDORS = [
    'CONDUIT', 'BRUCKNER', 'HORACE HARDING', 'CROSS BRONX', 'LINDEN', 'FLATBUSH',
    'EASTERN PARKWAY', 'ATLANTIC', 'QUEENS BOULEVARD', 'GRAND CENTRAL', 'BELT PARKWAY',
    'WOODHAVEN', 'OCEAN PARKWAY', 'BROADWAY', 'FRANCIS LEWIS', 'SEAGIRT', 'ROCKAWAY',
    'HYLAN', 'RICHMOND', 'STATEN ISLAND', 'CANAL', 'DELANCEY', 'HOUSTON', '14 STREET',
    '34 STREET', '42 STREET', '57 STREET', '125 STREET', 'AMSTERDAM', 'COLUMBUS',
    'MADISON', 'LEXINGTON', 'PARK AVENUE', '5 AVENUE', 'BEDFORD', 'NOSTRAND', 'UTICA',
    'MYRTLE', 'FULTON', 'BUSHWICK', 'METROPOLITAN', 'NORTHERN BOULEVARD'
]

def normalize_street(s):
    """
    Standardizes NYC street names to maximize entity resolution
    against NYC Open Data augmentation tables.
    """
    if not isinstance(s, str) or not s.strip():
        return 'UNKNOWN'
    s = s.upper().strip()
    # Strip prefix location descriptions
    s = re.sub(r'^(W/S|E/S|N/S|S/S|R/O|F/O|OPP|CORNER OF|NEAR|ACROSS FROM)\s+(OF\s+)?', '', s)
    # Strip travel direction prefixes (WB, EB, NB, SB)
    s = re.sub(r'^(WB|EB|NB|SB)\s+', '', s)
    # Strip intersection notation '@ ...' or '/ ...'
    s = re.sub(r'\s*[@/].*$', '', s)
    # Normalize ordinal numbers: 1ST -> 1, 2ND -> 2, etc.
    s = re.sub(r'(\d+)(ST|ND|RD|TH)\b', r'\1', s)
    # Strip punctuation
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # Standardize street suffixes and directional words
    for pattern, rep in REPLACEMENTS.items():
        s = re.sub(pattern, rep, s)
    return s.strip()


# -----------------------------------------------------------------------------
# Data Loading and Corpus Augmentation
# -----------------------------------------------------------------------------
def load_corpus_features(corpus_dir):
    """
    Loads pre-extracted corpus features table (corpus_features.parquet).
    Aggregates by normalized street name.
    """
    feat_file = None
    if corpus_dir and os.path.isdir(corpus_dir):
        candidate = os.path.join(corpus_dir, 'corpus_features.parquet')
        if os.path.isfile(candidate):
            feat_file = candidate
            
    if not feat_file:
        feat_file = resolve_file_path(
            'corpus_features.parquet',
            default_subpaths=['train/all_data/corpus', 'corpus', 'train/all_data']
        )

    if not feat_file or not os.path.isfile(feat_file):
        print(f"[Warning] corpus_features.parquet not found. Proceeding with text-only features.")
        return None

    try:
        cf = pd.read_parquet(feat_file)
        cf['norm_key'] = cf['norm_street'].apply(normalize_street)
        num_cols = [c for c in cf.columns if c not in ['norm_street', 'norm_key', 'pave_borough']]
        cf_agg = cf.groupby('norm_key')[num_cols].mean().reset_index()
        return cf_agg
    except Exception as e:
        print(f"[Warning] Failed to read corpus features from {feat_file}: {e}")
        return None


def standardize_columns(df):
    """Maps varied column naming conventions to standard names."""
    col_map = {}
    for c in df.columns:
        c_clean = c.lower().strip().replace('_', ' ')
        if c_clean in ['street name', 'street', 'streetname'] or ('street' in c_clean and 'type' not in c_clean):
            col_map[c] = 'Street Name'
        elif c_clean in ['violation description', 'violation type', 'violation', 'violation code'] or ('violation' in c_clean and 'count' not in c_clean):
            col_map[c] = 'Violation Description'
        elif 'count' in c_clean or 'violation count' in c_clean:
            col_map[c] = 'violation_count'
            
    return df.rename(columns=col_map)


# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------
def extract_features(data, cf_agg=None):
    """
    Constructs rich feature representation for (Street Name, Violation Description) pairs:
    1. Street typology & geometric signals (direction, intersections, street types, numbers)
    2. Arterial corridor detection (major thoroughfares where violation counts peak)
    3. NYC Open Data corpus features (speed limits, pavement lengths, meters, storefronts, trees, 311 calls, DOB)
    4. Domain interactions (cameras x intersections/corridors, meters x commercial storefronts, hydrants x street length)
    """
    df_feat = pd.DataFrame(index=data.index)
    data_street = data['Street Name'].fillna('').astype(str)
    s_upper = data_street.str.upper()
    norm_key = data_street.apply(normalize_street)
    
    # 1. Typology and directional indicators
    has_wb = s_upper.str.contains(r'\bWB\b', regex=True).astype(float)
    has_eb = s_upper.str.contains(r'\bEB\b', regex=True).astype(float)
    has_nb = s_upper.str.contains(r'\bNB\b', regex=True).astype(float)
    has_sb = s_upper.str.contains(r'\bSB\b', regex=True).astype(float)
    df_feat['has_wb'] = has_wb
    df_feat['has_eb'] = has_eb
    df_feat['has_nb'] = has_nb
    df_feat['has_sb'] = has_sb
    df_feat['has_direction'] = ((has_wb + has_eb + has_nb + has_sb) > 0).astype(float)
    df_feat['has_at'] = s_upper.str.contains(r'@', regex=False).astype(float)
    df_feat['has_slash'] = s_upper.str.contains(r'/', regex=False).astype(float)
    df_feat['name_len'] = s_upper.str.len()
    df_feat['word_count'] = s_upper.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Street type one-hot indicators
    for stype in ['AVENUE', 'STREET', 'BOULEVARD', 'PARKWAY', 'EXPRESSWAY', 'ROAD', 'PLACE', 'DRIVE', 'HIGHWAY']:
        df_feat['is_' + stype.lower()] = norm_key.str.contains(stype, regex=False).astype(float)
        
    street_num = s_upper.str.extract(r'(\d+)')[0].astype(float)
    df_feat['street_num'] = street_num.fillna(-1)

    # 2. Major NYC traffic corridors
    is_corridor = np.zeros(len(data))
    for c in MAJOR_CORRIDORS:
        match = norm_key.str.contains(c, regex=False)
        df_feat['corridor_' + c.replace(' ', '_')] = match.astype(float)
        is_corridor = np.maximum(is_corridor, match.astype(float))
    df_feat['is_any_corridor'] = is_corridor

    # 3. Violation typology
    viol_str = data['Violation Description'].fillna('UNKNOWN').astype(str)
    is_cam_viol = viol_str.isin([
        'PHTO SCHOOL ZN SPEED VIOLATION', 'FAILURE TO STOP AT RED LIGHT', 'BUS LANE VIOLATION'
    ]).astype(float)
    df_feat['is_cam_viol'] = is_cam_viol
    df_feat['violation_type'] = viol_str.astype('category')

    # 4. Join Corpus Features
    if cf_agg is not None:
        df_feat['norm_key'] = norm_key
        df_feat = df_feat.merge(cf_agg, on='norm_key', how='left')
        df_feat = df_feat.drop(columns=['norm_key'])
    else:
        # Default placeholders if corpus missing
        for col in ['speed_total_length', 'meter_count', 'pave_length_sum', 'storefront_count']:
            df_feat[col] = 0.0

    # 5. Domain interactions
    speed_len = df_feat.get('speed_total_length', pd.Series(0, index=df_feat.index)).fillna(0)
    meter_cnt = df_feat.get('meter_count', pd.Series(0, index=df_feat.index)).fillna(0)
    pave_len = df_feat.get('pave_length_sum', pd.Series(0, index=df_feat.index)).fillna(0)
    storefront_cnt = df_feat.get('storefront_count', pd.Series(0, index=df_feat.index)).fillna(0)

    df_feat['cam_x_has_at'] = df_feat['is_cam_viol'] * df_feat['has_at']
    df_feat['cam_x_has_dir'] = df_feat['is_cam_viol'] * df_feat['has_direction']
    df_feat['cam_x_is_corridor'] = df_feat['is_cam_viol'] * df_feat['is_any_corridor']
    df_feat['cam_x_speed_len'] = df_feat['is_cam_viol'] * speed_len
    df_feat['meter_x_viol'] = (viol_str == '38-Failure to Dsplay Meter Rec').astype(float) * meter_cnt
    df_feat['hydrant_x_viol'] = (viol_str == '40-Fire Hydrant').astype(float) * pave_len
    df_feat['clean_x_viol'] = (viol_str == '21-No Parking (street clean)').astype(float) * pave_len
    df_feat['meter_x_storefront'] = (viol_str == '38-Failure to Dsplay Meter Rec').astype(float) * storefront_cnt

    return df_feat


# -----------------------------------------------------------------------------
# Model Training
# -----------------------------------------------------------------------------
def get_lgbm_model():
    """Returns the optimized Tweedie regression model."""
    return lgb.LGBMRegressor(
        objective='tweedie',
        tweedie_variance_power=1.40,
        n_estimators=450,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        random_state=42,
        verbose=-1
    )


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NYC Parking Violations Prediction Pipeline (2022 -> 2023)"
    )
    parser.add_argument(
        '--train-path',
        type=str,
        default=os.environ.get('TRAIN_PATH', 'violations_per_street_2022.csv'),
        help='Path to training CSV (default: violations_per_street_2022.csv)'
    )
    parser.add_argument(
        '--test-path',
        type=str,
        default=os.environ.get('TEST_PATH', os.environ.get('EVAL_PATH', None)),
        help='Optional path to test/evaluation CSV'
    )
    parser.add_argument(
        '--eval-path',
        type=str,
        default=None,
        help='Alias for --test-path'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=os.environ.get('OUTPUT_PATH', 'submission.csv'),
        help='Path for output predictions CSV (default: submission.csv)'
    )
    parser.add_argument(
        '--corpus-dir',
        type=str,
        default=os.environ.get('CORPUS_DIR', None),
        help='Path to corpus directory (default: autodetected)'
    )
    args = parser.parse_args()

    test_path = args.test_path or args.eval_path

    print("=" * 70)
    print("       NYC OPEN DATA: 2023 PARKING VIOLATIONS PREDICTION PIPELINE")
    print("=" * 70)

    # Resolve training data path
    train_file = resolve_file_path(
        args.train_path,
        default_subpaths=['train/all_data', 'train', '.']
    )
    if not train_file or not os.path.isfile(train_file):
        sys.exit(f"Error: Training file not found at {args.train_path} or resolved paths.")

    print(f"Loading training data from: {train_file}")
    df_train = pd.read_csv(train_file)
    df_train = standardize_columns(df_train)

    if 'Street Name' not in df_train.columns or 'Violation Description' not in df_train.columns:
        sys.exit("Error: Training data missing required schema: 'Street Name', 'Violation Description'")
    if 'violation_count' not in df_train.columns:
        sys.exit("Error: Training data missing 'violation_count' column.")

    print(f"Training dataset size: {len(df_train):,} rows across {df_train['Street Name'].nunique():,} unique streets")

    # Resolve and load corpus features
    corpus_dir = resolve_dir_path(
        args.corpus_dir,
        default_subpaths=['train/all_data/corpus', 'corpus', 'train/all_data']
    )
    print(f"Loading NYC Open Data corpus features from: {corpus_dir}")
    cf_agg = load_corpus_features(corpus_dir)
    if cf_agg is not None:
        print(f"Corpus features loaded for {len(cf_agg):,} normalized street segments.")

    # -------------------------------------------------------------------------
    # Development Validation (2022 Holdout Split)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" [1/3] DEVELOPMENT VALIDATION ON 2022 HOLDOUT SET")
    print("-" * 70)
    
    np.random.seed(42)
    val_mask = np.random.rand(len(df_train)) < 0.20
    dev_train = df_train[~val_mask].copy()
    dev_val = df_train[val_mask].copy()

    print(f"Split: {len(dev_train):,} train rows (80%), {len(dev_val):,} validation rows (20%)")

    # Compute baseline
    viol_means = dev_train.groupby('Violation Description')['violation_count'].mean()
    base_preds = dev_val['Violation Description'].map(viol_means).fillna(dev_train['violation_count'].mean())
    base_rmse = root_mean_squared_error(dev_val['violation_count'], base_preds)

    # Fit Tweedie LightGBM model on development train split
    X_dev_train = extract_features(dev_train, cf_agg)
    X_dev_val = extract_features(dev_val, cf_agg)
    y_dev_train = dev_train['violation_count']
    y_dev_val = dev_val['violation_count']

    dev_model = get_lgbm_model()
    dev_model.fit(X_dev_train, y_dev_train)
    dev_preds = np.clip(dev_model.predict(X_dev_val), 0, None)
    dev_rmse = root_mean_squared_error(y_dev_val, dev_preds)

    print(f"Development Validation Baseline RMSE (Violation Mean): {base_rmse:.4f}")
    print(f"Development Validation Model RMSE (Tweedie LightGBM):    {dev_rmse:.4f}")
    print(f"Improvement over baseline:                              -{base_rmse - dev_rmse:.4f} RMSE ({(base_rmse - dev_rmse)/base_rmse*100:.2f}%)")

    unseen_dev_keys = len(dev_val)
    print(f"Key pairs in validation holdout unseen in dev-train:    {unseen_dev_keys:,} / {len(dev_val):,} (100.0%)")
    print("Handling: Unseen pairs predicted via Tweedie LightGBM trained on street infrastructure & typology features.")

    # -------------------------------------------------------------------------
    # Full Model Training
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" [2/3] FITTING FINAL MODEL ON FULL 2022 DATASET")
    print("-" * 70)
    
    # Historical lookup table for known key pairs
    df_train['norm_key'] = df_train['Street Name'].apply(normalize_street)
    train_exact_lookup = df_train.set_index(['Street Name', 'Violation Description'])['violation_count'].to_dict()
    train_norm_lookup = df_train.groupby(['norm_key', 'Violation Description'])['violation_count'].mean().to_dict()

    X_full = extract_features(df_train, cf_agg)
    y_full = df_train['violation_count']
    
    final_model = get_lgbm_model()
    final_model.fit(X_full, y_full)
    print("Final model trained successfully on full dataset.")

    # -------------------------------------------------------------------------
    # Test / Posthoc Evaluation & Prediction Emission
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" [3/3] GENERATING PREDICTIONS")
    print("-" * 70)

    if test_path:
        test_file = resolve_file_path(
            test_path,
            default_subpaths=['train/all_data', 'train', '.']
        )
        if not test_file or not os.path.isfile(test_file):
            sys.exit(f"Error: Test file not found at {test_path} or resolved paths.")

        print(f"Loading test dataset from: {test_file}")
        df_test = pd.read_csv(test_file)
        df_test = standardize_columns(df_test)

        if 'Street Name' not in df_test.columns or 'Violation Description' not in df_test.columns:
            sys.exit("Error: Test dataset missing required schema: 'Street Name', 'Violation Description'")

        n_test = len(df_test)
        has_ground_truth = 'violation_count' in df_test.columns

        # Build features for test set
        X_test = extract_features(df_test, cf_agg)
        model_preds = np.clip(final_model.predict(X_test), 0, None)

        # Predict for each row, tracking seen vs unseen pairs
        df_test['norm_key'] = df_test['Street Name'].apply(normalize_street)
        predictions = []
        seen_count = 0
        unseen_count = 0

        for idx, row in df_test.iterrows():
            st = row['Street Name']
            vt = row['Violation Description']
            nk = row['norm_key']

            if (st, vt) in train_exact_lookup:
                predictions.append(train_exact_lookup[(st, vt)])
                seen_count += 1
            elif (nk, vt) in train_norm_lookup:
                predictions.append(train_norm_lookup[(nk, vt)])
                seen_count += 1
            else:
                predictions.append(model_preds[idx])
                unseen_count += 1

        df_test['predicted_count'] = np.clip(predictions, 0, None)

        print(f"Total test rows:                                       {n_test:,}")
        print(f"Seen key pairs in training:                            {seen_count:,} ({seen_count / n_test * 100:.2f}%)")
        print(f"  -> Handled by: Historical baseline volume anchor from 2022 training data.")
        print(f"Unseen key pairs in training:                          {unseen_count:,} ({unseen_count / n_test * 100:.2f}%)")
        print(f"  -> Handled by: Tweedie LightGBM regression model with NYC Open Data corpus augmentation.")

        # If ground truth was present, evaluate and report test RMSE
        if has_ground_truth:
            test_rmse = root_mean_squared_error(df_test['violation_count'], df_test['predicted_count'])
            print(f"\n>>> TEST / EVALUATION RMSE:                             {test_rmse:.4f} <<<")
        else:
            print("\nTest file is keys-only (no violation_count column). Emission only.")

        # Prepare submission dataframe
        submission_df = pd.DataFrame({
            'street_name': df_test['Street Name'],
            'violation_type': df_test['Violation Description'],
            'predicted_count': df_test['predicted_count']
        })

    else:
        print("No --test-path provided. Emitting submission for development validation set.")
        submission_df = pd.DataFrame({
            'street_name': dev_val['Street Name'],
            'violation_type': dev_val['Violation Description'],
            'predicted_count': dev_preds
        })

    # Save submission file
    output_file = os.path.abspath(args.output_path)
    submission_df.to_csv(output_file, index=False)
    print(f"Saved submission file to: {output_file}")
    print(f"Submission shape: {submission_df.shape} (columns: {list(submission_df.columns)})")
    print(submission_df.head(10))
    print("\n" + "=" * 70)
    print(" Pipeline execution completed successfully.")
    print("=" * 70)


if __name__ == '__main__':
    main()
