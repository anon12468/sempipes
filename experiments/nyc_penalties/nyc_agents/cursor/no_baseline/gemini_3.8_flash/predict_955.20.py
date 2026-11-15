#!/usr/bin/env python3
"""
NYC Parking Violations Prediction Pipeline
Goal: Predict 2023 parking violation counts per (street_name, violation_type) pair
using 2022 data and corpus augmentation features.
"""

import os
import re
import sys
import argparse
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


# ---------------------------------------------------------------------------
# Path Resolution Helpers
# ---------------------------------------------------------------------------
def resolve_file(path, candidates):
    if path and os.path.exists(path):
        return path
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def resolve_dir(path, candidates):
    if path and os.path.isdir(path):
        return path
    for c in candidates:
        if os.path.isdir(c):
            return c
    return path


# ---------------------------------------------------------------------------
# Text Normalization & Feature Extraction Helpers
# ---------------------------------------------------------------------------
def normalize_street(s):
    """Normalize street names to reduce spelling and abbreviation variances."""
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = re.sub(r'[\.,#\(\)]', ' ', s)
    tokens = s.split()
    replacements = {
        'AV': 'AVE', 'AVENUE': 'AVE', 'STREET': 'ST', 'STR': 'ST', 'ROAD': 'RD',
        'BOULEVARD': 'BLVD', 'BOUL': 'BLVD', 'DRIVE': 'DR', 'LANE': 'LN',
        'PLACE': 'PL', 'COURT': 'CT', 'EXPRESSWAY': 'EXPY', 'EXPWY': 'EXPY',
        'EXP': 'EXPY', 'PARKWAY': 'PKWY', 'PKW': 'PKWY', 'TURNPIKE': 'TPKE',
        'EAST': 'E', 'WEST': 'W', 'NORTH': 'N', 'SOUTH': 'S'
    }
    new_tokens = []
    for t in tokens:
        m = re.match(r'^(\d+)(ST|ND|RD|TH)$', t)
        if m:
            t = m.group(1)
        t = replacements.get(t, t)
        new_tokens.append(t)
    return ' '.join(new_tokens)


def extract_base_street(s):
    """Extract base road corridor from directional camera / intersection names."""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r'^(EB|WB|NB|SB|E/B|W/B|N/B|S/B)\s+', '', s, flags=re.IGNORECASE)
    if '@' in s:
        s = s.split('@')[0].strip()
    return normalize_street(s)


CAMERA_TYPES = {
    'PHTO SCHOOL ZN SPEED VIOLATION',
    'BUS LANE VIOLATION',
    'FAILURE TO STOP AT RED LIGHT'
}


# ---------------------------------------------------------------------------
# Corpus Data Loader
# ---------------------------------------------------------------------------
class CorpusFeatureExtractor:
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.meters_feat = {}
        self.speed_feat = {}
        self.pave_feat = {}
        self.bike_feat = {}
        self.litter_feat = {}
        self.fac_feat = {}
        self.store_feat = {}
        self._load_corpus()

    def _load_corpus(self):
        if not self.corpus_dir or not os.path.isdir(self.corpus_dir):
            print(f"[Warning] Corpus directory '{self.corpus_dir}' not found. Using text features only.")
            return

        # 1. Parking Meters
        meters_file = os.path.join(self.corpus_dir, 'parking_meters_locations_and_status__693u-uax6.parquet')
        if os.path.exists(meters_file):
            try:
                df_m = pd.read_parquet(meters_file, columns=['On_Street', 'Status'])
                df_m['norm'] = df_m['On_Street'].dropna().apply(normalize_street)
                self.meters_feat = df_m.groupby('norm').agg(
                    n_meters=('Status', 'count'),
                    n_active_meters=('Status', lambda s: (s == 'Active').sum())
                ).to_dict('index')
            except Exception as e:
                pass

        # 2. Vision Zero Speed Limits
        speed_file = os.path.join(self.corpus_dir, 'vzv_speed_limits__5mad-ntua.parquet')
        if os.path.exists(speed_file):
            try:
                df_s = pd.read_parquet(speed_file, columns=['street', 'postvz_sl', 'Shape_Leng'])
                df_s['norm'] = df_s['street'].dropna().apply(normalize_street)
                self.speed_feat = df_s.groupby('norm').agg(
                    n_speed_segs=('Shape_Leng', 'count'),
                    total_speed_len=('Shape_Leng', 'sum'),
                    max_speed_limit=('postvz_sl', 'max'),
                    avg_speed_limit=('postvz_sl', 'mean')
                ).to_dict('index')
            except Exception as e:
                pass

        # 3. Street Pavement Ratings & Borough
        pave_file = os.path.join(self.corpus_dir, 'street_pavement_ratings__6yyb-pb25.parquet')
        if os.path.exists(pave_file):
            try:
                df_p = pd.read_parquet(pave_file, columns=['OnStreetName', 'BoroughName', 'SystemRating', 'LocationGeometry.STLength'])
                df_p['norm'] = df_p['OnStreetName'].dropna().apply(normalize_street)
                self.pave_feat = df_p.groupby('norm').agg(
                    n_pave_segs=('SystemRating', 'count'),
                    avg_pave_rating=('SystemRating', 'mean'),
                    pave_len=('LocationGeometry.STLength', 'sum'),
                    primary_borough=('BoroughName', lambda b: b.mode().iloc[0] if len(b) > 0 else 'UNKNOWN')
                ).to_dict('index')
            except Exception as e:
                pass

        # 4. Bicycle Parking
        bike_file = os.path.join(self.corpus_dir, 'bicycle_parking__592z-n7dk.parquet')
        if os.path.exists(bike_file):
            try:
                df_b = pd.read_parquet(bike_file, columns=['OnStreet'])
                df_b['norm'] = df_b['OnStreet'].dropna().apply(normalize_street)
                self.bike_feat = df_b.groupby('norm').size().to_dict()
            except Exception as e:
                pass

        # 5. DSNY Litter Baskets
        litter_file = os.path.join(self.corpus_dir, 'dsny_litter_basket_inventory__8znf-7b2c.parquet')
        if os.path.exists(litter_file):
            try:
                df_l = pd.read_parquet(litter_file, columns=['STREETNAME1'])
                df_l['norm'] = df_l['STREETNAME1'].dropna().apply(normalize_street)
                self.litter_feat = df_l.groupby('norm').size().to_dict()
            except Exception as e:
                pass

        # 6. City Facilities
        fac_file = os.path.join(self.corpus_dir, 'facilities__67g2-p84d.parquet')
        if os.path.exists(fac_file):
            try:
                df_f = pd.read_parquet(fac_file, columns=['STREETNAME'])
                df_f['norm'] = df_f['STREETNAME'].dropna().apply(normalize_street)
                self.fac_feat = df_f.groupby('norm').size().to_dict()
            except Exception as e:
                pass

        # 7. Storefronts
        store_file = os.path.join(self.corpus_dir, 'storefronts_reported_vacant_or_not__92iy-9c3n.parquet')
        if os.path.exists(store_file):
            try:
                df_st = pd.read_parquet(store_file, columns=['Property Street'])
                df_st['norm'] = df_st['Property Street'].dropna().apply(normalize_street)
                self.store_feat = df_st.groupby('norm').size().to_dict()
            except Exception as e:
                pass

    def build_features(self, df_input):
        feats = []
        for idx, row in df_input.iterrows():
            st = str(row['street_name'])
            vt = str(row['violation_type'])
            norm = normalize_street(st)
            base = extract_base_street(st)

            st_upper = st.upper()
            has_dir = int(bool(re.match(r'^(EB|WB|NB|SB|E/B|W/B|N/B|S/B)\s', st_upper)))
            has_at = int('@' in st)
            is_ave = int('AVE' in st_upper or 'AVENUE' in st_upper)
            is_st = int('ST' in st_upper or 'STREET' in st_upper)
            is_blvd = int('BLVD' in st_upper or 'BOULEVARD' in st_upper)
            is_expy = int('EXPY' in st_upper or 'EXPRESSWAY' in st_upper or 'EXP' in st_upper)
            is_pkwy = int('PKWY' in st_upper or 'PARKWAY' in st_upper)
            is_bway = int('BROADWAY' in st_upper)
            name_len = len(st)
            n_words = len(st.split())

            is_cam = int(vt in CAMERA_TYPES)
            is_speed_cam = int(vt == 'PHTO SCHOOL ZN SPEED VIOLATION')
            is_bus_cam = int(vt == 'BUS LANE VIOLATION')
            is_red_light = int(vt == 'FAILURE TO STOP AT RED LIGHT')
            is_meter = int(vt == '38-Failure to Dsplay Meter Rec')
            is_clean = int(vt == '21-No Parking (street clean)')
            is_hydrant = int(vt == '40-Fire Hydrant')
            is_sticker = int('Sticker' in vt)

            m_info = self.meters_feat.get(norm) or self.meters_feat.get(base) or {}
            n_meters = m_info.get('n_meters', 0)
            n_active_meters = m_info.get('n_active_meters', 0)

            s_info = self.speed_feat.get(norm) or self.speed_feat.get(base) or {}
            n_speed_segs = s_info.get('n_speed_segs', 0)
            total_speed_len = s_info.get('total_speed_len', 0.0)
            max_speed = s_info.get('max_speed_limit', 0)
            avg_speed = s_info.get('avg_speed_limit', 0.0)

            p_info = self.pave_feat.get(norm) or self.pave_feat.get(base) or {}
            n_pave_segs = p_info.get('n_pave_segs', 0)
            avg_pave_rating = p_info.get('avg_pave_rating', 0.0)
            pave_len = p_info.get('pave_len', 0.0)
            borough = p_info.get('primary_borough', 'UNKNOWN')

            n_bikes = self.bike_feat.get(norm, 0) or self.bike_feat.get(base, 0)
            n_litters = self.litter_feat.get(norm, 0) or self.litter_feat.get(base, 0)
            n_facs = self.fac_feat.get(norm, 0) or self.fac_feat.get(base, 0)
            n_stores = self.store_feat.get(norm, 0) or self.store_feat.get(base, 0)

            cam_dir = is_cam * has_dir
            meter_int = is_meter * n_meters
            speed_len_cam = is_cam * np.log1p(total_speed_len)
            clean_len = is_clean * np.log1p(total_speed_len)
            hydrant_len = is_hydrant * np.log1p(total_speed_len)

            feats.append({
                'name_len': name_len,
                'n_words': n_words,
                'has_dir': has_dir,
                'has_at': has_at,
                'is_ave': is_ave,
                'is_st': is_st,
                'is_blvd': is_blvd,
                'is_expy': is_expy,
                'is_pkwy': is_pkwy,
                'is_bway': is_bway,
                'is_cam': is_cam,
                'is_speed_cam': is_speed_cam,
                'is_bus_cam': is_bus_cam,
                'is_red_light': is_red_light,
                'is_meter': is_meter,
                'is_clean': is_clean,
                'is_hydrant': is_hydrant,
                'is_sticker': is_sticker,
                'n_meters': n_meters,
                'n_active_meters': n_active_meters,
                'n_speed_segs': n_speed_segs,
                'total_speed_len': total_speed_len,
                'log_speed_len': np.log1p(total_speed_len),
                'max_speed': max_speed,
                'avg_speed': avg_speed,
                'n_pave_segs': n_pave_segs,
                'avg_pave_rating': avg_pave_rating,
                'pave_len': pave_len,
                'n_bikes': n_bikes,
                'n_litters': n_litters,
                'n_facs': n_facs,
                'n_stores': n_stores,
                'cam_dir': cam_dir,
                'meter_int': meter_int,
                'speed_len_cam': speed_len_cam,
                'clean_len': clean_len,
                'hydrant_len': hydrant_len,
                'borough': borough,
                'violation_type': vt
            })
        X = pd.DataFrame(feats)
        for col in ['borough', 'violation_type']:
            X[col] = X[col].astype('category')
        return X


# ---------------------------------------------------------------------------
# Standardize DataFrame Schema
# ---------------------------------------------------------------------------
def standardize_df(df):
    """Map flexible input columns to standard names: street_name, violation_type, violation_count (optional)."""
    col_map = {}
    for c in df.columns:
        c_clean = str(c).strip().lower().replace(' ', '_')
        if c_clean in ['street_name', 'street', 'st_name']:
            col_map[c] = 'street_name'
        elif c_clean in ['violation_description', 'violation_type', 'violation']:
            col_map[c] = 'violation_type'
        elif c_clean in ['violation_count', 'count', 'target', 'violations']:
            col_map[c] = 'violation_count'
    standardized = df.rename(columns=col_map)
    if 'street_name' not in standardized.columns or 'violation_type' not in standardized.columns:
        raise ValueError(f"Could not find required columns in {df.columns.tolist()}. Expected street_name and violation_type.")
    standardized['street_name'] = standardized['street_name'].fillna('').astype(str)
    standardized['violation_type'] = standardized['violation_type'].fillna('').astype(str)
    return standardized


# ---------------------------------------------------------------------------
# Hierarchical Parking Violation Predictor
# ---------------------------------------------------------------------------
class HierarchicalViolationPredictor:
    def __init__(self, corpus_extractor):
        self.corpus_extractor = corpus_extractor
        self.exact_map = {}
        self.upper_map = {}
        self.norm_map = {}
        self.corridor_means = {}
        self.st_noncam_tot = {}
        self.norm_noncam_tot = {}
        self.v_shares_nc = {}
        self.global_vtype_means = {}
        self.global_mean = 100.0
        self.model = None

    def fit(self, train_df):
        print(f"Fitting predictor on {len(train_df)} training records...")
        trn = train_df.copy()
        trn['st_upper'] = trn['street_name'].str.upper().str.strip()
        trn['norm'] = trn['street_name'].apply(normalize_street)
        trn['base'] = trn['street_name'].apply(extract_base_street)
        y = trn['violation_count'].values
        self.global_mean = float(np.mean(y))

        # 1. Exact pair lookup
        self.exact_map = trn.set_index(['street_name', 'violation_type'])['violation_count'].to_dict()

        # 2. Case / strip lookup
        self.upper_map = trn.groupby(['st_upper', 'violation_type'])['violation_count'].mean().to_dict()

        # 3. Normalized street lookup
        self.norm_map = trn.groupby(['norm', 'violation_type'])['violation_count'].mean().to_dict()

        # 4. Camera corridor averages
        trn_cam = trn[trn['violation_type'].isin(CAMERA_TYPES)]
        if len(trn_cam) > 0:
            self.corridor_means = trn_cam.groupby(['base', 'violation_type'])['violation_count'].mean().to_dict()

        # 5. Non-camera street activity & violation shares
        trn_noncam = trn[~trn['violation_type'].isin(CAMERA_TYPES)]
        if len(trn_noncam) > 0:
            tot_noncam = trn_noncam['violation_count'].sum()
            self.v_shares_nc = (trn_noncam.groupby('violation_type')['violation_count'].sum() / tot_noncam).to_dict()
            self.st_noncam_tot = trn_noncam.groupby('street_name')['violation_count'].sum().to_dict()
            self.norm_noncam_tot = trn_noncam.groupby('norm')['violation_count'].sum().to_dict()

        # 6. Global violation type averages
        self.global_vtype_means = trn.groupby('violation_type')['violation_count'].mean().to_dict()

        # 7. LightGBM model for unseen streets and generalisation
        X_trn = self.corpus_extractor.build_features(trn)
        self.model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.model.fit(X_trn, y)
        print("Model training complete.")

    def predict(self, test_df, return_details=False):
        t_df = test_df.copy()
        t_df['st_upper'] = t_df['street_name'].str.upper().str.strip()
        t_df['norm'] = t_df['street_name'].apply(normalize_street)
        t_df['base'] = t_df['street_name'].apply(extract_base_street)

        # LightGBM predictions
        X_test = self.corpus_extractor.build_features(t_df)
        m_preds = np.clip(self.model.predict(X_test), 1.0, None)

        preds = []
        match_breakdown = {
            'exact_seen': 0,
            'upper_match': 0,
            'norm_match': 0,
            'camera_corridor_match': 0,
            'noncam_street_activity': 0,
            'noncam_norm_activity': 0,
            'model_prediction': 0
        }
        unseen_count = 0

        for i, (_, row) in enumerate(t_df.iterrows()):
            st = row['street_name']
            vt = row['violation_type']
            st_u = row['st_upper']
            norm = row['norm']
            base = row['base']

            # Level 1: Exact match in training data
            if (st, vt) in self.exact_map:
                pred = self.exact_map[(st, vt)]
                match_breakdown['exact_seen'] += 1

            # Level 2: Case/strip match (unseen exact pair)
            elif (st_u, vt) in self.upper_map:
                pred = self.upper_map[(st_u, vt)]
                match_breakdown['upper_match'] += 1
                unseen_count += 1

            # Level 3: Normalized street name match (unseen exact pair)
            elif (norm, vt) in self.norm_map:
                pred = self.norm_map[(norm, vt)]
                match_breakdown['norm_match'] += 1
                unseen_count += 1

            # Level 4: Camera corridor match
            elif vt in CAMERA_TYPES and (base, vt) in self.corridor_means:
                corridor_val = self.corridor_means[(base, vt)]
                pred = 0.7 * corridor_val + 0.3 * m_preds[i]
                match_breakdown['camera_corridor_match'] += 1
                unseen_count += 1

            # Level 5: Non-camera street activity scaling
            elif vt not in CAMERA_TYPES and st in self.st_noncam_tot:
                p_share = self.st_noncam_tot[st] * self.v_shares_nc.get(vt, 0.1)
                pred = 0.8 * p_share + 0.2 * m_preds[i]
                match_breakdown['noncam_street_activity'] += 1
                unseen_count += 1

            # Level 6: Non-camera normalized street activity scaling
            elif vt not in CAMERA_TYPES and norm in self.norm_noncam_tot:
                p_share = self.norm_noncam_tot[norm] * self.v_shares_nc.get(vt, 0.1)
                pred = 0.8 * p_share + 0.2 * m_preds[i]
                match_breakdown['noncam_norm_activity'] += 1
                unseen_count += 1

            # Level 7: Machine Learning Model (LightGBM on corpus & text)
            else:
                pred = m_preds[i]
                match_breakdown['model_prediction'] += 1
                unseen_count += 1

            # Ensure non-negative numeric output
            preds.append(max(1.0, float(pred)))

        preds = np.array(preds)
        if return_details:
            return preds, unseen_count, match_breakdown
        return preds


# ---------------------------------------------------------------------------
# Development Validation Routine (Grouped / K-Fold Holdout on 2022)
# ---------------------------------------------------------------------------
def run_development_validation(train_df, corpus_extractor, val_size=0.2, seed=42):
    print("=" * 70)
    print("RUNNING 2022 DEVELOPMENT VALIDATION")
    print(f"Splitting 2022 data ({len(train_df)} rows) into train ({1-val_size:.0%}) and holdout ({val_size:.0%})...")
    print("=" * 70)

    kf = KFold(n_splits=int(1 / val_size), shuffle=True, random_state=seed)
    trn_idx, val_idx = next(kf.split(train_df))

    trn_split = train_df.iloc[trn_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    val_predictor = HierarchicalViolationPredictor(corpus_extractor)
    val_predictor.fit(trn_split)

    preds, unseen_count, breakdown = val_predictor.predict(val_split, return_details=True)
    y_true = val_split['violation_count'].values
    dev_rmse = root_mean_squared_error(y_true, preds)

    # Compute baseline for comparison
    v_means = trn_split.groupby('violation_type')['violation_count'].mean()
    naive_preds = val_split['violation_type'].map(v_means).fillna(trn_split['violation_count'].mean()).values
    naive_rmse = root_mean_squared_error(y_true, naive_preds)

    print("\n--- Development Validation Results ---")
    print(f"Holdout Set Size: {len(val_split)} pairs")
    print(f"Key pairs unseen in training split: {unseen_count} / {len(val_split)} ({unseen_count / len(val_split):.1%})")
    print("Unseen Pairs Handling Breakdown:")
    for k, v in breakdown.items():
        print(f"  - {k}: {v} rows ({v / len(val_split):.1%})")
    print(f"\nNaive Baseline (Violation Type Mean) RMSE: {naive_rmse:.2f}")
    print(f"Development Pipeline RMSE:                 {dev_rmse:.2f}")
    print(f"Improvement over Baseline:                  {naive_rmse - dev_rmse:.2f} ({(naive_rmse - dev_rmse) / naive_rmse:.1%})")
    print("=" * 70)
    return dev_rmse


# ---------------------------------------------------------------------------
# Main CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict NYC parking violations per (street_name, violation_type) pair.")
    parser.add_argument('--train-path', type=str, default=os.getenv('TRAIN_PATH', 'violations_per_street_2022.csv'),
                        help="Path to training CSV (default: violations_per_street_2022.csv)")
    parser.add_argument('--test-path', '--eval-path', type=str, default=os.getenv('TEST_PATH', os.getenv('EVAL_PATH', None)),
                        dest='test_path', help="Optional path to test/eval CSV for inference / posthoc scoring")
    parser.add_argument('--corpus-dir', type=str, default=os.getenv('CORPUS_DIR', 'corpus'),
                        help="Directory containing corpus augmentation datasets (default: corpus)")
    parser.add_argument('--output-path', '--submission-path', type=str, default=os.getenv('OUTPUT_PATH', 'submission.csv'),
                        dest='output_path', help="Path to write submission CSV (default: submission.csv)")
    parser.add_argument('--val-size', type=float, default=0.2, help="Validation holdout fraction (default: 0.2)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    start_time = time.time()

    # 1. Resolve paths
    train_path = resolve_file(args.train_path, [
        args.train_path,
        'violations_per_street_2022.csv',
        'train/all_data/violations_per_street_2022.csv',
        'all_data/violations_per_street_2022.csv'
    ])
    corpus_dir = resolve_dir(args.corpus_dir, [
        args.corpus_dir,
        'corpus',
        'train/all_data/corpus',
        'all_data/corpus'
    ])

    print(f"Loading training data from: {train_path}")
    if not os.path.exists(train_path):
        print(f"[Error] Training file '{train_path}' not found.")
        sys.exit(1)

    train_raw = pd.read_csv(train_path)
    train_df = standardize_df(train_raw)
    print(f"Loaded {len(train_df)} training records across {train_df['street_name'].nunique()} unique streets.")

    # 2. Initialize Corpus Extractor
    print(f"Initializing corpus feature extractor from: {corpus_dir}")
    corpus_extractor = CorpusFeatureExtractor(corpus_dir)

    # 3. Always run and report Development Validation
    dev_rmse = run_development_validation(train_df, corpus_extractor, val_size=args.val_size, seed=args.seed)

    # 4. Determine Target for Submission & Scoring
    test_path = None
    if args.test_path:
        test_path = resolve_file(args.test_path, [args.test_path])

    if test_path and os.path.exists(test_path):
        print(f"\nProcessing posthoc test/eval file: {test_path}")
        eval_raw = pd.read_csv(test_path)
        eval_df = standardize_df(eval_raw)
        has_ground_truth = 'violation_count' in eval_df.columns
        target_df = eval_df
    else:
        print("\nNo test-path supplied. Generating submission for development evaluation set.")
        has_ground_truth = False
        target_df = train_df

    # 5. Train full predictor on entire training dataset
    full_predictor = HierarchicalViolationPredictor(corpus_extractor)
    full_predictor.fit(train_df)

    # 6. Generate predictions
    print(f"Generating predictions for {len(target_df)} rows...")
    predictions, unseen_count, breakdown = full_predictor.predict(target_df, return_details=True)

    # 7. Write submission.csv
    submission_df = pd.DataFrame({
        'street_name': target_df['street_name'],
        'violation_type': target_df['violation_type'],
        'predicted_count': np.round(predictions, 2)
    })
    submission_df.to_csv(args.output_path, index=False)
    print(f"\nSuccessfully wrote submission to: {args.output_path}")
    print(f"Submission shape: {submission_df.shape} (columns: {submission_df.columns.tolist()})")
    print(submission_df.head(5))

    # 8. Unseen Pairs Reporting
    print("\n" + "=" * 70)
    print("KEY PAIR COVERAGE & UNSEEN PAIRS REPORT")
    print("=" * 70)
    print(f"Total target key pairs: {len(target_df)}")
    print(f"Key pairs unseen in training: {unseen_count} ({unseen_count / len(target_df):.1%})")
    print("How unseen pairs were handled:")
    for strategy, count in breakdown.items():
        if strategy != 'exact_seen':
            print(f"  - {strategy:25}: {count:6d} rows ({count / len(target_df):.1%})")

    # 9. Posthoc Scoring (if ground truth available)
    if test_path and has_ground_truth:
        y_true = eval_df['violation_count'].values
        posthoc_rmse = root_mean_squared_error(y_true, predictions)
        print("\n" + "=" * 70)
        print("POSTHOC EVALUATION SCORING")
        print("=" * 70)
        print(f"Evaluated on ground truth from: {test_path}")
        print(f"Rows evaluated:                 {len(y_true)}")
        print(f"Development Validation RMSE:   {dev_rmse:.2f}")
        print(f"Posthoc Dataset RMSE:          {posthoc_rmse:.2f}")
        print("=" * 70)

    print(f"\nPipeline finished in {time.time() - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
