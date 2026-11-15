"""
Course High Enrollment Prediction Pipeline
Academic IT Data Warehouse - MIT Course Catalog & Enrollment
Task: Predict whether each course offering per term will have high enrollment (Y/N)
      relative to other courses in the same department and term (top quartile).
"""

import os
import re
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import lightgbm as lgb


# Default Paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_DIR = os.path.join(WORKSPACE_DIR, 'table_splits', 'train')
DEFAULT_TRAIN_GOLD = os.path.join(DEFAULT_TRAIN_DIR, 'gold_enrollment_train.csv')
DEFAULT_TEST_GOLD = os.path.join("/Users/USER/BEAVER/test", 'gold_enrollment_test.csv')
DEFAULT_MODEL_PATH = os.path.join(WORKSPACE_DIR, 'table_splits', 'lgbm_model.pkl')
DEFAULT_PREDS_PATH = os.path.join(WORKSPACE_DIR, 'table_splits', 'predictions.csv')

FEATURE_NAMES = [
    'lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean', 'roll3_log',
    'lag1_log', 'lag2_log', 'trend_roll_lag1', 'trend_lag1_lag2', 'ratio_lag1_lag2',
    'ratio_roll3_dept', 'ratio_lag1_dept', 'course_level', 'is_grad_level',
    'is_intro_level', 'is_thesis', 'is_urop', 'is_special', 'is_harvard',
    'total_units', 'lecture_units', 'lab_units', 'preparation_units',
    'title_len', 'title_words', 'title_seminar', 'title_thesis', 'title_research',
    'title_special', 'title_intro', 'title_workshop', 'title_lab', 'title_project',
    'title_advanced', 'title_colloquium', 'title_independent', 'title_practicum',
    'title_studio', 'n_sections', 'n_lecture', 'n_lab', 'n_recitation',
    'n_design', 'n_evening', 'n_with_time', 'n_ose', 'n_repeatable', 'n_fac_sec',
    'n_places', 'cluster_n', 'is_master', 'subj_hist_rate', 'subj_hist_cnt',
    'subj_hist_enr_mean', 'subj_hist_enr_max', 'subj_hist_enr_median',
    'subj_season_rate', 'subj_season_cnt', 'dept_hist_rate', 'dept_hist_cnt',
    'prefix_hist_rate', 'prefix_hist_cnt', 'fac_hist_rate', 'fac_hist_cnt',
    'dept_p75', 'cat_var_units', 'cat_prereq_len', 'OFFER_DEPT_CODE',
    'OFFER_SCHOOL_NAME', 'prefix', 'season', 'CLUSTER_TYPE', 'HASS_ATTRIBUTE',
    'GIR_ATTRIBUTE'
]

CAT_COLS = [
    'OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'prefix', 'season',
    'CLUSTER_TYPE', 'HASS_ATTRIBUTE', 'GIR_ATTRIBUTE'
]

TITLE_KEYWORDS = [
    'seminar', 'thesis', 'research', 'special', 'intro', 'workshop',
    'lab', 'project', 'advanced', 'colloquium', 'independent', 'practicum', 'studio'
]


def parse_course_level(sid):
    parts = str(sid).split('.')
    if len(parts) > 1 and parts[1][:1].isdigit():
        digits = re.findall(r'^\d+', parts[1])
        if digits:
            return float(digits[0][:2])
    return 0.0


def extract_features(keys_df, data_dir, priors, cat_maps):
    """
    Extract all 73 engineered features for the provided (TERM_CODE, SUBJECT_ID_SORT) keys.
    Vectorized and optimized for performance across hundreds of thousands of records.
    """
    df = keys_df[['TERM_CODE', 'SUBJECT_ID_SORT']].copy()
    df['season'] = df['TERM_CODE'].str[4:]
    df['prefix'] = df['SUBJECT_ID_SORT'].apply(lambda x: str(x).split('.')[0])

    target_terms = set(df['TERM_CODE'].unique())

    # 1. Subject Offered aggregations
    so_path = os.path.join(data_dir, 'SUBJECT_OFFERED.csv')
    if os.path.exists(so_path):
        so = pd.read_csv(so_path, low_memory=False)
        so = so[so['TERM_CODE'].isin(target_terms)].copy()

        so['is_lecture'] = (so['IS_LECTURE_SECTION'] == 'Y').astype(int)
        so['is_lab'] = (so['IS_LAB_SECTION'] == 'Y').astype(int)
        so['is_recitation'] = (so['IS_RECITATION_SECTION'] == 'Y').astype(int)
        so['is_design'] = (so['IS_DESIGN_SECTION'] == 'Y').astype(int)
        so['is_evening'] = so['MEET_TIME'].astype(str).str.contains('EVE').astype(int)
        so['has_time'] = so['MEET_TIME'].notnull().astype(int)
        so['is_ose'] = (so['IS_OSE_SUBJECT'] == 'Y').astype(int)
        so['is_repeatable'] = (so['IS_REPEATABLE_SUBJECT'] == 'Y').astype(int)
        so['has_fac'] = so['RESPONSIBLE_FACULTY_NAME'].notnull().astype(int)
        so['is_master'] = (so['IS_MASTER_SECTION'] == 'Y').astype(int)
        so['has_place'] = so['MEET_PLACE'].notnull().astype(int)

        so_agg = so.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg({
            'OFFER_DEPT_CODE': 'first',
            'OFFER_SCHOOL_NAME': 'first',
            'RESPONSIBLE_FACULTY_NAME': 'first',
            'is_lecture': ['count', 'sum'],
            'is_lab': 'sum',
            'is_recitation': 'sum',
            'is_design': 'sum',
            'is_evening': 'sum',
            'has_time': 'sum',
            'is_ose': 'sum',
            'is_repeatable': 'sum',
            'has_fac': 'sum',
            'has_place': 'sum',
            'is_master': 'max',
            'CLUSTER_TYPE': 'first',
        }).reset_index()

        so_agg.columns = [
            'TERM_CODE', 'SUBJECT_ID_SORT', 'OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'RESPONSIBLE_FACULTY_NAME',
            'n_sections', 'n_lecture', 'n_lab', 'n_recitation', 'n_design',
            'n_evening', 'n_with_time', 'n_ose', 'n_repeatable', 'n_fac_sec', 'n_places',
            'is_master', 'CLUSTER_TYPE'
        ]
        df = df.merge(so_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        for c in ['n_sections', 'n_lecture', 'n_lab', 'n_recitation', 'n_design', 'n_evening', 'n_with_time', 'n_ose', 'n_repeatable', 'n_fac_sec', 'n_places', 'is_master']:
            df[c] = 0
        df['OFFER_DEPT_CODE'] = 'MISSING'
        df['OFFER_SCHOOL_NAME'] = 'MISSING'
        df['RESPONSIBLE_FACULTY_NAME'] = np.nan
        df['CLUSTER_TYPE'] = 'MISSING'

    # 2. Subject Summary units & cluster size
    ss_path = os.path.join(data_dir, 'SUBJECT_SUMMARY.csv')
    if os.path.exists(ss_path):
        ss = pd.read_csv(ss_path, low_memory=False)
        ss = ss[ss['TERM_CODE'].isin(target_terms)][[
            'TERM_CODE', 'SUBJECT_ID_SORT', 'SUBJECT_TITLE',
            'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREP_UNITS', 'CLUSTER_LIST'
        ]].drop_duplicates(['TERM_CODE', 'SUBJECT_ID_SORT'])

        ss['cluster_n'] = ss['CLUSTER_LIST'].fillna('').apply(lambda s: len(str(s).split(',')) if str(s).strip() else 0)
        df = df.merge(ss[['TERM_CODE', 'SUBJECT_ID_SORT', 'SUBJECT_TITLE', 'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREP_UNITS', 'cluster_n']], on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        df['SUBJECT_TITLE'] = ''
        df['TOTAL_UNITS'] = 0
        df['LECTURE_UNITS'] = 0
        df['LAB_UNITS'] = 0
        df['PREP_UNITS'] = 0
        df['cluster_n'] = 0

    df['cluster_n'] = df['cluster_n'].fillna(0)

    # 3. Enrollment History features
    eh_path = os.path.join(data_dir, 'ENROLLMENT_HISTORY_FEATURES.csv')
    if os.path.exists(eh_path):
        eh = pd.read_csv(eh_path)
        df = df.merge(eh, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        df['lag1_enrollment'] = 0.0
        df['lag2_enrollment'] = 0.0
        df['roll3_enrollment_mean'] = 0.0

    df['lag1_enrollment'] = df['lag1_enrollment'].fillna(0.0)
    df['lag2_enrollment'] = df['lag2_enrollment'].fillna(0.0)
    df['roll3_enrollment_mean'] = df['roll3_enrollment_mean'].fillna(0.0)

    # 4. Catalog Metadata
    cis_path = os.path.join(data_dir, 'CIS_COURSE_CATALOG.csv')
    if os.path.exists(cis_path):
        cis = pd.read_csv(cis_path, low_memory=False)
        cis['sid'] = cis['SUBJECT_CODE'].astype(str) + '.' + cis['SUBJECT_NUMBER'].astype(str)
        cis_sub = cis[['sid', 'IS_VARIABLE_UNITS', 'PREREQUISITES', 'HASS_ATTRIBUTE', 'GIR_ATTRIBUTE']].drop_duplicates('sid')
        df = df.merge(cis_sub, left_on='SUBJECT_ID_SORT', right_on='sid', how='left')
    else:
        df['IS_VARIABLE_UNITS'] = 'N'
        df['PREREQUISITES'] = ''
        df['HASS_ATTRIBUTE'] = 'MISSING'
        df['GIR_ATTRIBUTE'] = 'MISSING'

    # 5. Priors
    df = df.merge(priors.get('dept_p75', pd.DataFrame({'OFFER_DEPT_CODE': [], 'dept_p75': []})), on='OFFER_DEPT_CODE', how='left')
    df['dept_p75'] = df['dept_p75'].fillna(10.0)

    df = df.merge(priors.get('subj_stat', pd.DataFrame({'SUBJECT_ID_SORT': []})), on='SUBJECT_ID_SORT', how='left')
    df['sid_season'] = df['SUBJECT_ID_SORT'].astype(str) + '|' + df['season'].astype(str)
    df = df.merge(priors.get('subj_season_stat', pd.DataFrame({'sid_season': []})), on='sid_season', how='left')
    df = df.merge(priors.get('dept_stat', pd.DataFrame({'OFFER_DEPT_CODE': []})), on='OFFER_DEPT_CODE', how='left')
    df = df.merge(priors.get('prefix_stat', pd.DataFrame({'prefix': []})), on='prefix', how='left')
    df['fac'] = df['RESPONSIBLE_FACULTY_NAME']
    df = df.merge(priors.get('fac_stat', pd.DataFrame({'fac': []})), on='fac', how='left')

    # 6. Engineered Numerical & Text Indicators
    df['course_level'] = df['SUBJECT_ID_SORT'].apply(parse_course_level)
    df['is_grad_level'] = df['SUBJECT_ID_SORT'].apply(lambda x: int('THG' in str(x) or ('G' in str(x).split('.')[-1] if '.' in str(x) else False)))
    df['is_intro_level'] = df['SUBJECT_ID_SORT'].apply(lambda x: int('.0' in str(x) or '.1' in str(x)))
    df['is_thesis'] = df['SUBJECT_ID_SORT'].apply(lambda x: int('TH' in str(x)))
    df['is_urop'] = df['SUBJECT_ID_SORT'].apply(lambda x: int('UR' in str(x)))
    df['is_special'] = df['SUBJECT_ID_SORT'].apply(lambda x: int('SP' in str(x) or 'S' in str(x).split('.')[-1]))
    df['is_harvard'] = df['SUBJECT_ID_SORT'].apply(lambda x: int(str(x).startswith('H')))

    df['roll3_log'] = np.log1p(df['roll3_enrollment_mean'].clip(lower=0))
    df['lag1_log'] = np.log1p(df['lag1_enrollment'].clip(lower=0))
    df['lag2_log'] = np.log1p(df['lag2_enrollment'].clip(lower=0))
    df['trend_roll_lag1'] = df['roll3_enrollment_mean'] - df['lag1_enrollment']
    df['trend_lag1_lag2'] = df['lag1_enrollment'] - df['lag2_enrollment']
    df['ratio_lag1_lag2'] = (df['lag1_enrollment'] + 1.0) / (df['lag2_enrollment'] + 1.0)
    df['ratio_roll3_dept'] = df['roll3_enrollment_mean'] / (df['dept_p75'] + 1.0)
    df['ratio_lag1_dept'] = df['lag1_enrollment'] / (df['dept_p75'] + 1.0)

    df['total_units'] = pd.to_numeric(df['TOTAL_UNITS'], errors='coerce').fillna(0)
    df['lecture_units'] = pd.to_numeric(df['LECTURE_UNITS'], errors='coerce').fillna(0)
    df['lab_units'] = pd.to_numeric(df['LAB_UNITS'], errors='coerce').fillna(0)
    df['preparation_units'] = pd.to_numeric(df['PREP_UNITS'], errors='coerce').fillna(0)

    t = df['SUBJECT_TITLE'].fillna('').astype(str).str.lower()
    df['title_len'] = t.str.len()
    df['title_words'] = t.str.split().apply(len)
    for kw in TITLE_KEYWORDS:
        df[f'title_{kw}'] = t.apply(lambda s: int(kw in s))

    df['cat_var_units'] = (df['IS_VARIABLE_UNITS'] == 'Y').astype(int)
    df['cat_prereq_len'] = df['PREREQUISITES'].fillna('').astype(str).str.len()

    # 7. Categorical encoding
    for col in CAT_COLS:
        df[col] = df[col].fillna('MISSING').astype(str)
        if col in cat_maps:
            known = set(cat_maps[col])
            df[col] = df[col].apply(lambda x: x if x in known else 'MISSING')
            df[col] = pd.Categorical(df[col], categories=cat_maps[col])

    # 8. Numeric null cleanup
    for f in FEATURE_NAMES:
        if f not in CAT_COLS:
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0.0)

    return df[FEATURE_NAMES]


def build_priors_from_train(train_gold_df, train_data_dir):
    """
    Build historical rate and department statistics strictly from historical training data.
    """
    ss_path = os.path.join(train_data_dir, 'SUBJECT_SUMMARY.csv')
    so_path = os.path.join(train_data_dir, 'SUBJECT_OFFERED.csv')

    train = train_gold_df.copy()
    train['y'] = (train['HIGH_ENROLLMENT'] == 'Y').astype(int)
    train['season'] = train['TERM_CODE'].str[4:]
    train['prefix'] = train['SUBJECT_ID_SORT'].apply(lambda x: str(x).split('.')[0])

    so = pd.read_csv(so_path, low_memory=False)
    so_map = so[['TERM_CODE', 'SUBJECT_ID_SORT', 'OFFER_DEPT_CODE', 'RESPONSIBLE_FACULTY_NAME']].drop_duplicates(['TERM_CODE', 'SUBJECT_ID_SORT'])
    train = train.merge(so_map, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    ss = pd.read_csv(ss_path, low_memory=False)
    ss_enr = ss[['TERM_CODE', 'SUBJECT_ID_SORT', 'SUBJECT_ENROLLMENT_FIFTH_WEEK']].drop_duplicates(['TERM_CODE', 'SUBJECT_ID_SORT'])
    train = train.merge(ss_enr, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Subject stat
    subj_stat = train.groupby('SUBJECT_ID_SORT').agg(
        subj_hist_rate=('y', 'mean'),
        subj_hist_cnt=('y', 'count'),
        subj_hist_enr_mean=('SUBJECT_ENROLLMENT_FIFTH_WEEK', 'mean'),
        subj_hist_enr_max=('SUBJECT_ENROLLMENT_FIFTH_WEEK', 'max'),
        subj_hist_enr_median=('SUBJECT_ENROLLMENT_FIFTH_WEEK', 'median')
    ).reset_index()

    # Subject x Season stat
    train['sid_season'] = train['SUBJECT_ID_SORT'].astype(str) + '|' + train['season'].astype(str)
    subj_season_stat = train.groupby('sid_season').agg(
        subj_season_rate=('y', 'mean'),
        subj_season_cnt=('y', 'count')
    ).reset_index()

    # Department stat
    dept_stat = train.dropna(subset=['OFFER_DEPT_CODE']).groupby('OFFER_DEPT_CODE').agg(
        dept_hist_rate=('y', 'mean'),
        dept_hist_cnt=('y', 'count')
    ).reset_index()

    # Prefix stat
    prefix_stat = train.groupby('prefix').agg(
        prefix_hist_rate=('y', 'mean'),
        prefix_hist_cnt=('y', 'count')
    ).reset_index()

    # Faculty stat
    fac_stat = train.dropna(subset=['RESPONSIBLE_FACULTY_NAME']).rename(columns={'RESPONSIBLE_FACULTY_NAME': 'fac'}).groupby('fac').agg(
        fac_hist_rate=('y', 'mean'),
        fac_hist_cnt=('y', 'count')
    ).reset_index()

    # Dept p75 on positive enrollment
    pos = train[train['SUBJECT_ENROLLMENT_FIFTH_WEEK'] > 0]
    dept_p75 = pos.dropna(subset=['OFFER_DEPT_CODE']).groupby('OFFER_DEPT_CODE')['SUBJECT_ENROLLMENT_FIFTH_WEEK'].quantile(0.75).reset_index()
    dept_p75.columns = ['OFFER_DEPT_CODE', 'dept_p75']

    return {
        'subj_stat': subj_stat,
        'subj_season_stat': subj_season_stat,
        'dept_stat': dept_stat,
        'prefix_stat': prefix_stat,
        'fac_stat': fac_stat,
        'dept_p75': dept_p75
    }


def train_model(train_data_dir=DEFAULT_TRAIN_DIR,
                train_gold_path=DEFAULT_TRAIN_GOLD,
                model_save_path=DEFAULT_MODEL_PATH,
                val_cutoff_year=2016):
    """
    Train LightGBM model using time-based validation strictly within train data.
    """
    print(f"[Training] Loading train gold labels from {train_gold_path}...")
    train_gold = pd.read_csv(train_gold_path)
    train_gold['year'] = train_gold['TERM_CODE'].str[:4].astype(int)

    # Time-based split: historical before cutoff, validation at or after cutoff
    train_split = train_gold[train_gold['year'] < val_cutoff_year].copy()
    val_split = train_gold[train_gold['year'] >= val_cutoff_year].copy()

    print(f"[Training] Time-based split: {len(train_split)} train samples (< {val_cutoff_year}), {len(val_split)} val samples (>= {val_cutoff_year})")

    # Build historical priors only from training split to prevent time leakage
    print("[Training] Building historical priors...")
    priors = build_priors_from_train(train_split, train_data_dir)

    # Establish category maps
    cat_maps = {
        'OFFER_DEPT_CODE': sorted(priors['dept_stat']['OFFER_DEPT_CODE'].astype(str).tolist() + ['MISSING']),
        'OFFER_SCHOOL_NAME': ['Non-MIT', 'Engineering', 'Architecture and Planning', 'Hum, Arts & Social Sciences',
                              'Sloan School of Management', 'Science', 'MIT, academic', 'Whitaker Coll of HST;  HST',
                              'Schwarzman Coll of Comp', 'MIT, non-academic', 'MISSING'],
        'prefix': sorted(priors['prefix_stat']['prefix'].astype(str).tolist() + ['MISSING']),
        'season': ['FA', 'SP', 'SU', 'JA', '10', 'MISSING'],
        'CLUSTER_TYPE': ['S', 'MISSING', 'M', 'J'],
        'HASS_ATTRIBUTE': ['MISSING', 'HS', 'HH', 'HA', 'HE', 'HA,HH'],
        'GIR_ATTRIBUTE': ['MISSING', 'CAL1', 'REST', 'CAL2', 'HE', 'LAB', 'HDL', 'HD4', 'HD5', 'HD3', 'HD1', 'HD2', 'LAB2', 'PHY1', 'RST2', 'CHEM', 'PHY2', 'BIOL']
    }

    print("[Training] Extracting features for train set...")
    X_train = extract_features(train_split, train_data_dir, priors, cat_maps)
    y_train = (train_split['HIGH_ENROLLMENT'] == 'Y').astype(int)

    print("[Training] Extracting features for validation set...")
    X_val = extract_features(val_split, train_data_dir, priors, cat_maps)
    y_val = (val_split['HIGH_ENROLLMENT'] == 'Y').astype(int)

    print("[Training] Fitting LightGBM Classifier...")
    clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_train)

    # Optimize threshold on validation slice for Macro F1
    val_probs = clf.predict_proba(X_val)[:, 1]
    best_th = 0.5
    best_f1 = 0.0
    for th in np.arange(0.20, 0.70, 0.02):
        f1 = f1_score(y_val, (val_probs >= th).astype(int), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"[Validation Result] Optimal Threshold: {best_th:.4f} | Validation Macro F1: {best_f1:.4f}")

    model_obj = {
        'classifier': clf,
        'features': FEATURE_NAMES,
        'cat_cols': CAT_COLS,
        'cat_maps': cat_maps,
        'threshold': best_th,
        'positive_rate': float(y_train.mean()),
        'priors': priors,
        'metrics': {
            'validation_macro_f1': best_f1,
            'threshold': best_th
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_obj, f)
    print(f"[Training] Model and artifacts saved to {model_save_path}")
    return model_obj


def predict(test_df_or_path,
            train_data_dir=DEFAULT_TRAIN_DIR,
            model_path=DEFAULT_MODEL_PATH,
            output_csv_path=DEFAULT_PREDS_PATH):
    """
    Generate predictions for a given test gold file or dataframe.
    Predicts one row per test summary row with target column HIGH_ENROLLMENT ('Y' / 'N').
    """
    t0 = time.time()
    if isinstance(test_df_or_path, str):
        print(f"[Prediction] Loading test input from {test_df_or_path}...")
        test_df = pd.read_csv(test_df_or_path)
    else:
        test_df = test_df_or_path.copy()

    assert 'TERM_CODE' in test_df.columns and 'SUBJECT_ID_SORT' in test_df.columns, \
        "Test input must contain columns TERM_CODE and SUBJECT_ID_SORT"

    if not os.path.exists(model_path):
        print(f"[Prediction] Model file {model_path} not found. Running training first...")
        train_model(train_data_dir=train_data_dir, model_save_path=model_path)

    with open(model_path, 'rb') as f:
        model_obj = pickle.load(f)

    clf = model_obj['classifier']
    feature_names = model_obj['features']
    cat_maps = model_obj['cat_maps']
    threshold = model_obj.get('threshold', 0.46)
    priors = model_obj['priors']

    print(f"[Prediction] Extracting features for {len(test_df)} test rows...")
    X_test = extract_features(test_df, train_data_dir, priors, cat_maps)

    print("[Prediction] Running model inference...")
    probs = clf.predict_proba(X_test[feature_names])[:, 1]
    preds = np.where(probs >= threshold, 'Y', 'N')

    output_df = pd.DataFrame({
        'TERM_CODE': test_df['TERM_CODE'],
        'SUBJECT_ID_SORT': test_df['SUBJECT_ID_SORT'],
        'HIGH_ENROLLMENT': preds
    })

    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        output_df.to_csv(output_csv_path, index=False)
        print(f"[Prediction] Wrote {len(output_df)} predictions to {output_csv_path} in {time.time() - t0:.2f}s")

    return output_df


def evaluate(test_gold_path, preds_path_or_df):
    """
    Evaluate predictions against test gold labels.
    Computes Macro F1, Precision, Recall, Accuracy, and confusion matrix.
    """
    test_gold = pd.read_csv(test_gold_path)
    if isinstance(preds_path_or_df, str):
        preds_df = pd.read_csv(preds_path_or_df)
    else:
        preds_df = preds_path_or_df

    assert len(test_gold) == len(preds_df), \
        f"Row count mismatch between gold ({len(test_gold)}) and preds ({len(preds_df)})"

    # Map labels to consistent string format 'N' / 'Y'
    label_map = {0: 'N', 1: 'Y', '0': 'N', '1': 'Y', 'N': 'N', 'Y': 'Y'}
    y_true = test_gold['HIGH_ENROLLMENT'].map(label_map)
    y_pred = preds_df['HIGH_ENROLLMENT'].map(label_map)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    prec_macro = precision_score(y_true, y_pred, average='macro')
    rec_macro = recall_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=['N', 'Y'])

    print("\n=======================================================")
    print("                 EVALUATION METRICS                     ")
    print("=======================================================")
    print(f"Total Test Samples:     {len(y_true)}")
    print(f"Accuracy:               {acc:.4%}")
    print(f"Macro F1 Score:         {macro_f1:.4f}")
    print(f"Macro Precision:        {prec_macro:.4f}")
    print(f"Macro Recall:           {rec_macro:.4f}")
    print("\nConfusion Matrix (Rows: True [N, Y], Cols: Pred [N, Y]):")
    print(f"  [[TN={cm[0,0]:5d}, FP={cm[0,1]:5d}],")
    print(f"   [FN={cm[1,0]:5d}, TP={cm[1,1]:5d}]]")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("=======================================================\n")

    return {
        'macro_f1': macro_f1,
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'confusion_matrix': cm
    }


def predict_and_evaluate(test_gold_path=DEFAULT_TEST_GOLD,
                         train_data_dir=DEFAULT_TRAIN_DIR,
                         model_path=DEFAULT_MODEL_PATH,
                         output_preds_path=DEFAULT_PREDS_PATH):
    """
    Unified entry point: predicts on test_gold_path and evaluates if labels exist.
    """
    print(f"\n[Pipeline] Starting Predict & Evaluate workflow on: {test_gold_path}")
    preds_df = predict(test_gold_path, train_data_dir=train_data_dir, model_path=model_path, output_csv_path=output_preds_path)

    test_df = pd.read_csv(test_gold_path)
    if 'HIGH_ENROLLMENT' in test_df.columns:
        metrics = evaluate(test_gold_path, preds_df)
        return preds_df, metrics
    else:
        print("[Pipeline] No HIGH_ENROLLMENT column in test input; predictions generated without evaluation.")
        return preds_df, None


def main():
    parser = argparse.ArgumentParser(description="Course High Enrollment Prediction Pipeline")
    parser.add_argument('--train', action='store_true', help="Train model from historical training tables")
    parser.add_argument('--predict', action='store_true', help="Generate predictions for test file")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate existing predictions against test gold")
    parser.add_argument('--all', action='store_true', help="Train, predict, and evaluate")
    parser.add_argument('--test-path', type=str, default=DEFAULT_TEST_GOLD, help="Path to test gold CSV")
    parser.add_argument('--train-dir', type=str, default=DEFAULT_TRAIN_DIR, help="Path to training data directory")
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help="Path to LGBM model pickle")
    parser.add_argument('--output-path', type=str, default=DEFAULT_PREDS_PATH, help="Path for output predictions CSV")

    args = parser.parse_args()

    if args.train:
        train_model(train_data_dir=args.train_dir, model_save_path=args.model_path)
    elif args.evaluate:
        evaluate(args.test_path, args.output_path)
    elif args.predict:
        predict(args.test_path, train_data_dir=args.train_dir, model_path=args.model_path, output_csv_path=args.output_path)
    else:
        # Default behavior: run predict & evaluate
        predict_and_evaluate(test_gold_path=args.test_path,
                             train_data_dir=args.train_dir,
                             model_path=args.model_path,
                             output_preds_path=args.output_path)


if __name__ == '__main__':
    main()
