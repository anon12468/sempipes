"""
Course High Enrollment Prediction Pipeline
------------------------------------------
Task:
    For each course offering per term, predict whether it will have high enrollment
    (Y or N) relative to other courses in the same department and term (top quartile
    among offerings with positive enrollment).

Target Column:
    HIGH_ENROLLMENT ('Y' or 'N')
Keys:
    TERM_CODE, SUBJECT_ID_SORT
Development Metric:
    Macro F1 on a time-based validation slice within train.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# Default directory configurations
DEFAULT_TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), 'table_splits', 'train')
DEFAULT_TEST_DATA_DIR = os.path.join('/Users/USER/BEAVER/', 'test/')


# ==============================================================================
# 1. Feature Extraction & Engineering Functions
# ==============================================================================

def load_and_aggregate_sections(subject_offered_path: str) -> pd.DataFrame:
    """
    Extracts and aggregates section-level features from SUBJECT_OFFERED.csv.
    Groups by (TERM_CODE, SUBJECT_ID_SORT).
    """
    cols = [
        'TERM_CODE', 'SUBJECT_ID_SORT', 'OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'HGN_CODE',
        'IS_LECTURE_SECTION', 'IS_LAB_SECTION', 'IS_RECITATION_SECTION', 'IS_DESIGN_SECTION',
        'IS_MASTER_SECTION', 'IS_REPEATABLE_SUBJECT', 'EVALUATE_THIS_SUBJECT'
    ]
    if not os.path.exists(subject_offered_path):
        return pd.DataFrame(columns=['TERM_CODE', 'SUBJECT_ID_SORT'])

    off = pd.read_csv(subject_offered_path, usecols=lambda c: c in cols, low_memory=False)
    if off.empty:
        return pd.DataFrame(columns=['TERM_CODE', 'SUBJECT_ID_SORT'])

    # Vectorized binary conversions
    binary_cols = [
        'IS_LECTURE_SECTION', 'IS_LAB_SECTION', 'IS_RECITATION_SECTION',
        'IS_DESIGN_SECTION', 'IS_MASTER_SECTION', 'IS_REPEATABLE_SUBJECT', 'EVALUATE_THIS_SUBJECT'
    ]
    for c in binary_cols:
        if c in off.columns:
            off[c] = (off[c] == 'Y').astype(int)
        else:
            off[c] = 0

    agg_dict = {
        'IS_LECTURE_SECTION': 'sum',
        'IS_LAB_SECTION': 'sum',
        'IS_RECITATION_SECTION': 'sum',
        'IS_DESIGN_SECTION': 'sum',
        'IS_MASTER_SECTION': 'sum',
        'IS_REPEATABLE_SUBJECT': 'max',
        'EVALUATE_THIS_SUBJECT': 'max',
    }
    if 'OFFER_DEPT_CODE' in off.columns:
        agg_dict['OFFER_DEPT_CODE'] = 'first'
    if 'OFFER_SCHOOL_NAME' in off.columns:
        agg_dict['OFFER_SCHOOL_NAME'] = 'first'
    if 'HGN_CODE' in off.columns:
        agg_dict['HGN_CODE'] = 'first'

    sec_agg = off.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(agg_dict).reset_index()
    counts = off.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).size().reset_index(name='section_count')
    sec_agg = sec_agg.merge(counts, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    return sec_agg


def build_historical_subject_priors(train_gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Computes expanding Bayesian prior high-enrollment rates per subject and department across time.
    Also returns static historical priors across the full training dataset for test inference.
    """
    gold = train_gold.copy()
    gold['term_year'] = gold['TERM_CODE'].str[:4].astype(int)
    gold['target'] = (gold['HIGH_ENROLLMENT'] == 'Y').astype(int)
    global_prior = gold['target'].mean()

    # Expanding priors per subject for training terms (prevents future leakage)
    yearly_subj = gold.groupby(['SUBJECT_ID_SORT', 'term_year'])['target'].agg(['sum', 'count']).reset_index()
    yearly_subj = yearly_subj.sort_values(['SUBJECT_ID_SORT', 'term_year'])
    yearly_subj['cum_sum'] = yearly_subj.groupby('SUBJECT_ID_SORT')['sum'].cumsum() - yearly_subj['sum']
    yearly_subj['cum_count'] = yearly_subj.groupby('SUBJECT_ID_SORT')['count'].cumsum() - yearly_subj['count']
    yearly_subj['subj_prior_rate'] = (yearly_subj['cum_sum'] + 0.1) / (yearly_subj['cum_count'] + 1.0)
    yearly_subj.rename(columns={'cum_count': 'subj_prior_count'}, inplace=True)
    expanding_subj_priors = yearly_subj[['SUBJECT_ID_SORT', 'term_year', 'subj_prior_rate', 'subj_prior_count']]

    # Full train historical subject lookup (for inference on future test data)
    full_subj = gold.groupby('SUBJECT_ID_SORT')['target'].agg(['sum', 'count']).reset_index()
    full_subj['subj_prior_rate'] = (full_subj['sum'] + 0.1) / (full_subj['count'] + 1.0)
    full_subj.rename(columns={'count': 'subj_prior_count'}, inplace=True)
    full_subj_priors = full_subj[['SUBJECT_ID_SORT', 'subj_prior_rate', 'subj_prior_count']]

    return expanding_subj_priors, full_subj_priors, global_prior


def extract_features(
    data_dir: str,
    gold_df: pd.DataFrame = None,
    expanding_priors: pd.DataFrame = None,
    full_priors: pd.DataFrame = None,
    global_prior: float = 0.081,
    is_training: bool = True
) -> pd.DataFrame:
    """
    Constructs the feature table by merging SUBJECT_OFFERED_SUMMARY, ENROLLMENT_HISTORY_FEATURES,
    and aggregated section features from SUBJECT_OFFERED, engineering ratios, department-relative
    statistics, and temporal/calendar signals.
    """
    summary_path = os.path.join(data_dir, 'SUBJECT_OFFERED_SUMMARY.csv')
    history_path = os.path.join(data_dir, 'ENROLLMENT_HISTORY_FEATURES.csv')
    offered_path = os.path.join(data_dir, 'SUBJECT_OFFERED.csv')

    # 1. Base Summary
    sum_df = pd.read_csv(summary_path, low_memory=False)
    if sum_df.empty:
        return pd.DataFrame()

    base_cols = ['TERM_CODE', 'SUBJECT_ID_SORT']
    for opt_c in ['TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'CLUSTER_TYPE', 'OFFER_DEPT_CODE', 'HGN_CODE']:
        if opt_c in sum_df.columns:
            base_cols.append(opt_c)

    df = sum_df[base_cols].copy()

    # 2. Section Aggregation
    sec_agg = load_and_aggregate_sections(offered_path)
    if not sec_agg.empty:
        # Merge, preferring section department if summary did not have it
        df = df.merge(sec_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Ensure department is populated (fallback to subject prefix e.g. '6' from '6.033')
    if 'OFFER_DEPT_CODE' not in df.columns:
        df['OFFER_DEPT_CODE'] = df['SUBJECT_ID_SORT'].str.split('.').str[0]
    else:
        df['OFFER_DEPT_CODE'] = df['OFFER_DEPT_CODE'].fillna(df['SUBJECT_ID_SORT'].str.split('.').str[0])

    # 3. Enrollment History Features
    if os.path.exists(history_path):
        hist_df = pd.read_csv(history_path)
        df = df.merge(hist_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    for col in ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # 4. Units Decomposition & Ratios
    for u in ['TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS']:
        if u not in df.columns:
            df[u] = 0.0
        else:
            df[u] = df[u].fillna(0.0)

    df['PREP_UNITS'] = np.maximum(0.0, df['TOTAL_UNITS'] - df['LECTURE_UNITS'] - df['LAB_UNITS'])
    df['lecture_ratio'] = df['LECTURE_UNITS'] / (df['TOTAL_UNITS'] + 1e-5)
    df['lab_ratio'] = df['LAB_UNITS'] / (df['TOTAL_UNITS'] + 1e-5)
    df['is_zero_units'] = (df['TOTAL_UNITS'] == 0).astype(int)

    # 5. Section Counts Defaults
    sec_cols = [
        'IS_LECTURE_SECTION', 'IS_LAB_SECTION', 'IS_RECITATION_SECTION',
        'IS_DESIGN_SECTION', 'IS_MASTER_SECTION', 'IS_REPEATABLE_SUBJECT',
        'EVALUATE_THIS_SUBJECT', 'section_count'
    ]
    for sc in sec_cols:
        if sc not in df.columns:
            df[sc] = 0
        else:
            df[sc] = df[sc].fillna(0)

    # 6. Enrollment History Transformations
    df['enroll_diff_1_2'] = df['lag1_enrollment'] - df['lag2_enrollment']
    df['enroll_diff_1_roll3'] = df['lag1_enrollment'] - df['roll3_enrollment_mean']
    df['has_positive_roll3'] = (df['roll3_enrollment_mean'] > 0).astype(int)
    df['log_roll3'] = np.log1p(np.maximum(0.0, df['roll3_enrollment_mean']))
    df['log_lag1'] = np.log1p(np.maximum(0.0, df['lag1_enrollment']))

    # 7. Season and Calendar Indicators
    df['season'] = df['TERM_CODE'].str[4:]
    df['term_year'] = pd.to_numeric(df['TERM_CODE'].str[:4], errors='coerce').fillna(2000).astype(int)
    df['is_fall'] = (df['season'] == 'FA').astype(int)
    df['is_spring'] = (df['season'] == 'SP').astype(int)
    df['is_iap'] = (df['season'] == 'JA').astype(int)
    df['is_summer'] = (df['season'] == 'SU').astype(int)
    df['is_regular_term'] = (df['season'].isin(['FA', 'SP'])).astype(int)

    # 8. Department-Term Relative Features
    grp = df.groupby(['TERM_CODE', 'OFFER_DEPT_CODE'])
    df['dept_roll3_rank'] = grp['roll3_enrollment_mean'].rank(pct=True).fillna(0.0)
    df['dept_roll3_mean'] = grp['roll3_enrollment_mean'].transform('mean').fillna(0.0)
    df['dept_roll3_max'] = grp['roll3_enrollment_mean'].transform('max').fillna(0.0)
    df['dept_roll3_diff_mean'] = df['roll3_enrollment_mean'] - df['dept_roll3_mean']
    df['dept_roll3_q75'] = grp['roll3_enrollment_mean'].transform(lambda x: x.quantile(0.75)).fillna(0.0)
    df['dept_roll3_above_q75'] = (df['roll3_enrollment_mean'] >= df['dept_roll3_q75']).astype(int)
    df['dept_course_count'] = grp['roll3_enrollment_mean'].transform('count').fillna(1)
    df['dept_sec_rank'] = grp['section_count'].rank(pct=True).fillna(0.0)

    # Positive-only Department Statistics (Exact Task Definition Alignment)
    pos_mask = df['roll3_enrollment_mean'] > 0
    pos_df = df[pos_mask]
    if not pos_df.empty:
        pos_grp = pos_df.groupby(['TERM_CODE', 'OFFER_DEPT_CODE'])
        pos_stats = pos_grp['roll3_enrollment_mean'].agg(
            pos_dept_q75=lambda x: x.quantile(0.75),
            pos_dept_mean='mean',
            pos_dept_count='count'
        ).reset_index()
        df = df.merge(pos_stats, on=['TERM_CODE', 'OFFER_DEPT_CODE'], how='left')
    else:
        df['pos_dept_q75'] = 0.0
        df['pos_dept_mean'] = 0.0
        df['pos_dept_count'] = 0.0

    df['pos_dept_q75'] = df['pos_dept_q75'].fillna(0.0)
    df['pos_dept_mean'] = df['pos_dept_mean'].fillna(0.0)
    df['pos_dept_count'] = df['pos_dept_count'].fillna(0.0)
    df['diff_from_pos_q75'] = df['roll3_enrollment_mean'] - df['pos_dept_q75']
    df['is_above_pos_q75'] = ((df['roll3_enrollment_mean'] >= df['pos_dept_q75']) & (df['roll3_enrollment_mean'] > 0)).astype(int)
    df['pos_ratio_in_dept'] = df['pos_dept_count'] / (df['dept_course_count'] + 1e-5)

    # 9. Subject Prefix and Categoricals
    df['subj_prefix'] = df['SUBJECT_ID_SORT'].str.split('.').str[0]
    for cat in ['OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'HGN_CODE', 'CLUSTER_TYPE', 'season', 'subj_prefix']:
        if cat not in df.columns:
            df[cat] = 'UNKNOWN'
        df[cat] = df[cat].fillna('UNKNOWN').astype(str)

    # 10. Bayesian Historical Subject Priors
    if is_training and expanding_priors is not None:
        df = df.merge(expanding_priors, on=['SUBJECT_ID_SORT', 'term_year'], how='left')
    elif full_priors is not None:
        df = df.merge(full_priors, on=['SUBJECT_ID_SORT'], how='left')

    if 'subj_prior_rate' not in df.columns:
        df['subj_prior_rate'] = global_prior
        df['subj_prior_count'] = 0
    else:
        df['subj_prior_rate'] = df['subj_prior_rate'].fillna(global_prior)
        df['subj_prior_count'] = df['subj_prior_count'].fillna(0)

    # Merge labels if provided
    if gold_df is not None:
        df = df.merge(gold_df[['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']], on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    return df


# ==============================================================================
# 2. Pipeline Implementations
# ==============================================================================

class BaselinePipeline:
    """
    Baseline model: uses raw un-engineered features with default classifier.
    """
    def __init__(self):
        self.features = ['TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean']
        self.model = LGBMClassifier(random_state=42, verbose=-1)

    def fit(self, train_df: pd.DataFrame):
        X = train_df[self.features].fillna(0)
        y = (train_df['HIGH_ENROLLMENT'] == 'Y').astype(int)
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].fillna(0)
        preds = self.model.predict(X)
        return np.where(preds == 1, 'Y', 'N')


class ImprovedPipeline:
    """
    Improved model:
    - Extracts section details and department hierarchy
    - Engineers credit unit ratios, enrollment deltas, and calendar indicators
    - Aggregates department-term relative ranks and positive-enrollment top-quartiles
    - Leverages historical Bayesian subject priors
    - Fits LightGBM with cost-sensitive weighting and calibrated macro-F1 threshold
    """
    def __init__(self, val_split_year: int = 2017):
        self.val_split_year = val_split_year
        self.model = None
        self.optimal_threshold = 0.38
        self.full_subject_priors = None
        self.global_prior = 0.081
        self.feature_cols = [
            'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREP_UNITS', 'lecture_ratio', 'lab_ratio',
            'is_zero_units', 'lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean',
            'enroll_diff_1_2', 'enroll_diff_1_roll3', 'has_positive_roll3', 'log_roll3', 'log_lag1',
            'IS_LECTURE_SECTION', 'IS_LAB_SECTION', 'IS_RECITATION_SECTION', 'IS_DESIGN_SECTION',
            'IS_MASTER_SECTION', 'IS_REPEATABLE_SUBJECT', 'EVALUATE_THIS_SUBJECT', 'section_count',
            'dept_roll3_rank', 'dept_roll3_mean', 'dept_roll3_max', 'dept_roll3_diff_mean',
            'dept_roll3_q75', 'dept_roll3_above_q75', 'dept_course_count', 'dept_sec_rank',
            'pos_dept_q75', 'pos_dept_mean', 'pos_dept_count', 'diff_from_pos_q75', 'is_above_pos_q75', 'pos_ratio_in_dept',
            'subj_prior_rate', 'subj_prior_count',
            'term_year', 'is_fall', 'is_spring', 'is_iap', 'is_summer', 'is_regular_term',
            'OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'HGN_CODE', 'CLUSTER_TYPE', 'season', 'subj_prefix'
        ]
        self.categorical_cols = [
            'OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'HGN_CODE', 'CLUSTER_TYPE', 'season', 'subj_prefix'
        ]

    def _prepare_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].copy()
        for cat in self.categorical_cols:
            X[cat] = X[cat].astype('category')
        return X

    def fit_and_validate(self, train_dir: str) -> dict:
        """
        Trains and validates using TRAIN_DATA_DIR only.
        Holds out terms >= val_split_year for development evaluation.
        """
        gold_path = os.path.join(train_dir, 'gold_enrollment_train.csv')
        gold_df = pd.read_csv(gold_path)

        # 1. Compute historical Bayesian priors strictly within train
        expanding_priors, full_priors, global_prior = build_historical_subject_priors(gold_df)
        self.full_subject_priors = full_priors
        self.global_prior = global_prior

        # 2. Extract features
        features_df = extract_features(
            data_dir=train_dir,
            gold_df=gold_df,
            expanding_priors=expanding_priors,
            global_prior=global_prior,
            is_training=True
        )

        # 3. Time-based validation split
        train_mask = features_df['term_year'] < self.val_split_year
        val_mask = features_df['term_year'] >= self.val_split_year

        X_train = self._prepare_matrix(features_df[train_mask])
        y_train = (features_df.loc[train_mask, 'HIGH_ENROLLMENT'] == 'Y').astype(int)

        X_val = self._prepare_matrix(features_df[val_mask])
        y_val = (features_df.loc[val_mask, 'HIGH_ENROLLMENT'] == 'Y').astype(int)

        # 4. Train LightGBM model
        self.model = LGBMClassifier(
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            scale_pos_weight=1.5,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X_train, y_train)

        # 5. Calibrate optimal threshold on validation slice
        val_probs = self.model.predict_proba(X_val)[:, 1]
        best_f1, best_thresh = 0.0, 0.5
        for th in np.linspace(0.2, 0.7, 51):
            preds = (val_probs >= th).astype(int)
            f1 = f1_score(y_val, preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(th)

        self.optimal_threshold = best_thresh

        # Evaluate on validation slice
        final_val_preds = (val_probs >= self.optimal_threshold).astype(int)
        val_macro_f1 = f1_score(y_val, final_val_preds, average='macro')
        report_str = classification_report(y_val, final_val_preds, target_names=['N', 'Y'], digits=4)

        return {
            'val_macro_f1': val_macro_f1,
            'optimal_threshold': self.optimal_threshold,
            'report': report_str,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }

    def predict(self, test_dir: str) -> pd.DataFrame:
        """
        Generates predictions for each course offering in test_dir.
        Returns a DataFrame with columns ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'].
        """
        features_df = extract_features(
            data_dir=test_dir,
            gold_df=None,
            full_priors=self.full_subject_priors,
            global_prior=self.global_prior,
            is_training=False
        )

        if features_df.empty:
            return pd.DataFrame(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'])

        X_test = self._prepare_matrix(features_df)
        probs = self.model.predict_proba(X_test)[:, 1]
        preds = np.where(probs >= self.optimal_threshold, 'Y', 'N')

        result = pd.DataFrame({
            'TERM_CODE': features_df['TERM_CODE'],
            'SUBJECT_ID_SORT': features_df['SUBJECT_ID_SORT'],
            'HIGH_ENROLLMENT': preds
        })
        return result


# ==============================================================================
# 3. Evaluation Function
# ==============================================================================

def evaluate_predictions(predictions_df: pd.DataFrame, gold_path_or_df) -> dict:
    """
    Evaluates predictions against gold ground truth.
    Supports either path to gold CSV or DataFrame.
    Returns metrics dict.
    """
    if isinstance(gold_path_or_df, str):
        if not os.path.exists(gold_path_or_df):
            print(f"[EVALUATION] Gold file '{gold_path_or_df}' not found.")
            return {}
        gold_df = pd.read_csv(gold_path_or_df)
    else:
        gold_df = gold_path_or_df

    if gold_df.empty or len(gold_df) == 0:
        print("[EVALUATION] Ground truth file has 0 rows (placeholder held-out split).")
        return {}

    merged = predictions_df.merge(gold_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], suffixes=('_pred', '_gold'))
    if merged.empty:
        print("[EVALUATION] No matching keys between predictions and ground truth.")
        return {}

    y_true = (merged['HIGH_ENROLLMENT_gold'] == 'Y').astype(int)
    y_pred = (merged['HIGH_ENROLLMENT_pred'] == 'Y').astype(int)

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rep = classification_report(y_true, y_pred, target_names=['N', 'Y'], digits=4)

    metrics = {
        'macro_f1': macro_f1,
        'accuracy': acc,
        'precision_macro': prec,
        'recall_macro': rec,
        'report': rep,
        'evaluated_samples': len(merged)
    }

    print("\n" + "=" * 50)
    print("           PREDICTION EVALUATION REPORT           ")
    print("=" * 50)
    print(f"Evaluated Offerings: {len(merged):,}")
    print(f"Development Metric (Macro F1): {macro_f1:.4f}")
    print(f"Accuracy:                      {acc:.4f}")
    print("\nDetailed Classification Report:")
    print(rep)
    print("=" * 50)

    return metrics


# ==============================================================================
# 4. Main CLI Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Course High Enrollment Prediction Pipeline")
    parser.add_argument('--train_dir', type=str, default=DEFAULT_TRAIN_DATA_DIR, help="Path to training data directory")
    parser.add_argument('--test_dir', type=str, default=DEFAULT_TEST_DATA_DIR, help="Path to test data directory")
    parser.add_argument('--val_year', type=int, default=2017, help="Validation cutoff year (e.g. 2017 holds out 2017-2018)")
    parser.add_argument('--mode', type=str, choices=['baseline', 'improved', 'compare'], default='compare', help="Pipeline mode to run")
    parser.add_argument('--output', type=str, default='predictions.csv', help="Path to save predictions CSV")
    parser.add_argument('--sanity_check', action='store_true', help="Sanity-check prediction and evaluation on latest train terms")
    args = parser.parse_args()

    print("=" * 60)
    print("  ACADEMIC COURSE HIGH ENROLLMENT PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Train Directory:    {args.train_dir}")
    print(f"Test Directory:     {args.test_dir}")
    print(f"Validation Year:    {args.val_year}")
    print(f"Execution Mode:     {args.mode}")

    # Step 1: Baseline pipeline if requested
    baseline_macro_f1 = None
    if args.mode in ['baseline', 'compare']:
        print("\n[1/3] Running Baseline Pipeline...")
        gold_df = pd.read_csv(os.path.join(args.train_dir, 'gold_enrollment_train.csv'))
        sum_df = pd.read_csv(os.path.join(args.train_dir, 'SUBJECT_OFFERED_SUMMARY.csv'), low_memory=False)
        hist_df = pd.read_csv(os.path.join(args.train_dir, 'ENROLLMENT_HISTORY_FEATURES.csv'))

        df_base = gold_df.merge(sum_df[['TERM_CODE', 'SUBJECT_ID_SORT', 'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS']], on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        df_base = df_base.merge(hist_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

        df_base['year'] = df_base['TERM_CODE'].str[:4].astype(int)
        train_mask = df_base['year'] < args.val_year
        val_mask = df_base['year'] >= args.val_year

        base_pipe = BaselinePipeline()
        base_pipe.fit(df_base[train_mask])
        base_preds = base_pipe.predict(df_base[val_mask])

        y_val = df_base.loc[val_mask, 'HIGH_ENROLLMENT'].values
        baseline_macro_f1 = f1_score(y_val, base_preds, average='macro')
        print(f"  -> Baseline Validation Macro F1: {baseline_macro_f1:.4f}")

    # Step 2: Improved pipeline
    improved_pipeline = ImprovedPipeline(val_split_year=args.val_year)
    print("\n[2/3] Training and Validating Improved Pipeline...")
    val_results = improved_pipeline.fit_and_validate(train_dir=args.train_dir)
    print(f"  -> Optimal Calibrated Threshold: {val_results['optimal_threshold']:.3f}")
    print(f"  -> Improved Validation Macro F1: {val_results['val_macro_f1']:.4f}")
    print("\nValidation Classification Report:")
    print(val_results['report'])

    if baseline_macro_f1 is not None:
        delta = val_results['val_macro_f1'] - baseline_macro_f1
        print(f"  -> Macro F1 Improvement over Baseline: +{delta:.4f} ({(delta/baseline_macro_f1)*100:+.2f}%)")

    # Step 3: Test prediction and evaluation if test_dir provided
    if args.test_dir:
        print(f"\n[3/3] Predicting on Test Directory: {args.test_dir}...")
        test_preds = improved_pipeline.predict(args.test_dir)
        print(f"  -> Generated {len(test_preds):,} predictions.")

        if not test_preds.empty:
            test_preds.to_csv(args.output, index=False)
            print(f"  -> Predictions saved to '{args.output}'.")

            # Check if gold test labels exist for evaluation
            gold_test_path = os.path.join(args.test_dir, 'gold_enrollment_test.csv')
            if os.path.exists(gold_test_path):
                print(f"  -> Evaluating predictions against '{gold_test_path}'...")
                evaluate_predictions(test_preds, gold_test_path)
        else:
            print("  -> Test directory contains placeholder/empty tables (0 offerings).")

    # Optional Sanity Check on validation slice
    if args.sanity_check:
        print("\n[Sanity Check] Evaluating prediction and evaluation path on validation slice...")
        gold_df = pd.read_csv(os.path.join(args.train_dir, 'gold_enrollment_train.csv'))
        val_terms = [t for t in gold_df['TERM_CODE'].unique() if int(t[:4]) >= args.val_year]
        sanity_gold = gold_df[gold_df['TERM_CODE'].isin(val_terms)]
        val_features = extract_features(
            data_dir=args.train_dir,
            gold_df=None,
            full_priors=improved_pipeline.full_subject_priors,
            global_prior=improved_pipeline.global_prior,
            is_training=False
        )
        val_features = val_features[val_features['TERM_CODE'].isin(val_terms)]
        X_val_s = improved_pipeline._prepare_matrix(val_features)
        probs_s = improved_pipeline.model.predict_proba(X_val_s)[:, 1]
        preds_s = np.where(probs_s >= improved_pipeline.optimal_threshold, 'Y', 'N')
        sanity_preds = pd.DataFrame({
            'TERM_CODE': val_features['TERM_CODE'],
            'SUBJECT_ID_SORT': val_features['SUBJECT_ID_SORT'],
            'HIGH_ENROLLMENT': preds_s
        })
        evaluate_predictions(sanity_preds, sanity_gold)

    print("\nPipeline execution completed successfully.")


if __name__ == '__main__':
    main()
