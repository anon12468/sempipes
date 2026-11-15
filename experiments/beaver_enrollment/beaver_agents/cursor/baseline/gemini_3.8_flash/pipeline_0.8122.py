#!/usr/bin/env python3
"""
BEAVER Institutional Course High-Enrollment Prediction Pipeline.

Task:
  For each course offering per term, predict whether it will have high enrollment
  (yes or no) relative to other courses in the same department and term
  (top quartile among offerings with positive enrollment).

Prediction keys:
  TERM_CODE, SUBJECT_ID_SORT
Target:
  HIGH_ENROLLMENT ('Y' or 'N')
Metric:
  Macro F1 score
"""

import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 4))
import re
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Default directory paths
DEFAULT_TRAIN_DIR = 'table_splits/train'
DEFAULT_TEST_DIR = 'table_splits/test'
DEFAULT_TRAIN_GOLD = os.path.join(DEFAULT_TRAIN_DIR, 'gold_enrollment_train.csv')
DEFAULT_TEST_GOLD = os.path.join(DEFAULT_TEST_DIR, 'gold_enrollment_test.csv')


# ==============================================================================
# Helper & Extraction Utilities
# ==============================================================================

def parse_subject_number(subject_id: str) -> float:
    """Extract numeric suffix digits from SUBJECT_ID_SORT (e.g. 6.001 -> 1, HAA.6904 -> 6904)."""
    try:
        parts = str(subject_id).split('.')
        if len(parts) > 1:
            digits = ''.join(c for c in parts[1] if c.isdigit())
            return float(digits[:5]) if digits else -1.0
    except Exception:
        pass
    return -1.0


def load_table(directory: str, filename: str, **kwargs) -> pd.DataFrame:
    """Load a table from a given directory if it exists and has content, else return empty DataFrame."""
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return pd.DataFrame()
    return pd.read_csv(filepath, **kwargs)


def aggregate_subject_offered(so_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized aggregation of section-level attributes from SUBJECT_OFFERED.csv."""
    if so_df.empty:
        return pd.DataFrame(columns=['TERM_CODE', 'SUBJECT_ID_SORT'])

    cols_to_check = [
        'IS_LECTURE_SECTION', 'IS_LAB_SECTION', 'IS_RECITATION_SECTION',
        'IS_DESIGN_SECTION', 'EVALUATE_THIS_SUBJECT', 'IS_REPEATABLE_SUBJECT'
    ]
    df_copy = so_df[['TERM_CODE', 'SUBJECT_ID_SORT']].copy()
    for c in cols_to_check:
        if c in so_df.columns:
            df_copy[c] = (so_df[c] == 'Y').astype(np.int8)
        else:
            df_copy[c] = 0

    agg_dict = {
        'IS_LECTURE_SECTION': ['count', 'max', 'sum'],
        'IS_LAB_SECTION': ['max', 'sum'],
        'IS_RECITATION_SECTION': ['max', 'sum'],
        'IS_DESIGN_SECTION': 'max',
        'EVALUATE_THIS_SUBJECT': 'max',
        'IS_REPEATABLE_SUBJECT': 'max'
    }

    agg = df_copy.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(agg_dict)
    agg.columns = ['sec_' + '_'.join(col) if isinstance(col, tuple) else 'sec_' + col for col in agg.columns]
    return agg.reset_index()


# ==============================================================================
# Feature Engineering & Pipeline Class
# ==============================================================================

class BeaverFeaturePipeline:
    """
    Comprehensive Feature Engineering, Feature Extraction, and Aggregation Pipeline
    designed specifically for department-relative high enrollment prediction.
    """

    def __init__(self):
        self.course_prior_stats_ = {}
        self.dept_prior_stats_ = {}
        self.course_season_prior_stats_ = {}
        self.global_prior_rate_ = 0.08
        self.feature_cols_ = []
        self.cat_cols_ = ['OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'HGN_CODE', 'CLUSTER_TYPE', 'SEASON', 'IS_REGULAR_TERM']

    def fit_historical_stats(self, train_summary: pd.DataFrame, train_gold: pd.DataFrame):
        """
        Compute historical subject and department performance from the training labels
        to be utilized for future test terms without data leakage.
        """
        merged = pd.merge(train_summary[['TERM_CODE', 'SUBJECT_ID_SORT', 'OFFER_DEPT_CODE']],
                          train_gold[['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']],
                          on=['TERM_CODE', 'SUBJECT_ID_SORT'])

        merged['Y'] = (merged['HIGH_ENROLLMENT'] == 'Y').astype(int)
        merged['SEASON'] = merged['TERM_CODE'].str[4:]
        merged['YEAR'] = pd.to_numeric(merged['TERM_CODE'].str[:4], errors='coerce').fillna(2000).astype(int)
        valid_terms = merged[merged['YEAR'] >= 1995]

        self.global_prior_rate_ = float(valid_terms['Y'].mean())

        # Subject historical prior stats
        course_grp = valid_terms.groupby('SUBJECT_ID_SORT')['Y'].agg(['sum', 'count']).reset_index()
        course_grp['rate'] = (course_grp['sum'] + self.global_prior_rate_) / (course_grp['count'] + 1.0)
        self.course_prior_stats_ = course_grp.set_index('SUBJECT_ID_SORT').to_dict('index')

        # Course-Season historical stats
        course_seas = valid_terms.groupby(['SUBJECT_ID_SORT', 'SEASON'])['Y'].agg(['sum', 'count']).reset_index()
        course_seas['rate'] = (course_seas['sum'] + self.global_prior_rate_) / (course_seas['count'] + 1.0)
        self.course_season_prior_stats_ = course_seas.set_index(['SUBJECT_ID_SORT', 'SEASON'])['rate'].to_dict()

        # Department historical prior stats
        dept_grp = valid_terms.groupby('OFFER_DEPT_CODE')['Y'].agg(['sum', 'count']).reset_index()
        dept_grp['rate'] = (dept_grp['sum'] + self.global_prior_rate_) / (dept_grp['count'] + 1.0)
        self.dept_prior_stats_ = dept_grp.set_index('OFFER_DEPT_CODE')['rate'].to_dict()

    def build_features(self, data_dir: str, is_train: bool = False, train_gold_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract and aggregate features from data directory.
        If is_train=True, historical target features use expanding/prior chronological windows.
        If is_train=False, pre-computed training historical statistics are applied.
        """
        summary = load_table(data_dir, 'SUBJECT_OFFERED_SUMMARY.csv', low_memory=False)
        if summary.empty:
            return pd.DataFrame()

        enh = load_table(data_dir, 'ENROLLMENT_HISTORY_FEATURES.csv')
        terms = load_table(data_dir, 'ACADEMIC_TERMS.csv')
        so = load_table(data_dir, 'SUBJECT_OFFERED.csv',
                        usecols=['TERM_CODE', 'SUBJECT_ID_SORT', 'IS_LECTURE_SECTION', 'IS_LAB_SECTION',
                                 'IS_RECITATION_SECTION', 'IS_DESIGN_SECTION', 'EVALUATE_THIS_SUBJECT',
                                 'IS_REPEATABLE_SUBJECT'])

        # Aggregate section attributes from SUBJECT_OFFERED
        so_agg = aggregate_subject_offered(so)

        # Merge base tables
        df = pd.merge(summary, enh, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        if not so_agg.empty:
            df = pd.merge(df, so_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

        term_cols = [c for c in ['TERM_CODE', 'ACADEMIC_YEAR', 'IS_REGULAR_TERM'] if c in terms.columns]
        if term_cols:
            df = pd.merge(df, terms[term_cols], on='TERM_CODE', how='left')

        # Year & Season
        df['YEAR'] = pd.to_numeric(df['TERM_CODE'].str[:4], errors='coerce').fillna(2000).astype(int)
        df['SEASON'] = df['TERM_CODE'].str[4:]
        if 'IS_REGULAR_TERM' not in df.columns:
            df['IS_REGULAR_TERM'] = df['SEASON'].isin(['FA', 'SP']).map({True: 'Y', False: 'N'})

        # 1. Feature Extraction: Subject metadata & Text length
        df['cluster_size'] = df['CLUSTER_LIST'].fillna('').apply(lambda s: len(str(s).split(',')) if str(s).strip() else 0)
        df['title_len'] = df['SUBJECT_TITLE'].fillna('').astype(str).apply(len)
        df['title_word_count'] = df['SUBJECT_TITLE'].fillna('').astype(str).apply(lambda s: len(s.split()))
        df['subject_num'] = df['SUBJECT_ID_SORT'].apply(parse_subject_number)
        df['is_intro_level'] = ((df['subject_num'] >= 0) & (df['subject_num'] < 100)).astype(int)
        df['is_advanced_level'] = (df['subject_num'] >= 500).astype(int)

        # 2. Feature Engineering: Units
        for u in ['TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREPARATION_UNITS']:
            if u not in df.columns:
                df[u] = 0.0
            else:
                df[u] = pd.to_numeric(df[u], errors='coerce').fillna(0.0)

        df['lecture_unit_ratio'] = df['LECTURE_UNITS'] / (df['TOTAL_UNITS'] + 1.0)

        # 3. Enrollment History Features & Dynamics
        for c in ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean']:
            if c not in df.columns:
                df[c] = 0.0
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

        df['has_enrollment_history'] = ((df['roll3_enrollment_mean'] > 0) | (df['lag1_enrollment'] > 0) | (df['lag2_enrollment'] > 0)).astype(int)
        df['lag_diff'] = df['lag1_enrollment'] - df['lag2_enrollment']
        df['lag_trend'] = df['lag1_enrollment'] - df['roll3_enrollment_mean']
        df['log_roll3'] = np.log1p(df['roll3_enrollment_mean'])
        df['log_lag1'] = np.log1p(df['lag1_enrollment'])

        # Fill section aggregation columns if missing
        sec_cols = [
            'sec_IS_LECTURE_SECTION_count', 'sec_IS_LECTURE_SECTION_max', 'sec_IS_LECTURE_SECTION_sum',
            'sec_IS_LAB_SECTION_max', 'sec_IS_LAB_SECTION_sum',
            'sec_IS_RECITATION_SECTION_max', 'sec_IS_RECITATION_SECTION_sum',
            'sec_IS_DESIGN_SECTION_max', 'sec_EVALUATE_THIS_SUBJECT_max', 'sec_IS_REPEATABLE_SUBJECT_max'
        ]
        for sc in sec_cols:
            if sc not in df.columns:
                df[sc] = 0

        # 4. Department-Term Relative Features (Top Quartile Task Definition)
        dept_col = 'OFFER_DEPT_CODE' if 'OFFER_DEPT_CODE' in df.columns else 'COURSE_NUMBER'
        df[dept_col] = df[dept_col].fillna('UNKNOWN')

        df['dept_term_course_count'] = df.groupby(['TERM_CODE', dept_col])['SUBJECT_ID_SORT'].transform('count')

        for feat in ['roll3_enrollment_mean', 'lag1_enrollment', 'sec_IS_LECTURE_SECTION_count']:
            grp = df.groupby(['TERM_CODE', dept_col])[feat]
            dept_mean = grp.transform('mean')
            dept_p75 = grp.transform(lambda s: s.quantile(0.75))
            df[f'{feat}_dept_mean'] = dept_mean
            df[f'{feat}_dept_p75'] = dept_p75
            df[f'{feat}_rel_mean'] = df[feat] / (dept_mean + 1.0)
            df[f'{feat}_diff_mean'] = df[feat] - dept_mean
            df[f'{feat}_diff_p75'] = df[feat] - dept_p75
            df[f'{feat}_dept_rank'] = grp.rank(pct=True)

        # Department rank among offerings with positive enrollment
        pos_mask = df['roll3_enrollment_mean'] > 0
        df['roll3_dept_rank_pos'] = 0.0
        if pos_mask.any():
            df.loc[pos_mask, 'roll3_dept_rank_pos'] = df[pos_mask].groupby(['TERM_CODE', dept_col])['roll3_enrollment_mean'].rank(pct=True)

        # 5. Prior Historical Target Statistics (Zero-Leakage)
        if is_train and train_gold_df is not None:
            # Chronological expanding window for training
            df = pd.merge(df, train_gold_df[['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']],
                          on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
            df['Y'] = (df['HIGH_ENROLLMENT'] == 'Y').astype(int)

            term_order = df[['TERM_CODE', 'YEAR']].drop_duplicates().sort_values(['YEAR', 'TERM_CODE'])['TERM_CODE'].tolist()
            term_to_idx = {t: i for i, t in enumerate(term_order)}
            df['term_idx'] = df['TERM_CODE'].map(term_to_idx)
            df = df.sort_values(['term_idx']).reset_index(drop=True)

            cum_high = df.groupby('SUBJECT_ID_SORT')['Y'].shift(1).groupby(df['SUBJECT_ID_SORT']).cumsum().fillna(0)
            cum_count = df.groupby('SUBJECT_ID_SORT').cumcount()
            df['course_prior_high_count'] = cum_high
            df['course_prior_offer_count'] = cum_count
            df['course_prior_high_rate'] = (cum_high + self.global_prior_rate_) / (cum_count + 1.0)
            df['course_ever_high_prior'] = (cum_high > 0).astype(int)

            cum_high_seas = df.groupby(['SUBJECT_ID_SORT', 'SEASON'])['Y'].shift(1).groupby([df['SUBJECT_ID_SORT'], df['SEASON']]).cumsum().fillna(0)
            cum_count_seas = df.groupby(['SUBJECT_ID_SORT', 'SEASON']).cumcount()
            df['course_season_prior_high_rate'] = (cum_high_seas + self.global_prior_rate_) / (cum_count_seas + 1.0)

            dept_cum_high = df.groupby(dept_col)['Y'].shift(1).groupby(df[dept_col]).cumsum().fillna(0)
            dept_cum_count = df.groupby(dept_col).cumcount()
            df['dept_prior_high_rate'] = (dept_cum_high + self.global_prior_rate_) / (dept_cum_count + 1.0)
        else:
            # Apply learned prior statistics for test/inference
            def get_course_stats(sid):
                stat = self.course_prior_stats_.get(sid, None)
                if stat:
                    return stat['sum'], stat['count'], stat['rate'], 1 if stat['sum'] > 0 else 0
                return 0.0, 0, self.global_prior_rate_, 0

            stats = [get_course_stats(sid) for sid in df['SUBJECT_ID_SORT']]
            df['course_prior_high_count'] = [s[0] for s in stats]
            df['course_prior_offer_count'] = [s[1] for s in stats]
            df['course_prior_high_rate'] = [s[2] for s in stats]
            df['course_ever_high_prior'] = [s[3] for s in stats]

            df['course_season_prior_high_rate'] = [
                self.course_season_prior_stats_.get((sid, seas), self.course_prior_stats_.get(sid, {}).get('rate', self.global_prior_rate_))
                for sid, seas in zip(df['SUBJECT_ID_SORT'], df['SEASON'])
            ]

            df['dept_prior_high_rate'] = [
                self.dept_prior_stats_.get(d, self.global_prior_rate_) for d in df[dept_col]
            ]

        # Categorical typing
        for col in self.cat_cols_:
            if col in df.columns:
                df[col] = df[col].astype('category')

        return df


# ==============================================================================
# Model Pipelines (Baseline & Improved)
# ==============================================================================

class BaselineModel:
    """
    Given baseline model utilizing ENROLLMENT_HISTORY_FEATURES.csv and Logistic Regression.
    """

    def __init__(self):
        self.clf = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42)
        self.features = ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean']

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_feat = X[self.features].fillna(0.0)
        self.clf.fit(X_feat, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_feat = X[self.features].fillna(0.0)
        return self.clf.predict(X_feat)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_feat = X[self.features].fillna(0.0)
        return self.clf.predict_proba(X_feat)[:, 1]


class ImprovedEnrollmentModel:
    """
    Improved LightGBM Gradient Boosted Decision Tree pipeline with engineered features,
    department relative aggregates, zero-leakage historical course encoding,
    and macro F1 threshold optimization.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.threshold = 0.35
        self.pipeline = BeaverFeaturePipeline()
        self.model = None
        self.feature_cols = [
            'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREPARATION_UNITS', 'lecture_unit_ratio',
            'cluster_size', 'title_len', 'title_word_count', 'subject_num', 'is_intro_level', 'is_advanced_level',
            'lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean', 'log_roll3', 'log_lag1',
            'has_enrollment_history', 'lag_diff', 'lag_trend',
            'course_prior_high_count', 'course_prior_offer_count', 'course_prior_high_rate', 'course_ever_high_prior',
            'course_season_prior_high_rate', 'dept_prior_high_rate', 'dept_term_course_count',
            'sec_IS_LECTURE_SECTION_count', 'sec_IS_LECTURE_SECTION_max', 'sec_IS_LECTURE_SECTION_sum',
            'sec_IS_LAB_SECTION_max', 'sec_IS_LAB_SECTION_sum',
            'sec_IS_RECITATION_SECTION_max', 'sec_IS_RECITATION_SECTION_sum',
            'sec_IS_DESIGN_SECTION_max', 'sec_EVALUATE_THIS_SUBJECT_max', 'sec_IS_REPEATABLE_SUBJECT_max',
            'roll3_enrollment_mean_dept_mean', 'roll3_enrollment_mean_dept_p75', 'roll3_enrollment_mean_rel_mean',
            'roll3_enrollment_mean_diff_mean', 'roll3_enrollment_mean_diff_p75', 'roll3_enrollment_mean_dept_rank',
            'roll3_dept_rank_pos',
            'lag1_enrollment_dept_mean', 'lag1_enrollment_dept_p75', 'lag1_enrollment_rel_mean',
            'lag1_enrollment_diff_mean', 'lag1_enrollment_diff_p75', 'lag1_enrollment_dept_rank',
            'sec_IS_LECTURE_SECTION_count_dept_mean', 'sec_IS_LECTURE_SECTION_count_rel_mean',
            'sec_IS_LECTURE_SECTION_count_diff_mean', 'sec_IS_LECTURE_SECTION_count_dept_rank',
            'OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'HGN_CODE', 'CLUSTER_TYPE', 'SEASON', 'IS_REGULAR_TERM'
        ]

    def fit(self, train_dir: str, train_gold_path: str, val_split_year: int = 2017):
        """
        Fit model using TRAIN_DATA_DIR and gold_enrollment_train.csv only.
        Uses time-based validation slice to optimize decision threshold for macro F1,
        then fits the final model.
        """
        print(f"[ImprovedModel] Loading train tables from {train_dir}...")
        gold_df = pd.read_csv(train_gold_path)
        summary_df = load_table(train_dir, 'SUBJECT_OFFERED_SUMMARY.csv', low_memory=False)

        # Fit historical priors on train tables
        print("[ImprovedModel] Fitting zero-leakage historical course priors...")
        self.pipeline.fit_historical_stats(summary_df, gold_df)

        # Extract features for all train rows
        print("[ImprovedModel] Building feature dataset...")
        df = self.pipeline.build_features(train_dir, is_train=True, train_gold_df=gold_df)

        # Filter out pre-1995 historical terms (where labels were not tracked)
        valid_df = df[df['YEAR'] >= 1995].copy()
        valid_df['Y'] = (valid_df['HIGH_ENROLLMENT'] == 'Y').astype(int)

        train_mask = valid_df['YEAR'] < val_split_year
        val_mask = valid_df['YEAR'] >= val_split_year

        X_train = valid_df.loc[train_mask, self.feature_cols]
        y_train = valid_df.loc[train_mask, 'Y']
        X_val = valid_df.loc[val_mask, self.feature_cols]
        y_val = valid_df.loc[val_mask, 'Y']

        print(f"[ImprovedModel] Time-based Validation: train {X_train.shape[0]} rows (terms < {val_split_year}), val {X_val.shape[0]} rows (terms >= {val_split_year})")

        # Train model for validation
        val_model = lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=35,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        val_model.fit(X_train, y_train)

        # Threshold optimization for Macro F1 on validation slice
        val_probs = val_model.predict_proba(X_val)[:, 1]
        best_t = 0.5
        best_f1 = 0.0
        for t in np.arange(0.15, 0.65, 0.01):
            score = f1_score(y_val, (val_probs >= t).astype(int), average='macro')
            if score > best_f1:
                best_f1 = score
                best_t = t

        self.threshold = round(float(best_t), 2)
        val_preds = (val_probs >= self.threshold).astype(int)
        val_macro_f1 = f1_score(y_val, val_preds, average='macro')
        val_auc = roc_auc_score(y_val, val_probs)

        print("\n" + "=" * 60)
        print(f"[Validation Results] Optimal Threshold: {self.threshold:.2f}")
        print(f"[Validation Results] Macro F1: {val_macro_f1:.4f} | ROC-AUC: {val_auc:.4f}")
        print("=" * 60)
        print(classification_report(y_val, val_preds, target_names=['N (Low)', 'Y (High)']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, val_preds))
        print("=" * 60 + "\n")

        # Fit final production model on all valid data (1995 - latest)
        print("[ImprovedModel] Training final model on all historical training data (1995-2018)...")
        X_all = valid_df[self.feature_cols]
        y_all = valid_df['Y']
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.035,
            num_leaves=35,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        self.model.fit(X_all, y_all)
        print("[ImprovedModel] Model training complete.")
        return val_macro_f1

    def predict(self, test_dir: str) -> pd.DataFrame:
        """
        Generate predictions for course offerings in test_dir.
        Returns DataFrame with columns: ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']
        aligned 1-to-1 with SUBJECT_OFFERED_SUMMARY in test_dir.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet! Call fit() first.")

        test_features_df = self.pipeline.build_features(test_dir, is_train=False)
        if test_features_df.empty:
            return pd.DataFrame(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'])

        X_test = test_features_df[self.feature_cols]
        probs = self.model.predict_proba(X_test)[:, 1]
        preds_binary = (probs >= self.threshold).astype(int)

        result_df = pd.DataFrame({
            'TERM_CODE': test_features_df['TERM_CODE'],
            'SUBJECT_ID_SORT': test_features_df['SUBJECT_ID_SORT'],
            'HIGH_ENROLLMENT': np.where(preds_binary == 1, 'Y', 'N')
        })
        return result_df


# ==============================================================================
# Evaluation Functionality
# ==============================================================================

def evaluate_predictions(preds_df: pd.DataFrame, gold_df: pd.DataFrame) -> dict:
    """
    Evaluates model predictions against ground truth labels.
    Computes Macro F1, Accuracy, Precision, Recall, Confusion Matrix, and Classification Report.
    """
    if preds_df.empty or gold_df.empty:
        print("[Evaluation] Warning: Predictions or Gold dataset is empty. Cannot compute metrics.")
        return {}

    merged = pd.merge(gold_df, preds_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], suffixes=('_true', '_pred'))
    if len(merged) != len(gold_df):
        print(f"[Evaluation] Note: Matched {len(merged)} of {len(gold_df)} gold rows.")

    y_true = (merged['HIGH_ENROLLMENT_true'] == 'Y').astype(int)
    y_pred = (merged['HIGH_ENROLLMENT_pred'] == 'Y').astype(int)

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=['N (Low)', 'Y (High)'])

    print("\n" + "#" * 60)
    print("                    EVALUATION RESULTS")
    print("#" * 60)
    print(f"Total Evaluated Rows: {len(merged)}")
    print(f"Benchmark Macro F1:  {macro_f1:.4f}")
    print(f"Accuracy:            {acc:.4f}")
    print(f"F1 Score (Class Y):  {f1_pos:.4f}")
    print(f"F1 Score (Class N):  {f1_neg:.4f}")
    print("-" * 60)
    print("Classification Report:")
    print(cr)
    print("Confusion Matrix:")
    print(cm)
    print("#" * 60 + "\n")

    return {
        'macro_f1': macro_f1,
        'accuracy': acc,
        'f1_y': f1_pos,
        'f1_n': f1_neg,
        'confusion_matrix': cm
    }


def evaluate_test(model: ImprovedEnrollmentModel, test_dir: str, gold_test_path: str, output_path: str = None) -> dict:
    """
    Executes test prediction and evaluation logic:
    Takes gold_enrollment_test.csv and test directory, generates predictions, and evaluates.
    """
    print(f"\n[Test Evaluation] Predicting on test directory: {test_dir}...")
    if not os.path.exists(test_dir):
        print(f"[Test Evaluation] Error: Test directory '{test_dir}' does not exist.")
        return {}

    preds_df = model.predict(test_dir)
    print(f"[Test Evaluation] Generated {len(preds_df)} predictions.")

    if output_path and not preds_df.empty:
        preds_df.to_csv(output_path, index=False)
        print(f"[Test Evaluation] Saved predictions to {output_path}")

    if not os.path.exists(gold_test_path) or os.path.getsize(gold_test_path) == 0:
        print(f"[Test Evaluation] Test gold file '{gold_test_path}' is empty or does not exist yet (placeholder).")
        print("[Test Evaluation] Ready for external evaluation when held-out test data is supplied.")
        return {'status': 'placeholder_test_dir', 'predictions_count': len(preds_df)}

    gold_test_df = pd.read_csv(gold_test_path)
    if gold_test_df.empty:
        print(f"[Test Evaluation] Test gold file '{gold_test_path}' has 0 rows.")
        return {'status': 'empty_gold', 'predictions_count': len(preds_df)}

    return evaluate_predictions(preds_df, gold_test_df)


# ==============================================================================
# Baseline Runner (Comparison)
# ==============================================================================

def run_baseline_comparison(train_dir: str, gold_path: str, val_split_year: int = 2017) -> float:
    """
    Runs the given baseline model on the time-based validation slice to provide
    a benchmark comparison for feature engineering improvements.
    """
    print("\n" + "=" * 60)
    print("               RUNNING BASELINE MODEL")
    print("=" * 60)
    gold = pd.read_csv(gold_path)
    enh = load_table(train_dir, 'ENROLLMENT_HISTORY_FEATURES.csv')

    df = pd.merge(gold, enh, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    df['YEAR'] = pd.to_numeric(df['TERM_CODE'].str[:4], errors='coerce').fillna(2000).astype(int)
    df = df[df['YEAR'] >= 1995].copy()

    train_mask = df['YEAR'] < val_split_year
    val_mask = df['YEAR'] >= val_split_year

    baseline = BaselineModel()
    X_train = df.loc[train_mask]
    y_train = (df.loc[train_mask, 'HIGH_ENROLLMENT'] == 'Y').astype(int)
    X_val = df.loc[val_mask]
    y_val = (df.loc[val_mask, 'HIGH_ENROLLMENT'] == 'Y').astype(int)

    baseline.fit(X_train, y_train)
    val_preds = baseline.predict(X_val)

    score = f1_score(y_val, val_preds, average='macro')
    print(f"[Baseline Result] Logistic Regression Macro F1: {score:.4f}")
    print(classification_report(y_val, val_preds, target_names=['N (Low)', 'Y (High)']))
    print("=" * 60 + "\n")
    return score


# ==============================================================================
# Main CLI Interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="BEAVER High-Enrollment Prediction Pipeline")
    parser.add_argument('--train_dir', type=str, default=DEFAULT_TRAIN_DIR, help="Path to training tables directory")
    parser.add_argument('--test_dir', type=str, default=DEFAULT_TEST_DIR, help="Path to test tables directory")
    parser.add_argument('--gold_train', type=str, default=DEFAULT_TRAIN_GOLD, help="Path to gold training labels CSV")
    parser.add_argument('--gold_test', type=str, default=DEFAULT_TEST_GOLD, help="Path to gold test labels CSV")
    parser.add_argument('--val_year', type=int, default=2017, help="Cutoff year for time-based validation holdout")
    parser.add_argument('--output_preds', type=str, default=None, help="Filepath to save test predictions")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'baseline', 'train', 'evaluate', 'sanity_check'],
                        help="Execution mode")

    args = parser.parse_args()

    print("=" * 60)
    print("  BEAVER ACADEMIC DATA WAREHOUSE - ENROLLMENT PREDICTION")
    print(f"  Train Dir: {args.train_dir}")
    print(f"  Test Dir:  {args.test_dir}")
    print(f"  Mode:      {args.mode}")
    print("=" * 60)

    if args.mode in ['all', 'baseline']:
        baseline_score = run_baseline_comparison(args.train_dir, args.gold_train, val_split_year=args.val_year)

    if args.mode in ['all', 'train', 'evaluate', 'sanity_check']:
        # Improved model
        model = ImprovedEnrollmentModel(random_state=42)
        val_score = model.fit(args.train_dir, args.gold_train, val_split_year=args.val_year)

        if args.mode in ['all', 'train']:
            if 'baseline_score' in locals():
                print(f"[Summary] Baseline Macro F1: {baseline_score:.4f} -> Improved Pipeline Macro F1: {val_score:.4f}")
                print(f"[Summary] Absolute Improvement: +{(val_score - baseline_score):.4f} Macro F1 points.")

        if args.mode in ['all', 'evaluate']:
            evaluate_test(model, args.test_dir, args.gold_test, output_path=args.output_preds)

        if args.mode == 'sanity_check':
            print("\n[Sanity Check] Running prediction path on training directory...")
            train_preds = model.predict(args.train_dir)
            print(f"[Sanity Check] Generated {len(train_preds)} predictions on train directory.")
            print("[Sanity Check] Evaluating predictions against gold_enrollment_train.csv...")
            gold_train_df = pd.read_csv(args.gold_train)
            evaluate_predictions(train_preds, gold_train_df)

    print("[Pipeline] Done.")


if __name__ == '__main__':
    main()
