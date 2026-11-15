import pandas as pd
import numpy as np

def create_features(merged_train_df):
    print("Performing advanced feature engineering (with department aggregates and relative features). ")
    features_df = merged_train_df.copy()

    # Example feature: Extract year and semester from TERM_CODE
    if 'TERM_CODE' in features_df.columns:
        features_df['TERM_YEAR'] = features_df['TERM_CODE'].str[:4].astype(int)
        features_df['TERM_SEMESTER'] = features_df['TERM_CODE'].str[4:]
        features_df = pd.get_dummies(features_df, columns=['TERM_SEMESTER'], prefix='semester', drop_first=True)

    # Example feature: Length of SUBJECT_ID_SORT
    if 'SUBJECT_ID_SORT' in features_df.columns:
        features_df['SUBJECT_ID_SORT_LEN'] = features_df['SUBJECT_ID_SORT'].apply(len)

    # New features from existing enrollment history features
    enrollment_cols = ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean']
    if all(col in features_df.columns for col in enrollment_cols):
        # Difference features
        features_df['lag1_minus_lag2'] = features_df['lag1_enrollment'] - features_df['lag2_enrollment']
        features_df['lag1_minus_roll3_mean'] = features_df['lag1_enrollment'] - features_df['roll3_enrollment_mean']

        # Ratio features (handle division by zero/NaNs)
        features_df['lag1_to_lag2_ratio'] = features_df['lag1_enrollment'] / features_df['lag2_enrollment'].replace(0, np.nan)
        features_df['lag1_to_roll3_mean_ratio'] = features_df['lag1_enrollment'] / features_df['roll3_enrollment_mean'].replace(0, np.nan)
        
        # Fill potential NaNs created by division with 0 or a suitable value
        features_df['lag1_to_lag2_ratio'] = features_df['lag1_to_lag2_ratio'].fillna(0)
        features_df['lag1_to_roll3_mean_ratio'] = features_df['lag1_to_roll3_mean_ratio'].fillna(0)

        # Polynomial features (e.g., square of lag1 enrollment)
        features_df['lag1_enrollment_sq'] = features_df['lag1_enrollment'] ** 2

        # --- Relative features based on department and term averages ---
        if 'TERM_CODE' in features_df.columns and 'SUBJECT_DEPT_CODE' in features_df.columns:
            # Calculate department-term averages for enrollment features
            dept_term_avg_enrollment = features_df.groupby(['TERM_CODE', 'SUBJECT_DEPT_CODE'])[enrollment_cols].transform('mean')
            dept_term_avg_enrollment.columns = [f'avg_dept_term_{col}' for col in enrollment_cols]
            features_df = pd.concat([features_df, dept_term_avg_enrollment], axis=1)

            # Create relative features
            for col in enrollment_cols:
                features_df[f'{col}_vs_dept_term_avg'] = features_df[col] / features_df[f'avg_dept_term_{col}'].replace(0, np.nan)
                features_df[f'{col}_vs_dept_term_avg'] = features_df[f'{col}_vs_dept_term_avg'].fillna(0)

    # Drop original 'SUBJECT_DEPT_CODE' and 'OFFER_DEPT_CODE' if they are not needed as raw features
    features_df.drop(columns=['SUBJECT_DEPT_CODE', 'OFFER_DEPT_CODE'], errors='ignore', inplace=True)

    # Fill any remaining NaNs created by new features with 0 or a strategy like mean/median
    features_df = features_df.fillna(0) # Simple imputation for now

    # Convert target variable to numerical for modeling
    if 'HIGH_ENROLLMENT' in features_df.columns:
        features_df['HIGH_ENROLLMENT'] = features_df['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    return features_df