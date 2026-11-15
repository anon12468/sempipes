import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import joblib

# Define paths
TRAIN_DATA_DIR = "table_splits/train/"
GOLD_ENROLLMENT_TRAIN_PATH = TRAIN_DATA_DIR + "gold_enrollment_train.csv"
ENROLLMENT_HISTORY_FEATURES_PATH = TRAIN_DATA_DIR + "ENROLLMENT_HISTORY_FEATURES.csv"
SUBJECT_OFFERED_PATH = TRAIN_DATA_DIR + "SUBJECT_OFFERED.csv"

# Frequency encoder custom transformer
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoding_maps = {}
        for col in X.columns:
            self.encoding_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            # Use .get() with a default value of 0 for unseen categories
            X_encoded[col] = X_encoded[col].map(self.encoding_maps[col]).fillna(0)
        return X_encoded

    def get_feature_names_out(self, input_features=None):
        # For frequency encoding, the feature names remain the same
        return input_features

def load_data():
    """Loads the necessary CSV files into pandas DataFrames."""
    gold_enrollment = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
    enrollment_history = pd.read_csv(ENROLLMENT_HISTORY_FEATURES_PATH)
    subject_offered = pd.read_csv(SUBJECT_OFFERED_PATH)
    return gold_enrollment, enrollment_history, subject_offered

def preprocess_data(gold_enrollment, enrollment_history, subject_offered):
    """
    Merges data and performs initial feature engineering.
    """
    # Merge gold enrollment with historical features
    df = pd.merge(gold_enrollment, enrollment_history, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Merge with subject offered details
    df = pd.merge(df, subject_offered, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Handle missing values - fill numerical NaNs with 0, and categorical NaNs with 'Unknown'
    for col in ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean']:
        df[col] = df[col].fillna(0)

    # Convert HIGH_ENROLLMENT to numerical (0 and 1)
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    # For binary features, fill NaN with a sensible default or 'N' and convert to numerical
    binary_features = [
        'IS_MASTER_SECTION', 'IS_LECTURE_SECTION', 'IS_LAB_SECTION',
        'IS_RECITATION_SECTION', 'IS_DESIGN_SECTION', 'EVALUATE_THIS_SUBJECT',
        'IS_OSE_SUBJECT', 'IS_CREATED_BY_DATA_WAREHOUSE', 'IS_REPEATABLE_SUBJECT'
    ]
    for col in binary_features:
        if col in df.columns:
            df[col] = df[col].fillna('N')
            df[col] = df[col].map({'Y': 1, 'N': 0}) # Ensure they are numerical

    # Extract year and semester from TERM_CODE
    df['YEAR'] = df['TERM_CODE'].str[:4].astype(int)
    df['SEMESTER'] = df['TERM_CODE'].str[4:]

    return df

def train_model(df):
    """
    Trains a RandomForestClassifier model and evaluates it using time-based validation.
    """
    # Define features and target
    X = df.drop(['HIGH_ENROLLMENT', 'TERM_CODE', 'SUBJECT_ID_SORT', 'SUBJECT_TITLE'], axis=1)
    y = df['HIGH_ENROLLMENT']

    # Identify categorical and numerical features
    # High cardinality categorical features for frequency encoding
    high_cardinality_features = ['MEET_TIME', 'MEET_PLACE']
    
    # Low cardinality categorical features for one-hot encoding
    low_cardinality_features = ['SEMESTER', 'OFFER_DEPT_CODE', 'OFFER_DEPT_NAME', 'OFFER_SCHOOL_NAME', 'CLUSTER_TYPE', 'CLUSTER_TYPE_DESC', 'FORM_TYPE', 'FORM_TYPE_DESC']
    numerical_features = ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean', 'YEAR']

    # The binary features are already handled in preprocess_data and are now numerical
    binary_features_processed = [
        'IS_MASTER_SECTION', 'IS_LECTURE_SECTION', 'IS_LAB_SECTION',
        'IS_RECITATION_SECTION', 'IS_DESIGN_SECTION', 'EVALUATE_THIS_SUBJECT',
        'IS_OSE_SUBJECT', 'IS_CREATED_BY_DATA_WAREHOUSE', 'IS_REPEATABLE_SUBJECT'
    ]

    # Fill NaN values in low cardinality categorical features with 'Unknown' for consistent encoding
    for col in low_cardinality_features:
        if col in X.columns:
            X[col] = X[col].fillna('Unknown')
            
    # Fill NaN values in high cardinality categorical features with 'Unknown'
    for col in high_cardinality_features:
        if col in X.columns:
            X[col] = X[col].fillna('Unknown')

    # Print diagnostic information
    print(f"Shape of X before encoding: {X.shape}")
    print("Unique values in low cardinality categorical features (OneHotEncoded):")
    for col in low_cardinality_features:
        if col in X.columns:
            print(f"  {col}: {X[col].nunique()} unique values")
    print("Unique values in high cardinality categorical features (FrequencyEncoded):")
    for col in high_cardinality_features:
        if col in X.columns:
            print(f"  {col}: {X[col].nunique()} unique values")

    # Combine all features for preprocessing
    features_to_use_low_cat = [f for f in low_cardinality_features if f in X.columns]
    features_to_use_high_cat = [f for f in high_cardinality_features if f in X.columns]
    features_to_use_num = [f for f in numerical_features if f in X.columns]
    features_to_use_bin = [f for f in binary_features_processed if f in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), features_to_use_low_cat),
            ('freq', FrequencyEncoder(), features_to_use_high_cat),
            ('num', 'passthrough', features_to_use_num),
            ('bin', 'passthrough', features_to_use_bin)
        ],
        remainder='drop'
    )

    # Fit and transform the features
    # Combine all feature names to select columns from X consistently
    all_feature_columns = features_to_use_low_cat + features_to_use_high_cat + features_to_use_num + features_to_use_bin
    X_processed = preprocessor.fit_transform(X[all_feature_columns])

    # Time-based split: Use latest year(s) for validation
    # Determine the latest year in the training data
    latest_year = df['YEAR'].max()
    
    # Define validation set as the latest year, and training set as all prior years
    X_train_df = df[df['YEAR'] < latest_year]
    y_train = df[df['YEAR'] < latest_year]['HIGH_ENROLLMENT']
    X_val_df = df[df['YEAR'] == latest_year]
    y_val = df[df['YEAR'] == latest_year]['HIGH_ENROLLMENT']

    # Ensure that X_train_df and X_val_df have the same columns and apply preprocessing
    # This is crucial for consistent feature sets between train and validation
    X_train_processed = preprocessor.transform(X_train_df[all_feature_columns])
    X_val_processed = preprocessor.transform(X_val_df[all_feature_columns])

    # Initialize and train the model
    model = LGBMClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train_processed, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val_processed)

    # Evaluate the model
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"Validation Macro F1 Score: {f1}")

    return model, preprocessor

if __name__ == "__main__":
    gold_enrollment, enrollment_history, subject_offered = load_data()
    df = preprocess_data(gold_enrollment, enrollment_history, subject_offered)
    model, preprocessor = train_model(df)
    print("Model training and validation complete.")
    
    # Save the trained model and preprocessor
    joblib.dump(model, 'enrollment_prediction_model.joblib')
    joblib.dump(preprocessor, 'feature_preprocessor.joblib')
    print("Trained model and preprocessor saved.")
