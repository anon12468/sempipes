import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import numpy as np

# Define paths - these would be adapted for test data later
TEST_DATA_DIR = "/Users/USER/BEAVER/test/" # Placeholder, will be TEST_DATA_DIR
GOLD_ENROLLMENT_TEST_PATH = "/Users/USER/BEAVER/test/gold_enrollment_test.csv" # Placeholder, will be gold_enrollment_test.csv
ENROLLMENT_HISTORY_FEATURES_PATH = TEST_DATA_DIR + "ENROLLMENT_HISTORY_FEATURES.csv"
SUBJECT_OFFERED_PATH = TEST_DATA_DIR + "SUBJECT_OFFERED.csv"

# Re-define FrequencyEncoder as it needs to be available for loading the preprocessor
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoding_maps = {}
        for col in X.columns:
            self.encoding_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            X_encoded[col] = X_encoded[col].map(self.encoding_maps[col]).fillna(0)
        return X_encoded

    def get_feature_names_out(self, input_features=None):
        return input_features

def load_test_data(test_gold_path=None):
    """Loads the necessary CSV files for prediction."""
    # For initial testing, we'll use train data as 'test' data
    if test_gold_path:
        gold_enrollment_test = pd.read_csv(test_gold_path)
    else:
        gold_enrollment_test = pd.read_csv(GOLD_ENROLLMENT_TEST_PATH)

    enrollment_history = pd.read_csv(ENROLLMENT_HISTORY_FEATURES_PATH)
    subject_offered = pd.read_csv(SUBJECT_OFFERED_PATH)
    return gold_enrollment_test, enrollment_history, subject_offered

def preprocess_test_data(gold_enrollment_test, enrollment_history, subject_offered, preprocessor):
    """
    Merges data and preprocesses it using the loaded preprocessor.
    """
    df_test = pd.merge(gold_enrollment_test, enrollment_history, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    df_test = pd.merge(df_test, subject_offered, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Handle missing values - fill numerical NaNs with 0
    for col in ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean']:
        df_test[col] = df_test[col].fillna(0)

    # Convert HIGH_ENROLLMENT to numerical (0 and 1) if it exists (for evaluation)
    if 'HIGH_ENROLLMENT' in df_test.columns:
        df_test['HIGH_ENROLLMENT'] = df_test['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    # For binary features, fill NaN with 'N' and convert to numerical
    binary_features = [
        'IS_MASTER_SECTION', 'IS_LECTURE_SECTION', 'IS_LAB_SECTION',
        'IS_RECITATION_SECTION', 'IS_DESIGN_SECTION', 'EVALUATE_THIS_SUBJECT',
        'IS_OSE_SUBJECT', 'IS_CREATED_BY_DATA_WAREHOUSE', 'IS_REPEATABLE_SUBJECT'
    ]
    for col in binary_features:
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna('N')
            df_test[col] = df_test[col].map({'Y': 1, 'N': 0}) # Ensure they are numerical

    # Extract year and semester from TERM_CODE
    df_test['YEAR'] = df_test['TERM_CODE'].str[:4].astype(int)
    df_test['SEMESTER'] = df_test['TERM_CODE'].str[4:]

    # Define features to use for prediction
    # These must match the features used during training
    high_cardinality_features = ['MEET_TIME', 'MEET_PLACE']
    low_cardinality_features = ['SEMESTER', 'OFFER_DEPT_CODE', 'OFFER_DEPT_NAME', 'OFFER_SCHOOL_NAME', 'CLUSTER_TYPE', 'CLUSTER_TYPE_DESC', 'FORM_TYPE', 'FORM_TYPE_DESC']
    numerical_features = ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean', 'YEAR']
    binary_features_processed = [
        'IS_MASTER_SECTION', 'IS_LECTURE_SECTION', 'IS_LAB_SECTION',
        'IS_RECITATION_SECTION', 'IS_DESIGN_SECTION', 'EVALUATE_THIS_SUBJECT',
        'IS_OSE_SUBJECT', 'IS_CREATED_BY_DATA_WAREHOUSE', 'IS_REPEATABLE_SUBJECT'
    ]

    # Fill NaN values in categorical features with 'Unknown' for consistent encoding
    for col in low_cardinality_features + high_cardinality_features:
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna('Unknown')

    all_feature_columns = [
        f for f in (low_cardinality_features + high_cardinality_features + numerical_features + binary_features_processed)
        if f in df_test.columns
    ]

    X_test = df_test[all_feature_columns]

    # Transform data using the fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)
    
    return df_test, X_test_processed

def make_predictions(model, X_test_processed, original_df_test):
    """
    Makes predictions and formats the output.
    """
    predictions = model.predict(X_test_processed)
    predictions_df = pd.DataFrame({
        'TERM_CODE': original_df_test['TERM_CODE'],
        'SUBJECT_ID_SORT': original_df_test['SUBJECT_ID_SORT'],
        'HIGH_ENROLLMENT': np.where(predictions == 1, 'Y', 'N')
    })
    return predictions_df

def evaluate_predictions(predictions_df, gold_enrollment_test):
    """
    Evaluates predictions against the gold standard if available.
    """
    # Merge predictions with gold standard to ensure correct alignment
    eval_df = pd.merge(predictions_df, gold_enrollment_test, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner',
                       suffixes=('_pred', '_gold'))

    # Map gold labels to numerical for F1 score calculation
    eval_df['HIGH_ENROLLMENT_gold_numeric'] = eval_df['HIGH_ENROLLMENT_gold'].map({'Y': 1, 'N': 0})
    eval_df['HIGH_ENROLLMENT_pred_numeric'] = eval_df['HIGH_ENROLLMENT_pred'].map({'Y': 1, 'N': 0})

    f1 = f1_score(eval_df['HIGH_ENROLLMENT_gold_numeric'], eval_df['HIGH_ENROLLMENT_pred_numeric'], average='macro')
    print(f"Macro F1 Score on test data: {f1}")
    return f1

if __name__ == "__main__":
    # Load the trained model and preprocessor
    try:
        model = joblib.load('enrollment_prediction_model.joblib')
        preprocessor = joblib.load('feature_preprocessor.joblib')
        print("Model and preprocessor loaded successfully.")
    except FileNotFoundError:
        print("Error: Model or preprocessor files not found. Please run train_model.py first.")
        exit()

    # Load test data (using training data as a placeholder for now)
    gold_enrollment_test, enrollment_history_test, subject_offered_test = load_test_data()
    
    # Preprocess test data
    original_df_test, X_test_processed = preprocess_test_data(gold_enrollment_test, enrollment_history_test, subject_offered_test, preprocessor)

    # Make predictions
    predictions_df = make_predictions(model, X_test_processed, original_df_test)
    print("Predictions made.")
    print(predictions_df.head())

    # Save predictions to a CSV file
    predictions_output_path = "predictions.csv"
    predictions_df.to_csv(predictions_output_path, index=False)
    print(f"Predictions saved to {predictions_output_path}")

    # Evaluate predictions if gold labels are available (always true for this placeholder)
    print("Evaluating predictions...")
    evaluate_predictions(predictions_df, gold_enrollment_test)
