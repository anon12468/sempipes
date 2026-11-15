import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def normalize_street_name(street_name):
    """Normalizes street names for consistent matching."""
    if isinstance(street_name, str):
        return street_name.upper().replace('.', '').replace('  ', ' ').strip()
    return street_name

def load_and_preprocess_data(train_path, corpus_features_path):
    """
    Loads and preprocesses the training and augmentation data.
    """
    # Load primary training data
    df_train = pd.read_csv(train_path)
    df_train.columns = ['street_name', 'violation_description', 'violation_count']

    # Load corpus features
    df_corpus = pd.read_parquet(corpus_features_path)
    df_corpus.rename(columns={'norm_street': 'street_name'}, inplace=True)

    # Normalize street names in both dataframes for merging
    df_train['street_name_normalized'] = df_train['street_name'].apply(normalize_street_name)
    df_corpus['street_name_normalized'] = df_corpus['street_name'].apply(normalize_street_name)

    # Merge dataframes
    df_merged = pd.merge(df_train, df_corpus, on='street_name_normalized', how='left')

    return df_merged, df_corpus # Return df_corpus as well

def train_and_predict(df_merged, df_corpus, test_path=None):
    """
    Trains a model and generates predictions.
    """
    # Feature Engineering (example: label encoding for violation_description)
    le = LabelEncoder()
    df_merged['violation_description_encoded'] = le.fit_transform(df_merged['violation_description'])

    # Define features and target
    features = [
        'violation_description_encoded',
        'meter_count',
        'active_meter_count',
        'speed_limit_mean',
        'speed_limit_max',
        'speed_total_length',
        'speed_segment_count',
        'pave_rating_mean',
        'pave_rating_min',
        'pave_length_sum',
        'pave_segment_count',
        'storefront_count',
        'storefront_vacant_count',
        'restaurant_count',
        'restaurant_score_mean',
        'speed_reducer_count',
        'tree_count',
        'bike_rack_count',
        'dob_violation_count',
        'service_311_count'
    ]
    target = 'violation_count'

    # Handle missing values by filling with 0 or mean for numerical features
    for col in features:
        if df_merged[col].dtype in ['float64', 'int64']:
            df_merged[col] = df_merged[col].fillna(0) # Changed to avoid FutureWarning

    # Split data for validation
    X = df_merged[features]
    y = df_merged[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Validation RMSE: {rmse}")

    # If test_path is provided, generate predictions for it
    if test_path:
        df_test = pd.read_csv(test_path)
        # Ensure test file has the expected columns, add violation_count if missing for consistent merge and prediction prep
        if 'violation_count' not in df_test.columns:
            df_test['violation_count'] = np.nan # Fill with NaN if not present, as it's optional
        df_test.columns = ['street_name', 'violation_description', 'violation_count']

        df_test['street_name_normalized'] = df_test['street_name'].apply(normalize_street_name)
        df_test_merged = pd.merge(df_test, df_corpus, on='street_name_normalized', how='left')

        # Encode violation_description for test data (using the same encoder fitted on training data)
        # Handle unseen labels in test data by assigning a default value (e.g., -1 or re-fitting with unseen labels if appropriate)
        # For simplicity, here we'll transform and if a new label appears, it will throw an error or be handled by default value for unseen labels.
        # A more robust approach might involve adding unseen labels to the LabelEncoder or using a different encoding strategy.
        df_test_merged['violation_description_encoded'] = df_test_merged['violation_description'].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1 # Assign -1 for unseen categories
        )

        # Handle missing values in test set features
        for col in features:
            if col not in df_test_merged.columns: # Add missing feature columns to test set if any
                df_test_merged[col] = 0
            if df_test_merged[col].dtype in ['float64', 'int64']:
                df_test_merged[col] = df_test_merged[col].fillna(0) # Changed to avoid FutureWarning

        X_test = df_test_merged[features]
        # Align columns - very important for consistent feature set between train and test
        # This ensures that the order and presence of columns in X_test match X_train
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for c in missing_cols:
            X_test[c] = 0
        X_test = X_test[X_train.columns]

        test_predictions = model.predict(X_test)
        df_submission = pd.DataFrame({
            'street_name': df_test['street_name'],
            'violation_type': df_test['violation_description'],
            'predicted_count': test_predictions.round().astype(int).clip(min=0)
        })
        df_submission.to_csv('submission.csv', index=False)
        print("Predictions saved to submission.csv")

        # Report RMSE if ground truth is available in test_path and not all NaNs
        if 'violation_count' in df_test.columns and df_test['violation_count'].isnull().sum() < len(df_test):
            # Only consider rows where violation_count is not NaN for RMSE calculation
            valid_rows = df_test['violation_count'].notnull()
            test_rmse = np.sqrt(mean_squared_error(df_test[valid_rows]['violation_count'], test_predictions[valid_rows]))
            print(f"Test RMSE: {test_rmse}")
        else:
            print("Test file does not contain ground truth (violation_count) for RMSE calculation.")

        # Report how many key pairs in test/eval were unseen in training and how those were handled.
        # An unseen key pair is one where 'street_name_normalized' from the test set did not find a match in df_corpus.
        unseen_street_names_count = df_test_merged['street_name_normalized'].isnull().sum()
        print(f"Number of unseen street_name_normalized in test/eval: {unseen_street_names_count}")
        unseen_violation_types_count = (df_test_merged['violation_description_encoded'] == -1).sum()
        print(f"Number of unseen violation_description in test/eval: {unseen_violation_types_count}")

    return rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict parking violations.")
    parser.add_argument("--train-path", type=str, default="violations_per_street_2022.csv",
                        help="Path to the training data CSV file.")
    parser.add_argument("--test-path", type=str, default="/Users/USER/Downloads/violations_per_street_2023.csv",
                        help="Path to the test data CSV file (optional).")
    parser.add_argument("--corpus-features-path", type=str, default="corpus/corpus_features.parquet",
                        help="Path to the corpus features Parquet file.")
    args = parser.parse_args()

    df_merged, df_corpus_data = load_and_preprocess_data(args.train_path, args.corpus_features_path)
    train_and_predict(df_merged, df_corpus_data, args.test_path)
