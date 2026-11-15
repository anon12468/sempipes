
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import argparse
import re
import numpy as np

def clean_street_name(street_name):
    """Cleans street names for better matching."""
    if isinstance(street_name, str):
        street_name = street_name.upper()
        street_name = re.sub(r'\\b(AVENUE|AVE)\\b', 'AVE', street_name)
        street_name = re.sub(r'\\b(STREET|ST)\\b', 'ST', street_name)
        street_name = re.sub(r'\\b(ROAD|RD)\\b', 'RD', street_name)
        street_name = re.sub(r'\\b(BOULEVARD|BLVD)\\b', 'BLVD', street_name)
        street_name = re.sub(r'\\b(PLACE|PL)\\b', 'PL', street_name)
        street_name = re.sub(r'\\b(COURT|CT)\\b', 'CT', street_name)
        street_name = re.sub(r'[^A-Z0-9\\s]', '', street_name) # Remove special characters
        return street_name.strip()
    return street_name

def main():
    parser = argparse.ArgumentParser(description="Predict parking violations.")
    parser.add_argument("--train-path", type=str, default="train/all_data/violations_per_street_2022.csv",
                        help="Path to the training data CSV file.")
    parser.add_argument("--test-path", type=str, default="/Users/USER/Downloads/violations_per_street_2023.csv",
                        help="Path to the test data CSV file (optional).")
    args = parser.parse_args()

    # Load and preprocess training data
    df_train = pd.read_csv(args.train_path)
    df_train.columns = ['street_name', 'violation_type', 'violation_count']
    df_train['street_name_clean'] = df_train['street_name'].apply(clean_street_name)
    df_train['violation_type_clean'] = df_train['violation_type'].apply(lambda x: x.upper().strip() if isinstance(x, str) else x)

    # Load and preprocess parking meter data for augmentation
    df_meters = pd.read_parquet("train/all_data/corpus/parking_meters_locations_and_status__693u-uax6.parquet")
    df_meters['On_Street_clean'] = df_meters['On_Street'].apply(clean_street_name)

    # Aggregate meter data per street
    df_meters_agg = df_meters.groupby('On_Street_clean').agg(
        num_meters=('ObjectID', 'count'),
        num_unique_boroughs=('Borough', 'nunique'),
        avg_latitude=('Latitude', 'mean'),
        avg_longitude=('Longitude', 'mean')
    ).reset_index()
    df_meters_agg = df_meters_agg.rename(columns={'On_Street_clean': 'street_name_clean'})

    # Join training data with meter data
    df_train = pd.merge(df_train, df_meters_agg, on='street_name_clean', how='left')

    # Fill NaN values for new features
    df_train['num_meters'] = df_train['num_meters'].fillna(0)
    df_train['num_unique_boroughs'] = df_train['num_unique_boroughs'].fillna(0)
    df_train['avg_latitude'] = df_train['avg_latitude'].fillna(df_train['avg_latitude'].mean())
    df_train['avg_longitude'] = df_train['avg_longitude'].fillna(df_train['avg_longitude'].mean())

    # Feature engineering for violation type
    df_train = pd.get_dummies(df_train, columns=['violation_type_clean'], prefix='violation')

    # Drop original string columns that are not features. Keep 'street_name_clean' for key tracking.
    df_train = df_train.drop(columns=['street_name', 'violation_type'], errors='ignore')

    # Define features and target
    # Identify one-hot encoded violation columns and other numerical features
    features = [col for col in df_train.columns if col.startswith('violation_') or \
                col in ['num_meters', 'num_unique_boroughs', 'avg_latitude', 'avg_longitude']]
    target = 'violation_count'

    # Ensure that 'violation_count' and 'street_name_clean' are not accidentally in features
    if target in features:
        features.remove(target)
    if 'street_name_clean' in features: # Explicitly remove if it somehow gets in
        features.remove('street_name_clean')

    X = df_train[features]
    y = df_train[target]
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("X_train head:\n", X_train.head())
    print("X_train columns:\n", X_train.columns)

    # Train a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Validation RMSE: {val_rmse}")

    # Handle unseen key pairs
    unseen_train_keys = set(df_train['street_name_clean'].unique())

    if args.test_path:
        df_test = pd.read_csv(args.test_path)
        df_test.columns = ['street_name', 'violation_type'] # Assuming violation_count is optional in test
        df_test['street_name_clean'] = df_test['street_name'].apply(clean_street_name)
        df_test['violation_type_clean'] = df_test['violation_type'].apply(lambda x: x.upper().strip() if isinstance(x, str) else x)

        # Merge with meter data
        df_test = pd.merge(df_test, df_meters_agg, on='street_name_clean', how='left')
        df_test['num_meters'] = df_test['num_meters'].fillna(0)
        df_test['num_unique_boroughs'] = df_test['num_unique_boroughs'].fillna(0)
        df_test['avg_latitude'] = df_test['avg_latitude'].fillna(df_train['avg_latitude'].mean()) # Use training mean
        df_test['avg_longitude'] = df_test['avg_longitude'].fillna(df_train['avg_longitude'].mean()) # Use training mean

        # Align columns with training data (important for one-hot encoding)
        df_test_processed = pd.get_dummies(df_test, columns=['violation_type_clean'], prefix='violation')
        missing_cols = set(X_train.columns) - set(df_test_processed.columns)
        for c in missing_cols:
            df_test_processed[c] = 0
        df_test_processed = df_test_processed[X_train.columns]

        # Predict on test data
        test_predictions = model.predict(df_test_processed[features])
        df_test['predicted_count'] = np.maximum(0, test_predictions.round()).astype(int) # Ensure non-negative and int

        # Report unseen key pairs
        unseen_test_keys = set(df_test['street_name_clean'].unique()) - unseen_train_keys
        print(f"Number of unseen street_name_clean in test data: {len(unseen_test_keys)}")
        print(f"Handling of unseen keys: Predicted with available features; imputed values for meter data.")

        # Generate submission file
        submission = df_test[['street_name', 'violation_type', 'predicted_count']]
        submission.to_csv("submission.csv", index=False)
        print("Generated submission.csv")

        # If violation_count is present in test file, calculate RMSE
        if 'violation_count' in df_test.columns:
            test_rmse = np.sqrt(mean_squared_error(df_test['violation_count'], df_test['predicted_count']))
            print(f"Test RMSE: {test_rmse}")

if __name__ == "__main__":
    main()
