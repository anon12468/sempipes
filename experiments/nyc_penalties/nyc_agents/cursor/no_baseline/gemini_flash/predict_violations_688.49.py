
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pyarrow.parquet as pq
import lightgbm as lgb

def clean_street_name(df, column_name):
    """Cleans and standardizes street names for merging."""
    df[column_name] = df[column_name].str.upper().str.strip().replace(r'[^A-Z0-9 ]', '', regex=True).fillna('UNKNOWN_STREET')
    return df

def main():
    parser = argparse.ArgumentParser(description="Predict parking violations.")
    parser.add_argument("--train-path", type=str, default="violations_per_street_2022.csv",
                        help="Path to the training data CSV file (default: violations_per_street_2022.csv)")
    parser.add_argument("--test-path", type=str, default="/Users/USER/Downloads/violations_per_street_2023.csv",
                        help="Path to the test data CSV file (optional). If provided, predictions will be generated.")
    
    args = parser.parse_args()

    corpus_dir = os.path.join(os.path.dirname(__file__), 'corpus')

    # --- Load and preprocess parking meter data ---
    parking_meters_path = os.path.join(corpus_dir, 'parking_meters_locations_and_status__693u-uax6.parquet')
    try:
        parking_meters_df = pd.read_parquet(parking_meters_path)
        parking_meters_df = clean_street_name(parking_meters_df, 'On_Street')
        
        # Aggregate parking meter features by street
        street_parking_features = parking_meters_df.groupby('On_Street').agg(
            num_meters=('Meter Number', 'count'),
            has_meters=('Meter Number', lambda x: 1) # Binary feature: does street have meters?
        ).reset_index()
        street_parking_features.rename(columns={'On_Street': 'street_name_cleaned'}, inplace=True)

    except FileNotFoundError:
        print(f"Warning: Parking meters file not found at {parking_meters_path}. Continuing without this feature.")
        street_parking_features = pd.DataFrame({'street_name_cleaned': [], 'num_meters': [], 'has_meters': []})
    except Exception as e:
        print(f"Error loading or processing parking meters data: {e}. Continuing without this feature.")
        street_parking_features = pd.DataFrame({'street_name_cleaned': [], 'num_meters': [], 'has_meters': []})


    # --- Load and preprocess training data ---
    try:
        train_df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), args.train_path))
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}")
        return

    train_df_full = clean_street_name(train_df_full, 'Street Name')
    train_df_full['violation_type'] = train_df_full['Violation Description'].fillna('UNKNOWN_VIOLATION')
    train_df_full.rename(columns={'Street Name': 'street_name_cleaned'}, inplace=True)

    # Merge with parking meter features
    train_df_full = pd.merge(train_df_full, street_parking_features, on='street_name_cleaned', how='left')
    train_df_full['num_meters'] = train_df_full['num_meters'].fillna(0)
    train_df_full['has_meters'] = train_df_full['has_meters'].fillna(0)


    # Create a combined key for prediction
    train_df_full['key'] = train_df_full['street_name_cleaned'] + '_' + train_df_full['violation_type']

    # --- Feature Engineering and Model Training with LightGBM ---
    # Define features and target
    categorical_features = ['street_name_cleaned', 'violation_type']
    numerical_features = ['num_meters', 'has_meters']

    X = train_df_full[categorical_features + numerical_features]
    y = train_df_full['violation_count']

    # Convert categorical features to 'category' dtype for LightGBM
    for col in categorical_features:
        X[col] = X[col].astype('category')

    # Store known keys from the training split for unseen key reporting
    # Split 2022 data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_keys = (X_train['street_name_cleaned'].astype(str) + '_' + X_train['violation_type'].astype(str)).unique()

    # Train a LGBMRegressor
    model = lgb.LGBMRegressor(objective='regression_l1', 
                              n_estimators=500,
                              learning_rate=0.05,
                              num_leaves=31,
                              n_jobs=-1,
                              random_state=42)
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)], 
              categorical_feature=[col for col in categorical_features if col in X_train.columns])

    # Predict on validation set
    val_predictions = model.predict(X_val)
    
    # Ensure predictions are non-negative
    val_predictions[val_predictions < 0] = 0

    # Calculate RMSE on validation set
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Validation RMSE with LGBMRegressor: {rmse:.2f}")

    # Report unseen key pairs in validation set
    val_combined_keys = (X_val['street_name_cleaned'].astype(str) + '_' + X_val['violation_type'].astype(str))
    unseen_val_keys = val_combined_keys[~val_combined_keys.isin(train_keys)]
    print(f"Number of unseen (street_name, violation_type) pairs in validation set: {len(unseen_val_keys)}")
    print("Handling of unseen keys in validation: LGBMRegressor handles them as new categories, effectively using a default split for them.")

    # If test-path is provided, generate predictions for it
    if args.test_path:
        try:
            test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), args.test_path))
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}")
            return
        
        test_df = clean_street_name(test_df, 'Street Name')
        test_df['violation_type'] = test_df['Violation Description'].fillna('UNKNOWN_VIOLATION')
        test_df.rename(columns={'Street Name': 'street_name_cleaned'}, inplace=True)
        test_df['key'] = test_df['street_name_cleaned'] + '_' + test_df['violation_type']

        # Merge with parking meter features for test data
        test_df = pd.merge(test_df, street_parking_features, on='street_name_cleaned', how='left')
        test_df['num_meters'] = test_df['num_meters'].fillna(0)
        test_df['has_meters'] = test_df['has_meters'].fillna(0)

        X_test = test_df[categorical_features + numerical_features]

        # Convert categorical features to 'category' dtype for LightGBM in test set
        for col in categorical_features:
            X_test[col] = X_test[col].astype('category')
            # Ensure test categories are consistent with training categories
            X_test[col] = X_test[col].cat.set_categories(X[col].cat.categories)

        test_predictions = model.predict(X_test)

        # Ensure predicted_count is non-negative and integer-like
        test_predictions[test_predictions < 0] = 0
        test_predictions = np.round(test_predictions).astype(int)

        # Generate submission file
        submission_df = pd.DataFrame({
            'street_name': test_df['street_name_cleaned'],
            'violation_type': test_df['violation_type'],
            'predicted_count': test_predictions
        })
        submission_df.to_csv("submission.csv", index=False)
        print("Submission file 'submission.csv' generated.")

        # Report unseen key pairs in test set
        test_combined_keys = (test_df['street_name_cleaned'].astype(str) + '_' + test_df['violation_type'].astype(str))
        unseen_test_keys = test_combined_keys[~test_combined_keys.isin(train_keys)]
        print(f"Number of unseen (street_name, violation_type) pairs in test set: {len(unseen_test_keys)}")
        print("Handling of unseen keys in test set: LGBMRegressor handles them as new categories, effectively using a default split for them.")

        # If violation_count is present in test_df, report RMSE
        if 'violation_count' in test_df.columns:
            # Ensure ground truth is not used as feature, only for scoring
            test_rmse = np.sqrt(mean_squared_error(test_df['violation_count'], test_predictions))
            print(f"Test RMSE: {test_rmse:.2f}")

if __name__ == "__main__":
    main()
