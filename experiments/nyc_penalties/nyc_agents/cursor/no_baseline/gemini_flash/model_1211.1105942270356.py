import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import re
import lightgbm as lgb

def load_data(path):
    """
    Loads the CSV data from the given path.
    """
    df = pd.read_csv(path)
    return df

def extract_street_name(address):
    if isinstance(address, str):
        # Regex to capture street name from an address like "1139 57 STREET"
        # It tries to capture words after the number and before any apartment/unit numbers.
        match = re.search(r'\d+\s+([A-Z0-9\s]+?)(?:(?:AVE|STREET|ROAD|BLVD|DR|PL|CT|LN|RD|PKWY|TER|WAY)\b|$)', address.upper())
        if match:
            street = match.group(1).strip()
            # Remove common address suffixes that might not be part of the street name itself
            street = re.sub(r'\s+(NORTH|SOUTH|EAST|WEST|NW|NE|SW|SE|APT|UNIT|BLDG|FL|RM|STE)\b.*$', '', street)
            return street.strip()
        # Fallback for addresses without a leading number
        match = re.search(r'^([A-Z0-9\s]+?)(?:(?:AVE|STREET|ROAD|BLVD|DR|PL|CT|LN|RD|PKWY|TER|WAY)\b|$)', address.upper())
        if match:
            return match.group(1).strip()
    return "UNKNOWN_STREET"

def load_and_preprocess_tree_data(tree_data_path):
    """
    Loads and preprocesses the tree census data to create aggregated features per street.
    """
    tree_df = pd.read_parquet(tree_data_path)
    
    # Extract street name from address
    tree_df['extracted_street_name'] = tree_df['address'].apply(extract_street_name)

    # Convert condition to numeric (e.g., Good=3, Fair=2, Poor=1)
    condition_mapping = {'Good': 3, 'Fair': 2, 'Poor': 1}
    tree_df['condition_numeric'] = tree_df['status'].map(condition_mapping).fillna(0) # Fill NaN with 0 for unknown status

    # Aggregate features per extracted street name
    aggregated_tree_features_by_street = tree_df.groupby('extracted_street_name').agg(
        total_trees=('OBJECTID', 'count'),
        avg_tree_dbh=('tree_dbh', 'mean'),
        avg_tree_condition=('condition_numeric', 'mean'),
        sidewalk_crack_ratio=('sidw_crack', lambda x: (x == 'Yes').mean())
    ).reset_index()
    
    return aggregated_tree_features_by_street

def main():
    parser = argparse.ArgumentParser(description="Predict parking violations.")
    parser.add_argument("--train-path", type=str, default="train/all_data/violations_per_street_2022.csv",
                        help="Path to the training data CSV file.")
    parser.add_argument("--test-path", type=str, default="/Users/USER/Downloads/violations_per_street_2023.csv",
                        help="Path to the test data CSV file (optional).")
    parser.add_argument("--tree-data-path", type=str, default="train/all_data/corpus/2005_street_tree_census__29bw-z7pj.parquet",
                        help="Path to the tree census data parquet file.")
    
    args = parser.parse_args()

    # Load training data
    train_df = load_data(args.train_path)

    # Load and preprocess tree data for augmentation
    aggregated_tree_features_by_street = load_and_preprocess_tree_data(args.tree_data_path)

    # Standardize 'Street Name' in train_df for merging
    train_df['Street_Name_cleaned'] = train_df['Street Name'].apply(lambda x: x.upper().replace('.', '').strip())
    aggregated_tree_features_by_street['extracted_street_name'] = aggregated_tree_features_by_street['extracted_street_name'].apply(lambda x: x.upper().replace('.', '').strip())

    # Merge tree features into training data
    train_df = pd.merge(
        train_df,
        aggregated_tree_features_by_street,
        left_on='Street_Name_cleaned',
        right_on='extracted_street_name',
        how='left'
    )
    
    # Drop the temporary merging columns and fill NaN for new features
    train_df = train_df.drop(columns=['Street_Name_cleaned', 'extracted_street_name'], errors='ignore')
    
    tree_feature_names = [col for col in aggregated_tree_features_by_street.columns if col not in ['extracted_street_name']]
    # Fill NaN for tree features with 0 or mean/median - for now, fill with 0
    for col in tree_feature_names:
        train_df[col] = train_df[col].fillna(0)


    # Prepare features and target
    categorical_features = ['Street Name', 'Violation Description']
    numerical_features = tree_feature_names

    X = train_df[categorical_features + numerical_features]
    y = train_df['violation_count']

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing steps for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])

    # --- LightGBM Model ---
    lgbm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=42))
    ])
    
    lgbm_pipeline.fit(X_train, y_train)
    lgbm_val_predictions = lgbm_pipeline.predict(X_val)
    lgbm_val_predictions[lgbm_val_predictions < 0] = 0
    lgbm_rmse = np.sqrt(mean_squared_error(y_val, lgbm_val_predictions))
    print(f"LightGBM Validation RMSE: {lgbm_rmse}")

    # --- RandomForestRegressor Model ---
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) # n_jobs=-1 for parallel processing
    ])

    rf_pipeline.fit(X_train, y_train)
    rf_val_predictions = rf_pipeline.predict(X_val)
    rf_val_predictions[rf_val_predictions < 0] = 0
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_val_predictions))
    print(f"RandomForestRegressor Validation RMSE: {rf_rmse}")

    # Decide which model to use for test set predictions based on validation RMSE
    if lgbm_rmse <= rf_rmse:
        best_model_pipeline = lgbm_pipeline
        print("LightGBM selected for test predictions.")
    else:
        best_model_pipeline = rf_pipeline
        print("RandomForestRegressor selected for test predictions.")

    if args.test_path:
        test_df = load_data(args.test_path)
        
        # Add tree features to test_df
        test_df['Street_Name_cleaned'] = test_df['Street Name'].apply(lambda x: x.upper().replace('.', '').strip())
        test_df = pd.merge(
            test_df,
            aggregated_tree_features_by_street,
            left_on='Street_Name_cleaned',
            right_on='extracted_street_name',
            how='left'
        )
        test_df = test_df.drop(columns=['Street_Name_cleaned', 'extracted_street_name'], errors='ignore')
        for col in tree_feature_names:
            test_df[col] = test_df[col].fillna(0)

        # Identify unseen key pairs
        train_keys = X[categorical_features].set_index(categorical_features).index
        test_keys = test_df[categorical_features].set_index(categorical_features).index
        unseen_keys_count = len(test_keys.difference(train_keys))
        print(f"Number of unseen (street_name, violation_type) pairs in test set: {unseen_keys_count}")

        # Prepare test data for prediction
        X_test = test_df[categorical_features + numerical_features]

        # Generate predictions for test data using the best model
        test_predictions = best_model_pipeline.predict(X_test)
        test_predictions[test_predictions < 0] = 0 # Ensure non-negative predictions
        
        # Create submission file
        submission_df = pd.DataFrame({
            'street_name': test_df['Street Name'],
            'violation_type': test_df['Violation Description'],
            'predicted_count': test_predictions
        })
        submission_df.to_csv("submission.csv", index=False)
        print("submission.csv generated.")

        # If violation_count is present in test_df, report RMSE
        if 'violation_count' in test_df.columns:
            test_rmse = np.sqrt(mean_squared_error(test_df['violation_count'], test_predictions))
            print(f"Test RMSE: {test_rmse}")
        
    else:
        print("No test-path provided. Skipping test prediction and submission.csv generation.")


if __name__ == "__main__":
    main()
