import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import FeatureHasher
import numpy as np
import argparse
import pyarrow.parquet as pq

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def load_and_merge_tree_data(violations_df, tree_data_path):
    """Loads tree census data and merges it with the violations data."""
    tree_df = pd.read_parquet(tree_data_path)
    tree_df['Street Name'] = tree_df['address'].str.extract(r'\d+\s+(.*)')
    tree_counts = tree_df.groupby('Street Name').size().reset_index(name='tree_count')
    merged_df = pd.merge(violations_df, tree_counts, on='Street Name', how='left')
    merged_df['tree_count'] = merged_df['tree_count'].fillna(0)
    return merged_df

def load_and_merge_meter_data(violations_df, meter_data_path):
    """Loads parking meter data and merges it with the violations data."""
    meter_df = pd.read_parquet(meter_data_path)
    meter_counts = meter_df.groupby('On_Street').size().reset_index(name='meter_count')
    active_meters = meter_df[meter_df['Status'] == 'Active'].groupby('On_Street').size().reset_index(name='active_meter_count')
    merged_df = pd.merge(violations_df, meter_counts, left_on='Street Name', right_on='On_Street', how='left')
    merged_df = pd.merge(merged_df, active_meters, left_on='Street Name', right_on='On_Street', how='left')
    merged_df['meter_count'] = merged_df['meter_count'].fillna(0)
    merged_df['active_meter_count'] = merged_df['active_meter_count'].fillna(0)
    merged_df = merged_df.drop(columns=['On_Street_x', 'On_Street_y'], errors='ignore')
    return merged_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict parking violations.')
    parser.add_argument('--train-path', type=str, default='/Users/USER/nyc_open_data/train/all_data/violations_per_street_2022.csv', help='Path to the training data file.')
    parser.add_argument('--test-path', type=str, help='Path to the test data file.', default="/Users/USER/Downloads/violations_per_street_2023.csv")
    parser.add_argument('--tree-data-path', type=str, default='train/all_data/corpus/2005_street_tree_census__29bw-z7pj.parquet', help='Path to the tree census data.')
    parser.add_argument('--meter-data-path', type=str, default='train/all_data/corpus/parking_meters_locations_and_status__693u-uax6.parquet', help='Path to the parking meter data.')
    args = parser.parse_args()

    print(f"Loading training data from: {args.train_path}")
    train_df = load_data(args.train_path)
    print("Training data loaded successfully.")

    train_df = load_and_merge_tree_data(train_df, args.tree_data_path)
    train_df = load_and_merge_meter_data(train_df, args.meter_data_path)

    h = FeatureHasher(n_features=100, input_type='string')
    hashed_features = h.fit_transform(train_df['Street Name'].apply(lambda x: [x]))
    hashed_features_df = pd.DataFrame(hashed_features.toarray())

    violation_dummies = pd.get_dummies(train_df['Violation Description'], prefix='violation')

    X = pd.concat([hashed_features_df, violation_dummies, train_df[['tree_count', 'meter_count', 'active_meter_count']]], axis=1)
    X.columns = X.columns.astype(str)
    y = train_df['violation_count']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"Validation RMSE with combined features: {rmse}")

    if args.test_path:
        print(f"Loading test data from: {args.test_path}")
        test_df = load_data(args.test_path)
        print("Test data loaded successfully.")

        test_df = load_and_merge_tree_data(test_df, args.tree_data_path)
        test_df = load_and_merge_meter_data(test_df, args.meter_data_path)

        hashed_features_test = h.transform(test_df['Street Name'].apply(lambda x: [x]))
        hashed_features_test_df = pd.DataFrame(hashed_features_test.toarray())
        
        violation_dummies_test = pd.get_dummies(test_df['Violation Description'], prefix='violation')
        
        X_test = pd.concat([hashed_features_test_df, violation_dummies_test, test_df[['tree_count', 'meter_count', 'active_meter_count']]], axis=1)
        X_test.columns = X_test.columns.astype(str)

        # Align columns
        train_cols = X.columns
        missing_cols = set(train_cols) - set(X_test.columns)
        for c in missing_cols:
            X_test[c] = 0
        X_test = X_test[train_cols]
        
        predictions = model.predict(X_test)
        
        submission_df = pd.DataFrame({
            'street_name': test_df['Street Name'],
            'violation_type': test_df['Violation Description'],
            'predicted_count': predictions
        })
        submission_df.to_csv('submission.csv', index=False)
        print("submission.csv created successfully.")

        if 'violation_count' in test_df.columns:
            test_rmse = np.sqrt(mean_squared_error(test_df['violation_count'], predictions))
            print(f"Test RMSE with combined features: {test_rmse}")
