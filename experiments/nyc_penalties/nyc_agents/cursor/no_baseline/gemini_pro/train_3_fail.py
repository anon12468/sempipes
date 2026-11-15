import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def load_and_preprocess_meters(path):
    """Loads and preprocesses the parking meter data."""
    df = pd.read_parquet(path)
    # Group by street name and count meters
    meter_counts = df.groupby('On_Street').size().reset_index(name='meter_count')
    # Rename for consistency
    meter_counts = meter_counts.rename(columns={'On_Street': 'Street Name'})
    return meter_counts

def main(train_path, test_path=None):
    """
    Main function to run the violation prediction pipeline.
    """
    # Load the training data
    train_df = pd.read_csv(train_path)
    print("Training data loaded successfully.")
    print(f"Shape of training data: {train_df.shape}")

    # Load and merge parking meter data
    print("Loading parking meter data for feature engineering...")
    meter_counts_df = load_and_preprocess_meters('train/all_data/corpus/parking_meters_locations_and_status__693u-uax6.parquet')
    train_df = pd.merge(train_df, meter_counts_df, on='Street Name', how='left')
    # Fill streets with no meters with 0
    train_df['meter_count'].fillna(0, inplace=True)
    print("Parking meter data merged.")

    # Convert categorical variables to numerical using one-hot encoding
    train_df = pd.get_dummies(train_df, columns=['Violation Description'], drop_first=True)

    # Prepare data for modeling
    features = [col for col in train_df.columns if col not in ['Street Name', 'violation_count']]
    X = train_df[features]
    y = train_df['violation_count']

    if test_path:
        print(f"\nLoading test data from {test_path}...")
        test_df = pd.read_csv(test_path)
        print(f"Shape of test data: {test_df.shape}")

        # Preprocess test data
        test_df = pd.merge(test_df, meter_counts_df, on='Street Name', how='left')
        test_df['meter_count'].fillna(0, inplace=True)
        test_df = pd.get_dummies(test_df, columns=['Violation Description'], drop_first=True)
        
        # Align columns between train and test
        train_cols = set(X.columns)
        test_cols = set(test_df.columns)
        
        missing_in_test = list(train_cols - test_cols)
        for col in missing_in_test:
            test_df[col] = 0
            
        missing_in_train = list(test_cols - train_cols)
        for col in missing_in_train:
            X[col] = 0 # This should be handled more carefully in a real scenario

        test_df = test_df[X.columns] # Ensure order is the same

        # Train model on full training data
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions
        predictions = model.predict(test_df)
        predictions[predictions < 0] = 0 # Cannot have negative violations
        
        submission_df = pd.read_csv(test_path)[['Street Name', 'Violation Description']]
        submission_df['predicted_count'] = predictions.round()
        submission_df.to_csv('submission.csv', index=False)
        print("\nsubmission.csv generated successfully.")
        
        if 'violation_count' in pd.read_csv(test_path).columns:
            true_values = pd.read_csv(test_path)['violation_count']
            rmse = np.sqrt(mean_squared_error(true_values, predictions))
            print(f"RMSE on test data: {rmse}")

    else:
        print("\nNo test path provided. Performing validation on 2022 data.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        val_preds = model.predict(X_val)
        val_preds[val_preds < 0] = 0 # Clip at 0
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Validation RMSE with Linear Regression and meter data: {rmse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parking Violation Prediction')
    parser.add_argument('--train-path', type=str, default='train/all_data/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, help='Optional path to the test data CSV file.', default="/Users/USER/Downloads/violations_per_street_2023.csv")
    args = parser.parse_args()

    main(args.train_path, args.test_path)
