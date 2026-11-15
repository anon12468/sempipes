import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(file_path):
    """Loads data from a CSV file."""
    df = pd.read_csv(file_path)
    # Rename columns for easier access
    df.columns = ['street_name', 'violation_type', 'violation_count']
    return df

def train_model(train_df):
    """
    Trains a simple baseline model by storing the mean violation count for each (street_name, violation_type) pair.
    """
    model = train_df.groupby(['street_name', 'violation_type'])['violation_count'].mean().reset_index()
    model.rename(columns={'violation_count': 'predicted_count'}, inplace=True)
    return model

def predict(model, df_to_predict, overall_mean_train):
    """
    Generates predictions based on the trained model.
    For unseen pairs, it uses the overall mean from the training data.
    """
    # Normalize street_name and violation_type for consistent merging
    df_to_predict['street_name'] = df_to_predict['street_name'].str.upper()
    df_to_predict['violation_type'] = df_to_predict['violation_type'].str.upper()

    # Merge test data with the trained model to get predictions
    predictions = pd.merge(df_to_predict, model, on=['street_name', 'violation_type'], how='left')

    # Handle unseen street_name, violation_type pairs
    predictions['predicted_count'] = predictions['predicted_count'].fillna(overall_mean_train)
    
    # Ensure predicted_count is non-negative
    predictions['predicted_count'] = predictions['predicted_count'].apply(lambda x: max(0, x))

    return predictions

def main():
    parser = argparse.ArgumentParser(description="Predict NYC parking violations.")
    parser.add_argument('--train-path', type=str, default='train/all_data/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default="/Users/USER/Downloads/violations_per_street_2023.csv",
                        help='Path to the test data CSV file (optional).')
    
    args = parser.parse_args()

    print(f"Loading training data from: {args.train_path}")
    train_data = load_data(args.train_path)
    train_data['street_name'] = train_data['street_name'].str.upper() # Normalize street names
    train_data['violation_type'] = train_data['violation_type'].str.upper() # Normalize violation types

    # Split training data for validation (80/20 split)
    train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=42)
    
    # Calculate overall mean of violation_count from the training set for unseen predictions
    overall_mean_train = train_set['violation_count'].mean()

    print("Training model...")
    model = train_model(train_set)

    # Evaluate on validation set
    val_predictions = predict(model, val_set.copy(), overall_mean_train)
    # Check if 'violation_count' column exists in val_predictions before calculating RMSE
    if 'violation_count' in val_predictions.columns:
        rmse_val = np.sqrt(mean_squared_error(val_predictions['violation_count'], val_predictions['predicted_count']))
        print(f"Validation RMSE: {rmse_val:.2f}")
    else:
        print("Validation set does not contain 'violation_count' for RMSE calculation.")

    # Identify and report unseen key pairs in validation set
    train_keys = train_set.set_index(['street_name', 'violation_type']).index
    val_keys = val_set.set_index(['street_name', 'violation_type']).index
    unseen_val_pairs = val_set[~val_keys.isin(train_keys)]
    print(f"Number of unseen key pairs in validation set: {len(unseen_val_pairs)}")

    if args.test_path:
        print(f"Loading test data from: {args.test_path}")
        test_data = load_data(args.test_path)
        
        print("Generating predictions for test data...")
        test_predictions = predict(model, test_data.copy(), overall_mean_train)

        # Generate submission.csv
        submission_df = test_predictions[['street_name', 'violation_type', 'predicted_count']]
        submission_df.to_csv('submission.csv', index=False)
        print("Submission file 'submission.csv' generated.")

        # If test data has 'violation_count', calculate RMSE
        if 'violation_count' in test_predictions.columns:
            rmse_test = np.sqrt(mean_squared_error(test_predictions['violation_count'], test_predictions['predicted_count']))
            print(f"Test RMSE: {rmse_test:.2f}")
        else:
            print("Test set does not contain 'violation_count' for RMSE calculation.")
        
        # Identify and report unseen key pairs in test set
        test_keys = test_data.set_index(['street_name', 'violation_type']).index
        unseen_test_pairs = test_data[~test_keys.isin(train_keys)]
        print(f"Number of unseen key pairs in test set: {len(unseen_test_pairs)}")

if __name__ == "__main__":
    main()