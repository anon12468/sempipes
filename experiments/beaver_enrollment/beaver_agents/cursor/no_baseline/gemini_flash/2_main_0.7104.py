import pandas as pd
import argparse
from data_loader import load_data
from feature_engineering import create_features
from model import train_model, predict
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split

def run_prediction_and_evaluation(data_dir, gold_file_path, model=None, X_train_cols=None, is_test=False, y_train_split=None):
    print(f"\n--- {'Running on Test Data' if is_test else 'Running on Training/Validation Data'} ---")
    print("Loading data...")
    raw_df, gold_df_full = load_data(data_dir, gold_file_path)

    if raw_df.empty or gold_df_full.empty:
        print(f"Error: {'Test' if is_test else 'Training'} data or gold data is empty. Exiting.")
        return None, None, None # Return None to indicate failure or no results

    # Merge gold labels (if not test, or if test has gold for evaluation)
    merged_df = pd.merge(raw_df, gold_df_full, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
    print("Merged gold enrollment data.")

    print("Creating features...")
    features_df = create_features(merged_df)

    X = features_df.drop(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'], errors='ignore')
    y = features_df['HIGH_ENROLLMENT']

    X = pd.get_dummies(X, drop_first=True)

    if is_test:
        if model is None or X_train_cols is None:
            print("Error: Model or training columns not provided for test data prediction.")
            return None, None, None
        
        # Align test features with training features
        missing_cols = set(X_train_cols) - set(X.columns)
        for c in missing_cols:
            X[c] = 0
        X = X[X_train_cols] # Ensure same column order

        print("Making predictions...")
        predictions_array = predict(model, X)

        keys_df = features_df[['TERM_CODE', 'SUBJECT_ID_SORT']].copy()
        predictions_df = keys_df.copy()
        predictions_df['HIGH_ENROLLMENT'] = predictions_array

        print("Evaluating model...")
        evaluate_model(predictions_df, gold_df_full)
        return predictions_df, gold_df_full, X.columns # Return predictions and gold for potential further use

    else: # Training/Validation phase
        print("Performing time-based validation split...")
        if 'TERM_YEAR' not in features_df.columns:
            print("Error: 'TERM_YEAR' not found in features_df. Cannot perform time-based split.")
            return None, None, None

        latest_year = features_df['TERM_YEAR'].max()
        X_train_split = X[features_df['TERM_YEAR'] < latest_year]
        y_train_split = y[features_df['TERM_YEAR'] < latest_year]
        X_val_split = X[features_df['TERM_YEAR'] == latest_year]
        y_val_split = y[features_df['TERM_YEAR'] == latest_year]

        print(f"Training data shape: {X_train_split.shape}")
        print(f"Validation data shape: {X_val_split.shape}")

        neg_count = y_train_split.value_counts()[0] 
        pos_count = y_train_split.value_counts()[1] 
        scale_pos_weight = neg_count / pos_count
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

        print("Training model...")
        trained_model = train_model(X_train_split, y_train_split, scale_pos_weight)

        print("Making predictions on validation set...")
        predictions_array_val = predict(trained_model, X_val_split)

        val_keys = features_df[features_df['TERM_YEAR'] == latest_year][['TERM_CODE', 'SUBJECT_ID_SORT']].copy()
        predictions_df_val = val_keys.copy()
        predictions_df_val['HIGH_ENROLLMENT'] = predictions_array_val

        print("Evaluating model on validation set...")
        evaluate_model(predictions_df_val, gold_df_full)

        return trained_model, X_train_split.columns, y_train_split # Return model and training columns for test phase

def main():
    parser = argparse.ArgumentParser(description="Predict high enrollment for course offerings.")
    parser.add_argument('--train_data_dir', type=str, default="table_splits/train",
                        help="Path to the training data directory.")
    parser.add_argument('--gold_train_file_path', type=str, default="table_splits/train/gold_enrollment_train.csv",
                        help="Path to the gold training enrollment file.")
    parser.add_argument('--test_data_dir', type=str, default="/Users/USER/BEAVER/test/",
                        help="Path to the held-out test data directory (optional).")
    parser.add_argument('--gold_test_file_path', type=str, default="/Users/USER/BEAVER/test/gold_enrollment_test.csv",
                        help="Path to the gold test enrollment file (optional).")
    
    args = parser.parse_args()

    # Run training and validation phase
    trained_model, X_train_cols, y_train_split = run_prediction_and_evaluation(
        args.train_data_dir, args.gold_train_file_path, is_test=False
    )

    if trained_model is None or X_train_cols is None:
        print("Training and validation phase failed. Exiting.")
        return

    # Run prediction and evaluation on test data if paths are provided
    if args.test_data_dir and args.gold_test_file_path:
        run_prediction_and_evaluation(
            args.test_data_dir, args.gold_test_file_path, model=trained_model, X_train_cols=X_train_cols, is_test=True, y_train_split=y_train_split
        )
    else:
        print("No test data paths provided. Skipping test data prediction and evaluation.")

    print("Process complete.")

if __name__ == "__main__":
    main()
