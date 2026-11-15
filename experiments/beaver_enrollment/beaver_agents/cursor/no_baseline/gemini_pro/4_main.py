import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import os
import argparse

# --- Configuration ---
# This will be overridden by command-line arguments, but we keep a default.
TRAIN_DATA_DIR = "table_splits/train"
TARGET_FILE = "gold_enrollment_train.csv"
SUBJECT_OFFERED_FILE = "SUBJECT_OFFERED.csv"
ENROLLMENT_HISTORY_FILE = "ENROLLMENT_HISTORY_FEATURES.csv"
COURSE_CATALOG_FILE = "CIS_COURSE_CATALOG.csv"

# --- 1. Load Data ---
def load_data(data_dir):
    """Loads all the necessary CSVs into pandas DataFrames."""
    print(f"Loading data from {data_dir}...")
    try:
        gold_df = pd.read_csv(os.path.join(data_dir, TARGET_FILE))
        subject_df = pd.read_csv(os.path.join(data_dir, SUBJECT_OFFERED_FILE), on_bad_lines='skip', low_memory=False)
        enrollment_df = pd.read_csv(os.path.join(data_dir, ENROLLMENT_HISTORY_FILE))
        catalog_df = pd.read_csv(os.path.join(data_dir, COURSE_CATALOG_FILE), on_bad_lines='skip', low_memory=False)
        print("Data loaded successfully.")
        return gold_df, subject_df, enrollment_df, catalog_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure the data is in the specified directory.")
        return None, None, None, None

# --- 2. Merge Data & Preprocess ---
def merge_and_preprocess(gold_df, subject_df, enrollment_df, catalog_df):
    """Merges the dataframes and performs initial preprocessing."""
    print("Merging and preprocessing data...")

    # Merge gold with enrollment features
    merged_df = pd.merge(gold_df, enrollment_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Merge with subject offered data
    merged_df = pd.merge(merged_df, subject_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Create a subject identifier from catalog to merge
    catalog_df['SUBJECT_ID_SORT_proxy'] = catalog_df['SUBJECT_CODE'].astype(str) + '.' + catalog_df['SUBJECT_NUMBER'].astype(str)
    
    # For the catalog, we'll just take the most recent entry for each subject
    catalog_df = catalog_df.sort_values('ACADEMIC_YEAR', ascending=False).drop_duplicates('SUBJECT_ID_SORT_proxy')
    
    # Merge with catalog data
    # A significant number of SUBJECT_ID_SORT values don't have a direct match in the catalog's proxy.
    # We will perform a left merge and handle the NaNs.
    merged_df = pd.merge(merged_df, catalog_df, left_on='MASTER_SUBJECT_ID_SORT', right_on='SUBJECT_ID_SORT_proxy', how='left')

    # Convert target to binary
    merged_df['HIGH_ENROLLMENT'] = merged_df['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    
    # Extract year from term code for time-based split
    merged_df['YEAR'] = merged_df['TERM_CODE'].str.extract('(\\d{4})').astype(int)

    print("Data merged and preprocessed.")
    return merged_df

# --- 3. Feature Engineering ---
def create_features(df, top_categories_map=None):
    """Creates features for the model.
    
    If top_categories_map is None (training mode), it learns the top categories and returns a tuple (features, top_categories_map).
    Otherwise (prediction mode), it uses the provided map and returns only the features DataFrame.
    """
    is_training = top_categories_map is None
    if is_training:
        top_categories_map = {}

    print("Creating features...")
    
    # Use a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Numerical features
    numerical_features = df_copy[['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean', 
                             'LECTURE_UNITS', 'LAB_UNITS', 'PREPARATION_UNITS', 'TOTAL_UNITS']].fillna(0)
    
    # Categorical features
    categorical_cols = [
        'IS_LECTURE_SECTION', 'IS_LAB_SECTION', 'OFFER_SCHOOL_NAME', 'FORM_TYPE_DESC',
        'GRADE_TYPE_DESC', 'GIR_ATTRIBUTE_DESC', 'COMM_REQ_ATTRIBUTE_DESC', 
        'IS_OFFERED_FALL_TERM', 'IS_OFFERED_SPRING_TERM'
    ]
    
    high_cardinality_cols = ['RESPONSIBLE_FACULTY_NAME', 'OFFER_DEPT_NAME']
    
    if is_training:
        # Learn top categories
        for col in high_cardinality_cols:
            if col in df_copy.columns:
                top_categories_map[col] = df_copy[col].value_counts().nlargest(20).index
    
    # Apply top categories
    for col in high_cardinality_cols:
        if col in df_copy.columns and col in top_categories_map:
            df_copy.loc[:, col] = df_copy[col].where(df_copy[col].isin(top_categories_map[col]), 'Other')

    categorical_cols.extend(high_cardinality_cols)

    features = numerical_features
    for col in categorical_cols:
        if col in df_copy.columns:
            dummies = pd.get_dummies(df_copy[col], prefix=col, dummy_na=True)
            features = pd.concat([features, dummies], axis=1)
            
    # Boolean features
    for col in ['IS_VARIABLE_UNITS', 'IS_OFFERED_THIS_YEAR']:
        if col in df_copy.columns:
            features[col] = df_copy[col].apply(lambda x: 1 if x == 'Y' else 0)

    print("Features created.")
    
    if is_training:
        return features, top_categories_map
    else:
        return features

# --- 4. Train Model ---
def train_model(X_train, y_train):
    """Trains a RandomForestClassifier."""
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model trained.")
    return model

# --- 5. Predict and Evaluate ---
def predict_and_evaluate(model, X_test, y_test):
    """Makes predictions and evaluates the model."""
    print("Predicting and evaluating...")
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f"Macro F1 Score: {f1:.4f}")
    return f1

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict course enrollment.")
    parser.add_argument('--data_dir', type=str, default=TRAIN_DATA_DIR, help='Directory containing the training data.')
    parser.add_argument('--test_csv', type=str, help='Path to the test CSV file with TERM_CODE and SUBJECT_ID_SORT.', default='/Users/USER/BEAVER/test/gold_enrollment_test.csv')
    args = parser.parse_args()

    gold_df, subject_df, enrollment_df, catalog_df = load_data(args.data_dir)
    
    if gold_df is not None:
        merged_df = merge_and_preprocess(gold_df, subject_df, enrollment_df, catalog_df)
        
        # --- Time-based Split for Training/Validation ---
        train_df = merged_df[merged_df['YEAR'] < 2016].copy()
        
        # Use all available data for training the final model
        print(f"Training final model on {len(train_df)} samples.")
        X_train, top_categories = create_features(train_df)
        y_train = train_df['HIGH_ENROLLMENT']
        
        model = train_model(X_train, y_train)
        print("Final model trained.")

        # --- Prediction ---
        if args.test_csv:
            print(f"\\n--- Predicting on {args.test_csv} ---")
            try:
                test_keys_df = pd.read_csv(args.test_csv)
                
                if 'HIGH_ENROLLMENT' in test_keys_df.columns:
                    print("Found 'HIGH_ENROLLMENT' column in test file. Will calculate F1 score.")
                    test_keys_df['original_index'] = range(len(test_keys_df))
                
                # Ensure the test keys are merged with the full feature set
                test_df = pd.merge(test_keys_df, merged_df.drop(columns=['HIGH_ENROLLMENT'], errors='ignore'), on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
                
                y_test_true = None
                if 'HIGH_ENROLLMENT' in test_df.columns:
                    # Filter y_test_true to match the rows that were successfully merged
                    valid_indices = test_df['original_index'].dropna().astype(int)
                    y_test_true = test_keys_df.loc[valid_indices, 'HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

                if not test_df.empty:
                    # Drop rows that couldn't be merged
                    test_df = test_df.dropna(subset=['YEAR']).copy() # YEAR is a proxy for a successful merge
                    
                    # Create features for the test set
                    X_test = create_features(test_df, top_categories_map=top_categories)

                    # Align columns with the training set
                    missing_in_test = set(X_train.columns) - set(X_test.columns)
                    for c in missing_in_test:
                        X_test[c] = 0
                    X_test = X_test.reindex(columns=X_train.columns).fillna(0)

                    # Make predictions
                    predictions = model.predict(X_test)
                    test_df['HIGH_ENROLLMENT_PRED'] = ['Y' if x == 1 else 'N' for x in predictions]

                    # Save predictions
                    predictions_df = test_df[['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT_PRED']]
                    predictions_df.to_csv("predictions.csv", index=False)
                    print("Predictions saved to predictions.csv")

                    # Evaluate if true labels were available
                    if y_test_true is not None:
                        # Ensure y_test_true and predictions have the same length
                        if len(y_test_true) == len(predictions):
                            f1 = f1_score(y_test_true, predictions, average='macro')
                            print(f"\\n--- Test Set Evaluation ---")
                            print(f"Macro F1 Score on test set: {f1:.4f}")
                        else:
                            print(f"Could not calculate F1 score due to length mismatch: y_true ({len(y_test_true)}) vs y_pred ({len(predictions)})")

                else:
                    print("No matching data found for the provided test keys.")
            except FileNotFoundError:
                print(f"Error: Test file not found at {args.test_csv}")
        else:
            print("\\nNo test file provided. To make predictions, run with --test_csv path/to/your/test.csv")

