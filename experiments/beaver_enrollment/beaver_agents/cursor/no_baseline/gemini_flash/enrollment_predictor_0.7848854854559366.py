import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
import os

TRAIN_DATA_DIR = "table_splits/train/"
TEST_DATA_DIR = "/Users/USER/BEAVER/test/" # Placeholder path for test data, not available yet

GOLD_ENROLLMENT_TRAIN_FILE = TRAIN_DATA_DIR + "gold_enrollment_train.csv"
SUBJECT_OFFERED_TRAIN_FILE = TRAIN_DATA_DIR + "SUBJECT_OFFERED.csv"
ENROLLMENT_HISTORY_TRAIN_FILE = TRAIN_DATA_DIR + "ENROLLMENT_HISTORY_FEATURES.csv"

GOLD_ENROLLMENT_TEST_FILE = TEST_DATA_DIR + "gold_enrollment_test.csv" # Placeholder, not available yet
SUBJECT_OFFERED_TEST_FILE = TEST_DATA_DIR + "SUBJECT_OFFERED.csv" # Placeholder, not available yet
ENROLLMENT_HISTORY_TEST_FILE = TEST_DATA_DIR + "ENROLLMENT_HISTORY_FEATURES.csv" # Placeholder, not available yet

def load_data(data_dir, gold_file, subject_offered_file, enrollment_history_file):
    """Loads the gold enrollment data and relevant course offering data for a given data directory."""
    print(f"Loading gold enrollment from {gold_file}...")
    gold_df = pd.read_csv(gold_file)
    print("Gold enrollment loaded.")
    
    print(f"Loading subject offered data from {subject_offered_file}...")
    subject_offered_df = pd.read_csv(subject_offered_file, low_memory=False) # low_memory=False to handle mixed types
    print("Subject offered data loaded.")
    
    print(f"Loading enrollment history from {enrollment_history_file}...")
    enrollment_history_df = pd.read_csv(enrollment_history_file)
    print("Enrollment history loaded.")
    
    print("Merging gold labels with subject offered data...")
    df = pd.merge(gold_df, subject_offered_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    print("First merge complete.")
    
    print("Merging with enrollment history features...")
    df = pd.merge(df, enrollment_history_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    print("Second merge complete.")
    
    return df

def feature_engineer(df, is_training=True):
    """Creates features from the raw data."
    If is_training is False, it assumes HIGH_ENROLLMENT column might not be present (for test data).
    """
    print("Starting feature engineering...")
    # Convert target to numerical if in training mode
    if is_training:
        df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})
    
    # Handle TERM_CODE: Extract year and semester
    df['TERM_YEAR'] = df['TERM_CODE'].str[:4].astype(int)
    df['TERM_SEMESTER'] = df['TERM_CODE'].str[4:]
    
    # Fill NaN for categorical features before one-hot encoding
    for col in ['OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'TERM_SEMESTER']:
        df[col] = df[col].fillna('UNKNOWN')

    # Binary features from SUBJECT_OFFERED.csv
    binary_cols = ['IS_MASTER_SECTION', 'IS_LECTURE_SECTION', 'IS_LAB_SECTION', 
                   'IS_RECITATION_SECTION', 'IS_DESIGN_SECTION', 'EVALUATE_THIS_SUBJECT',
                   'IS_OSE_SUBJECT', 'IS_CREATED_BY_DATA_WAREHOUSE', 'IS_REPEATABLE_SUBJECT']
    for col in binary_cols:
        df[col] = df[col].map({'Y': 1, 'N': 0, True: 1, False: 0}).fillna(0) # Handle both Y/N and True/False

    # Numerical features (enrollment history)
    numerical_features = ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean']
    for col in numerical_features:
        df[col] = df[col].fillna(0) # Fill NaN for enrollment history with 0
    
    # Categorical features for one-hot encoding
    categorical_features = ['OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'TERM_SEMESTER']
    
    # All features for the model
    features = ['TERM_YEAR'] + binary_cols + numerical_features + categorical_features
    target = 'HIGH_ENROLLMENT'
    
    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['TERM_YEAR'] + binary_cols + numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    print("Feature engineering complete.")
    
    return df, features, target, preprocessor

def train_model(X_train, y_train, preprocessor):
    """Trains a RandomForestClassifier model within a pipeline."""
    print("Starting model training...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    pipeline.fit(X_train, y_train)
    print("Model training complete.")
    return pipeline

def evaluate_model(model, X_val, y_val):
    """Evaluates the model and returns macro F1 score."""
    print("Starting model evaluation...")
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    print("Model evaluation complete.")
    return f1

def predict_enrollment(model, new_data_df, features):
    """Makes predictions on new data using the trained model.
    new_data_df should contain columns similar to subject_offered_df and enrollment_history_df.
    """
    print("Starting prediction on new data...")

    # Apply the same feature engineering steps as training, but without target conversion
    new_data_df['TERM_YEAR'] = new_data_df['TERM_CODE'].str[:4].astype(int)
    new_data_df['TERM_SEMESTER'] = new_data_df['TERM_CODE'].str[4:]

    for col in ['OFFER_DEPT_CODE', 'OFFER_SCHOOL_NAME', 'TERM_SEMESTER']:
        new_data_df[col] = new_data_df[col].fillna('UNKNOWN')

    binary_cols = ['IS_MASTER_SECTION', 'IS_LECTURE_SECTION', 'IS_LAB_SECTION', 
                   'IS_RECITATION_SECTION', 'IS_DESIGN_SECTION', 'EVALUATE_THIS_SUBJECT',
                   'IS_OSE_SUBJECT', 'IS_CREATED_BY_DATA_WAREHOUSE', 'IS_REPEATABLE_SUBJECT']
    for col in binary_cols:
        new_data_df[col] = new_data_df[col].map({'Y': 1, 'N': 0, True: 1, False: 0}).fillna(0)

    numerical_features = ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean']
    for col in numerical_features:
        new_data_df[col] = new_data_df[col].fillna(0)

    X_new = new_data_df[features]
    predictions = model.predict(X_new)
    print("Prediction complete.")
    return pd.Series(predictions).map({1: 'Y', 0: 'N'})

def run_training_and_validation():
    """Main function to run the training and validation process."""
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, SUBJECT_OFFERED_TRAIN_FILE, ENROLLMENT_HISTORY_TRAIN_FILE)
    print("Data loaded and merged. Shape:", df.shape)

    df_processed, features, target, preprocessor = feature_engineer(df, is_training=True)
    
    print("Splitting data for time-based validation...")
    # Time-based validation split: use latest terms for validation
    # Assuming TERM_YEAR is sortable and higher values mean later terms
    unique_years = sorted(df_processed['TERM_YEAR'].unique())
    
    # Use the latest 20% of years for validation
    split_index = int(len(unique_years) * 0.8)
    train_years = unique_years[:split_index]
    val_years = unique_years[split_index:]
    
    X = df_processed[features]
    y = df_processed[target]
    
    X_train = X[df_processed['TERM_YEAR'].isin(train_years)]
    y_train = y[df_processed['TERM_YEAR'].isin(train_years)]
    X_val = X[df_processed['TERM_YEAR'].isin(val_years)]
    y_val = y[df_processed['TERM_YEAR'].isin(val_years)]
    print("Data split complete.")
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    
    model = train_model(X_train, y_train, preprocessor)
    f1 = evaluate_model(model, X_val, y_val)
    
    print(f"Validation Macro F1 Score: {f1}")
    
    return model, features

def run_test_prediction_and_evaluation(trained_model, trained_features):
    """Runs prediction and evaluation on the held-out test data.
    This function will only work when TEST_DATA_DIR and gold_enrollment_test.csv are available.
    """
    print("\n--- Attempting prediction and evaluation on test data ---")
    if not os.path.exists(TEST_DATA_DIR) or not os.path.exists(GOLD_ENROLLMENT_TEST_FILE):
        print(f"Test data directory ({TEST_DATA_DIR}) or gold test file ({GOLD_ENROLLMENT_TEST_FILE}) not found.")
        print("Cannot perform prediction and evaluation on test data until these files are available.")
        return

    # Load test data
    test_df = load_data(TEST_DATA_DIR, GOLD_ENROLLMENT_TEST_FILE, SUBJECT_OFFERED_TEST_FILE, ENROLLMENT_HISTORY_TEST_FILE)
    print("Test data loaded and merged. Shape:", test_df.shape)

    # Feature engineer test data (without converting target, as it's used for evaluation)
    # Note: preprocessor from training should be used for transforming, but for consistency in feature creation
    # we call feature_engineer, then extract X_test and y_test for evaluation.
    test_df_processed, _, _, _ = feature_engineer(test_df, is_training=False) 
    
    X_test = test_df_processed[trained_features]
    y_test = test_df_processed['HIGH_ENROLLMENT'] # Gold labels for evaluation
    
    print("Generating predictions on test data...")
    predictions_series = predict_enrollment(trained_model, test_df, trained_features) # Pass original test_df to predict_enrollment

    # Map y_test to numerical for f1_score
    y_test_numerical = y_test.map({'Y': 1, 'N': 0})
    
    f1_test = f1_score(y_test_numerical, predictions_series.map({'Y': 1, 'N': 0}), average='macro')
    print(f"Test Macro F1 Score: {f1_test}")
    
    # Optionally save predictions to a file
    predictions_output_df = test_df[['TERM_CODE', 'SUBJECT_ID_SORT']].copy()
    predictions_output_df['HIGH_ENROLLMENT_PREDICTED'] = predictions_series
    output_path = "test_predictions.csv"
    predictions_output_df.to_csv(output_path, index=False)
    print(f"Test predictions saved to {output_path}")


if __name__ == "__main__":
    start_time = time.time()
    trained_model, used_features = run_training_and_validation()
    end_time = time.time()
    print(f"Total training and validation execution time: {end_time - start_time:.2f} seconds")
    print("Training and validation complete.")

    # --- Demonstrate prediction and evaluation on test data (requires test data files) ---
    run_test_prediction_and_evaluation(trained_model, used_features)

    print("\nScript finished.")
