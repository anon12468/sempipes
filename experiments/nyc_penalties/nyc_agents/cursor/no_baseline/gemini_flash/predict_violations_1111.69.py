import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import re

def clean_street_name(street_name):
    """Cleans street names for better matching."""
    if isinstance(street_name, str):
        street_name = street_name.upper()
        # Remove common street type suffixes and abbreviations
        street_name = re.sub(r'\b(STREET|ST|AVE|AVENUE|ROAD|RD|DRIVE|DR|PLACE|PL|PLAZA|BOULEVARD|BLVD|TERRACE|TCE|LANE|LN|PARKWAY|PKWY|COURT|CT|WAY|WY|EXPRESSWAY|EXP|FREEWAY|FWY|HIGHWAY|HWY|TURNPIKE|TPKE|CIRCLE|CIR|SQUARE|SQ)\b', '', street_name)
        # Remove periods and extra spaces
        street_name = re.sub(r'[.\s]+', ' ', street_name).strip()
    return street_name
    
def main():
    parser = argparse.ArgumentParser(description="Predict NYC parking violations.")
    parser.add_argument("--train-path", type=str, default="train/all_data/violations_per_street_2022.csv",
                        help="Path to the training data CSV file.")
    parser.add_argument("--test-path", type=str, default="/Users/USER/Downloads/violations_per_street_2023.csv",
                        help="Path to the test data CSV file (optional).")
    args = parser.parse_args()

    # 1. Data Loading
    try:
        train_df = pd.read_csv(args.train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}")
        return

    test_df = None
    if args.test_path:
        try:
            test_df = pd.read_csv(args.test_path)
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}")
            return
    
    print(f"Training data loaded from {args.train_path}")
    if test_df is not None:
        print(f"Test data loaded from {args.test_path}")

    # Clean street names in the main dataset
    train_df['cleaned_street_name'] = train_df['Street Name'].apply(clean_street_name)
    if test_df is not None:
        test_df['cleaned_street_name'] = test_df['Street Name'].apply(clean_street_name)

    # 2. Feature Engineering & Augmentation

    # Augment with street tree census data
    street_tree_path = "train/all_data/corpus/2005_street_tree_census__29bw-z7pj.parquet"
    try:
        street_tree_df = pd.read_parquet(street_tree_path)
        print(f"Street tree data loaded from {street_tree_path}")
        
        if 'address' in street_tree_df.columns:
            street_tree_df['cleaned_street_name'] = street_tree_df['address'].astype(str).apply(lambda x: clean_street_name(x.split(',')[0].strip()))
            
            street_tree_agg = street_tree_df.groupby('cleaned_street_name').size().reset_index(name='tree_count')
            
            train_df = pd.merge(train_df, street_tree_agg, on='cleaned_street_name', how='left')
            train_df['tree_count'] = train_df['tree_count'].fillna(0) # Fill NaN for streets with no tree data
            
            if test_df is not None:
                test_df = pd.merge(test_df, street_tree_agg, on='cleaned_street_name', how='left')
                test_df['tree_count'] = test_df['tree_count'].fillna(0)
            
            print("Street tree data augmented successfully.")
        else:
            print("Warning: Could not find 'address' column in street tree data. Skipping tree data augmentation.")

    except FileNotFoundError:
        print(f"Warning: Street tree data not found at {street_tree_path}. Skipping augmentation.")
    except Exception as e:
        print(f"Error loading or processing street tree data: {e}. Skipping augmentation.")
        
    # Augment with parking meter data
    parking_meter_path = "train/all_data/corpus/parking_meters_locations_and_status__693u-uax6.parquet"
    try:
        parking_meter_df = pd.read_parquet(parking_meter_path)
        print(f"Parking meter data loaded from {parking_meter_path}")

        street_col = None
        for col in ['street_name', 'on_street', 'street', 'location', 'address']:
            if col in parking_meter_df.columns:
                street_col = col
                break
        
        if street_col:
            parking_meter_df['cleaned_street_name'] = parking_meter_df[street_col].astype(str).apply(clean_street_name)
            parking_meter_agg = parking_meter_df.groupby('cleaned_street_name').size().reset_index(name='parking_meter_count')
            
            train_df = pd.merge(train_df, parking_meter_agg, on='cleaned_street_name', how='left')
            train_df['parking_meter_count'] = train_df['parking_meter_count'].fillna(0)
            
            if test_df is not None:
                test_df = pd.merge(test_df, parking_meter_agg, on='cleaned_street_name', how='left')
                test_df['parking_meter_count'] = test_df['parking_meter_count'].fillna(0)
            print(f"Parking meter data augmented successfully using column '{street_col}'.")
        else:
            print("Warning: Could not find a suitable street name column ('street_name', 'on_street', 'street', 'location', 'address') in parking meter data. Skipping parking meter data augmentation.")
    except FileNotFoundError:
        print(f"Warning: Parking meter data not found at {parking_meter_path}. Skipping augmentation.")
    except Exception as e:
        print(f"Error loading or processing parking meter data: {e}. Skipping augmentation.")

    # Feature from Violation Description: presence of keywords
    train_df['is_parking_violation'] = train_df['Violation Description'].apply(lambda x: 1 if re.search(r'PARKING', x.upper()) else 0)
    train_df['is_standing_violation'] = train_df['Violation Description'].apply(lambda x: 1 if re.search(r'STANDING', x.upper()) else 0)

    if test_df is not None:
        test_df['is_parking_violation'] = test_df['Violation Description'].apply(lambda x: 1 if re.search(r'PARKING', x.upper()) else 0)
        test_df['is_standing_violation'] = test_df['Violation Description'].apply(lambda x: 1 if re.search(r'STANDING', x.upper()) else 0)

    # Combine 'Street Name' and 'Violation Description' for a unique key
    train_df['key'] = train_df['Street Name'] + '_' + train_df['Violation Description']
    
    # Simple feature engineering: target encoding for 'Street Name' and 'Violation Description'
    # This is a basic approach and will be improved. Note: using .transform('mean') on the full dataset here can lead to data leakage if not careful.
    # For a robust solution, cross-validation or proper time-based splits would be needed for target encoding.
    street_name_mean_violations = train_df.groupby('Street Name')['violation_count'].transform('mean')
    train_df['street_name_mean_violations'] = street_name_mean_violations
    
    violation_desc_mean_violations = train_df.groupby('Violation Description')['violation_count'].transform('mean')
    train_df['violation_desc_mean_violations'] = violation_desc_mean_violations
    
    features = ['street_name_mean_violations', 'violation_desc_mean_violations', 'is_parking_violation', 'is_standing_violation']
    
    if 'tree_count' in train_df.columns:
        features.append('tree_count')
    if 'parking_meter_count' in train_df.columns:
        features.append('parking_meter_count')

    target = 'violation_count'

    # Handle potential NaN values from target encoding on unseen categories in future steps
    train_df = train_df.fillna(0)

    # 3. Model Training
    X = train_df[features]
    y = train_df[target]

    # Split 2022 data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 4. Prediction & Evaluation on Validation Set
    val_predictions = model.predict(X_val)
    val_predictions = np.maximum(0, val_predictions).round().astype(int) # Ensure non-negative and integer
    rmse_val = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"\nValidation RMSE (with augmented features): {rmse_val:.2f}")

    # 5. Prediction for Test Data (if provided)
    if test_df is not None:
        # Preprocess test_df similar to train_df
        test_df['key'] = test_df['Street Name'] + '_' + test_df['Violation Description']
        
        # Apply the same target encoding from training data
        test_df['street_name_mean_violations'] = test_df['Street Name'].map(train_df.groupby('Street Name')['violation_count'].mean()).fillna(0)
        test_df['violation_desc_mean_violations'] = test_df['Violation Description'].map(train_df.groupby('Violation Description')['violation_count'].mean()).fillna(0)
        
        # Features from Violation Description for test_df
        test_df['is_parking_violation'] = test_df['Violation Description'].apply(lambda x: 1 if re.search(r'PARKING', x.upper()) else 0)
        test_df['is_standing_violation'] = test_df['Violation Description'].apply(lambda x: 1 if re.search(r'STANDING', x.upper()) else 0)

        # Handle NaNs from merges (e.g., tree_count, parking_meter_count) before final feature selection
        test_df = test_df.fillna(0)

        X_test = test_df[features]
        test_predictions = model.predict(X_test)
        test_predictions = np.maximum(0, test_predictions).round().astype(int)

        # Count unseen pairs
        unseen_pairs_count = test_df[test_df['key'].isin(train_df['key']) == False].shape[0]
        print(f"\nNumber of (street_name, violation_type) pairs in test set unseen in training: {unseen_pairs_count}")

        # If 'violation_count' is present in test_df, report RMSE
        if 'violation_count' in test_df.columns:
            rmse_test = np.sqrt(mean_squared_error(test_df['violation_count'], test_predictions))
            print(f"Test RMSE: {rmse_test:.2f}")

        # 6. Generate submission.csv
        submission_df = pd.DataFrame({
            'street_name': test_df['Street Name'],
            'violation_type': test_df['Violation Description'],
            'predicted_count': test_predictions
        })
        submission_df.to_csv('submission.csv', index=False)
        print("\nsubmission.csv generated.")
    else:
        print("\nNo test-path provided, skipping test prediction and submission.csv generation.")

if __name__ == "__main__":
    main()