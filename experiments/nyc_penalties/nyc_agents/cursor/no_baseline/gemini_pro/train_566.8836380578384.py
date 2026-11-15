import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def clean_data(df):
    """Cleans and preprocesses the data."""
    # Standardize street names
    df['Street Name'] = df['Street Name'].str.upper().str.strip()
    # Replace common abbreviations
    replacements = {
        'AVENUE': 'AVE',
        'STREET': 'ST',
        'BOULEVARD': 'BLVD',
        'PLACE': 'PL',
        'ROAD': 'RD',
        'DRIVE': 'DR'
    }
    for old, new in replacements.items():
        df['Street Name'] = df['Street Name'].str.replace(r'\b' + old + r'\b', new, regex=True)
    return df

def feature_engineering(df):
    """Creates new features."""
    df['street_name_len'] = df['Street Name'].str.len()
    df['street_name_code'] = pd.factorize(df['Street Name'])[0]
    df['violation_desc_code'] = pd.factorize(df['Violation Description'])[0]
    return df

def load_and_merge_external_data(df, path, column_name, aggregation):
    """Loads external data, aggregates it, and merges it with the main dataframe."""
    try:
        ext_df = pd.read_parquet(path)
        print(f"\nExternal data {path} loaded successfully. Columns: {ext_df.columns.tolist()}")
        
        # Clean street names in external data
        if 'On_Street' in ext_df.columns:
            ext_df.rename(columns={'On_Street': 'Street Name'}, inplace=True)
        elif 'street' in ext_df.columns:
            ext_df.rename(columns={'street': 'Street Name'}, inplace=True)
        elif 'street_name' in ext_df.columns:
            ext_df.rename(columns={'street_name': 'Street Name'}, inplace=True)

        if 'Street Name' in ext_df.columns:
            ext_df = clean_data(ext_df)
            
            # Aggregate data
            agg_df = ext_df.groupby('Street Name').size().reset_index(name=column_name)
            
            # Merge with main dataframe
            df = pd.merge(df, agg_df, on='Street Name', how='left')
            df[column_name] = df[column_name].fillna(0)
            print(f"Feature '{column_name}' merged successfully.")
        else:
            print(f"Warning: 'Street Name' column not found in {path}. Skipping this file.")

    except FileNotFoundError:
        print(f"Warning: The file {path} was not found.")
    except Exception as e:
        print(f"An error occurred while processing {path}: {e}")

    return df

def load_and_merge_vehicle_data(df, path):
    """Loads and merges vehicle count data."""
    try:
        vehicle_df = pd.read_parquet(path)
        print(f"\nExternal data {path} loaded successfully.")
        
        # Rename street name column
        vehicle_df.rename(columns={'Roadway Name': 'Street Name'}, inplace=True)
        vehicle_df = clean_data(vehicle_df)
        
        # Sum up traffic counts
        count_columns = [col for col in vehicle_df.columns if ':' in col]
        vehicle_df['total_traffic'] = vehicle_df[count_columns].sum(axis=1)
        
        # Aggregate by street name
        agg_df = vehicle_df.groupby('Street Name')['total_traffic'].mean().reset_index()
        
        # Merge with main dataframe
        df = pd.merge(df, agg_df, on='Street Name', how='left')
        df['total_traffic'] = df['total_traffic'].fillna(0)
        print("Feature 'total_traffic' merged successfully.")
        
    except FileNotFoundError:
        print(f"Warning: The file {path} was not found.")
    except Exception as e:
        print(f"An error occurred while processing {path}: {e}")
        
    return df

def load_and_merge_pavement_data(df, path):
    """Loads and merges pavement rating data."""
    try:
        pavement_df = pd.read_parquet(path)
        print(f"\nExternal data {path} loaded successfully.")
        
        # Rename street name column
        pavement_df.rename(columns={'OnStreetName': 'Street Name'}, inplace=True)
        pavement_df = clean_data(pavement_df)
        
        # Aggregate by street name
        agg_df = pavement_df.groupby('Street Name')['SystemRating'].mean().reset_index()
        agg_df.rename(columns={'SystemRating': 'pavement_rating'}, inplace=True)
        
        # Merge with main dataframe
        df = pd.merge(df, agg_df, on='Street Name', how='left')
        df['pavement_rating'] = df['pavement_rating'].fillna(df['pavement_rating'].mean())
        print("Feature 'pavement_rating' merged successfully.")
        
    except FileNotFoundError:
        print(f"Warning: The file {path} was not found.")
    except Exception as e:
        print(f"An error occurred while processing {path}: {e}")
        
    return df

def main():
    parser = argparse.ArgumentParser(description='Parking Violation Prediction')
    parser.add_argument('--train-path', type=str, default='train/all_data/violations_per_street_2022.csv',
                        help='Path to the training data file.')
    parser.add_argument('--test-path', type=str, help='Path to the test data file.', default="/Users/USER/Downloads/violations_per_street_2023.csv")
    args = parser.parse_args()

    # Load the training data
    train_df = load_data(args.train_path)

    if train_df is not None:
        print("Training data loaded successfully.")

        # Clean and preprocess data
        train_df = clean_data(train_df)

        # Load and merge external data
        corpus_path = 'train/all_data/corpus/'
        train_df = load_and_merge_external_data(train_df, corpus_path + 'parking_meters_locations_and_status__693u-uax6.parquet', 'parking_meter_count', 'count')
        train_df = load_and_merge_vehicle_data(train_df, corpus_path + 'vehicle_classification_counts_2011_2025__96ay-ea4r.parquet')
        train_df = load_and_merge_pavement_data(train_df, corpus_path + 'street_pavement_ratings__6yyb-pb25.parquet')

        # Feature Engineering
        train_df = feature_engineering(train_df)

        # Log transform the target variable
        train_df['violation_count_log'] = np.log1p(train_df['violation_count'])

        # Define features and target
        features = ['street_name_len', 'street_name_code', 'violation_desc_code', 
                    'parking_meter_count', 'total_traffic', 'pavement_rating']
        target = 'violation_count_log'
        
        # Fill NaNs in features
        for col in features:
            if train_df[col].isnull().any():
                train_df[col] = train_df[col].fillna(0)

        X = train_df[features]
        y = train_df[target]

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a LightGBM model
        params = {
            'objective': 'regression_l1',
            'metric': 'rmse',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        # Evaluate the model
        y_pred_val = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        print(f"\nValidation RMSE: {rmse}")
        
        # Generate predictions on test data if provided
        if args.test_path:
            test_df = load_data(args.test_path)
            if test_df is not None:
                print("\nTest data loaded successfully.")
                
                # Store original street name and violation description for submission file
                original_street_name = test_df['Street Name']
                original_violation_desc = test_df['Violation Description']

                # Apply same transformations as on training data
                test_df = clean_data(test_df)
                test_df = load_and_merge_external_data(test_df, corpus_path + 'parking_meters_locations_and_status__693u-uax6.parquet', 'parking_meter_count', 'count')
                test_df = load_and_merge_vehicle_data(test_df, corpus_path + 'vehicle_classification_counts_2011_2025__96ay-ea4r.parquet')
                test_df = load_and_merge_pavement_data(test_df, corpus_path + 'street_pavement_ratings__6yyb-pb25.parquet')
                test_df = feature_engineering(test_df)

                # Ensure all feature columns exist and fill NaNs
                for col in features:
                    if col not in test_df.columns:
                        test_df[col] = 0
                test_df[features] = test_df[features].fillna(0)
                
                X_test = test_df[features]
                
                # Predict
                y_pred_test_log = model.predict(X_test)
                
                # Inverse transform
                y_pred_test = np.expm1(y_pred_test_log)

                # Test RMSE
                test_rmse = np.sqrt(mean_squared_error(test_df['violation_count'], y_pred_test))
                print(f"Test RMSE: {test_rmse}")
                
                # Create submission file
                submission_df = pd.DataFrame({
                    'street_name': original_street_name,
                    'violation_type': original_violation_desc,
                    'predicted_count': y_pred_test
                })
                
                submission_df.to_csv('submission.csv', index=False)
                print("\nsubmission.csv file generated successfully.")

if __name__ == '__main__':
    main()
