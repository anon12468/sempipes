
import argparse
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="train/all_data/violations_per_street_2022.csv", help="Path to the training data.")
    parser.add_argument("--test-path", default="/Users/USER/Downloads/violations_per_street_2023.csv", help="Path to the test data. If not provided, the training data will be used for submission.")
    args = parser.parse_args()

    print(f"Loading training data from {args.train_path}")
    df = pd.read_csv(args.train_path)
    print("Training data loaded successfully")
    
    df.columns = df.columns.str.strip()
    df['Street Name'] = df['Street Name'].str.upper().str.strip()
    
    # --- Feature Engineering ---
    print("Starting feature engineering...")
    # 1. Parking Meter Feature
    meter_df = pd.read_parquet('train/all_data/corpus/parking_meters_locations_and_status__693u-uax6.parquet')
    meter_df['On_Street'] = meter_df['On_Street'].str.upper().str.strip()
    meter_counts = meter_df.groupby('On_Street').size().reset_index(name='meter_count')
    df = pd.merge(df, meter_counts, left_on='Street Name', right_on='On_Street', how='left')
    df['meter_count'] = df['meter_count'].fillna(0)
    df = df.drop(columns=['On_Street'])

    # 2. Pavement Rating Feature
    pavement_df = pd.read_parquet('train/all_data/corpus/street_pavement_ratings__6yyb-pb25.parquet')
    pavement_df['OnStreetName'] = pavement_df['OnStreetName'].str.upper().str.strip()
    pavement_rating = pavement_df.groupby('OnStreetName')['SystemRating'].mean().reset_index(name='avg_pavement_rating')
    df = pd.merge(df, pavement_rating, left_on='Street Name', right_on='OnStreetName', how='left')
    median_rating = df['avg_pavement_rating'].median()
    df['avg_pavement_rating'] = df['avg_pavement_rating'].fillna(median_rating)
    df = df.drop(columns=['OnStreetName'])
    print("Feature engineering complete.")

    # Categorical Features
    df['street_name_cat'] = df['Street Name'].astype('category').cat.codes
    df['violation_desc_cat'] = df['Violation Description'].astype('category').cat.codes

    features = ['street_name_cat', 'violation_desc_cat', 'meter_count', 'avg_pavement_rating']
    target = 'violation_count'

    X_train = df[features]
    y_train = df[target]

    print(f"Training model on {len(df)} records...")
    
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 1200, # Increased estimators for final model
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Determine test path
    test_path = args.test_path if args.test_path else args.train_path
    print(f"Generating predictions for {test_path}")
    test_df = pd.read_csv(test_path)
    test_df.columns = test_df.columns.str.strip()
    test_df['Street Name'] = test_df['Street Name'].str.upper().str.strip()

    # Apply same feature engineering to test data
    test_df = pd.merge(test_df, meter_counts, left_on='Street Name', right_on='On_Street', how='left').drop(columns=['On_Street'])
    test_df['meter_count'] = test_df['meter_count'].fillna(0)
    
    test_df = pd.merge(test_df, pavement_rating, left_on='Street Name', right_on='OnStreetName', how='left').drop(columns=['OnStreetName'])
    test_df['avg_pavement_rating'] = test_df['avg_pavement_rating'].fillna(median_rating)

    test_df['street_name_cat'] = pd.Categorical(test_df['Street Name'], categories=df['Street Name'].astype('category').cat.categories).codes
    test_df['violation_desc_cat'] = pd.Categorical(test_df['Violation Description'], categories=df['Violation Description'].astype('category').cat.categories).codes
    
    # Handle unseen categories in test data by assigning a new category
    unseen_street_code = len(df['Street Name'].astype('category').cat.categories)
    unseen_violation_code = len(df['Violation Description'].astype('category').cat.categories)
    test_df['street_name_cat'] = test_df['street_name_cat'].replace(-1, unseen_street_code)
    test_df['violation_desc_cat'] = test_df['violation_desc_cat'].replace(-1, unseen_violation_code)

    X_test = test_df[features]
    test_preds = model.predict(X_test)

    submission = pd.DataFrame({
        'street_name': test_df['Street Name'],
        'violation_type': test_df['Violation Description'],
        'predicted_count': test_preds
    })
    
    submission['predicted_count'] = submission['predicted_count'].clip(lower=0).round()
    
    # Calculate and report test performance if ground truth is available
    if 'violation_count' in test_df.columns:
        test_rmse = np.sqrt(mean_squared_error(test_df['violation_count'], submission['predicted_count']))
        print(f"Test RMSE: {test_rmse}")

    # Report on unseen key pairs
    train_keys = set(zip(df['Street Name'], df['Violation Description']))
    test_keys = set(zip(test_df['Street Name'], test_df['Violation Description']))
    unseen_keys = test_keys - train_keys
    print(f"Number of key pairs in test set: {len(test_keys)}")
    print(f"Number of key pairs unseen in training: {len(unseen_keys)}")
    print("Unseen keys were handled by assigning them a new category code during prediction.")

    submission.to_csv("submission.csv", index=False)
    print("submission.csv created successfully.")

if __name__ == "__main__":
    main()
