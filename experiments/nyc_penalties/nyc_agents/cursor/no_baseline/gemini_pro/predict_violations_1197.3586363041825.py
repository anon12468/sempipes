
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import argparse
import numpy as np

def main(train_path, test_path=None):
    # Load data
    df = pd.read_csv(train_path)

    # Feature engineering
    df_meters = pd.read_parquet('train/all_data/corpus/parking_meters_locations_and_status__693u-uax6.parquet')
    meter_counts = df_meters.groupby('On_Street').size().reset_index(name='meter_count')
    meter_counts.rename(columns={'On_Street': 'Street Name'}, inplace=True)
    df = pd.merge(df, meter_counts, on='Street Name', how='left').fillna(0)

    # Preprocessing
    df['violation_type'] = df['Violation Description'].astype('category').cat.codes
    df['street_name_encoded'] = df['Street Name'].astype('category').cat.codes

    X = df[['street_name_encoded', 'violation_type', 'meter_count']]
    y = df['violation_count']

    # Train-test split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Validation
    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f'Validation RMSE: {rmse}')

    # Prediction
    if test_path:
        df_test = pd.read_csv(test_path)
        df_test = pd.merge(df_test, meter_counts, on='Street Name', how='left').fillna(0)
        df_test['violation_type'] = df_test['Violation Description'].astype('category').cat.codes
        df_test['street_name_encoded'] = df_test['Street Name'].astype('category').cat.codes
        
        X_test = df_test[['street_name_encoded', 'violation_type', 'meter_count']]
        df_test['predicted_count'] = model.predict(X_test)
        
        # Handle unseen keys
        unseen_mask = ~df_test['Street Name'].isin(df['Street Name']) | ~df_test['Violation Description'].isin(df['Violation Description'])
        print(f"Number of unseen key pairs in test set: {unseen_mask.sum()}")
        # For now, we are not doing anything special for unseen keys, the model will predict based on encoded values.
        
        submission = df_test[['Street Name', 'Violation Description', 'predicted_count']]
        submission.columns = ['street_name', 'violation_type', 'predicted_count']
        submission['predicted_count'] = submission['predicted_count'].round().astype(int)
        submission.loc[submission['predicted_count'] < 0, 'predicted_count'] = 0
        submission.to_csv('submission.csv', index=False)
        print("submission.csv created.")

        if 'violation_count' in df_test.columns:
            test_rmse = np.sqrt(mean_squared_error(df_test['violation_count'], df_test['predicted_count']))
            print(f'Test RMSE: {test_rmse}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', default='train/all_data/violations_per_street_2022.csv')
    parser.add_argument('--test-path', default="/Users/USER/Downloads/violations_per_street_2023.csv")
    args = parser.parse_args()
    main(args.train_path, args.test_path)
