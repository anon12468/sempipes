import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(data_path='table_splits/train/'):
    """
    Loads, merges, and preprocesses the data to create a unique entry for each course offering.
    """
    print("Loading and preprocessing data...")
    
    # Load the tables
    subject_summary = pd.read_csv(data_path + 'SUBJECT_SUMMARY.csv', low_memory=False)
    subject_offered = pd.read_csv(data_path + 'SUBJECT_OFFERED.csv', low_memory=False)

    # Merge the tables
    df = pd.merge(subject_summary, subject_offered, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    # Clean up column names
    df = df.loc[:,~df.columns.duplicated()]
    df.columns = [col.replace('_x', '').replace('_y', '') for col in df.columns]

    # Handle null MASTER_SUBJECT_ID_SORT
    df['MASTER_SUBJECT_ID_SORT'] = df['MASTER_SUBJECT_ID_SORT'].fillna(df['SUBJECT_ID_SORT'])

    # Filter for master sections
    df_master = df[df['IS_MASTER_SECTION'] == 'Y'].copy()

    # Aggregate cross-listed courses
    agg_funcs = {col: 'first' for col in df_master.columns if col not in ['TERM_CODE', 'MASTER_SUBJECT_ID_SORT']}
    numeric_enrollment_cols = [
        'SUBJECT_ENROLLMENT_FIRST_WEEK', 'CLUSTER_ENROLLMENT_FIRST_WEEK',
        'SUBJECT_ENROLLMENT_FIFTH_WEEK', 'CLUSTER_ENROLLMENT_FIFTH_WEEK',
        'SUBJECT_ENROLLMENT_CREDIT', 'SUBJECT_ENROLLMENT_LISTEN',
        'CLUSTER_ENROLLMENT_CREDIT', 'CLUSTER_ENROLLMENT_LISTEN'
    ]
    for col in numeric_enrollment_cols:
        if col in df_master.columns:
            agg_funcs[col] = 'sum'
    
    df_agg = df_master.groupby(['TERM_CODE', 'MASTER_SUBJECT_ID_SORT']).agg(agg_funcs).reset_index()

    print("Data loaded and preprocessed.")
    return df_agg

def feature_engineering(df):
    """
    Engineers features for the model.
    """
    print("Engineering features...")

    # Term features
    df['YEAR'] = df['TERM_CODE'].str[:4].astype(int)
    df['SEMESTER'] = df['TERM_CODE'].str[4:]
    
    # Sort by year to create historical features
    df = df.sort_values(by=['YEAR', 'SEMESTER'])

    # Historical Enrollment Features
    df['HISTORICAL_AVG_ENROLLMENT'] = df.groupby('MASTER_SUBJECT_ID_SORT')['SUBJECT_ENROLLMENT_FIFTH_WEEK'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['HISTORICAL_MAX_ENROLLMENT'] = df.groupby('MASTER_SUBJECT_ID_SORT')['SUBJECT_ENROLLMENT_FIFTH_WEEK'].transform(lambda x: x.rolling(window=5, min_periods=1).max().shift(1))

    # Instructor Features
    df['INSTRUCTOR_AVG_ENROLLMENT'] = df.groupby('RESPONSIBLE_FACULTY_NAME')['SUBJECT_ENROLLMENT_FIFTH_WEEK'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))

    # Course-Level Features
    df['COURSE_OFFERING_COUNT'] = df.groupby('MASTER_SUBJECT_ID_SORT').cumcount()

    # Enrollment features
    df['ENROLLMENT_CHANGE'] = df['SUBJECT_ENROLLMENT_FIFTH_WEEK'] - df['SUBJECT_ENROLLMENT_FIRST_WEEK']

    # Target variable
    if 'HIGH_ENROLLMENT' in df.columns:
        df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Handle missing values
    for col in ['RESPONSIBLE_FACULTY_NAME', 'MEET_TIME', 'MEET_PLACE']:
        df[col] = df[col].fillna('Unknown')
    
    for col in ['TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREP_UNITS', 'DESIGN_UNITS', 'HISTORICAL_AVG_ENROLLMENT', 'HISTORICAL_MAX_ENROLLMENT', 'INSTRUCTOR_AVG_ENROLLMENT']:
        df[col] = df[col].fillna(0)

    # Encode categorical features
    for col in ['SEMESTER', 'DEPARTMENT_NAME', 'SCHOOL_NAME', 'RESPONSIBLE_FACULTY_NAME', 'MEET_TIME', 'MEET_PLACE']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Select features
    features = [
        'YEAR', 'SEMESTER', 'DEPARTMENT_NAME', 'SCHOOL_NAME', 'TOTAL_UNITS', 
        'LECTURE_UNITS', 'LAB_UNITS', 'PREP_UNITS', 'DESIGN_UNITS',
        'SUBJECT_ENROLLMENT_FIRST_WEEK', 'ENROLLMENT_CHANGE', 'RESPONSIBLE_FACULTY_NAME',
        'MEET_TIME', 'MEET_PLACE', 'HISTORICAL_AVG_ENROLLMENT', 'HISTORICAL_MAX_ENROLLMENT',
        'INSTRUCTOR_AVG_ENROLLMENT', 'COURSE_OFFERING_COUNT'
    ]
    
    # Ensure all feature columns are numeric
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print("Features engineered.")
    return df, features

def train_model(df, features):
    """
    Trains a model and evaluates it.
    """
    print("Training model...")
    
    gold_enrollment = pd.read_csv('table_splits/train/gold_enrollment_train.csv')
    df = pd.merge(gold_enrollment, df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
    df, features = feature_engineering(df)

    # Time-based split
    train_df = df[df['YEAR'] < 2016]
    val_df = df[df['YEAR'] >= 2016]

    X_train = train_df[features]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[features]
    y_val = val_df['HIGH_ENROLLMENT']

    # Handle class imbalance with SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train a simple model
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate the model
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average='macro')
    print(f"Validation Macro F1 Score: {f1}")

    # Save the model
    joblib.dump(model, 'enrollment_model.pkl')
    joblib.dump(features, 'features.pkl')
    print("Model and features saved.")

    return model, features

def predict_and_evaluate(model, features, gold_file_path):
    """
    Makes predictions on a test set and evaluates the results.
    """
    print(f"Making predictions on {gold_file_path}...")

    # Load the gold standard data
    gold_standard = pd.read_csv(gold_file_path)
    
    # Load the feature data
    all_data = load_and_preprocess_data('table_splits/train/')
    
    # Merge to get the rows we need to predict on
    predict_df = pd.merge(gold_standard, all_data, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Feature engineering
    predict_df, features = feature_engineering(predict_df)

    # Make predictions
    X_predict = predict_df[features].fillna(0)
    predictions = model.predict(X_predict)

    # Evaluate
    gold_standard['HIGH_ENROLLMENT'] = gold_standard['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    f1 = f1_score(gold_standard['HIGH_ENROLLMENT'], predictions, average='macro')
    print(f"Macro F1 Score on {gold_file_path}: {f1}")


if __name__ == '__main__':
    df = load_and_preprocess_data()
    model, features = train_model(df, None)
    
    # Predict and evaluate on the test set
    predict_and_evaluate(model, features, '/Users/USER/BEAVER/test/gold_enrollment_test.csv')
