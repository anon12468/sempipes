
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

def train_and_evaluate():
    # Load data
    data_dir = 'table_splits/train/'
    gold_enrollment = pd.read_csv(data_dir + 'gold_enrollment_train.csv')
    subject_summary = pd.read_csv(data_dir + 'SUBJECT_OFFERED_SUMMARY.csv', low_memory=False)
    enrollment_history = pd.read_csv(data_dir + 'ENROLLMENT_HISTORY_FEATURES.csv')

    # Merge data
    df = pd.merge(gold_enrollment, subject_summary, on=['TERM_CODE', 'SUBJECT_ID_SORT'])
    df = pd.merge(df, enrollment_history, on=['TERM_CODE', 'SUBJECT_ID_SORT'])

    # Feature Engineering
    term_codes = sorted(df['TERM_CODE'].unique())
    df['TERM_CODE'] = pd.Categorical(df['TERM_CODE'], categories=term_codes, ordered=True)
    df['OFFER_DEPT_NAME'] = df['OFFER_DEPT_NAME'].astype('category')
    df['OFFER_SCHOOL_NAME'] = df['OFFER_SCHOOL_NAME'].astype('category')
    
    # Label encode target
    le = LabelEncoder()
    df['HIGH_ENROLLMENT'] = le.fit_transform(df['HIGH_ENROLLMENT'])

    # Features and target
    features = [
        'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREPARATION_UNITS',
        'lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean',
        'OFFER_DEPT_NAME', 'OFFER_SCHOOL_NAME'
    ]
    target = 'HIGH_ENROLLMENT'

    # Time-based split
    val_term = next(term for term in df['TERM_CODE'].cat.categories if term.startswith('2017'))
    train_df = df[df['TERM_CODE'] < val_term]
    val_df = df[df['TERM_CODE'] >= val_term]

    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]
    
    # Handle categorical features by converting them to integers
    for col in X_train.select_dtypes(include=['category']).columns:
        X_train[col] = X_train[col].cat.codes
        X_val[col] = X_val[col].cat.codes

    # Train LightGBM model
    lgb_clf = lgb.LGBMClassifier(objective='binary', random_state=42)
    lgb_clf.fit(X_train, y_train, categorical_feature=['OFFER_DEPT_NAME', 'OFFER_SCHOOL_NAME'])

    # Evaluate model
    y_pred = lgb_clf.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    print(f'Macro F1 Score on validation set: {macro_f1:.4f}')

    # Save the model
    joblib.dump(lgb_clf, 'lgbm_model.pkl')
    print("Model saved to lgbm_model.pkl")

    return lgb_clf

def predict(model, test_data_dir):
    # This is a placeholder for the final prediction logic
    # It will load test data, preprocess it, and predict
    print("\nPrediction function called. (Placeholder)")
    
if __name__ == '__main__':
    model = train_and_evaluate()
