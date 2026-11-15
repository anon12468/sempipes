
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

def predict_and_evaluate(model_path, test_data_dir, gold_labels_path):
    # Load the trained model
    lgb_clf = joblib.load(model_path)

    # Load test data
    subject_summary = pd.read_csv(test_data_dir + 'SUBJECT_OFFERED_SUMMARY.csv')
    enrollment_history = pd.read_csv(test_data_dir + 'ENROLLMENT_HISTORY_FEATURES.csv')
    gold_enrollment_test = pd.read_csv(gold_labels_path)

    # Merge test data
    df_test = pd.merge(gold_enrollment_test, subject_summary, on=['TERM_CODE', 'SUBJECT_ID_SORT'])
    df_test = pd.merge(df_test, enrollment_history, on=['TERM_CODE', 'SUBJECT_ID_SORT'])

    # Feature Engineering
    df_test['OFFER_DEPT_NAME'] = df_test['OFFER_DEPT_NAME'].astype('category')
    df_test['OFFER_SCHOOL_NAME'] = df_test['OFFER_SCHOOL_NAME'].astype('category')

    # Features
    features = [
        'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREPARATION_UNITS',
        'lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean',
        'OFFER_DEPT_NAME', 'OFFER_SCHOOL_NAME'
    ]
    
    X_test = df_test[features]
    
    # Handle categorical features
    for col in X_test.select_dtypes(include=['category']).columns:
        X_test[col] = X_test[col].cat.codes

    # Predict
    y_pred = lgb_clf.predict(X_test)

    # Label encode gold standard
    le = LabelEncoder()
    y_true = le.fit_transform(df_test['HIGH_ENROLLMENT'])

    # Evaluate
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Macro F1 Score on test set: {macro_f1:.4f}')

    # Save predictions
    df_test['PREDICTION'] = le.inverse_transform(y_pred)
    df_test[['TERM_CODE', 'SUBJECT_ID_SORT', 'PREDICTION']].to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == '__main__':
    # This assumes the model is saved and the test data is in the same structure
    predict_and_evaluate(
        'lgbm_model.pkl',
        '/Users/USER/BEAVER/test/',
        '/Users/USER/BEAVER/test/gold_enrollment_test.csv'
    )
