import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_and_evaluate():
    # Load the data
    gold_enrollment = pd.read_csv('table_splits/train/gold_enrollment_train.csv')
    subject_offered = pd.read_csv('table_splits/train/SUBJECT_OFFERED.csv', encoding='ISO-8859-1', low_memory=False)
    subject_summary = pd.read_csv('table_splits/train/SUBJECT_SUMMARY.csv', encoding='ISO-8859-1', low_memory=False)

    # Merge the data
    data = pd.merge(gold_enrollment, subject_offered, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    data = pd.merge(data, subject_summary, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Feature Engineering
    data['TERM_CODE'] = data['TERM_CODE'].astype(str)
    data['YEAR'] = data['TERM_CODE'].str[:4].astype(int)
    data['SEMESTER'] = data['TERM_CODE'].str[4:]
    data['IS_FALL'] = (data['SEMESTER'] == 'FA').astype(int)
    data['IS_SPRING'] = (data['SEMESTER'] == 'SP').astype(int)
    data['IS_SUMMER'] = (data['SEMESTER'] == 'SU').astype(int)
    
    # Fill missing values
    for col in ['TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREP_UNITS', 'DESIGN_UNITS']:
        data[col] = data[col].fillna(data[col].median())

    # Label Encoding for categorical features
    categorical_features = ['SEMESTER', 'SUBJECT_ID_SORT', 'DEPARTMENT_CODE', 'SCHOOL_CODE']
    for col in categorical_features:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # Target variable
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Time-based split
    train_data = data[data['YEAR'] < 2016]
    val_data = data[data['YEAR'] >= 2016]

    features = ['YEAR', 'SEMESTER', 'SUBJECT_ID_SORT', 'DEPARTMENT_CODE', 'SCHOOL_CODE',
                'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREP_UNITS', 'DESIGN_UNITS',
                'IS_FALL', 'IS_SPRING', 'IS_SUMMER']
    
    X_train = train_data[features]
    y_train = train_data['HIGH_ENROLLMENT']
    X_val = val_data[features]
    y_val = val_data['HIGH_ENROLLMENT']

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_val)
    f1 = f1_score(y_val, predictions, average='macro')
    print(f"Validation Macro F1 Score: {f1}")
    
    return model

def predict_and_evaluate(model):
    # This function will be used for the final evaluation on the test set
    # For now, it will predict and evaluate on the validation set as a placeholder
    gold_enrollment_test = pd.read_csv('/Users/USER/BEAVER/test/gold_enrollment_test.csv')
    subject_offered = pd.read_csv('/Users/USER/BEAVER/test/SUBJECT_OFFERED.csv', encoding='ISO-8859-1', low_memory=False)
    subject_summary = pd.read_csv('/Users/USER/BEAVER/test/SUBJECT_SUMMARY.csv', encoding='ISO-8859-1', low_memory=False)

    # Merge the data
    data = pd.merge(gold_enrollment_test, subject_offered, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    data = pd.merge(data, subject_summary, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Feature Engineering
    data['TERM_CODE'] = data['TERM_CODE'].astype(str)
    data['YEAR'] = data['TERM_CODE'].str[:4].astype(int)
    data['SEMESTER'] = data['TERM_CODE'].str[4:]
    data['IS_FALL'] = (data['SEMESTER'] == 'FA').astype(int)
    data['IS_SPRING'] = (data['SEMESTER'] == 'SP').astype(int)
    data['IS_SUMMER'] = (data['SEMESTER'] == 'SU').astype(int)
    
    # Fill missing values
    for col in ['TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREP_UNITS', 'DESIGN_UNITS']:
        data[col] = data[col].fillna(data[col].median())

    # Label Encoding for categorical features
    categorical_features = ['SEMESTER', 'SUBJECT_ID_SORT', 'DEPARTMENT_CODE', 'SCHOOL_CODE']
    for col in categorical_features:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            
    # Target variable
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    features = ['YEAR', 'SEMESTER', 'SUBJECT_ID_SORT', 'DEPARTMENT_CODE', 'SCHOOL_CODE',
                'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREP_UNITS', 'DESIGN_UNITS',
                'IS_FALL', 'IS_SPRING', 'IS_SUMMER']

    X_test = data[features]
    y_test = data['HIGH_ENROLLMENT']
    
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f"Test Macro F1 Score: {f1}")

if __name__ == "__main__":
    model = train_and_evaluate()
    predict_and_evaluate(model)
