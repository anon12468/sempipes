import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# Load the data
gold_enrollment = pd.read_csv('table_splits/train/gold_enrollment_train.csv')
subject_summary = pd.read_csv('table_splits/train/SUBJECT_OFFERED_SUMMARY.csv', low_memory=False)
enrollment_history = pd.read_csv('table_splits/train/ENROLLMENT_HISTORY_FEATURES.csv')

# Merge the dataframes
data = pd.merge(gold_enrollment, subject_summary, on=['TERM_CODE', 'SUBJECT_ID_SORT'])
data = pd.merge(data, enrollment_history, on=['TERM_CODE', 'SUBJECT_ID_SORT'])

# Preprocessing and Feature Engineering
le = LabelEncoder()
data['TERM_CODE_encoded'] = le.fit_transform(data['TERM_CODE'])

# Fill missing numerical data
for col in ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean', 'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREPARATION_UNITS']:
    data[col] = data[col].fillna(0)

# Identify categorical and numerical features
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
# Remove identifiers and the target variable from categorical features
categorical_features.remove('TERM_CODE')
categorical_features.remove('SUBJECT_ID_SORT')
categorical_features.remove('HIGH_ENROLLMENT')
if 'SUBJECT_TITLE' in categorical_features: # Title is too high-cardinality
    categorical_features.remove('SUBJECT_TITLE')
if 'MASTER_SUBJECT_ID_SORT' in categorical_features: # Another high-cardinality ID
    categorical_features.remove('MASTER_SUBJECT_ID_SORT')
if 'CLUSTER_LIST' in categorical_features:
    categorical_features.remove('CLUSTER_LIST')
if 'RESPONSIBLE_FACULTY_NAME' in categorical_features:
    categorical_features.remove('RESPONSIBLE_FACULTY_NAME')


numerical_features = data.select_dtypes(include=['number']).columns.tolist()

# One-hot encode categorical features
data = pd.get_dummies(data, columns=categorical_features, dummy_na=True, drop_first=True)

# Update feature lists after one-hot encoding
features = [col for col in data.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT', 'SUBJECT_TITLE', 'MASTER_SUBJECT_ID_SORT', 'CLUSTER_LIST', 'RESPONSIBLE_FACULTY_NAME']]

target = 'HIGH_ENROLLMENT'
X = data[features]
y = data[target]

# Time-based split
sorted_terms = sorted(data['TERM_CODE'].unique())
split_point = int(len(sorted_terms) * 0.8)
train_terms = sorted_terms[:split_point]
validation_terms = sorted_terms[split_point:]

train_data = data[data['TERM_CODE'].isin(train_terms)]
validation_data = data[data['TERM_CODE'].isin(validation_terms)]

X_train = train_data[features]
y_train = train_data[target]
X_validation = validation_data[features]
y_validation = validation_data[target]

# Align columns after one-hot encoding
train_cols = X_train.columns
val_cols = X_validation.columns

missing_in_val = set(train_cols) - set(val_cols)
for c in missing_in_val:
    X_validation[c] = 0

missing_in_train = set(val_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_validation = X_validation[X_train.columns]

# Train the model
model = LogisticRegression(class_weight='balanced', max_iter=2000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_validation)

# Evaluate the model
f1 = f1_score(y_validation, y_pred, average='macro')
print(f"Macro F1 Score on the validation set: {f1}")

# Add evaluation logic as final step
def predict_and_evaluate(model, test_data_path, gold_file_path):
    # Load test data and preprocess it similarly to the training data
    subject_summary_test = pd.read_csv(f'{test_data_path}/SUBJECT_OFFERED_SUMMARY.csv', low_memory=False)
    enrollment_history_test = pd.read_csv(f'{test_data_path}/ENROLLMENT_HISTORY_FEATURES.csv')
    test_data = pd.merge(subject_summary_test, enrollment_history_test, on=['TERM_CODE', 'SUBJECT_ID_SORT'])
    
    # Preprocessing
    test_data['TERM_CODE_encoded'] = le.transform(test_data['TERM_CODE'])
    for col in ['lag1_enrollment', 'lag2_enrollment', 'roll3_enrollment_mean', 'TOTAL_UNITS', 'LECTURE_UNITS', 'LAB_UNITS', 'PREPARATION_UNITS']:
        test_data[col] = test_data[col].fillna(0)
    
    test_data = pd.get_dummies(test_data, columns=categorical_features, dummy_na=True, drop_first=True)
    
    X_test = test_data[features]
    
    # Align columns
    test_cols = X_test.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
    X_test = X_test[train_cols]
    
    test_data['HIGH_ENROLLMENT_pred'] = model.predict(X_test)
    
    gold_enrollment_test = pd.read_csv(gold_file_path)
    evaluation_data = pd.merge(test_data, gold_enrollment_test, on=['TERM_CODE', 'SUBJECT_ID_SORT'])
    
    f1 = f1_score(evaluation_data['HIGH_ENROLLMENT'], evaluation_data['HIGH_ENROLLMENT_pred'], average='macro')
    print(f"Final Macro F1 Score on the test set: {f1}")
    return f1

print(predict_and_evaluate(model, '/Users/USER/BEAVER/test', '/Users/USER/BEAVER/test/gold_enrollment_test.csv'))
