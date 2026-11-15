
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Load the gold enrollment data
gold_enrollment_path = "table_splits/train/gold_enrollment_train.csv"
gold_enrollment_df = pd.read_csv(gold_enrollment_path)

# Display the first few rows and column information
print("Gold Enrollment Data Head:")
print(gold_enrollment_df.head())
print("\nGold Enrollment Data Info:")
gold_enrollment_df.info()

# Preprocessing
# Encode the target variable HIGH_ENROLLMENT
le = LabelEncoder()
gold_enrollment_df['HIGH_ENROLLMENT_ENCODED'] = le.fit_transform(gold_enrollment_df['HIGH_ENROLLMENT'])

# For simplicity, we'll use TERM_CODE and SUBJECT_ID_SORT as features
# In a real scenario, we would join with other tables in TRAIN_DATA_DIR
# to get richer features.

# Convert TERM_CODE to numerical (e.g., year and semester)
def parse_term_code(term_code):
    year = int(str(term_code)[:4])
    semester_code = str(term_code)[4:]
    semester_map = {'FA': 0, 'SP': 1, 'JA': 2} # Fall, Spring, IAP (January)
    return year, semester_map.get(semester_code, -1)

term_data = gold_enrollment_df['TERM_CODE'].apply(lambda x: pd.Series(parse_term_code(x)))
gold_enrollment_df['YEAR'] = term_data[0]
gold_enrollment_df['SEMESTER'] = term_data[1]

# Convert SUBJECT_ID_SORT to numerical using LabelEncoder
gold_enrollment_df['SUBJECT_ID_SORT_ENCODED'] = le.fit_transform(gold_enrollment_df['SUBJECT_ID_SORT'])

features = ['YEAR', 'SEMESTER', 'SUBJECT_ID_SORT_ENCODED']
X = gold_enrollment_df[features]
y = gold_enrollment_df['HIGH_ENROLLMENT_ENCODED']

# Split data for validation
# Using a time-based split as suggested: latest train years held out for validation
# Find the unique years
unique_years = sorted(gold_enrollment_df['YEAR'].unique())
train_years = unique_years[:-1] # All but the latest year for training
val_year = unique_years[-1] # The latest year for validation

X_train = X[gold_enrollment_df['YEAR'].isin(train_years)]
y_train = y[gold_enrollment_df['YEAR'].isin(train_years)]
X_val = X[gold_enrollment_df['YEAR'] == val_year]
y_val = y[gold_enrollment_df['YEAR'] == val_year]

print(f"\nTraining data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model using macro F1 score
f1 = f1_score(y_val, y_pred, average='macro')
print(f"\nMacro F1 Score on validation set (using latest year {val_year} for validation): {f1}")

# --- Prediction logic for unseen data (example using train data for sanity check) ---
def predict_high_enrollment(df_to_predict, model, label_encoder, features):
    df = df_to_predict.copy()
    term_data = df['TERM_CODE'].apply(lambda x: pd.Series(parse_term_code(x)))
    df['YEAR'] = term_data[0]
    df['SEMESTER'] = term_data[1]
    df['SUBJECT_ID_SORT_ENCODED'] = label_encoder.transform(df['SUBJECT_ID_SORT']) # Use the same encoder fitted on training data

    X_predict = df[features]
    predictions_encoded = model.predict(X_predict)
    df['HIGH_ENROLLMENT'] = label_encoder.inverse_transform(predictions_encoded)
    return df[['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]

# --- Prediction logic for unseen data (using actual test data) ---
# When TEST_DATA_DIR and gold_enrollment_test.csv become available, uncomment and use the following:
test_gold_enrollment_path = "/Users/USER/BEAVER/test/gold_enrollment_test.csv"
test_df = pd.read_csv(test_gold_enrollment_path)

print("\nPredicting on actual test data:")
test_predictions_df = predict_high_enrollment(test_df, model, le, features)
print(test_predictions_df.head())

# Save the predictions to a CSV file in the required format
test_predictions_df.to_csv("predictions.csv", index=False)

# For now, keeping the sanity check with training data since test data is not available
print("\nPredicting on a sample of training data for sanity check (replace with test data when available):")
sample_to_predict = gold_enrollment_df.sample(5, random_state=42) # Take a small sample
predictions_df = predict_high_enrollment(sample_to_predict, model, le, features)
print(predictions_df)
