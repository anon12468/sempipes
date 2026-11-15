import pandas as pd
from xgboost import XGBClassifier

def train_model(X_train, y_train, scale_pos_weight):
    print("Training an XGBoost Classifier with scale_pos_weight (fixed hyperparameters).")
    # Using fixed, reasonable hyperparameters for XGBoost
    model = XGBClassifier(objective='binary:logistic', 
                          eval_metric='logloss', 
                          use_label_encoder=False, 
                          n_estimators=100, 
                          max_depth=5, 
                          learning_rate=0.1, 
                          subsample=0.8, 
                          colsample_bytree=0.8, 
                          gamma=0.1, 
                          random_state=42,
                          scale_pos_weight=scale_pos_weight) # Handle class imbalance
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    print("Making predictions.")
    predictions = model.predict(X_test)
    return predictions