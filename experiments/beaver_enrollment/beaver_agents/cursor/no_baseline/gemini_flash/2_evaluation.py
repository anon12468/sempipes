import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def evaluate_model(predictions_df, gold_df):
    print("Evaluating the model.")
    
    # Ensure consistent column types for merging
    predictions_df['TERM_CODE'] = predictions_df['TERM_CODE'].astype(str)
    predictions_df['SUBJECT_ID_SORT'] = predictions_df['SUBJECT_ID_SORT'].astype(str)
    gold_df['TERM_CODE'] = gold_df['TERM_CODE'].astype(str)
    gold_df['SUBJECT_ID_SORT'] = gold_df['SUBJECT_ID_SORT'].astype(str)

    merged_df = pd.merge(predictions_df, gold_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], suffixes= ('_pred', '_gold'))

    if not merged_df.empty:
        y_true = merged_df['HIGH_ENROLLMENT_gold']
        y_pred = merged_df['HIGH_ENROLLMENT_pred']

        # Convert 'Y'/'N' from gold labels to numerical
        y_true_numeric = y_true.apply(lambda x: 1 if x == 'Y' else 0)
        # Predictions are already 0/1, no conversion needed for y_pred
        y_pred_numeric = y_pred 

        f1 = f1_score(y_true_numeric, y_pred_numeric, average='macro')
        precision = precision_score(y_true_numeric, y_pred_numeric, average='macro')
        recall = recall_score(y_true_numeric, y_pred_numeric, average='macro')
        accuracy = accuracy_score(y_true_numeric, y_pred_numeric)

        print(f"Macro F1 Score: {f1:.4f}")
        print(f"Macro Precision Score: {precision:.4f}")
        print(f"Macro Recall Score: {recall:.4f}")
        print(f"Accuracy Score: {accuracy:.4f}")
    else:
        print("No common entries between predictions and gold labels for evaluation.")