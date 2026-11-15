import pandas as pd
import os

def load_data(data_dir, gold_file_path):
    all_data = {}
    relevant_files = [
        "ENROLLMENT_HISTORY_FEATURES.csv",
        "SUBJECT_OFFERED.csv"
    ]

    for filename in relevant_files:
        filepath = os.path.join(data_dir, filename)
        df_name = filename.replace(".csv", "")
        try:
            all_data[df_name] = pd.read_csv(filepath, low_memory=False)
            print(f"Loaded {filename}")
        except FileNotFoundError:
            print(f"Error: {filename} not found in {data_dir}")
            return pd.DataFrame(), pd.DataFrame() # Return empty dataframes on error

    try:
        gold_enrollment_df = pd.read_csv(gold_file_path, low_memory=False)
        print(f"Loaded {os.path.basename(gold_file_path)}")
    except FileNotFoundError:
        print(f"Error: Gold file not found at {gold_file_path}")
        return pd.DataFrame(), pd.DataFrame()

    training_df = all_data["ENROLLMENT_HISTORY_FEATURES"].copy()

    # --- Aggregation from SUBJECT_OFFERED.csv ---
    if "SUBJECT_OFFERED" in all_data:
        subject_offered_df = all_data["SUBJECT_OFFERED"].copy()
        subject_offered_df['TERM_CODE'] = subject_offered_df['TERM_CODE'].astype(str)
        subject_offered_df['OFFER_DEPT_CODE'] = subject_offered_df['OFFER_DEPT_CODE'].astype(str)

        # Convert binary flags to numeric for summation
        binary_cols = [
            'IS_MASTER_SECTION', 'IS_LECTURE_SECTION', 'IS_LAB_SECTION',
            'IS_RECITATION_SECTION', 'IS_DESIGN_SECTION', 'EVALUATE_THIS_SUBJECT',
            'IS_OSE_SUBJECT', 'IS_CREATED_BY_DATA_WAREHOUSE', 'IS_REPEATABLE_SUBJECT'
        ]
        for col in binary_cols:
            if col in subject_offered_df.columns:
                subject_offered_df[col] = subject_offered_df[col].apply(lambda x: 1 if x == 'Y' else 0)

        # Calculate department- and term-level aggregates
        dept_term_aggregates = subject_offered_df.groupby(['TERM_CODE', 'OFFER_DEPT_CODE']).agg(
            num_subjects_in_dept_term=('SUBJECT_ID_SORT', 'nunique'),
            num_sections_in_dept_term=('MASTER_COURSE_NUMBER', 'count'),
            num_master_sections=('IS_MASTER_SECTION', 'sum'),
            num_lecture_sections=('IS_LECTURE_SECTION', 'sum'),
            num_lab_sections=('IS_LAB_SECTION', 'sum'),
            num_recitation_sections=('IS_RECITATION_SECTION', 'sum'),
            num_design_sections=('IS_DESIGN_SECTION', 'sum'),
            num_faculty_in_dept_term=('RESPONSIBLE_FACULTY_NAME', 'nunique'),
            num_evaluated_subjects=('EVALUATE_THIS_SUBJECT', 'sum'),
            num_ose_subjects=('IS_OSE_SUBJECT', 'sum'),
            num_dw_created_subjects=('IS_CREATED_BY_DATA_WAREHOUSE', 'sum'),
            num_repeatable_subjects=('IS_REPEATABLE_SUBJECT', 'sum'),
        ).reset_index()
        print("Calculated advanced department-term aggregates from SUBJECT_OFFERED")

        # Extract department code from SUBJECT_ID_SORT in training_df for merging
        training_df['SUBJECT_DEPT_CODE'] = training_df['SUBJECT_ID_SORT'].str.split('.').str[0]

        # Merge these aggregates into training_df
        training_df = pd.merge(training_df, dept_term_aggregates,
                               left_on=['TERM_CODE', 'SUBJECT_DEPT_CODE'], 
                               right_on=['TERM_CODE', 'OFFER_DEPT_CODE'], 
                               how='left', suffixes=("", "_dept_agg"))
        
        # Retain OFFER_DEPT_CODE (now _dept_agg) if needed in feature engineering for relative features
        # or drop temporary SUBJECT_DEPT_CODE if no longer needed

        print("Merged advanced department-term aggregates into training_df")
    
    # Ensure TERM_CODE and SUBJECT_ID_SORT are strings for consistent merging later
    training_df['TERM_CODE'] = training_df['TERM_CODE'].astype(str)
    training_df['SUBJECT_ID_SORT'] = training_df['SUBJECT_ID_SORT'].astype(str)
    gold_enrollment_df['TERM_CODE'] = gold_enrollment_df['TERM_CODE'].astype(str)
    gold_enrollment_df['SUBJECT_ID_SORT'] = gold_enrollment_df['SUBJECT_ID_SORT'].astype(str)

    return training_df, gold_enrollment_df