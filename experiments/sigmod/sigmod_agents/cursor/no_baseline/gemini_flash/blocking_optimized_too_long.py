from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re
import numpy as np

def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

def block_scalable(X, attr):
    """
    This function performs scalable blocking using the specified attribute.
    It avoids O(k^2) operations within blocking buckets for pair generation.
    :param X: dataframe
    :param attr: attribute used for blocking (e.g., 'name')
    :return: candidate set of tuple pairs (real IDs)
    """

    # 1. Build Inverted Index (mapping token to list of DataFrame indices)
    inverted_index_df_idx = defaultdict(list)
    for df_idx, row in tqdm(X.iterrows(), total=X.shape[0], desc="Building Inverted Index"):
        tokens = tokenize(row[attr])
        for token in tokens:
            inverted_index_df_idx[token].append(df_idx)

    # 2. Generate Candidates and Co-occurrence Scores
    #    This approach avoids O(k^2) within a single token's posting list.
    #    Instead, it iterates through records and identifies partners based on shared tokens,
    #    accumulating a count of shared tokens for each unique pair.
    candidate_cooccurrence_scores = defaultdict(int)

    for df_idx, row in tqdm(X.iterrows(), total=X.shape[0], desc="Calculating Co-occurrence Scores"):
        record_tokens = tokenize(row[attr])
        current_record_id = row['id']
        
        # Use a set to store unique partner DataFrame indices for the current record
        # to avoid redundant processing within this record's token iteration.
        unique_partners_for_current_record = set()

        for token in record_tokens:
            df_indices_for_token = inverted_index_df_idx[token]
            for other_df_idx in df_indices_for_token:
                # Ensure we only consider pairs (current_record_id, other_record_id) where current_record_id < other_record_id
                # This handles self-pairs, duplicates, and the left < right constraint.
                other_record_id = X['id'][other_df_idx]
                if current_record_id < other_record_id:
                    unique_partners_for_current_record.add(other_df_idx)
        
        # Now, for each unique partner found for the current record, increment their co-occurrence score.
        for partner_df_idx in unique_partners_for_current_record:
            partner_record_id = X['id'][partner_df_idx]
            # The pair (current_record_id, partner_record_id) is already ordered (current_record_id < partner_record_id)
            candidate_cooccurrence_scores[(current_record_id, partner_record_id)] += 1

    # 3. Sort Candidate Pairs by Co-occurrence Score and Truncate
    #    Pairs with higher shared token counts are considered more similar.
    sorted_candidates_with_scores = sorted(candidate_cooccurrence_scores.items(), key=lambda item: item[1], reverse=True)

    final_candidate_pairs_real_ids = []
    for (id1, id2), _ in sorted_candidates_with_scores:
        final_candidate_pairs_real_ids.append((id1, id2))
        
        if len(final_candidate_pairs_real_ids) >= 2000000:
            break

    return final_candidate_pairs_real_ids

def save_output(X2_candidate_pairs):
    expected_cand_size_X2 = 2000000

    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    if len(X2_candidate_pairs) < expected_cand_size_X2:
        # Fill with unique dummy pairs if not enough candidates, ensuring id1 < id2 and no self-pairs
        num_dummies = expected_cand_size_X2 - len(X2_candidate_pairs)
        dummy_pairs = []
        # Generate unique negative IDs for dummy pairs to avoid conflicts and satisfy constraints
        for i in range(num_dummies):
            # Example: (-1, -2), (-3, -4), ...
            dummy_id1 = -(i * 2 + 1)
            dummy_id2 = -(i * 2 + 2)
            dummy_pairs.append((dummy_id2, dummy_id1)) # Ensure id1 < id2 (i.e., -2 < -1)
        X2_candidate_pairs.extend(dummy_pairs)

    output_df = pd.DataFrame(X2_candidate_pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)


# read the datasets
X2 = pd.read_csv("../SIGMOD_full_data/X2.csv") # Read from sample_data for dev


# perform blocking
X2_candidate_pairs = block_scalable(X2, attr="name")

# save results
save_output(X2_candidate_pairs)
