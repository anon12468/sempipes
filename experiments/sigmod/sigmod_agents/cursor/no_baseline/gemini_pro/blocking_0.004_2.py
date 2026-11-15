from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re

def clean_and_tokenize(text):
    """
    Cleans and tokenizes the input text.
    - Converts to lowercase
    - Extracts alphanumeric tokens
    """
    if not isinstance(text, str):
        return []
    return re.findall(r'[a-z0-9]+', text.lower())

def generate_candidates(X, limit):
    """
    Generates candidate pairs using a scalable blocking strategy.
    """
    print("Building inverted index...")
    token_to_ids = defaultdict(list)
    for i, row in tqdm(X.iterrows(), total=len(X)):
        name = row['name']
        tokens = clean_and_tokenize(name)
        for token in tokens:
            token_to_ids[token].append(i)

    # Sort posting lists by length (process rarer tokens first)
    # and filter out very common tokens to avoid noise.
    print("Sorting and filtering blocks...")
    all_posting_lists = token_to_ids.values()
    
    # Filter out posting lists that are too short (no pairs) or too long (likely stopwords)
    # The max length of 5000 is a heuristic to balance recall and performance.
    filtered_posting_lists = [lst for lst in all_posting_lists if 1 < len(lst) < 5000]
    
    # Sorting by length processes more discriminative blocks first.
    sorted_posting_lists = sorted(filtered_posting_lists, key=len)

    candidate_pairs = set()
    window_size = 10  # Sliding window size

    print("Generating candidate pairs...")
    for ids in tqdm(sorted_posting_lists):
        if len(candidate_pairs) >= limit:
            break

        sorted_ids = sorted(ids) # Sort internal list of doc IDs

        for i in range(len(sorted_ids)):
            current_id_index = sorted_ids[i]
            
            # Create pairs with subsequent ids in the window
            for j in range(i + 1, min(i + window_size, len(sorted_ids))):
                next_id_index = sorted_ids[j]

                # Get real IDs from the dataframe
                real_id1 = X.loc[current_id_index, 'id']
                real_id2 = X.loc[next_id_index, 'id']

                # Ensure consistent order (smaller id first) and no self-pairs
                if real_id1 < real_id2:
                    candidate_pairs.add((real_id1, real_id2))
                elif real_id2 < real_id1:
                    candidate_pairs.add((real_id2, real_id1))

                if len(candidate_pairs) >= limit:
                    break
            if len(candidate_pairs) >= limit:
                break
    
    return list(candidate_pairs)

def save_output(X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 2000000 pairs for dataset X2
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 2000000 pairs for dataset X2
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X2_candidate_pairs
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)


# read the datasets
print("Reading data...")
X2 = pd.read_csv("../SIGMOD_full_data/X2.csv")
X2 = X2.astype(str) # Ensure all columns are strings

# perform blocking
print("Starting blocking process...")
X2_candidate_pairs = generate_candidates(X2, limit=2000000)

# save results
print(f"Generated {len(X2_candidate_pairs)} candidate pairs. Saving to output.csv...")
save_output(X2_candidate_pairs)
print("Done.")

