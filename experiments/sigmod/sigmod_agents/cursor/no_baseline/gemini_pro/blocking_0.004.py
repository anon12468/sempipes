import pandas as pd
from collections import defaultdict
import re
from tqdm import tqdm

def tokenize(text):
    """
    A simple tokenizer that cleans and splits text into tokens.
    """
    if not isinstance(text, str):
        return []
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Split into words
    tokens = text.split()
    return tokens

def sorted_neighborhood(block, window_size):
    """
    Generates candidate pairs from a block using the Sorted Neighborhood Method.
    """
    if len(block) < 2:
        return []

    # The block is already sorted by a secondary key (the record's name)
    # before being passed to this function.
    
    candidate_pairs = []
    
    for i in range(len(block)):
        # Define the end of the window
        window_end = min(i + 1 + window_size, len(block))
        # Generate pairs within the window
        for j in range(i + 1, window_end):
            # Ensure pair is ordered (id1 < id2)
            pair = tuple(sorted((block[i][0], block[j][0])))
            candidate_pairs.append(pair)
            
    return candidate_pairs

def block_dataset(X, window_size=5, max_block_size=500):
    """
    Performs blocking on the dataset using a token-based inverted index
    and the Sorted Neighborhood Method.
    """
    # Build inverted index
    token_to_records = defaultdict(list)
    for i, row in tqdm(X.iterrows(), total=X.shape[0], desc="Building index"):
        name = row['name']
        tokens = tokenize(name)
        for token in tokens:
            token_to_records[token].append((i, name)) # Store index and name

    # Generate candidate pairs
    candidate_pairs = set()
    for token in tqdm(token_to_records, desc="Generating pairs"):
        block = token_to_records[token]
        
        # Filter out very large blocks (likely stopwords or uninformative tokens)
        if len(block) > max_block_size or len(block) < 2:
            continue
            
        # Sort the block by the name (secondary key)
        block.sort(key=lambda x: x[1])

        # Apply Sorted Neighborhood Method
        pairs_from_block = sorted_neighborhood(block, window_size)
        candidate_pairs.update(pairs_from_block)

    return list(candidate_pairs)


def save_output(candidate_pairs, X):
    """
    Saves the candidate set to output.csv, ensuring it has the correct size and format.
    """
    expected_cand_size = 2000000
    
    # Convert local indices to real instance ids
    candidate_pairs_real_ids = []
    for id1, id2 in candidate_pairs:
        real_id1 = X.iloc[id1]['id']
        real_id2 = X.iloc[id2]['id']
        # Ensure correct order
        if real_id1 < real_id2:
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))
    
    # In a real scenario with a large number of candidates, we would sort by a similarity
    # metric (like Jaccard) before truncating. For this improved blocking, we assume
    # the quality is high enough to truncate directly if needed.
    
    # Remove duplicates that might have been created by id swapping
    candidate_pairs_real_ids = list(set(candidate_pairs_real_ids))

    print(f"Generated {len(candidate_pairs_real_ids)} unique candidate pairs.")

    # Truncate or pad to meet the exact output size requirement
    if len(candidate_pairs_real_ids) > expected_cand_size:
        # A more advanced strategy could be to rank pairs before truncating.
        # For now, we take a sample to meet the size requirement.
        # This is better than arbitrary truncation.
        # To make it deterministic, we sort before taking the first N.
        candidate_pairs_real_ids.sort()
        final_pairs = candidate_pairs_real_ids[:expected_cand_size]
    else:
        final_pairs = candidate_pairs_real_ids
        # Pad if necessary
        padding = [(0, 0)] * (expected_cand_size - len(final_pairs))
        final_pairs.extend(padding)

    output_df = pd.DataFrame(final_pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)
    print(f"Saved {len(final_pairs)} pairs to output.csv")


if __name__ == "__main__":
    # Read the dataset
    print("Reading data...")
    X2 = pd.read_csv("../SIGMOD_full_data/X2.csv")

    # Perform blocking
    print("Performing blocking...")
    # These parameters can be tuned for performance and recall
    candidate_pairs = block_dataset(X2, window_size=10, max_block_size=100)

    # Save results
    print("Saving results...")
    save_output(candidate_pairs, X2)

    print("Done.")

