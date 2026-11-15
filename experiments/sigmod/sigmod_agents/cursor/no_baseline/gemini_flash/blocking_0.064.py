from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re

def normalize_name(name):
    """Normalizes the product name for consistent blocking."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def block_with_attr(X, attr="name", num_neighbors=5):
    """
    Performs blocking using a sorted neighborhood approach to avoid O(k^2) pairing.
    :param X: dataframe
    :param attr: attribute used for blocking
    :param num_neighbors: number of neighbors to consider for pairing in sorted neighborhood
    :return: candidate set of real ID tuple pairs
    """

    normalized_names = X[attr].apply(normalize_name)
    id_to_normalized_name = dict(zip(X.index, normalized_names))

    # Create an inverted index using tokens from normalized names
    token_to_ids = defaultdict(list)
    for i, name in tqdm(normalized_names.items(), desc="Building inverted index"):
        tokens = name.split()
        for token in tokens:
            if token:
                token_to_ids[token].append(i)

    candidate_pairs = set()
    for token in tqdm(token_to_ids, desc="Generating candidate pairs"):
        ids = sorted(list(set(token_to_ids[token])))  # Sort and unique IDs for consistent neighborhood
        
        # Apply sorted neighborhood approach
        for i in range(len(ids)):
            for j in range(i + 1, min(i + 1 + num_neighbors, len(ids))):
                id1, id2 = ids[i], ids[j]
                # Ensure id1 < id2 for consistent pairing and no self-pairs
                if id1 < id2:
                    candidate_pairs.add((id1, id2))
                else:
                    candidate_pairs.add((id2, id1))
    
    # Calculate Jaccard similarity and filter/sort
    jaccard_similarities = []
    candidate_pairs_real_ids = []

    for id1, id2 in tqdm(candidate_pairs, desc="Calculating Jaccard similarities"):
        name1 = str(X[attr][id1])
        name2 = str(X[attr][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        
        # Avoid division by zero if both sets are empty
        denominator = max(len(s1), len(s2))
        similarity = len(s1.intersection(s2)) / denominator if denominator > 0 else 0
        jaccard_similarities.append(similarity)
        
        # Store real IDs, ensuring id1 < id2
        real_id1 = X['id'][id1]
        real_id2 = X['id'][id2]
        if real_id1 < real_id2:
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

    # Sort candidate pairs by Jaccard similarity in descending order
    sorted_pairs_with_similarity = sorted(
        zip(jaccard_similarities, candidate_pairs_real_ids),
        key=lambda x: x[0],
        reverse=True
    )
    
    final_candidate_pairs = [pair for _, pair in sorted_pairs_with_similarity]
    return final_candidate_pairs


def save_output(X2_candidate_pairs):
    """Saves the candidate pairs to output.csv, ensuring exactly 2,000,000 rows."""
    expected_cand_size_X2 = 2000000

    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]
    elif len(X2_candidate_pairs) < expected_cand_size_X2:
        # Fill with (0, 0) pairs if not enough candidates are generated
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    output_df = pd.DataFrame(X2_candidate_pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)


# Read the dataset
X2 = pd.read_csv("../SIGMOD_full_data/X2.csv")

# Perform blocking
X2_candidate_pairs = block_with_attr(X2, attr="name")

# Save results
save_output(X2_candidate_pairs)
