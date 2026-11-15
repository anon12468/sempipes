from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re
import heapq

def generate_blocking_keys(text):
    """
    Generates blocking keys from a given text.
    Improved: lowercasing, removing non-alphanumeric, splitting into words.
    """
    if not isinstance(text, str):
        return []
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters, replacing them with spaces
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    # Split into words and return unique words as keys
    return list(set(text.split()))

def calculate_jaccard_similarity(name1, name2):
    """
    Calculates Jaccard similarity between two strings.
    """
    s1 = set(name1.lower().split())
    s2 = set(name2.lower().split())
    if not s1 or not s2:
        return 0.0
    # Handle case where both sets are empty to avoid ZeroDivisionError
    union_len = len(s1.union(s2))
    if union_len == 0:
        return 0.0
    return len(s1.intersection(s2)) / union_len

def block_with_sorted_neighborhood(X, attr, window_size=5, target_cand_size=2000000):
    """
    Performs blocking using a sorted neighborhood approach and maintains a min-heap
    of the top `target_cand_size` candidate pairs based on Jaccard similarity.
    :param X: dataframe
    :param attr: attribute used for blocking
    :param window_size: The size of the neighborhood window for comparison
    :param target_cand_size: The target number of candidate pairs to output
    :return: A list of the top `target_cand_size` candidate pairs (real_id1, real_id2)
    """
    # Min-heap to store (jaccard_similarity, (id1, id2)) for top candidates
    candidate_heap = [] # Stores (similarity, pair_tuple)

    # Inverted index: key -> list of (record_id, original_index_in_X)
    inverted_index = defaultdict(list)

    print("Building inverted index...")
    for original_index, row in tqdm(X.iterrows(), total=X.shape[0]):
        record_id = row['id']
        name_attr = row[attr]
        blocking_keys = generate_blocking_keys(name_attr)
        for key in blocking_keys:
            inverted_index[key].append((record_id, original_index))

    print("Generating candidate pairs using sorted neighborhood with Jaccard filtering...")
    for key in tqdm(inverted_index):
        sorted_bucket = sorted(inverted_index[key], key=lambda x: x[0]) # x[0] is record_id

        for i in range(len(sorted_bucket)):
            for j in range(i + 1, min(i + 1 + window_size, len(sorted_bucket))):
                id1_real, original_index1 = sorted_bucket[i]
                id2_real, original_index2 = sorted_bucket[j]

                # Ensure no self-pairs
                if id1_real == id2_real:
                    continue

                # Ensure consistent ordering for pairs (id1 < id2)
                pair = tuple(sorted((id1_real, id2_real)))

                # Calculate Jaccard similarity
                name1 = str(X.loc[original_index1, attr])
                name2 = str(X.loc[original_index2, attr])
                jaccard_sim = calculate_jaccard_similarity(name1, name2)

                # Push to heap, maintaining size limit
                if len(candidate_heap) < target_cand_size:
                    heapq.heappush(candidate_heap, (jaccard_sim, pair))
                else:
                    # If current similarity is greater than the smallest in heap, replace it
                    if jaccard_sim > candidate_heap[0][0]: # candidate_heap[0][0] is the smallest similarity
                        heapq.heapreplace(candidate_heap, (jaccard_sim, pair))

    # Extract pairs from the heap. These are already the top `target_cand_size` by similarity.
    # Sort by similarity in descending order to output the most similar pairs first.
    final_candidate_pairs_with_sim = sorted(candidate_heap, key=lambda x: x[0], reverse=True)
    final_candidate_pairs = [pair for sim, pair in final_candidate_pairs_with_sim]

    return final_candidate_pairs

def save_output(X2_candidate_pairs, expected_cand_size_X2=2000000):
    """
    Saves the candidate pairs to output.csv, ensuring exactly `expected_cand_size_X2` pairs.
    """
    # The block_with_sorted_neighborhood function now ensures the exact size,
    # so this truncation/padding logic is primarily for robustness or if called directly.
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]
    elif len(X2_candidate_pairs) < expected_cand_size_X2:
        # Pad with (0,0) if fewer than expected. 
        # These are dummy pairs to meet the exact output size requirement.
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    output_df = pd.DataFrame(X2_candidate_pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)


# read the datasets
X2 = pd.read_csv("../SIGMOD_full_data/X2.csv")

# perform blocking
X2_candidate_pairs = block_with_sorted_neighborhood(X2, attr="name", window_size=5, target_cand_size=2000000)

print(f"Generated {len(X2_candidate_pairs)} candidate pairs after Jaccard filtering.")

# save results
save_output(X2_candidate_pairs)