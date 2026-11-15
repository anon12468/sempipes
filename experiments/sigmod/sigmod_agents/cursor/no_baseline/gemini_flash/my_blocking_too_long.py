
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
import random

def generate_ngrams(text, n=2):
    """Generate n-grams from a given text."""
    words = text.lower().split()
    if len(words) < n:
        return []
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

def block_optimized(X, attributes, small_bucket_threshold=50, snm_window_size=50):
    """
    Performs optimized blocking on the dataframe X using specified attributes.
    Uses full pairing for small buckets and Sorted Neighborhood Method (SNM) for large buckets
    to avoid O(k^2) operations.
    """
    inverted_index = defaultdict(set)
    # Using original 'id' from X as record_id directly
    for i, row in tqdm(X.iterrows(), total=X.shape[0], desc="Building Inverted Index"):
        record_id = row['id']
        for attr in attributes:
            if pd.notna(row[attr]):
                text = str(row[attr])
                # Use a combination of single words and bi-grams for blocking keys
                tokens = re.findall(r'\b\w+\b', text.lower()) # single words
                ngrams = generate_ngrams(text, n=2) # bi-grams
                blocking_keys = set(tokens + ngrams)

                for key in blocking_keys:
                    inverted_index[key].add(record_id)

    candidate_pairs = set()
    for key, record_ids in tqdm(inverted_index.items(), desc="Generating Candidates"):
        ids_list = sorted(list(record_ids))
        k = len(ids_list)

        if k > 1:
            if k <= small_bucket_threshold: # For small buckets, generate all pairs
                for i in range(k):
                    for j in range(i + 1, k):
                        if ids_list[i] < ids_list[j]:
                            candidate_pairs.add((ids_list[i], ids_list[j]))
            else: # For large buckets, use Sorted Neighborhood Method (SNM)
                for i in range(k):
                    for j in range(i + 1, min(i + 1 + snm_window_size, k)):
                        if ids_list[i] < ids_list[j]:
                            candidate_pairs.add((ids_list[i], ids_list[j]))

    return list(candidate_pairs)

def compute_jaccard_similarity_on_names(record1_name, record2_name):
    """Computes jaccard similarity between two sets of tokens from product names."""
    s1 = set(str(record1_name).lower().split()) if pd.notna(record1_name) else set()
    s2 = set(str(record2_name).lower().split()) if pd.notna(record2_name) else set()
    if not s1 and not s2:
        return 0.0
    union_len = len(s1.union(s2))
    if union_len == 0: # Avoid division by zero if both sets are empty
        return 0.0
    return len(s1.intersection(s2)) / union_len

def save_output(X2_candidate_pairs, id_to_record_map, output_filename="output.csv", expected_cand_size_X2=2000000):
    """
    Saves the candidate set for X2 to a CSV file, ensuring exact size.
    Sorts by Jaccard similarity if more candidates than expected.
    """
    print(f"Initial number of candidates: {len(X2_candidate_pairs)}")

    if len(X2_candidate_pairs) > expected_cand_size_X2:
        print(f"More candidates than expected ({len(X2_candidate_pairs)} > {expected_cand_size_X2}). Sorting by Jaccard similarity...")
        jaccard_similarities = []
        candidate_pairs_with_jaccard = []

        for lid, rid in tqdm(X2_candidate_pairs, desc="Computing Jaccard similarities"):
            name1 = id_to_record_map[lid]['name']
            name2 = id_to_record_map[rid]['name']
            sim = compute_jaccard_similarity_on_names(name1, name2)
            candidate_pairs_with_jaccard.append((sim, (lid, rid)))

        # Sort candidate pairs by jaccard similarity in descending order
        candidate_pairs_with_jaccard.sort(key=lambda x: x[0], reverse=True)
        X2_candidate_pairs_sorted = [pair for sim, pair in candidate_pairs_with_jaccard]

        final_candidates = X2_candidate_pairs_sorted[:expected_cand_size_X2]
    else:
        final_candidates = X2_candidate_pairs

    if len(final_candidates) < expected_cand_size_X2:
        print(f"Fewer candidates than expected ({len(final_candidates)} < {expected_cand_size_X2}). Padding with (0,0)...")
        final_candidates.extend([(0, 0)] * (expected_cand_size_X2 - len(final_candidates)))

    output_df = pd.DataFrame(final_candidates, columns=["lid", "rid"])
    output_df.to_csv(output_filename, index=False)
    print(f"Output saved to {output_filename} with {len(output_df)} pairs.")

# Main execution (for X2 blocking)
if __name__ == "__main__":
    # It's crucial to use the full X2.csv for the final run, but sample_data/X2.csv for dev.
    # The full X2.csv was not found in 'data/X2.csv'. Using sample_data/X2.csv for now.
    X2 = pd.read_csv("/Users/USER/SIGMOD_full_data/X2.csv") # Using sample data for initial testing

    # Pre-build id_to_record_map for efficient lookup during Jaccard calculation
    id_to_record_map = X2.set_index('id').to_dict('index')

    # Use 'name' and 'brand' attributes for blocking
    # small_bucket_threshold: buckets smaller than this will have all pairs generated.
    # snm_window_size: for larger buckets, compare each record with its N neighbors.
    # These parameters will need careful tuning on the FULL dataset to hit 2M candidates and maximize recall.
    X2_candidate_pairs = block_optimized(X2, attributes=["name", "brand"], small_bucket_threshold=50, snm_window_size=50)

    save_output(X2_candidate_pairs, id_to_record_map, output_filename="output.csv")
