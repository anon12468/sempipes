import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
import time

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    # Expanded stopwords list
    stopwords = {'usb', 'sd', 'gb', 'de', 'mb', 's', 'go', 'pro', 'card', 'carte', 'class', 'extreme', 'ultra', 'micro', 'drive', 'flash', 'with', 'to', 'for', 'in', 'and', 'the', 'a', 'an'}
    return [token for token in tokens if token.isalnum() and token not in stopwords]

def generate_pairs_sliding_window(ids, window_size):
    """Generates candidate pairs using a sliding window to avoid O(k^2) complexity."""
    pairs = set()
    ids = sorted(list(set(ids)))
    if len(ids) > 1:
        for i in range(len(ids)):
            for j in range(i + 1, min(i + window_size + 1, len(ids))):
                pair = tuple(sorted((ids[i], ids[j])))
                pairs.add(pair)
    return pairs


def main():
    start_time = time.time()
    
    # Load data
    try:
        X = pd.read_csv("../SIGMOD_full_data/X2.csv")
    except FileNotFoundError:
        print("../SIGMOD_full_data/X2.csv not found, using ../SIGMOD_full_data/X2.csv")
        X = pd.read_csv("../SIGMOD_full_data/X2.csv")

    # Clean and tokenize names
    X['name_clean'] = X['name'].apply(clean_text)

    # Inverted Index
    inverted_index = defaultdict(list)
    for i, tokens in enumerate(X['name_clean']):
        for token in tokens:
            inverted_index[token].append(i)

    # Candidate pair generation
    candidate_pairs = set()
    max_pairs = 2000000
    window_size = 5 # Small window to keep pairs manageable for frequent tokens

    # Sort tokens by frequency to process less frequent (more informative) ones first
    sorted_tokens = sorted(inverted_index.keys(), key=lambda t: len(inverted_index[t]))

    for token in tqdm(sorted_tokens, desc="Generating Candidates"):
        if len(candidate_pairs) >= max_pairs:
            break
        
        ids = inverted_index[token]
        
        # Skip very common tokens/blocks if we are getting too many pairs
        if len(ids) > 1000:
             continue

        new_pairs = generate_pairs_sliding_window(ids, window_size)
        
        # Add new pairs, making sure not to exceed max_pairs
        if len(candidate_pairs) + len(new_pairs) > max_pairs:
            needed = max_pairs - len(candidate_pairs)
            candidate_pairs.update(list(new_pairs)[:needed])
        else:
            candidate_pairs.update(new_pairs)

    print(f"Generated {len(candidate_pairs)} candidate pairs.")

    # Convert to real IDs and format for output
    candidate_pairs_real_ids = []
    id_map = X['id'].to_dict()

    for id1_idx, id2_idx in tqdm(candidate_pairs, desc="Mapping to real IDs"):
        real_id1 = id_map[id1_idx]
        real_id2 = id_map[id2_idx]
        # Ensure order and no self-pairs
        if real_id1 < real_id2:
            candidate_pairs_real_ids.append((real_id1, real_id2))
        elif real_id2 < real_id1:
            candidate_pairs_real_ids.append((real_id2, real_id1))

    # Final output formatting
    output_df = pd.DataFrame(list(set(candidate_pairs_real_ids)), columns=["left_instance_id", "right_instance_id"])
    
    if len(output_df) > max_pairs:
        output_df = output_df.sample(n=max_pairs, random_state=42).sort_values(by=['left_instance_id', 'right_instance_id'])
    elif len(output_df) < max_pairs:
        padding = pd.DataFrame([(0,0)] * (max_pairs - len(output_df)), columns=["left_instance_id", "right_instance_id"])
        output_df = pd.concat([output_df, padding])

    output_df.to_csv("output.csv", index=False)
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
