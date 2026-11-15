import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
import math

def clean_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

# read the datasets
X2 = pd.read_csv("../SIGMOD_full_data/X2.csv")

# Data cleaning for sorting
X2['brand'] = X2['brand'].fillna('')
X2['price'] = X2['price'].fillna(0)

# Clean the 'name' attribute
X2['name_cleaned'] = X2['name'].apply(clean_text)

# Build inverted index and count document frequencies
inverted_index = defaultdict(list)
doc_freq = defaultdict(int)
for i, row in tqdm(X2.iterrows(), total=X2.shape[0], desc="Building Inverted Index"):
    for token in row['name_cleaned']:
        inverted_index[token].append((row['id'], row['brand'], row['price']))
    for token in set(row['name_cleaned']):
        doc_freq[token] += 1

# Calculate IDF
num_docs = len(X2)
idf = {token: math.log(num_docs / (df + 1)) for token, df in doc_freq.items()}

# Sort tokens by IDF
sorted_tokens = sorted(idf.items(), key=lambda x: x[1], reverse=True)

candidate_pairs = set()
CANDIDATE_LIMIT = 2000000
WINDOW_SIZE = 10

for token, _ in tqdm(sorted_tokens, desc="Generating Candidates"):
    if len(candidate_pairs) >= CANDIDATE_LIMIT * 1.1: # Generate a few more to have buffer
        break
    
    postings = inverted_index[token]
    if len(postings) > 1 and len(postings) < 1000: # Heuristic to avoid very large blocks for now
        # Sort by brand and price to bring similar items together
        postings.sort(key=lambda x: (x[1], x[2]))

        if len(postings) < 100: # For small blocks, O(k^2) is fine
             for i in range(len(postings)):
                for j in range(i + 1, len(postings)):
                    id1, id2 = postings[i][0], postings[j][0]
                    if id1 < id2:
                        candidate_pairs.add((id1, id2))
                    else:
                        candidate_pairs.add((id2, id1))
        else: # Sliding window for larger blocks
            for i in range(len(postings)):
                for j in range(i + 1, min(i + WINDOW_SIZE, len(postings))):
                    id1, id2 = postings[i][0], postings[j][0]
                    if id1 < id2:
                        candidate_pairs.add((id1, id2))
                    else:
                        candidate_pairs.add((id2, id1))

print(f"\nGenerated {len(candidate_pairs)} candidate pairs.")

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Create a mapping from id to cleaned name tokens for Jaccard calculation
id_to_tokens = pd.Series(X2.name_cleaned.values, index=X2.id).to_dict()

def save_output(candidate_pairs, id_to_tokens):
    pairs_list = list(candidate_pairs)
    
    # Rank pairs by Jaccard similarity
    ranked_pairs = []
    for id1, id2 in tqdm(pairs_list, desc="Ranking pairs"):
        tokens1 = set(id_to_tokens.get(id1, []))
        tokens2 = set(id_to_tokens.get(id2, []))
        similarity = jaccard_similarity(tokens1, tokens2)
        ranked_pairs.append(((id1, id2), similarity))
    
    ranked_pairs.sort(key=lambda x: x[1], reverse=True)
    
    sorted_pairs_list = [pair for pair, sim in ranked_pairs]

    if len(sorted_pairs_list) > CANDIDATE_LIMIT:
        sorted_pairs_list = sorted_pairs_list[:CANDIDATE_LIMIT]

    if len(sorted_pairs_list) < CANDIDATE_LIMIT:
        sorted_pairs_list.extend([(0, 0)] * (CANDIDATE_LIMIT - len(sorted_pairs_list)))
    
    output_df = pd.DataFrame(sorted_pairs_list, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)

save_output(candidate_pairs, id_to_tokens)
print("Saved output.csv")
