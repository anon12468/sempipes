import pandas as pd
from collections import defaultdict
import re
from itertools import combinations
import random

def preprocess(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens

def build_inverted_index(data, columns):
    inverted_index = defaultdict(list)
    for index, row in data.iterrows():
        for col in columns:
            tokens = preprocess(row[col])
            for token in tokens:
                inverted_index[token].append(row['id'])
    return inverted_index

def generate_candidate_pairs(inverted_index, max_pairs_per_token=1000):
    candidate_pairs = set()
    for token in inverted_index:
        ids = sorted(list(set(inverted_index[token])))
        if len(ids) > 1:
            if len(ids) > max_pairs_per_token:
                # Down-sample to avoid quadratic complexity
                ids = random.sample(ids, max_pairs_per_token)
            
            # Efficient pair generation using a sliding window approach
            window_size = 5
            for i in range(len(ids)):
                for j in range(i + 1, min(i + window_size, len(ids))):
                    pair = tuple(sorted((ids[i], ids[j])))
                    candidate_pairs.add(pair)
    return candidate_pairs

def save_output(candidate_pairs, output_file, num_pairs):
    if len(candidate_pairs) > num_pairs:
        candidate_pairs = random.sample(list(candidate_pairs), num_pairs)
    
    output_df = pd.DataFrame(list(candidate_pairs), columns=['lid', 'rid'])
    
    if len(output_df) < num_pairs:
        padding = pd.DataFrame([(0, 0)] * (num_pairs - len(output_df)), columns=['lid', 'rid'])
        output_df = pd.concat([output_df, padding], ignore_index=True)

    output_df.to_csv(output_file, index=False)

# Load the data
try:
    X2 = pd.read_csv("../SIGMOD_full_data/X2.csv")
except FileNotFoundError:
    X2 = pd.read_csv("../SIGMOD_full_data/X2.csv")

# Build inverted index
columns_to_index = ['name', 'brand', 'description']
inverted_index = build_inverted_index(X2, columns_to_index)

# Generate candidate pairs
candidate_pairs = generate_candidate_pairs(inverted_index)

# Save the output
save_output(candidate_pairs, "output.csv", 2000000)
