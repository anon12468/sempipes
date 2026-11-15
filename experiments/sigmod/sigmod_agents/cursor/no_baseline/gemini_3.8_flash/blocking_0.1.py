import os
import sys
import time
import re
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd

# Stopwords representing generic linguistic tokens, web entities, and general media types
STOPWORDS = {
    'usb', 'sd', 'sdhc', 'sdxc', 'microsd', 'gb', 'go', 'mb', 'mo', 'tb', 'to',
    'class', 'uhs', 'card', 'carte', 'flash', 'de', 'la', 'le', 'en', 'un', 'du',
    'et', 'und', 'the', 'with', 'for', 'pour', 'mit', 'speed', 'read', 'write',
    'memory', 'memoria', 'speicherkarte', 'drive', 'stick', 'cle', 'chiavetta',
    'pendrive', 'direct', 'tesco', 'fnac', 'amazon', 'sous', 'blister', 'negro',
    'noir', 'black', 'white', 'blanc', 'blanco', 'gris', 'grey', 'silver', 'silber',
    'type', 'gen', 'rot', 'red', 'rojo', 'rouge', 'blau', 'blue', 'bleu', 'azul',
    'vesa', 'nbsp', 'quot', 'amp', 'null', 'nan'
}

KNOWN_BRANDS = [
    'sandisk', 'sony', 'kingston', 'lexar', 'samsung', 'toshiba',
    'intenso', 'pny', 'transcend', 'panasonic', 'apple'
]


def extract_tokens(text, brand_hint=''):
    """
    Extract informative tokens, normalized capacities, brands, model codes,
    and composite 2-grams from a product record.
    """
    text = str(text).lower()
    text = re.sub(r'&[a-z0-9]+;', ' ', text)

    # Normalize capacities (e.g., '16 gb', '16go' -> '16gb')
    caps = re.findall(r'\b(\d+)\s*(?:gb|go|g|tb|to)\b', text)
    norm_caps = set(f'{c}gb' for c in caps)
    text = re.sub(r'\b(\d+)\s*(?:gb|go|g)\b', r' \1gb ', text)
    text = re.sub(r'\b(\d+)\s*(?:tb|to)\b', r' \1tb ', text)

    # Detect brand
    brand = ''
    brand_hint = str(brand_hint).lower() if brand_hint else ''
    for b in KNOWN_BRANDS:
        if b in brand_hint or b in text:
            brand = b
            break

    # Extract words
    words = re.findall(r'[a-z0-9]+', text)
    filtered = [w for w in words if len(w) >= 2 and w not in STOPWORDS]

    # Extract model alphanumeric codes (e.g., cz80, s70, c20m, 3503470)
    models = [
        w for w in filtered
        if (any(c.isdigit() for c in w) and any(c.isalpha() for c in w) and len(w) >= 3
            and not w.endswith('gb') and not w.endswith('mb') and not w.endswith('tb'))
        or (w.isdigit() and len(w) >= 5)
    ]

    toks = set(filtered) | norm_caps

    # Composite tokens
    for c in norm_caps:
        if brand:
            toks.add(f'{brand}_{c}')
        for m in models[:2]:
            toks.add(f'{m}_{c}')
    if brand:
        for m in models[:2]:
            toks.add(f'{brand}_{m}')

    # Word 2-grams
    for idx in range(len(filtered) - 1):
        if filtered[idx] != filtered[idx + 1]:
            toks.add(f'{filtered[idx]}_{filtered[idx + 1]}')

    return toks


def block_with_attr(X, attr="name"):
    """
    Highly scalable blocking function designed for up to 2 million records.
    Combines:
    1. Query-driven Inverted Index candidate search for rare/discriminative tokens (O(1) per record)
    2. Multi-hash MinHash Sorted Neighborhood Method (SNM) with window comparison (O(N log N + N*w))
    Guarantees strictly linear work inside any bucket and strictly bounded candidate generation.
    """
    N = len(X)
    ids = X['id'].to_numpy(dtype=np.int64)
    names = X[attr].fillna('').astype(str).tolist()
    brands = X['brand'].fillna('').astype(str).tolist() if 'brand' in X.columns else [''] * N

    # Step 1: Token extraction
    doc_tokens = [extract_tokens(names[i], brands[i]) for i in range(N)]

    pair_counts = defaultdict(int)

    # Step 2: Query-time inverted index for rare discriminative tokens
    # Safeguard: discard tokens that appear in > 30 documents (prevents frequent-token explosion)
    inv_index = defaultdict(list)
    for i, toks in enumerate(doc_tokens):
        for t in toks:
            inv_index[t].append(i)

    MAX_POSTING = 30
    inv_filtered = {t: pl for t, pl in inv_index.items() if 1 < len(pl) <= MAX_POSTING}
    k_query = 8 if N < 50000 else 2

    for i in range(N):
        scores = defaultdict(int)
        for t in doc_tokens[i]:
            if t in inv_filtered:
                for j in inv_filtered[t]:
                    if j != i:
                        scores[j] += 1
        if scores:
            top_j = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k_query]
            for j, s in top_j:
                u, v = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                packed = (u << 32) | v
                pair_counts[packed] += s

    # Step 3: Vectorized MinHash Sorted Neighborhood Method (SNM)
    num_hashes = 48
    prime = 2147483647
    rng = np.random.RandomState(42)
    coeff_a = rng.randint(1, prime, size=num_hashes, dtype=np.int64)
    coeff_b = rng.randint(0, prime, size=num_hashes, dtype=np.int64)

    all_toks = sorted(list(set().union(*doc_tokens)))
    tok_to_id = {t: i for i, t in enumerate(all_toks)}
    tok_hashes = np.array(
        [int(hashlib.md5(t.encode('utf-8')).hexdigest()[:8], 16) for t in all_toks],
        dtype=np.int64
    )

    sigs = np.empty((N, num_hashes), dtype=np.int64)
    for i in range(N):
        indices = [tok_to_id[t] for t in doc_tokens[i] if t in tok_to_id]
        if indices:
            th = tok_hashes[indices]
            vals = (coeff_a[:, None] * th[None, :] + coeff_b[:, None]) % prime
            sigs[i] = vals.min(axis=1)
        else:
            sigs[i] = prime

    w_range = [1, 2, 3] if N < 50000 else [1, 2]
    max_cands = 3500000

    for k in range(num_hashes):
        col = sigs[:, k]
        order = np.argsort(col)
        sorted_col = col[order]
        for w in w_range:
            mask = (sorted_col[:-w] == sorted_col[w:]) & (sorted_col[:-w] < prime)
            u_idx = order[:-w][mask]
            v_idx = order[w:][mask]
            for u, v in zip(u_idx, v_idx):
                u_id, v_id = (ids[u], ids[v]) if ids[u] < ids[v] else (ids[v], ids[u])
                packed = (u_id << 32) | v_id
                pair_counts[packed] += 1

        # Periodic memory safeguard: prune low-count pairs if total candidates exceed threshold
        if len(pair_counts) > max_cands:
            pruned = {p: c for p, c in pair_counts.items() if c > 1}
            if len(pruned) >= 2000000:
                pair_counts = defaultdict(int, pruned)

    # Step 4: Extract top candidate pairs ranked by similarity score
    target_count = 2000000
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    cand_pairs = []
    for packed, _ in sorted_pairs[:target_count]:
        cand_pairs.append((packed >> 32, packed & 0xFFFFFFFF))

    # If N >= 2001 (e.g. full dataset) and we need to fill up to target_count:
    # fill with valid, distinct pairs with lid < rid from the dataset
    if len(cand_pairs) < target_count and N >= 2001:
        seen = set(cand_pairs)
        for step in [1, 2, 3, 4]:
            for i in range(N - step):
                u, v = (ids[i], ids[i + step]) if ids[i] < ids[i + step] else (ids[i + step], ids[i])
                if (u, v) not in seen:
                    cand_pairs.append((u, v))
                    seen.add((u, v))
                    if len(cand_pairs) == target_count:
                        break
            if len(cand_pairs) == target_count:
                break

    # If N < 2001 (small sample subset where total distinct pairs < target_count):
    # pad remainder to ensure output.csv has exactly target_count rows
    if len(cand_pairs) < target_count:
        cand_pairs.extend([(0, 0)] * (target_count - len(cand_pairs)))

    return cand_pairs


def save_output(candidate_pairs, output_path="output.csv"):
    """
    Save candidate pairs to output CSV with columns: lid, rid.
    Guarantees exactly 2,000,000 rows.
    """
    expected_cand_size = 2000000

    if len(candidate_pairs) > expected_cand_size:
        candidate_pairs = candidate_pairs[:expected_cand_size]
    elif len(candidate_pairs) < expected_cand_size:
        candidate_pairs.extend([(0, 0)] * (expected_cand_size - len(candidate_pairs)))

    lids = [p[0] for p in candidate_pairs]
    rids = [p[1] for p in candidate_pairs]

    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        table = pa.Table.from_arrays([pa.array(lids), pa.array(rids)], names=["lid", "rid"])
        pacsv.write_csv(table, output_path)
    except Exception:
        df = pd.DataFrame({"lid": lids, "rid": rids})
        df.to_csv(output_path, index=False)


def main():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        input_path = sys.argv[1]
    elif os.path.exists("../SIGMOD_full_data/X2.csv"):
        input_path = "../SIGMOD_full_data/X2.csv"
    else:
        input_path = "../SIGMOD_full_data/X2.csv"

    output_path = sys.argv[2] if len(sys.argv) > 2 else "../SIGMOD_full_data/output.csv"

    print(f"Reading input from {input_path}...")
    X2 = pd.read_csv(input_path)

    print("Running blocking algorithm...")
    start_time = time.time()
    candidates = block_with_attr(X2, attr="name")
    elapsed = time.time() - start_time
    print(f"Blocking finished in {elapsed:.2f} seconds.")

    print(f"Saving exactly {len(candidates)} candidate pairs to {output_path}...")
    save_output(candidates, output_path=output_path)
    print("Done!")


if __name__ == "__main__":
    main()
