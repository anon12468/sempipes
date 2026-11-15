import os
import sys
import re
import html
import time
from collections import defaultdict
import pandas as pd
import numpy as np

# Canonical known brands for electronic products in X2
KNOWN_BRANDS = [
    'sandisk', 'sony', 'kingston', 'lexar', 'toshiba', 'intenso',
    'samsung', 'transcend', 'pny', 'corsair', 'verbatim', 'panasonic',
    'hp', 'seagate', 'wd', 'western digital', 'crucial', 'patriot',
    'silicon power', 'adata', 'philips', 'delkin', 'integral', 'kioxia'
]

# Frequent product series / family names in X2
FAMILY_KEYWORDS = [
    'cruzer glide', 'cruzer blade', 'cruzer fit', 'cruzer dial', 'cruzer switch',
    'cruzer snap', 'cruzer edge', 'cruzer force', 'cruzer extreme', 'cruzer orbit',
    'ultra plus', 'ultra dual', 'ultra fit', 'extreme pro', 'extreme plus',
    'jumpdrive s70', 'jumpdrive s75', 'jumpdrive s50', 'jumpdrive s25', 'jumpdrive s35',
    'jumpdrive v10', 'jumpdrive v20', 'jumpdrive v30', 'jumpdrive c20c', 'jumpdrive c20m',
    'jumpdrive p20', 'professional 633x', 'professional 1000x',
    'datatraveler 100 g3', 'datatraveler 101 g2', 'datatraveler g4', 'datatraveler se9',
    'datatraveler microduo', 'datatraveler', 'hyperx savage', 'hyperx fury', 'hyperx predator',
    'canvas select', 'canvas go', 'canvas react',
    'transmemory u202', 'transmemory u301', 'transmemory u302', 'transmemory u363',
    'transmemory', 'exceria pro n101', 'exceria pro n401', 'exceria pro', 'exceria n302',
    'exceria m302', 'exceria',
    'basic line', 'rainbow line', 'premium line', 'speed line', 'alu line', 'micro line',
    'microvault click', 'microvault usm', 'microvault',
    'evo plus', 'evo select', 'evo', 'pro plus', 'pro select', 'pro endurance',
    'attache 4', 'attache'
]

# Multilingual noise words (German, French, Italian, Spanish, English e-commerce terms)
NOISE_WORDS = {
    'carte', 'memoire', 'mémoire', 'memoria', 'scheda', 'karta', 'speicherkarte', 'card',
    'stick', 'cle', 'clé', 'chiavetta', 'pendrive', 'drive', 'flash', 'usb',
    'de', 'la', 'le', 'les', 'en', 'pour', 'avec', 'zu', 'und', 'von', 'mit',
    'sur', 'dans', 'achat', 'rayon', 'blister', 'direct', 'tesco', 'fnac', 'amazon',
    'online', 'shop', 'noir', 'blanc', 'schwarz', 'weiss', 'black', 'white',
    'blanche', 'argento', 'zilver', 'silver', 'rouge', 'rot', 'red', 'azul',
    'bleu', 'blau', 'blue', 'verde', 'vert', 'green', 'grise', 'gris', 'grau',
    'grey', 'gray', 'viola', 'violet', 'lila', 'purple',
    'speicher', 'laufwerk', 'disque', 'disco', 'unita', 'unidad', 'lecteur',
    'sous', 'sans', 'con', 'sin', 'ohne',
    'frustfreie', 'verpackung', 'packaging', 'original', 'retail', 'bulk',
    'mb', 's', 'sek', 'mo', 'seconde', 'r', 'w', 'speed', 'vitesse'
}

CAP_STANDARD = {'1gb', '2gb', '4gb', '8gb', '16gb', '32gb', '64gb', '128gb', '256gb', '512gb', '1tb', '2tb', '4tb'}


def preprocess_row(name, brand_col):
    """
    Extract structured attributes and normalized tokens from a product row.
    """
    text = (str(name) + ' ' + (str(brand_col) if pd.notna(brand_col) else '')).lower()
    text = html.unescape(text)

    # Normalize common compounds
    text = re.sub(r'\bdata\s+traveler\b', 'datatraveler', text)
    text = re.sub(r'\bjump\s+drive\b', 'jumpdrive', text)
    text = re.sub(r'\btrans\s+memory\b', 'transmemory', text)
    text = re.sub(r'\bmicro\s+sdhc\b', 'microsdhc', text)
    text = re.sub(r'\bmicro\s+sdxc\b', 'microsdxc', text)
    text = re.sub(r'\bmicro\s+sd\b', 'microsd', text)
    text = re.sub(r'\bsd\s+hc\b', 'sdhc', text)
    text = re.sub(r'\bsd\s+xc\b', 'sdxc', text)

    # Normalize capacities
    text = re.sub(r'(\d+)\s*(go|gigas?)\b', r'\1gb', text)
    text = re.sub(r'(\d+)\s*(gb|tb|mb)\b', r'\1\2', text)
    text = re.sub(r'\b10[02][04]gb\b', '1tb', text)
    text = re.sub(r'\b32768mb\b', '32gb', text)
    text = re.sub(r'\b16384mb\b', '16gb', text)
    text = re.sub(r'\b65536mb\b', '64gb', text)
    text = re.sub(r'\b8192mb\b', '8gb', text)

    # Detect brand
    brand = ''
    for b in KNOWN_BRANDS:
        if b in text:
            brand = 'wd' if b == 'western digital' else b
            break

    # Detect capacity set
    caps = set(re.findall(r'\b\d+(?:gb|tb|mb)\b', text))

    # Detect product family
    family = ''
    for fam in FAMILY_KEYWORDS:
        if fam in text:
            family = fam.replace(' ', '_')
            break

    # Extract model codes and alphanumeric identifiers
    tokens = re.findall(r'[a-z0-9]+', text)
    models = []
    for t in tokens:
        if t in CAP_STANDARD:
            continue
        if any(c.isalpha() for c in t) and any(c.isdigit() for c in t):
            if t not in {'3d', '4k', '2k', '1x', '2x', '3x', '3g', '4g', '5g', 'c10', 'u1', 'u3', 'v10', 'v30', 'v60', 'v90'}:
                models.append(t)
        elif t.isdigit() and len(t) >= 5:
            models.append(t)

    content_tokens = [t for t in tokens if t not in NOISE_WORDS and len(t) > 1]

    return {
        'brand': brand,
        'caps': caps,
        'family': family,
        'models': set(models),
        'content_tokens': content_tokens,
        'clean_set': set(content_tokens)
    }


def block_with_attr(X, attr="name"):
    """
    Performs scalable blocking on dataset X.
    Guaranteed linear complexity O(N), finishing in minutes on 2M records.
    Strictly avoids O(k^2) pairwise generation in any bucket.
    """
    n_records = len(X)
    print(f"Starting blocking on {n_records} records...")

    # Step 1: Preprocess records
    brand_col = X['brand'] if 'brand' in X.columns else [None] * n_records
    name_col = X[attr] if attr in X.columns else X.iloc[:, 1]
    ids = list(X['id'] if 'id' in X.columns else X.iloc[:, 0])

    items = [
        preprocess_row(name_col[i], brand_col[i])
        for i in range(n_records)
    ]

    # Step 2: Build discriminative features per item
    item_features = []
    for it in items:
        feats = set()
        b = it['brand']
        fam = it['family']
        for m in it['models']:
            feats.add('m:' + m)
            if b:
                feats.add('bm:' + b + '_' + m)
        if fam:
            feats.add('fam:' + fam)
            if b:
                feats.add('bfam:' + b + '_' + fam)
            for c in it['caps']:
                feats.add('famc:' + fam + '_' + c)
        for c in it['caps']:
            if b:
                feats.add('bc:' + b + '_' + c)
            for t in it['content_tokens']:
                feats.add('bct:' + b + '_' + c + '_' + t)
        tok_list = sorted(list(it['content_tokens']))
        for i in range(len(tok_list)):
            for j in range(i + 1, min(i + 4, len(tok_list))):
                feats.add('btt:' + b + '_' + tok_list[i] + '_' + tok_list[j])
        for t in it['content_tokens']:
            if b:
                feats.add('bt:' + b + '_' + t)
            feats.add('t:' + t)
        item_features.append(feats)

    # Step 3: Count document frequencies and filter inverted index
    df_count = defaultdict(int)
    for feats in item_features:
        for f in feats:
            df_count[f] += 1

    MAX_DF = 100
    inverted_index = defaultdict(list)
    for idx, feats in enumerate(item_features):
        for f in feats:
            c = df_count[f]
            if 1 < c <= MAX_DF:
                inverted_index[f].append(idx)

    idf = {f: np.log(n_records / df_count[f]) for f in inverted_index}

    # Step 4: Top-K candidate search per item
    # Target size: 2,000,000 candidate pairs
    TARGET_PAIRS = 2_000_000
    K_TOP = 30 if n_records <= 100_000 else 3
    tier_cands = [[] for _ in range(K_TOP)]

    print(f"Querying inverted index (K_TOP={K_TOP})...")
    for idx in range(n_records):
        id_i = ids[idx]
        scores = defaultdict(float)
        it1 = items[idx]
        for f in item_features[idx]:
            if f in inverted_index:
                w = idf[f]
                if f.startswith('m:') or f.startswith('bm:'):
                    w *= 5.0
                elif f.startswith('fam') or f.startswith('bfam'):
                    w *= 3.0
                for other in inverted_index[f]:
                    if other != idx:
                        id_j = ids[other]
                        if id_i == id_j:
                            continue
                        it2 = items[other]
                        b1, b2 = it1['brand'], it2['brand']
                        if b1 and b2 and b1 != b2:
                            continue
                        c1, c2 = it1['caps'], it2['caps']
                        if c1 and c2 and not (c1 & c2):
                            continue
                        scores[other] += w

        if scores:
            s1 = it1['clean_set']
            for other in scores:
                s2 = items[other]['clean_set']
                inter = len(s1 & s2)
                if inter > 0:
                    jaccard = inter / (len(s1) + len(s2) - inter)
                    scores[other] += jaccard * 5.0

            top_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:K_TOP]
            for rank, (other, sc) in enumerate(top_list):
                id_j = ids[other]
                lid = min(id_i, id_j)
                rid = max(id_i, id_j)
                tier_cands[rank].append((sc, (lid, rid)))

    # Step 5: Collect unique pairs by tier in order of confidence
    seen_pairs = set()
    ordered_pairs = []
    for r in range(K_TOP):
        tier_cands[r].sort(key=lambda x: x[0], reverse=True)
        for sc, p in tier_cands[r]:
            if p not in seen_pairs:
                seen_pairs.add(p)
                ordered_pairs.append(p)
                if len(ordered_pairs) == TARGET_PAIRS:
                    break
        if len(ordered_pairs) == TARGET_PAIRS:
            break

    print(f"Retrieved {len(ordered_pairs)} unique candidate pairs.")
    return ordered_pairs


def save_output(X2_candidate_pairs, max_id=0):
    """
    Save candidate pairs to output.csv with exactly 2,000,000 rows.
    Guarantees: columns 'lid', 'rid'; lid < rid; no self-pairs; no duplicates.
    """
    expected_cand_size = 2_000_000

    if len(X2_candidate_pairs) > expected_cand_size:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size]

    # If fewer than expected (e.g. dev sample data), pad with distinct valid dummy pairs
    if len(X2_candidate_pairs) < expected_cand_size:
        needed = expected_cand_size - len(X2_candidate_pairs)
        dummy_base = max(100_000_000, max_id + 10_000_000)
        for k in range(needed):
            X2_candidate_pairs.append((dummy_base + 2 * k, dummy_base + 2 * k + 1))

    output_df = pd.DataFrame(X2_candidate_pairs, columns=["lid", "rid"])
    output_df.to_csv("output.csv", index=False)
    print(f"Successfully saved {len(output_df)} pairs to output.csv")


if __name__ == "__main__":
    start_time = time.time()

    # Determine input path
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        data_path = sys.argv[1]
    elif os.path.exists("../SIGMOD_full_data/X2.csv"):
        data_path = "../SIGMOD_full_data/X2.csv"
    else:
        raise FileNotFoundError("Could not find ../SIGMOD_full_data/X2.csv")

    print(f"Reading dataset from {data_path}...")
    X2 = pd.read_csv(data_path)

    # Perform blocking
    candidates = block_with_attr(X2, attr="name")

    # Save output
    max_rec_id = int(X2['id'].max()) if 'id' in X2.columns else 0
    save_output(candidates, max_id=max_rec_id)
    print(f"Total execution finished in {time.time() - start_time:.2f} seconds.")
