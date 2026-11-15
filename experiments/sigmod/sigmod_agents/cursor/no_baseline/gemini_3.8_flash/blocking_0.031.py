import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# Known electronic brands for brand normalization
BRANDS = [
    'sandisk', 'samsung', 'sony', 'kingston', 'toshiba', 'intenso', 
    'lexar', 'transcend', 'pny', 'seagate', 'wd', 'western digital', 
    'corsair', 'crucial', 'panasonic', 'hp', 'canon', 'nikon', 
    'olympus', 'fujifilm', 'apple', 'dell', 'lenovo', 'asus', 'acer', 
    'logilink', 'verbatim', 'patriot', 'silicon power', 'adata'
]

# Common product series / model families in electronics
FAMILIES = [
    'cruzer glide', 'cruzer extreme', 'cruzer blade', 'cruzer fit', 
    'cruzer dial', 'cruzer switch', 'cruzer snap', 'cruzer force',
    'extreme pro', 'extreme plus', 'ultra plus', 'ultra fit', 'ultra flair', 
    'ultra dual', 'jumpdrive s70', 'jumpdrive s25', 'jumpdrive s50', 
    'jumpdrive s57', 'jumpdrive c20m', 'jumpdrive c20c', 'jumpdrive v30', 
    'jumpdrive p20', 'datatraveler 100', 'datatraveler 101', 'datatraveler g2', 
    'datatraveler g3', 'datatraveler g4', 'datatraveler se9', 'datatraveler micro', 
    'canvas select', 'canvas go', 'canvas react', 'transmemory u202', 
    'transmemory u363', 'transmemory u364', 'exceria pro', 'exceria plus', 
    'basic line', 'rainbow line', 'speed line', 'alu line', 'slim line', 
    'business line', 'premium line', 'micro line', 'evo plus', 'evo select', 
    'pro plus', 'pro endurance', 'bar plus'
]

BRAND_REGEX = re.compile(r'\b(' + '|'.join(BRANDS) + r')\b')
CAP_REGEX = re.compile(r'\b(\d+)\s*(gb|go|tb|to|mb)\b')
UNIT_REGEX = re.compile(r'^(\d+(gb|go|tb|to|mb|g|mo)|u[1-3]|c\d+|class\d+)$')
CODE_REGEX = re.compile(r'^[a-z]+\d+[a-z0-9]*$|^\d+[a-z]+\d*$')


def clean_text(s):
    """Normalize text: decode HTML entities, unify synonyms, keep alphanumeric."""
    s = str(s).lower()
    s = re.sub(r'&[a-z]+;', ' ', s)
    s = s.replace('cruizer', 'cruzer')
    s = re.sub(r'\bdata\s*traveler\b', 'datatraveler', s)
    s = re.sub(r'\bdt\s*(\d+)', r'datatraveler \1', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return s.strip()


def get_brand(name_clean, b_val):
    """Extract canonical brand name from explicit column or text."""
    b_str = clean_text(b_val) if pd.notna(b_val) else ''
    m = BRAND_REGEX.search(b_str)
    if m:
        return m.group(1)
    m = BRAND_REGEX.search(name_clean)
    if m:
        return m.group(1)
    return ''


def get_capacity(name_raw):
    """Extract storage capacity and format as zero-padded key for sorting."""
    t = str(name_raw).lower()
    m = CAP_REGEX.findall(t)
    for num, unit in m:
        val = int(num)
        if unit in ('go', 'gb'):
            return f'{val:05d}gb'
        elif unit in ('to', 'tb'):
            return f'{val * 1024:05d}gb'
        elif unit == 'mb':
            if val in (16000, 16384):
                return f'{16:05d}gb'
            elif val in (32000, 32768):
                return f'{32:05d}gb'
            elif val in (64000, 65536):
                return f'{64:05d}gb'
            elif val in (128000, 131072):
                return f'{128:05d}gb'
            elif val in (256000, 262144):
                return f'{256:05d}gb'
    return ''


def get_model_codes(name_raw):
    """Extract alphanumeric model numbers and MPN codes (with vendor prefixes stripped)."""
    t = str(name_raw).lower()
    tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', t)
    codes = []
    for token in tokens:
        sub_tokens = [token] + token.split('-')
        for prefix in ('ljd', 'sdcz', 'sdsd', 'thn', 'usm'):
            if token.startswith(prefix) and len(token) > len(prefix) + 2:
                sub_tokens.append(token[len(prefix):])
        for p in sub_tokens:
            if CODE_REGEX.match(p) and not UNIT_REGEX.match(p):
                codes.append(p)
            elif re.match(r'^\d{6,8}$', p):
                codes.append(p)
    return list(dict.fromkeys(codes))


def get_family(name_clean):
    """Match common product line / family name."""
    for fam in FAMILIES:
        if fam in name_clean:
            return fam
    return ''


def block_with_attr(X, attr="name"):
    """
    Performs entity resolution blocking on dataset X.
    Uses multi-pass Sorted Neighborhood Method (SNM) and MinHash LSH
    with strictly linear-time sliding windows to avoid any O(k^2) bucket explosion.
    
    :param X: dataframe of records
    :param attr: primary attribute used for blocking (default 'name')
    :return: list of candidate pairs (lid, rid) where lid < rid
    """
    N = len(X)
    MAX_PAIRS = 2000000
    
    # 1. Feature extraction in O(N)
    print("Extracting features...")
    items = []
    doc_tokens = []
    has_brand_col = 'brand' in X.columns
    
    names = X[attr].tolist()
    brands = X['brand'].tolist() if has_brand_col else [None] * N
    ids = X['id'].astype(np.int64).tolist()
    
    for i in tqdm(range(N), desc="Feature extraction"):
        rid = ids[i]
        raw_name = names[i]
        b_val = brands[i]
        
        c_name = clean_text(raw_name)
        brand = get_brand(c_name, b_val)
        cap = get_capacity(raw_name)
        codes = get_model_codes(raw_name)
        fam = get_family(c_name)
        
        tokens = [w for w in c_name.split() if len(w) > 1]
        sorted_tokens = sorted(tokens)
        first_code = codes[0] if codes else ''
        second_code = codes[1] if len(codes) > 1 else ''
        last_code = codes[-1] if codes else ''
        
        doc_tokens.append(set(tokens))
        items.append({
            'id': rid,
            'idx': i,
            'brand': brand,
            'cap': cap,
            'fam': fam,
            'first_code': first_code,
            'second_code': second_code,
            'last_code': last_code,
            'sorted_tokens': ' '.join(sorted_tokens),
            'rev_sorted_tokens': ' '.join(sorted_tokens[::-1]),
            'first_3_tokens': ' '.join(sorted_tokens[:3]),
            'last_3_tokens': ' '.join(sorted_tokens[-3:]),
            'clean_name': c_name
        })

    candidate_pairs = set()

    def add_pair(idx1, idx2):
        if len(candidate_pairs) >= MAX_PAIRS:
            return
        b1, b2 = items[idx1]['brand'], items[idx2]['brand']
        if b1 and b2 and b1 != b2:
            return
        c1, c2 = items[idx1]['cap'], items[idx2]['cap']
        if c1 and c2 and c1 != c2:
            return
        id1, id2 = items[idx1]['id'], items[idx2]['id']
        if id1 == id2:
            return
        if id1 < id2:
            candidate_pairs.add((id1, id2))
        else:
            candidate_pairs.add((id2, id1))

    # 2. Multi-Pass Sorted Neighborhood Method (SNM)
    # High-precision keys: window W=10
    high_prec_keys = [
        lambda x: (x['brand'], x['cap'], x['fam'], x['first_code'], x['sorted_tokens']),
        lambda x: (x['brand'], x['cap'], x['first_code'], x['sorted_tokens']),
        lambda x: (x['brand'], x['cap'], x['second_code'], x['sorted_tokens']),
        lambda x: (x['brand'], x['cap'], x['last_code'], x['rev_sorted_tokens']),
        lambda x: (x['brand'], x['first_code'], x['cap'], x['sorted_tokens']),
        lambda x: (x['fam'], x['cap'], x['brand'], x['sorted_tokens']),
        lambda x: (x['first_code'], x['brand'], x['cap']) if x['first_code'] else ('zzz',),
        lambda x: (x['second_code'], x['brand'], x['cap']) if x['second_code'] else ('zzz',),
        lambda x: (x['last_code'], x['brand'], x['cap']) if x['last_code'] else ('zzz',),
    ]

    # Medium-precision keys: window W=5
    med_prec_keys = [
        lambda x: (x['brand'], x['cap'], x['first_3_tokens']),
        lambda x: (x['brand'], x['cap'], x['last_3_tokens']),
        lambda x: (x['cap'], x['first_code'], x['brand'], x['sorted_tokens']),
        lambda x: (x['sorted_tokens']),
        lambda x: (x['rev_sorted_tokens']),
        lambda x: (x['brand'], x['sorted_tokens']),
        lambda x: (x['clean_name']),
    ]

    print("Running high-precision sorted neighborhood passes...")
    for key_fn in tqdm(high_prec_keys, desc="High-precision SNM"):
        if len(candidate_pairs) >= MAX_PAIRS:
            break
        sorted_items = sorted(items, key=key_fn)
        for i in range(N):
            if len(candidate_pairs) >= MAX_PAIRS:
                break
            for w in range(1, 11):
                if i + w < N:
                    add_pair(sorted_items[i]['idx'], sorted_items[i + w]['idx'])

    print("Running medium-precision sorted neighborhood passes...")
    for key_fn in tqdm(med_prec_keys, desc="Medium-precision SNM"):
        if len(candidate_pairs) >= MAX_PAIRS:
            break
        sorted_items = sorted(items, key=key_fn)
        for i in range(N):
            if len(candidate_pairs) >= MAX_PAIRS:
                break
            for w in range(1, 6):
                if i + w < N:
                    add_pair(sorted_items[i]['idx'], sorted_items[i + w]['idx'])

    # 3. Banded MinHash LSH via Sorting (W=3)
    if len(candidate_pairs) < MAX_PAIRS:
        print("Running MinHash LSH passes...")
        NUM_HASHES = 32
        np.random.seed(42)
        seeds = np.random.randint(0, 2**31 - 1, size=NUM_HASHES)

        def minhash(tokens, seed):
            min_h = 2**63 - 1
            for t in tokens:
                h = hash((t, seed))
                if h < min_h:
                    min_h = h
            return min_h

        signatures = np.zeros((N, NUM_HASHES), dtype=np.int64)
        for i in range(N):
            for h_idx in range(NUM_HASHES):
                signatures[i, h_idx] = minhash(doc_tokens[i], seeds[h_idx])

        for h_idx in tqdm(range(NUM_HASHES), desc="MinHash SNM"):
            if len(candidate_pairs) >= MAX_PAIRS:
                break
            sort_keys = [(items[i]['brand'], items[i]['cap'], signatures[i, h_idx], i) for i in range(N)]
            sort_keys.sort()
            for i in range(N):
                if len(candidate_pairs) >= MAX_PAIRS:
                    break
                for w in range(1, 4):
                    if i + w < N:
                        b1, c1, h1, idx1 = sort_keys[i]
                        b2, c2, h2, idx2 = sort_keys[i + w]
                        if h1 == h2:
                            add_pair(idx1, idx2)

    print(f"Candidate generation completed. Pairs collected: {len(candidate_pairs)}")
    return list(candidate_pairs)


def save_output(X2_candidate_pairs, output_path="output.csv", max_real_id=None):
    """
    Saves exactly 2,000,000 candidate pairs to output.csv.
    Ensures columns: lid, rid (where lid < rid, no self-pairs, no duplicates).
    If X2_candidate_pairs has fewer than 2,000,000 pairs (e.g. on sample_data dev subset),
    pads with unique non-overlapping dummy candidate pairs satisfying lid < rid.
    """
    expected_cand_size_X2 = 2000000
    candidate_list = list(X2_candidate_pairs)

    if len(candidate_list) > expected_cand_size_X2:
        candidate_list = candidate_list[:expected_cand_size_X2]

    n_cur = len(candidate_list)
    if n_cur < expected_cand_size_X2:
        needed = expected_cand_size_X2 - n_cur
        base_id = 100000000
        if max_real_id is not None and max_real_id >= base_id:
            base_id = max_real_id + 1000

        k = np.arange(needed, dtype=np.int64)
        dummy_lids = base_id + 2 * k
        dummy_rids = base_id + 2 * k + 1

        if n_cur > 0:
            cur_lids = np.array([p[0] for p in candidate_list], dtype=np.int64)
            cur_rids = np.array([p[1] for p in candidate_list], dtype=np.int64)
            all_lids = np.concatenate([cur_lids, dummy_lids])
            all_rids = np.concatenate([cur_rids, dummy_rids])
        else:
            all_lids = dummy_lids
            all_rids = dummy_rids
    else:
        all_lids = np.array([p[0] for p in candidate_list], dtype=np.int64)
        all_rids = np.array([p[1] for p in candidate_list], dtype=np.int64)

    output_df = pd.DataFrame({"lid": all_lids, "rid": all_rids})
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(output_df)} candidate pairs to {output_path}")


if __name__ == "__main__":
    # Locate X2 dataset (current directory or sample_data/)
    data_path = "../SIGMOD_full_data/X2.csv" if os.path.exists("../SIGMOD_full_data/X2.csv") else "./X2.csv"
    print(f"Reading dataset from {data_path}...")
    X2 = pd.read_csv(data_path)

    # Perform blocking
    X2_candidate_pairs = block_with_attr(X2, attr="name")

    # Save output
    max_id = int(X2['id'].max()) if 'id' in X2.columns else None
    save_output(X2_candidate_pairs, max_real_id=max_id)
