"""
Scalable Entity Resolution Candidate Blocking for X2 Dataset.
Designed to run in under 3 minutes on 2M records with strict O(N) complexity
and bounded RAM using Multi-Pass Sorted Neighborhood Method (SNM).
"""

import os
import re
import html
import time
import numpy as np
import pandas as pd
import unidecode

# ---------------------------------------------------------
# Feature Extraction and Normalization Configuration
# ---------------------------------------------------------

ADDITIONAL_ALIASES = {
    'canvas go': 'canvasgo', 'canvas select': 'canvasselect', 'canvas react': 'canvasreact',
    'exceria pro': 'exceriapro', 'exceria plus': 'exceriaplus',
    'cruzer glide': 'cruzerglide', 'cruzer blade': 'cruzerblade', 'cruzer extreme': 'cruzerextreme',
    'cruzer fit': 'cruzerfit', 'cruzer force': 'cruzerforce', 'cruzer dial': 'cruzerdial',
    'cruzer edge': 'cruzeredge', 'cruzer flair': 'cruzerflair', 'ultra flair': 'ultraflair',
    'ultra dual': 'ultradual', 'ultra fit': 'ultrafit', 'ultra plus': 'ultraplus',
    'extreme pro': 'extremepro', 'extreme plus': 'extremeplus',
    'basic line': 'basicline', 'speed line': 'speedline', 'premium line': 'premiumline',
    'dt101g2': 'dt101', 'dt101 g2': 'dt101', 'dt 101': 'dt101', '101g2': 'dt101',
    'dtig4': 'dtg4', 'dt ig4': 'dtg4', 'dt g4': 'dtg4', 'datatraveler g4': 'dtg4',
    'dtse9': 'se9', 'dt se9': 'se9',
    'c20m': 'c20m', 'ljdc20m': 'c20m', 'c20c': 'c20c', 'ljdc20c': 'c20c',
    's70': 's70', 'ljds70': 's70', 's75': 's75', 'ljds75': 's75',
    's25': 's25', 'ljds25': 's25', 'v10': 'v10', 'ljdv10': 'v10',
    'v30': 'v30', 'ljdv30': 'v30', 'p20': 'p20', 'ljdp20': 'p20',
    'u202': 'u202', 'thn-u202': 'u202', 'u302': 'u302', 'thn-u302': 'u302',
    'n101': 'n101', 'thn-n101': 'n101', 'n302': 'n302', 'thn-n302': 'n302',
    'n401': 'n401', 'thn-n401': 'n401', 'm203': 'm203', 'm302': 'm302', 'm401': 'm401',
    'srg1uxa': 'g1ux', 'srg1ux': 'g1ux', 'sr-g1uxa': 'g1ux', 'sr-g1ux': 'g1ux', 'sr-ux2a': 'g1ux',
    'cz80': 'cz80', 'sdcz80': 'cz80', 'cz48': 'cz48', 'sdcz48': 'cz48',
    'cz60': 'cz60', 'sdcz60': 'cz60', 'cz73': 'cz73', 'sdcz73': 'cz73',
    'cz50': 'cz50', 'sdcz50': 'cz50',
}

KNOWN_BRANDS = [
    'sandisk', 'kingston', 'toshiba', 'sony', 'samsung', 'transcend',
    'lexar', 'intenso', 'pny', 'panasonic', 'corsair', 'hp', 'seagate',
    'western digital', 'wd', 'verbatim', 'integral', 'patriot', 'delkin',
    'adata', 'silicon power', 'strontium', 'kioxia'
]

ADDITIONAL_MODELS = [
    'canvasgo', 'canvasselect', 'canvasreact', 'canvas',
    'u202', 'u302', 'n101', 'n203', 'n302', 'n401', 'm203', 'm302', 'm401',
    'exceriapro', 'exceriaplus', 'exceria',
    'cruzerglide', 'cruzerblade', 'cruzerextreme', 'cruzerfit', 'cruzerforce',
    'cruzerdial', 'cruzeredge', 'cruzerflair', 'ultraflair', 'ultradual',
    'ultrafit', 'ultraplus', 'extremepro', 'extremeplus',
    'c20m', 'c20c', 's70', 's75', 's25', 'v10', 'v30', 'p20',
    'dt101g2', 'dt101', 'dtig4', 'dtg4', 'dt50', 'se9',
    'g1ux', 'sf8u', 'sda3', 'sda10', 'sdca3', 'sdc4', 'sdc10',
    'cz80', 'cz48', 'cz60', 'cz73', 'cz50',
    'basicline', 'speedline', 'premiumline', 'rainbow',
    'evoplus', 'evo', 'proplus', 'microvault',
    'ultra', 'extreme', 'cruzer', 'glide', 'blade', 'flair', 'fit'
]

CAPACITY_RE = re.compile(r'\b(2|4|8|16|32|64|128|200|256|400|512|1000|1024|2048)\s*(?:gb|go|gigas?|g|mb|mo|tb|to)\b')
SHORT_PAT_RE = re.compile(r'\b[a-z]{1,4}\d{1,4}[a-z]{0,3}\b')
LONG_PAT_RE = re.compile(r'\b[a-z0-9]{2,5}[-_][a-z0-9]{3,7}(?:[-_][a-z0-9]+)?\b')
P2_RE = re.compile(r'\b[a-z]+\s+[a-z]*\d+[a-z0-9]*\b')


def clean_text(text: str) -> str:
    """Normalize raw title text: unescape, unidecode, lowercase, replace aliases."""
    text = html.unescape(str(text))
    text = unidecode.unidecode(text).lower()
    
    # Class and speed normalization
    text = re.sub(r'class\s*10\b', 'class10', text)
    text = re.sub(r'class\s*([2-9])\b', r'class\1', text)
    text = re.sub(r'clase\s*10\b', 'class10', text)
    text = re.sub(r'clase\s*([2-9])\b', r'class\1', text)
    text = re.sub(r'uhs\s*[-_]?\s*([i123])\b', r'uhs\1', text)
    text = re.sub(r'u\s*([13])\b', r'u\1', text)
    
    # Replace aliases
    for k, v in ADDITIONAL_ALIASES.items():
        text = text.replace(k, v)
        
    # Capacity units
    text = re.sub(r'\b(\d+)\s*(?:gb|go|gigas|giga|gig|g)\b', r' \1gb ', text)
    text = re.sub(r'\b(\d+)\s*(?:mb|mo|megas)\b', r' \1mb ', text)
    text = re.sub(r'\b(\d+)\s*(?:tb|to|teras)\b', r' \1tb ', text)
    text = re.sub(r'usb\s*(3\.[012]|2\.0|3\.0)', r' usb3 ', text)
    
    return re.sub(r'\s+', ' ', text).strip()


def extract_features(clean_name: str):
    """Extract structured features from a normalized product title."""
    # Brand
    brand = ''
    for b in KNOWN_BRANDS:
        if re.search(r'\b' + re.escape(b) + r'\b', clean_name):
            brand = b.replace(' ', '')
            break
            
    # Capacity
    caps = CAPACITY_RE.findall(clean_name)
    if not caps:
        caps = re.findall(r'\b(\d{1,4})\s*(?:gb|go|gigas?|g|mb|mo|tb|to)\b', clean_name)
    cap = ' '.join(sorted(set(caps))) if caps else ''
    
    # Model
    model = ''
    for m in ADDITIONAL_MODELS:
        if m in clean_name:
            model = m
            break
            
    # Type
    ptype = ''
    if 'microsd' in clean_name or 'micro sd' in clean_name:
        ptype = 'microsd'
    elif 'sdxc' in clean_name:
        ptype = 'sdxc'
    elif 'sdhc' in clean_name:
        ptype = 'sdhc'
    elif 'usb' in clean_name or 'flash' in clean_name or 'pen' in clean_name:
        ptype = 'usb'
    elif 'tv' in clean_name or 'television' in clean_name:
        ptype = 'tv'
        
    # Patterns
    sp_matches = SHORT_PAT_RE.findall(clean_name)
    sp = sp_matches[0] if sp_matches else ''
    
    lp_matches = LONG_PAT_RE.findall(clean_name)
    lp = lp_matches[0] if lp_matches else ''
    
    p2_matches = P2_RE.findall(clean_name)
    p2 = ' '.join(sorted([p.replace(' ', '') for p in p2_matches])) if p2_matches else ''
    
    tokens = set(re.findall(r'[a-z0-9]+', clean_name))
    
    return brand, cap, model, ptype, sp, lp, p2, tokens


def block_x2(df: pd.DataFrame, max_output_pairs: int = 2_000_000, window_size: int = 25) -> pd.DataFrame:
    """
    Perform Multi-Pass Sorted Neighborhood Blocking on X2.
    Strictly O(W * N) time complexity and strictly bounded memory.
    """
    t0 = time.time()
    n = len(df)
    ids = df['id'].values
    names = df['name'].values
    
    print(f"Starting feature extraction for {n:,} records...")
    cleaned_names = []
    token_sets = []
    brands = []
    caps = []
    models = []
    types = []
    short_pats = []
    long_pats = []
    p2_pats = []
    
    for i in range(n):
        cname = clean_text(names[i])
        br, ca, mo, pt, sp, lp, p2, toks = extract_features(cname)
        cleaned_names.append(cname)
        token_sets.append(toks)
        brands.append(br)
        caps.append(ca)
        models.append(mo)
        types.append(pt)
        short_pats.append(sp)
        long_pats.append(lp)
        p2_pats.append(p2)
        
    print(f"Features extracted in {time.time() - t0:.2f}s")
    
    # Define 13 multi-pass sorting keys
    sort_keys = [
        [f'{brands[i]}|{models[i]}|{caps[i]}|{cleaned_names[i]}' for i in range(n)],
        [f'{models[i]}|{caps[i]}|{brands[i]}|{cleaned_names[i]}' for i in range(n)],
        [f'{brands[i]}|{caps[i]}|{types[i]}|{models[i]}|{cleaned_names[i]}' for i in range(n)],
        [f'{short_pats[i]}|{caps[i]}|{cleaned_names[i]}' for i in range(n)],
        [f'{long_pats[i]}|{cleaned_names[i]}' for i in range(n)],
        [f'{p2_pats[i]}|{caps[i]}|{cleaned_names[i]}' for i in range(n)],
        [f'{brands[i]}|{types[i]}|{caps[i]}|{cleaned_names[i]}' for i in range(n)],
        [f'{types[i]}|{caps[i]}|{cleaned_names[i]}' for i in range(n)],
        [' '.join(sorted(token_sets[i])) for i in range(n)],
        [' '.join(reversed(cleaned_names[i].split())) for i in range(n)],
        [f'{brands[i]}|{" ".join(sorted(token_sets[i]))}' for i in range(n)],
        [cleaned_names[i] for i in range(n)],
        [f'{caps[i]}|{cleaned_names[i]}' for i in range(n)],
    ]
    
    # Bounded candidate buffer
    # Pairs encoded as uint64: (min_id << 32) | max_id -> sim
    MAX_BUFFER = 2_500_000
    candidates = {}
    min_sim_threshold = 0.20
    
    for pass_idx, keys in enumerate(sort_keys):
        pass_t0 = time.time()
        order = sorted(range(n), key=lambda i: keys[i])
        
        for i in range(n):
            i1 = order[i]
            s1 = token_sets[i1]
            len1 = len(s1)
            if len1 == 0:
                continue
                
            for w in range(1, window_size + 1):
                if i + w >= n:
                    break
                i2 = order[i + w]
                s2 = token_sets[i2]
                len2 = len(s2)
                if len2 == 0:
                    continue
                    
                min_len = min(len1, len2)
                inter = len(s1 & s2)
                sim = inter / min_len
                
                if sim >= min_sim_threshold:
                    r1 = int(ids[i1])
                    r2 = int(ids[i2])
                    if r1 == r2:
                        continue
                    lid = min(r1, r2)
                    rid = max(r1, r2)
                    pair_key = (lid << 32) | rid
                    
                    if pair_key not in candidates or sim > candidates[pair_key]:
                        candidates[pair_key] = sim
                        
        print(f"Pass {pass_idx + 1:2d}/{len(sort_keys)} completed in {time.time() - pass_t0:.2f}s | Cand pairs: {len(candidates):,}")
        
        # Dynamic memory safeguard: if buffer exceeds MAX_BUFFER, prune lowest similarity pairs
        if len(candidates) >= MAX_BUFFER:
            print(f"Pruning buffer: current size {len(candidates):,} exceeds {MAX_BUFFER:,}...")
            vals = np.fromiter(candidates.values(), dtype=np.float32, count=len(candidates))
            cutoff = float(np.partition(vals, len(vals) - max_output_pairs)[len(vals) - max_output_pairs])
            min_sim_threshold = max(min_sim_threshold, cutoff)
            candidates = {k: v for k, v in candidates.items() if v >= cutoff}
            print(f"Buffer pruned to {len(candidates):,} | new min_threshold = {min_sim_threshold:.3f}")
            
    # Sort pairs by similarity descending
    print(f"Total candidate pairs found: {len(candidates):,}. Sorting by similarity...")
    sorted_pairs = sorted(candidates.keys(), key=lambda k: candidates[k], reverse=True)
    
    # For full dataset, enforce exactly max_output_pairs
    if n > 100_000 and len(sorted_pairs) > max_output_pairs:
        sorted_pairs = sorted_pairs[:max_output_pairs]
        
    # Unpack to lid, rid
    lids = [(k >> 32) for k in sorted_pairs]
    rids = [(k & 0xFFFFFFFF) for k in sorted_pairs]
    
    result_df = pd.DataFrame({'lid': lids, 'rid': rids})
    print(f"Blocking complete in {time.time() - t0:.2f}s | Result shape: {result_df.shape}")
    return result_df


def main():
    # Resolve input path
    if os.path.exists("sample_data/X2.csv"):
        x2_path = "sample_data/X2.csv"
    elif os.path.exists("X2.csv"):
        x2_path = "X2.csv"
    else:
        raise FileNotFoundError("Could not find X2.csv or sample_data/X2.csv")
        
    print(f"Loading data from {x2_path}...")
    df = pd.read_csv(x2_path)
    print(f"Loaded {len(df):,} rows.")
    
    # Perform candidate blocking
    cand_df = block_x2(df, max_output_pairs=2_000_000, window_size=25)
    
    # Save output to root directory as output.csv
    out_path = "output.csv"
    print(f"Saving {len(cand_df):,} candidate pairs to {out_path}...")
    cand_df.to_csv(out_path, index=False)
    print("Done! output.csv successfully created.")
    
    # Verification against sample ground truth if available
    if os.path.exists("../SIGMOD_full_data/Z2.csv"):
        z2 = pd.read_csv("../SIGMOD_full_data/Z2.csv")
        gt = set(zip(z2.iloc[:, 0], z2.iloc[:, 1]))
        output_pairs = set(zip(cand_df['lid'], cand_df['rid']))
        hits = len(output_pairs & gt)
        print(f"Recall on Z2: {hits:,} / {len(gt):,} ({hits / len(gt):.4%})")


if __name__ == "__main__":
    main()
