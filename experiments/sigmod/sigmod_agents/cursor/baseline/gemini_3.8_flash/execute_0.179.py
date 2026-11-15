import os
import sys
import time
import math
import re
import unicodedata
from collections import defaultdict
import pandas as pd
import numpy as np

# Domain patterns and aliases for X2 electronic products
x2_aliases = {
    'class': ['classe', 'clase', 'clas ', 'klasse', 'cl '],
    'uhsi': ['uhs1', 'uhs-i', 'ultra high-speed'],
    'type-c': ['typec', 'type c', 'usb-c', 'usbc'],
    'intenso premium line': ['3534490'],
    'kingston hxs ': ['hyperx', 'savage'],
    'sony g1ux': ['serie ux'],
    ' kingston dt101 ': ['dtig4', ' 101 ', 'dt101g2'],
    ' kingston ultimate ': ['sda10', 'sda3'], 
    'extreme ': ['extrem '], 
    'att4': ['attach']
}

x2_clean_pattern_1 = r'&(nbsp|amp|reg|[a-z]?acute|quot|trade);?|[|;:/,‰+©\(\)\\][psn]*|(?<=usb)[\s][m]*(?=[23][\.\s])|(?<=usb)-[\w]+\s(?=[23][\.\s])|(?<=[a-z])[\s]+gb|(?<=data|jump)[t\s](?=trave|drive)|(?<=extreme|exceria)[\s](?=pro[\s]|plus)|(?<=class)[\s_](?=10|[234]\b)|(?<=gen)[\s_](?=[134\s][0]*)'
x2_class10_pattern = r'(10 class|class 10|class(?=[\w]+10\b)|cl\s10)'

x2_clean_reg_remove = re.compile(x2_clean_pattern_1)
x2_clean_reg_class10 = re.compile(x2_class10_pattern)
x2_brand_reg = re.compile(r'\b(intenso|lexar|logilink|pny|samsung|sandisk|kingston|sony|toshiba|transcend)\b')
x2_model_reg = re.compile(r'\b(datatraveler|extreme[p]?|exceria[p]?|dual[\s]*(?!=sim)|evo|xqd|ssd|cruzer[\w+]*|glide|blade|basic|fit|force|basic line|jump\s?drive|hxs|rainbow|speed line|premium line|att4|attach|serie u|r-serie|beast|fury|impact|a400|sd[hx]c|uhs[i12][i1]*|note\s?9|ultra)')
x2_model_num_reg = re.compile(r'\b([\(]*[\w]+[-]*[\d]+[-]*[\w]+[-]*[\d+]*|[\d]+[\w]|[\w][\d]+)')
x2_mem_reg = re.compile(r'([1-9]{1,3})[-\s]*[g][bo]?')
x2_modelnum_long_pattern = r'(thn-[a-z][\w]+|ljd[\w+][-][\w]+|ljd[sc][\w]+[-][\w]+|lsdmi[\d]+[\w]+|lsd[0-9]{1,3}[gb]+[\w]+|ljds[0-9]{2}[-][\w]+|usm[0-9]{1,3}[\w]+|sdsq[a-z]+[-][0-9]+[a-z]+[-][\w]+|sdsd[a-z]+[-][0-9]+[\w]+[-]*[\w]*|sdcz[\w]+|mk[\d]+|sr-g1[\w]+)'
x2_modelnum_short_pattern = r'\b(c20[mc]|sda[0-9]{1,2}|g1ux|s[72][05]|[unm][23]02|p20|g4|dt101|se9|[asm][0-9]{2})'
x2_feature_pattern = r'\b(usb[23]|type-c|uhs[i]{1,2}|class[0134]{1,2}|gen[1-9]{1,2}|u[23](?=[\s\.])|sd[hx]c|otg|lte|[45]g[-\s]lte|[0-9]+(?=-inch)|[0-9]{2,3}r[0-9]{2,3}w|[0-9]{2,3}(?=[\smbo/p]{3}))'

x2_feature_reg = re.compile(x2_feature_pattern)
x2_longpat_reg = re.compile(x2_modelnum_long_pattern)
x2_shortpat_reg = re.compile(x2_modelnum_short_pattern)
x2_wwd_reg = re.compile(r'\w+\s\w+\d+')

def x2_clean_data(name):
    try:
        import unidecode
        name = unidecode.unidecode(str(name).lower())
    except ImportError:
        name = unicodedata.normalize('NFKD', str(name).lower()).encode('ascii', 'ignore').decode('ascii')
    
    for key, val in x2_aliases.items():
        for word in val:
            name = name.replace(word, key)
            
    name = x2_clean_reg_remove.sub(' ', name)
    name = x2_clean_reg_class10.sub('class10', name)
    name = name.replace('class 4 ', 'class4 ')
    name = name.replace('class 3 ', 'class3 ')
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def x2_find_brand(cleaned):
    brands = x2_brand_reg.findall(cleaned)
    return " ".join(sorted(set(brands))) if brands else ''

def x2_find_models_and_type(cleaned):
    models = x2_model_reg.findall(cleaned)
    return " ".join(sorted(set(models))) if models else ''

def x2_find_memcapacity(cleaned):
    memCandidates = x2_mem_reg.findall(cleaned)
    if memCandidates:
        mem_capacity = sorted(list(set([re.sub('[^0-9a-z]+', '', i) for i in memCandidates])))
        return " ".join(mem_capacity)
    return ''

def x2_find_model_num(cleaned):
    model_num = x2_model_num_reg.findall(cleaned)
    if model_num:
        model_num = [x for x in model_num if len(x) > 5]
        return " ".join(model_num)
    return ''

def x2_find_features(cleaned):
    feats = x2_feature_reg.findall(cleaned)
    return ' '.join(sorted(set(feats))) if feats else ''

def x2_find_long_pattern(cleaned):
    m = x2_longpat_reg.search(cleaned)
    return m.group(0) if m else ''

def x2_find_short_pattern(cleaned):
    shorts = x2_shortpat_reg.findall(cleaned)
    return ' '.join(sorted(set(shorts))) if shorts else ''

def block_with_attr(X, attr="name"):
    """
    Feasible, linear-time blocking pipeline designed for 2,000,000 records.
    - Zero combinations inside any bucket
    - Zero O(k^2) work
    - Multi-pass Sorted Neighborhood (window size W=2)
    - Inverted index with posting-list safeguards and query-time top-K retrieval
    """
    N = len(X)
    TARGET_PAIRS = 2000000
    K_PER_QUERY = min(40, max(1, TARGET_PAIRS // max(1, N)))
    W_SNM = 3 if N < 10000 else 2
    MAX_POSTING = 100

    BRANDS = ['intenso', 'lexar', 'logilink', 'pny', 'samsung', 'sandisk', 'kingston', 'sony', 'toshiba', 'transcend']

    ids = X['id'].values
    names = X[attr].values
    brand_col_vals = X['brand'].values if 'brand' in X.columns else [None] * N

    records = []
    for i in range(N):
        raw_name = str(names[i])
        brand_val = str(brand_col_vals[i]).lower() if pd.notna(brand_col_vals[i]) else ''
        cleaned = x2_clean_data(raw_name)

        brand = x2_find_brand(cleaned)
        if not brand and brand_val:
            for b in BRANDS:
                if b in brand_val:
                    brand = b
                    break

        model = x2_find_models_and_type(cleaned)
        mem = x2_find_memcapacity(cleaned)
        model_num = x2_find_model_num(cleaned)
        feat = x2_find_features(cleaned)
        long_pat = x2_find_long_pattern(cleaned)
        short_pat = x2_find_short_pattern(cleaned)

        wwd = x2_wwd_reg.findall(raw_name)
        wwd_str = ' '.join(sorted([w.lower() for w in wwd])) if wwd else ''

        words = [w for w in cleaned.split() if len(w) >= 2]

        records.append({
            'id': ids[i],
            'cleaned': cleaned,
            'brand': brand,
            'model': model,
            'mem': mem,
            'model_num': model_num,
            'feat': feat,
            'long_pat': long_pat,
            'short_pat': short_pat,
            'wwd': wwd_str,
            'words': set(words)
        })

    candidate_pair_scores = {}

    # STEP 1: Multi-Pass Sorted Neighborhood Method (SNM)
    passes = [
        [('_'.join(sorted(r['cleaned'].split())), i) for i, r in enumerate(records)],
        [(f"{r['long_pat']}", i) for i, r in enumerate(records)],
        [(f"{r['brand']}_{r['model']}_{r['mem']}", i) for i, r in enumerate(records)],
        [(f"{r['brand']}_{r['short_pat']}_{r['mem']}", i) for i, r in enumerate(records)],
        [(f"{r['brand']}_{r['model_num']}", i) for i, r in enumerate(records)],
        [(f"{r['wwd']}", i) for i, r in enumerate(records)],
    ]

    for p_keys in passes:
        p_keys.sort(key=lambda x: x[0])
        for i in range(N):
            k_i, id_i = p_keys[i]
            clean_k = k_i.replace('_', '').strip()
            if not clean_k or len(clean_k) < 3:
                continue
            for w in range(1, W_SNM + 1):
                if i + w < N:
                    k_j, id_j = p_keys[i + w]
                    if k_i == k_j:
                        real_i = records[id_i]['id']
                        real_j = records[id_j]['id']
                        if real_i != real_j:
                            pair = (min(real_i, real_j), max(real_i, real_j))
                            if pair not in candidate_pair_scores:
                                r_i = records[id_i]
                                r_j = records[id_j]
                                overlap = len(r_i['words'].intersection(r_j['words'])) / max(1, min(len(r_i['words']), len(r_j['words'])))
                                candidate_pair_scores[pair] = 10.0 + 5.0 * overlap

    # STEP 2: Inverted Index with Query-time Safeguards and Top-K Selection
    tokens_per_record = []
    token_counts = defaultdict(int)

    for r in records:
        toks = set()
        toks.add('TITLE_' + '_'.join(sorted(r['cleaned'].split())))
        if r['long_pat']:
            toks.add('LP_' + r['long_pat'])
        if r['short_pat']:
            toks.add('SP_' + r['short_pat'])
            if r['mem']:
                toks.add(f'SPM_{r["short_pat"]}_{r["mem"]}')
        if r['brand'] and r['model'] and r['mem']:
            toks.add(f'BMM_{r["brand"]}_{r["model"]}_{r["mem"]}')
        if r['brand'] and r['model_num']:
            for mn in r['model_num'].split():
                toks.add(f'BMN_{r["brand"]}_{mn}')
        if r['brand'] and r['feat']:
            toks.add(f'BF_{r["brand"]}_{r["feat"]}')
        if r['brand'] and r['mem']:
            toks.add(f'BM_{r["brand"]}_{r["mem"]}')
        if r['wwd']:
            toks.add('WWD_' + r['wwd'])
        for w in r['words']:
            if len(w) >= 3:
                toks.add('W_' + w)
                
        tokens_per_record.append(toks)
        for t in toks:
            token_counts[t] += 1

    inv_idx = defaultdict(list)
    idf = {}
    for t, cnt in token_counts.items():
        if cnt <= MAX_POSTING:
            boost = 1.0
            if t.startswith('TITLE_'): boost = 6.0
            elif t.startswith('LP_'): boost = 5.0
            elif t.startswith('SPM_'): boost = 4.0
            elif t.startswith('BMM_'): boost = 3.5
            elif t.startswith('BMN_'): boost = 3.5
            elif t.startswith('WWD_'): boost = 3.0
            elif t.startswith('SP_'): boost = 2.0
            elif t.startswith('BF_'): boost = 1.5
            elif t.startswith('BM_'): boost = 1.0
            elif t.startswith('W_'): boost = 0.8
            idf[t] = boost * math.log((N + 1) / cnt)

    for idx, toks in enumerate(tokens_per_record):
        for t in toks:
            if t in idf:
                inv_idx[t].append(idx)

    for i in range(N):
        r_i = records[i]
        scores = defaultdict(float)
        for t in tokens_per_record[i]:
            if t in idf:
                w = idf[t]
                for j in inv_idx[t]:
                    if j != i:
                        scores[j] += w

        if not scores:
            continue

        valid_cands = []
        for j, score in scores.items():
            r_j = records[j]
            # Brand conflict check
            if r_i['brand'] and r_j['brand'] and r_i['brand'] != r_j['brand']:
                continue
            # Containment overlap coefficient
            overlap = len(r_i['words'].intersection(r_j['words'])) / max(1, min(len(r_i['words']), len(r_j['words'])))
            final_score = score + 6.0 * overlap
            valid_cands.append((j, final_score))

        valid_cands.sort(key=lambda x: x[1], reverse=True)
        for j, sc in valid_cands[:K_PER_QUERY]:
            real_i = r_i['id']
            real_j = records[j]['id']
            pair = (min(real_i, real_j), max(real_i, real_j))
            if pair not in candidate_pair_scores or sc > candidate_pair_scores[pair]:
                candidate_pair_scores[pair] = sc

    sorted_pairs = sorted(candidate_pair_scores.items(), key=lambda x: x[1], reverse=True)
    out_pairs = [p for p, _ in sorted_pairs]
    return out_pairs

def save_output(candidate_pairs, output_path="output.csv"):
    expected_size = 2000000

    if len(candidate_pairs) > expected_size:
        candidate_pairs = candidate_pairs[:expected_size]
    elif len(candidate_pairs) < expected_size:
        existing_set = set(candidate_pairs)
        pad_k = 1
        while len(candidate_pairs) < expected_size:
            p = (0, pad_k)
            if p not in existing_set:
                existing_set.add(p)
                candidate_pairs.append(p)
            pad_k += 1

    output_df = pd.DataFrame(candidate_pairs, columns=["lid", "rid"])
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(output_df)} pairs to {output_path}")

if __name__ == "__main__":
    # Auto-detect input path
    candidate_paths = ["X2.csv", "sample_data/X2.csv", "../sample_data/X2.csv", "../X2.csv"]
    input_file = "sample_data/X2.csv"
    for p in candidate_paths:
        if os.path.exists(p):
            input_file = p
            break

    print(f"Loading {input_file}...")
    X2 = pd.read_csv(input_file)
    candidate_pairs = block_with_attr(X2, attr="name")

    # Output path
    out_file = "output.csv"
    if not os.path.exists("sample_data") and os.path.exists("../sample_data"):
        out_file = "../output.csv"
    save_output(candidate_pairs, out_file)
