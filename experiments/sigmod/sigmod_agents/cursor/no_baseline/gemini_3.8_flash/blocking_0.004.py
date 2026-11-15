import os
import re
import numpy as np
import pandas as pd


MB_TO_GB = {
    '1024': '0001gb', '2048': '0002gb', '4096': '0004gb', '8192': '0008gb',
    '16384': '0016gb', '32768': '0032gb', '65536': '0064gb', '131072': '0128gb', '262144': '0256gb'
}

BRAND_LIST = [
    'sandisk', 'kingston', 'sony', 'toshiba', 'lexar', 'samsung', 'intenso',
    'transcend', 'pny', 'corsair', 'hp', 'wd', 'western digital', 'seagate',
    'verbatim', 'philips', 'panasonic', 'silicon power', 'delkin', 'patriot',
    'netac', 'kioxia', 'crucial', 'adata', 'sabrent'
]

BRAND_INFERENCE = {
    'cruzer': 'sandisk', 'sandisk': 'sandisk',
    'jumpdrive': 'lexar', 'lexar': 'lexar',
    'datatraveler': 'kingston', 'kingston': 'kingston',
    'exceria': 'toshiba', 'transmemory': 'toshiba', 'toshiba': 'toshiba',
    'evo': 'samsung', 'samsung': 'samsung',
    'intenso': 'intenso', 'sony': 'sony', 'pny': 'pny', 'transcend': 'transcend'
}
brand_inf_pattern = re.compile(r'\b(' + '|'.join(BRAND_INFERENCE.keys()) + r')\b')

FAMILIES = [
    'cruzer glide', 'cruzer blade', 'cruzer extreme', 'cruzer force', 'cruzer fit',
    'cruzer switch', 'cruzer dial', 'cruzer ultra', 'cruzer edge', 'cruzer orbit', 'cruzer facet', 'cruzer',
    'extreme pro', 'extreme plus', 'ultra plus', 'ultra dual', 'ultra flare', 'ultra fit', 'ultra', 'extreme',
    'jumpdrive s70', 'jumpdrive s75', 'jumpdrive s50', 'jumpdrive v30', 'jumpdrive c20m', 'jumpdrive c20c',
    'jumpdrive s45', 'jumpdrive s25', 'jumpdrive s35', 'jumpdrive m20', 'jumpdrive s55', 'jumpdrive',
    'datatraveler 100', 'datatraveler g4', 'datatraveler g2', 'datatraveler dt101', 'dt101 g2', 'dt101g2',
    'datatraveler se9', 'datatraveler micro', 'datatraveler',
    'canvas react', 'canvas select', 'canvas go', 'canvas',
    'basic line', 'rainbow line', 'premium line', 'alu line', 'speed line', 'ultra line',
    'slim line', 'micro line', 'business line', 'deluxe line', 'energy line',
    'exceria pro', 'exceria plus', 'exceria', 'transmemory',
    'evo plus', 'evo select', 'pro plus', 'fit plus', 'bar plus', 'evo',
    'micro vault', 'microvault',
    'microsdxc', 'microsdhc', 'microsd', 'sdxc', 'sdhc'
]
fam_pattern = re.compile(r'\b(' + '|'.join(FAMILIES) + r')\b')

re_mb = re.compile(r'\b(\d+)\s*(?:mb|mo)\b')
re_gb = re.compile(r'\b(\d+)\s*(gb|go|tb|to|g)\b')
re_compact_gb = re.compile(r'(\d+)(gb|go|g)')
re_tokens = re.compile(r'[a-zA-Z0-9]+')
re_subcode = re.compile(r'[a-zA-Z]+\d+[a-zA-Z]*|\d+[a-zA-Z]+')
re_is_cap = re.compile(r'^\d+(?:gb|go|mb|mo|tb|to|g)$')

STOP_WORDS = {
    'de', 'la', 'le', 'en', 'pour', 'and', 'the', 'with', 'mit', 'con',
    'di', 'der', 'die', 'das', 'fur', 'in', 'of', 'a', 'to', 'un', 'una',
    'el', 'los'
}


def extract_features(X, attr='name'):
    names = X[attr].fillna('').astype(str).values
    brands_col = X['brand'].fillna('').astype(str).values if 'brand' in X.columns else [''] * len(names)
    ids = X['id'].values

    n = len(names)
    keys_brand = []
    keys_cap = []
    keys_fam = []
    keys_m1 = []
    keys_m2 = []
    keys_pure_num = []
    keys_sorted_toks = []
    keys_rev_toks = []
    token_sets = []

    for i in range(n):
        text = (names[i] + ' ' + brands_col[i]).lower().replace('&nbsp;', ' ').replace('&amp;', '&').replace('cruizer', 'cruzer')

        mb = brand_inf_pattern.search(text)
        b = BRAND_INFERENCE[mb.group(1)] if mb else ''

        cap = ''
        mmb = re_mb.search(text)
        if mmb and mmb.group(1) in MB_TO_GB:
            cap = MB_TO_GB[mmb.group(1)]
        else:
            mgb = re_gb.search(text)
            if mgb:
                num = int(mgb.group(1))
                unit = mgb.group(2).replace('go', 'gb').replace('g', 'gb').replace('to', 'tb')
                cap = f'{num:04d}{unit}'
            else:
                mc = re_compact_gb.search(text)
                if mc:
                    num = int(mc.group(1))
                    unit = mc.group(2).replace('go', 'gb').replace('g', 'gb')
                    cap = f'{num:04d}{unit}'

        mf = fam_pattern.search(text)
        fam = mf.group(1).replace(' ', '_') if mf else ''

        toks = re_tokens.findall(text)
        alphanums = []
        pure_nums = []
        meaningful = []
        for t in toks:
            if len(t) > 1 and t not in STOP_WORDS:
                if t not in BRAND_INFERENCE:
                    meaningful.append(t)
                if any(c.isalpha() for c in t) and any(c.isdigit() for c in t) and not re_is_cap.match(t):
                    alphanums.append(t)
                    subs = re_subcode.findall(t)
                    for s in subs:
                        if len(s) >= 3 and not re_is_cap.match(s):
                            alphanums.append(s)
                elif t.isdigit() and len(t) >= 4:
                    pure_nums.append(t)

        alphanums_unique = sorted(set(alphanums), key=lambda x: (len(x), x))
        m1 = alphanums_unique[0] if alphanums_unique else (pure_nums[0] if pure_nums else '')
        m2 = alphanums_unique[1] if len(alphanums_unique) > 1 else ''
        pnum = pure_nums[0] if pure_nums else ''

        sorted_m = sorted(set(meaningful))
        stoks = ' '.join(sorted_m)
        rtoks = ' '.join(reversed(sorted_m))

        keys_brand.append(b)
        keys_cap.append(cap)
        keys_fam.append(fam)
        keys_m1.append(m1)
        keys_m2.append(m2)
        keys_pure_num.append(pnum)
        keys_sorted_toks.append(stoks)
        keys_rev_toks.append(rtoks)
        token_sets.append(set(meaningful))

    return {
        'id': ids,
        'brand': keys_brand,
        'cap': keys_cap,
        'fam': keys_fam,
        'm1': keys_m1,
        'm2': keys_m2,
        'pnum': keys_pure_num,
        'stoks': keys_sorted_toks,
        'rtoks': keys_rev_toks,
        'toks': token_sets
    }


def block_with_attr(X, attr='name'):
    '''
    Scalable Multi-Pass Sorted Neighborhood Blocking (MPSNM).
    Guarantees O(W * N) time per pass and O(1) memory per candidate, strictly avoiding O(k^2) bucket explosion.
    :param X: dataframe with records to block
    :param attr: attribute used for blocking (default 'name')
    :return: candidate set of tuple pairs (lid, rid) with lid < rid
    '''
    feats = extract_features(X, attr=attr)
    ids = feats['id']
    tok_sets = feats['toks']
    brands = feats['brand']
    n = len(ids)

    def make_key(*parts):
        return ['_'.join(str(p) for p in tup) for tup in zip(*parts)]

    passes = [
        # Tier 1: Model codes + Brand + Cap
        make_key(feats['brand'], feats['cap'], feats['m1'], feats['stoks']),
        make_key(feats['m1'], feats['brand'], feats['cap']),
        make_key(feats['m1'], feats['cap'], feats['brand']),
        make_key(feats['brand'], feats['fam'], feats['cap'], feats['stoks']),
        make_key(feats['fam'], feats['cap'], feats['brand']),
        make_key(feats['pnum'], feats['brand'], feats['cap']),
        make_key(feats['pnum'], feats['cap']),
        make_key(feats['brand'], feats['cap'], feats['m2']),
        make_key(feats['m2'], feats['brand'], feats['cap']),
        make_key(feats['m2'], feats['cap']),
        make_key(feats['brand'], feats['fam'], feats['cap']),
        # Tier 2: Canonical sorted tokens
        feats['stoks'],
        feats['rtoks'],
        # Tier 3: Brand + Tokens, Cap + Tokens
        make_key(feats['brand'], feats['cap'], feats['stoks']),
        make_key(feats['cap'], feats['brand'], feats['fam'], feats['stoks']),
        make_key(feats['brand'], feats['fam']),
        make_key(feats['cap'], feats['stoks']),
        make_key(feats['brand'], feats['stoks']),
    ]

    target_cand_size = 2000000
    cand_set = set()

    # Precompute sorted index orders once across all passes
    sorted_indices_list = [np.argsort(k) for k in passes]

    max_w = 20
    for w in range(1, max_w + 1):
        for sorted_indices in sorted_indices_list:
            for i in range(n - w):
                idx1, idx2 = sorted_indices[i], sorted_indices[i + w]
                # Filter out pairs with no shared tokens or brand (e.g. alphabet boundary jumps)
                if (not tok_sets[idx1].isdisjoint(tok_sets[idx2])) or (bool(brands[idx1]) and brands[idx1] == brands[idx2]):
                    id1, id2 = ids[idx1], ids[idx2]
                    if id1 > id2:
                        id1, id2 = id2, id1
                    if id1 != id2 and (id1, id2) not in cand_set:
                        cand_set.add((id1, id2))
                        if len(cand_set) >= target_cand_size:
                            break
            if len(cand_set) >= target_cand_size:
                break
        if len(cand_set) >= target_cand_size:
            break

    return list(cand_set)


def save_output(candidate_pairs, filename='output.csv'):
    '''
    Saves candidate pairs to output.csv with exactly 2,000,000 rows.
    Ensures lid < rid, no self-pairs, no duplicate pairs.
    :param candidate_pairs: list of (lid, rid) tuples
    :param filename: output CSV path
    '''
    target_size = 2000000
    if len(candidate_pairs) > target_size:
        candidate_pairs = candidate_pairs[:target_size]
    elif len(candidate_pairs) < target_size:
        needed = target_size - len(candidate_pairs)
        max_id = max((max(lid, rid) for lid, rid in candidate_pairs), default=0)
        base = max(100_000_000, max_id + 1)
        # lid = base + i, rid = base + needed + i (strictly lid < rid, no duplicates, no self-pairs)
        dummy_pairs = [(base + i, base + needed + i) for i in range(needed)]
        candidate_pairs = list(candidate_pairs) + dummy_pairs

    df_out = pd.DataFrame(candidate_pairs, columns=['lid', 'rid'])
    df_out.to_csv(filename, index=False)


# Read input dataset (checks workspace X2.csv first, then sample_data/X2.csv)
data_path = '../SIGMOD_full_data/X2.csv' if os.path.exists('../SIGMOD_full_data/X2.csv') else './SIGMOD_full_data/X2.csv'
X2 = pd.read_csv(data_path)

# Perform blocking
X2_candidate_pairs = block_with_attr(X2, attr='name')

# Save results
save_output(X2_candidate_pairs, filename='../SIGMOD_full_data/X2_candidate_pairs.csv')
