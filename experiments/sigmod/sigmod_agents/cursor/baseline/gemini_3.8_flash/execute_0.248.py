import os
import sys
import time
import re
from collections import defaultdict
import pandas as pd
import unidecode

'''
High-Performance Entity Resolution Blocking on X2 (Electronic Products)
Multi-Pass Sorted Neighborhood Method (SNM) with Transitive Closure
Linear-time O(W * N) complexity, strictly bounded memory footprint.
'''

# ---------------------------------------------------------
# Feature Extraction Patterns & Regexes
# ---------------------------------------------------------
x2_clean_pattern_1 = r'&(nbsp|amp|reg|[a-z]?acute|quot|trade);?|[|;:/,‰+©\(\)\\][psn]*|(?<=usb)[\s][m]*(?=[23][\.\s])|(?<=usb)-[\w]+\s(?=[23][\.\s])|(?<=[a-z])[\s]+gb|(?<=data|jump)[t\s](?=trave|drive)|(?<=extreme|exceria)[\s](?=pro[\s]|plus)|(?<=class)[\s_](?=10|[234]\b)|(?<=gen)[\s_](?=[134\s][0]*)'
x2_class10_pattern = r'(10 class|class 10|class(?=[\w]+10\b)|cl\s10)'

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

x2_model_num_pattern = r'\b([\(]*[\w]+[-]*[\d]+[-]*[\w]+[-]*[\d+]*|[\d]+[\w]|[\w][\d]+)'
x2_brand_pattern = r'\b(intenso|lexar|logilink|pny|samsung|sandisk|kingston|sony|toshiba|transcend)\b'
x2_model_pattern = r'\b(datatraveler|extreme[p]?|exceria[p]?|dual[\s]*(?!=sim)|evo|xqd|ssd|cruzer[\w+]*|glide|blade|basic|fit|force|basic line|jump\s?drive|hxs|rainbow|speed line|premium line|att4|attach|serie u|r-serie|beast|fury|impact|a400|sd[hx]c|uhs[i12][i1]*|note\s?9|ultra)'
x2_mem_pattern = r'([1-9]{1,3})[-\s]*[g][bo]?'
x2_modelnum_long_pattern = r'(thn-[a-z][\w]+|ljd[\w+][-][\w]+|ljd[sc][\w]+[-][\w]+|lsdmi[\d]+[\w]+|lsd[0-9]{1,3}[gb]+[\w]+|ljds[0-9]{2}[-][\w]+|usm[0-9]{1,3}[\w]+|sdsq[a-z]+[-][0-9]+[a-z]+[-][\w]+|sdsd[a-z]+[-][0-9]+[\w]+[-]*[\w]*|sdcz[\w]+|mk[\d]+|sr-g1[\w]+)'
x2_modelnum_short_pattern = r'\b(c20[mc]|sda[0-9]{1,2}|g1ux|s[72][05]|[unm][23]02|p20|g4|dt101|se9|[asm][0-9]{2})'
x2_feature_pattern = r'\b(usb[23]|type-c|uhs[i]{1,2}|class[0134]{1,2}|gen[1-9]{1,2}|u[23](?=[\s\.])|sd[hx]c|otg|lte|[45]g[-\s]lte|[0-9]+(?=-inch)|[0-9]{2,3}r[0-9]{2,3}w|[0-9]{2,3}(?=[\smbo/p]{3}))'

x2_clean_reg_remove = re.compile(x2_clean_pattern_1)
x2_clean_reg_class10 = re.compile(x2_class10_pattern)
x2_brand_reg = re.compile(x2_brand_pattern)
x2_model_reg = re.compile(x2_model_pattern)
x2_model_num_reg = re.compile(x2_model_num_pattern)
x2_mem_reg = re.compile(x2_mem_pattern)
x2_longpat_reg = re.compile(x2_modelnum_long_pattern)
x2_shortpat_reg = re.compile(x2_modelnum_short_pattern)
x2_feature_reg = re.compile(x2_feature_pattern)


def x2_clean_data(name):
    name = unidecode.unidecode(str(name).lower())
    for key, val in x2_aliases.items():
        for word in val:
            name = name.replace(word, key)
    name = x2_clean_reg_remove.sub('', name)
    name = x2_clean_reg_class10.sub('class10', name)
    name = name.replace('class 4 ', 'class4 ').replace('class 3 ', 'class3 ')
    name = ' '.join(name.split())
    return name


def x2_find_brand(low_title):
    brands = x2_brand_reg.findall(low_title)
    if brands:
        return ' '.join(sorted(set(brands)))
    return ''


def x2_find_memcapacity(low_title):
    mem_candidates = x2_mem_reg.findall(low_title)
    if mem_candidates:
        mem_capacity = sorted(set([re.sub('[^0-9a-z]+', '', i) for i in mem_candidates]))
        return ' '.join(mem_capacity)
    return ''


def x2_find_models_and_type(low_title):
    models = x2_model_reg.findall(low_title)
    if models:
        return ' '.join(sorted(set(models)))
    return ''


def x2_find_model_num(low_title):
    model_num = x2_model_num_reg.findall(low_title)
    if model_num:
        model_num = [x for x in model_num if len(x) > 5]
        return ' '.join(model_num)
    return ''


def find_all_occurences_sorted_unique_compreg(compiled_reg, name):
    res = compiled_reg.findall(name)
    if res:
        return ' '.join(sorted(set(res)))
    return ''


def x2_find_single_occurence(pattern, name):
    res = pattern.search(name) if hasattr(pattern, 'search') else re.search(pattern, name)
    if res is not None:
        return res.group()
    return ''


def block_x2(df):
    '''
    Build blocking buckets and generate candidate pairs using Multi-Pass SNM
    and transitive closure.
    Strictly bounded O(W * N) time and bounded memory footprint.
    '''
    n_records = len(df)
    real_ids = list(df['id'])
    raw_names = [str(name) for name in df['name']]
    
    cleaned_names = []
    words_list = []
    
    pattern2id_1 = defaultdict(list)
    pattern2id_2 = defaultdict(list)
    
    print(f'Extracting features and building inverted indexes for {n_records} records...')
    for i in range(n_records):
        attr_i = raw_names[i]
        pattern_1 = attr_i.lower()
        
        # 1. Whole sorted words pattern
        sorted_title_words = ' '.join(sorted(pattern_1.split()))
        pattern2id_1[sorted_title_words].append(i)
        
        # 2. Alphanumeric sequences
        p2 = re.findall(r'\w+\s\w+\d+', attr_i)
        if p2:
            p2 = sorted([str(it).lower() for it in p2])
            pattern2id_2[' '.join(p2)].append(i)
            
        clean_title = x2_clean_data(pattern_1)
        cleaned_names.append(clean_title)
        words_list.append(set(re.findall(r'[a-z0-9]{2,}', clean_title)))
        
        brand = x2_find_brand(clean_title)
        mem = x2_find_memcapacity(clean_title)
        model = x2_find_models_and_type(clean_title)
        model_num = x2_find_model_num(clean_title)
        features = find_all_occurences_sorted_unique_compreg(x2_feature_reg, clean_title)
        long_pattern = x2_find_single_occurence(x2_longpat_reg, clean_title)
        short_pattern = find_all_occurences_sorted_unique_compreg(x2_shortpat_reg, clean_title)
        
        if features:
            pattern2id_2[' '.join([brand, features])].append(i)
        if long_pattern:
            pattern2id_2[long_pattern].append(i)
        if short_pattern:
            pattern2id_2[short_pattern].append(i)
        pattern2id_2[' '.join([brand, mem])].append(i)
        pattern2id_2[' '.join([brand, mem, model])].append(i)
        pattern2id_2[' '.join([brand, model_num])].append(i)
        
        if mem and model:
            pattern2id_2['MEM_MOD:' + mem + ' ' + model].append(i)
        if model and model_num:
            pattern2id_2['MOD_NUM:' + model + ' ' + model_num].append(i)

    # Combine all blocking buckets
    all_buckets = list(pattern2id_1.values()) + list(pattern2id_2.values())
    print(f'Total buckets generated: {len(all_buckets)}')
    
    # ---------------------------------------------------------
    # Multi-Pass Sorted Neighborhood Method (SNM)
    # ---------------------------------------------------------
    WINDOW = 30
    MAX_BUCKET_SIZE = 250
    MAX_CANDIDATES = 2_500_000
    min_sim = 0.10
    candidate_pairs = {}
    
    print('Executing Multi-Pass Sorted Neighborhood Method (SNM)...')
    for id_list in all_buckets:
        id_list = list(set(id_list))
        k = len(id_list)
        if k < 2 or k > MAX_BUCKET_SIZE:
            continue
            
        sort_keys = [
            lambda idx: cleaned_names[idx],
            lambda idx: ' '.join(sorted(words_list[idx])),
            lambda idx: ' '.join(sorted(words_list[idx], reverse=True)),
            lambda idx: ' '.join(sorted(words_list[idx], key=lambda w: len(w), reverse=True)),
        ]
        
        for sk in sort_keys:
            sorted_ids = sorted(id_list, key=sk)
            for i in range(k):
                u_idx = sorted_ids[i]
                w_u = words_list[u_idx]
                r_u = real_ids[u_idx]
                
                for j in range(i + 1, min(i + 1 + WINDOW, k)):
                    v_idx = sorted_ids[j]
                    if u_idx == v_idx:
                        continue
                    r_v = real_ids[v_idx]
                    pair = (r_u, r_v) if r_u < r_v else (r_v, r_u)
                    if pair in candidate_pairs:
                        continue
                        
                    w_v = words_list[v_idx]
                    denom = min(len(w_u), len(w_v))
                    if denom > 0:
                        sim = len(w_u.intersection(w_v)) / denom
                        if sim >= min_sim:
                            candidate_pairs[pair] = sim
                            
        # Safeguard: prevent memory explosion
        if len(candidate_pairs) > MAX_CANDIDATES:
            sorted_cands = sorted(candidate_pairs.items(), key=lambda x: x[1], reverse=True)[:2000000]
            candidate_pairs = dict(sorted_cands)
            min_sim = sorted_cands[-1][1]
            
    print(f'Candidate pairs after SNM: {len(candidate_pairs)}')
    
    # ---------------------------------------------------------
    # Transitive Closure on High-Confidence Edges
    # ---------------------------------------------------------
    print('Computing transitive closure on high-confidence candidate cliques...')
    adj = defaultdict(set)
    for (u, v), sim in candidate_pairs.items():
        if sim >= 0.30:
            adj[u].add(v)
            adj[v].add(u)
            
    visited = set()
    for node in list(adj.keys()):
        if node not in visited:
            comp = []
            q = [node]
            visited.add(node)
            for curr in q:
                comp.append(curr)
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
            if 2 <= len(comp) <= 30:
                for i in range(len(comp)):
                    u = comp[i]
                    for j in range(i + 1, len(comp)):
                        v = comp[j]
                        pair = (u, v) if u < v else (v, u)
                        if pair not in candidate_pairs:
                            candidate_pairs[pair] = 0.50
                            
    print(f'Total candidates after transitive closure: {len(candidate_pairs)}')
    
    # ---------------------------------------------------------
    # Ranking & Formatting to exactly 2,000,000 pairs
    # ---------------------------------------------------------
    sorted_cand_pairs = [p for p, _ in sorted(candidate_pairs.items(), key=lambda x: x[1], reverse=True)]
    TARGET_ROWS = 2_000_000
    
    if len(sorted_cand_pairs) > TARGET_ROWS:
        final_pairs = sorted_cand_pairs[:TARGET_ROWS]
    else:
        needed = TARGET_ROWS - len(sorted_cand_pairs)
        dummy_pairs = [(-TARGET_ROWS - k, -k) for k in range(1, needed + 1)]
        final_pairs = sorted_cand_pairs + dummy_pairs
        
    return final_pairs


def save_output(candidate_pairs, out_path='output.csv'):
    print(f'Saving {len(candidate_pairs)} pairs to {out_path}...')
    df = pd.DataFrame(candidate_pairs, columns=['lid', 'rid'])
    df.to_csv(out_path, index=False)
    print(f'Successfully saved {out_path}')


def main():
    t_start = time.time()
    
    # Locate X2.csv dataset
    candidates = ['../SIGMOD_full_data/X2.csv']
    x2_path = None
    for p in candidates:
        if os.path.exists(p):
            x2_path = p
            break
            
    if x2_path is None:
        raise FileNotFoundError('Could not find ../SIGMOD_full_data/X2.csv')
        
    print(f'Loading dataset from {x2_path}...')
    df_x2 = pd.read_csv(x2_path)
    
    final_pairs = block_x2(df_x2)
    save_output(final_pairs, out_path='output.csv')
    
    print(f'Completed successfully in {time.time() - t_start:.2f} seconds.')


if __name__ == '__main__':
    main()
