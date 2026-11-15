from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re
import unidecode


#X2 PATTERNS ----> Regex patterns for extracting features from the X2 Dataset.
x2_clean_pattern_1 = r'&(nbsp|amp|reg|[a-z]?acute|quot|trade);?|[|;:/,‰+©\(\)\\][psn]*|(?<=usb)[\s][m]*(?=[23][\.\s])|(?<=usb)-[\w]+\s(?=[23][\.\s])|(?<=[a-z])[\s]+gb|(?<=data|jump)[t\s](?=trave|drive)|(?<=extreme|exceria)[\s](?=pro[\s]|plus)|(?<=class)[\s_](?=10|[234]\b)|(?<=gen)[\s_](?=[134\s][0]*)'
x2_class10_pattern = r'(10 class|class 10|class(?=[\w]+10\b)|cl\s10)'
x2_memory_clean_pattern = r'\b(msd|microvault|sd-karte|speicherkarte|minneskort|memóriakártya|flashgeheugenkaart|geheugenkaart|speicherkarten|memoriakartya|[-\s]+kaart|memory|memoria|memoire|mémoire|mamoria|tarjeta|carte|karta)'
x2_usb_clean_pattern = r'\b(flash[\s-]*drive|flash[\s-]*disk|pen[\s]*drive|micro-usb|usb-flashstation|usb-flash|usb-minne|usb-stick|speicherstick|flashgeheugen|flash|vault)'
x2_check_adapter_pattern = r'\b(adapter|adaptateur|adaptador|adattatore)'
x2_check_colors_pattern = r'silver|white|black|blue|purple|burgundy|red|green'
x2_speedrw_pattern = r'\b[0-9]{2,3}r[0-9]{2,3}w'

x2_aliases = {
    'class': ['classe','clase', 'clas ','klasse', 'cl '],
    'uhsi':['uhs1','uhs-i', 'ultra high-speed'],
    'type-c': ['typec','type c','usb-c','usbc'],
    'intenso premium line': ['3534490'],
    'kingston hxs ': ['hyperx', 'savage'],
    'sony g1ux': ['serie ux'],
    ' kingston dt101 ': ['dtig4', ' 101 ', 'dt101g2'],
    ' kingston ultimate ':['sda10', 'sda3'], 
    'extreme ': ['extrem '], 
    'att4': ['attach']
}

x2_model_num_pattern = r'\b([\(]*[\w]+[-]*[\d]+[-]*[\w]+[-]*[\d+]*|[\d]+[\w]|[\w][\d]+)'
x2_brand_pattern = r'\b(intenso|lexar|logilink|pny|samsung|sandisk|kingston|sony|toshiba|transcend)\b'
x2_model_pattern=  r'\b(datatraveler|extreme[p]?|exceria[p]?|dual[\s]*(?!=sim)|evo|xqd|ssd|cruzer[\w+]*|glide|blade|basic|fit|force|basic line|jump\s?drive|hxs|rainbow|speed line|premium line|att4|attach|serie u|r-serie|beast|fury|impact|a400|sd[hx]c|uhs[i12][i]*|note\s?9|ultra)'
x2_tv_phone_pattern = r'\b(tv|(?<=dual[\s-])*sim|lte|[45]g\b|[oq]*led_[u]*hd|led|galaxy|iphone|oneplus|[0-9]{1,2}[.]*[0-9]*(?=[-\s]*["inch]+))'
x2_mem_pattern = r'([1-9]{1,3})[-\s]*[g][bo]?'
x2_modelnum_long_pattern = r'(thn-[a-z][\w]+|ljd[\w+][-][\w]+|ljd[sc][\w]+[-][\w]+|lsdmi[\d]+[\w]+|lsd[0-9]{1,3}[gb]+[\w]+|ljds[0-9]{2}[-][\w]+|usm[0-9]{1,3}[\w]+|sdsq[a-z]+[-][0-9]+[a-z]+[-][\w]+|sdsd[a-z]+[-][0-9]+[\w]+[-]*[\w]*|sdcz[\w]+|mk[\d]+|sr-g1[\w]+)'
x2_modelnum_short_pattern = r'\b(c20[mc]|sda[0-9]{1,2}|g1ux|s[72][05]|[unm][23]02|p20|g4|dt101|se9|[asm][0-9]{2})' 
x2_feature_pattern = r'\b(usb[23]|type-c|uhs[i]{1,2}|class[0134]{1,2}|gen[1-9]{1,2}|u[23](?=[\s\.])|sd[hx]c|otg|lte|[45]g[-\s]lte|[0-9]+(?=-inch)|[0-9]{2,3}r[0-9]{2,3}w|[0-9]{2,3}(?=[\smbo/p]{3}))'

x2_clean_reg_remove = re.compile(x2_clean_pattern_1)
x2_clean_reg_memcard = re.compile(x2_memory_clean_pattern)
x2_clean_reg_usb = re.compile(x2_usb_clean_pattern)
x2_clean_reg_class10 = re.compile(x2_class10_pattern)

x2_brand_reg = re.compile(x2_brand_pattern)
x2_model_reg = re.compile(x2_model_pattern)
x2_model_num_reg = re.compile(x2_model_num_pattern)
x2_tvp_reg = re.compile(x2_tv_phone_pattern)
x2_mem_reg = re.compile(x2_mem_pattern)
x2_longpat_reg = re.compile(x2_modelnum_long_pattern)
x2_shortpat_reg = re.compile(x2_modelnum_short_pattern)
x2_feature_reg = re.compile(x2_feature_pattern)


def x2_find_ptype(low_title):

    ptype = re.search(r'xqd|ssd|tv|lte', low_title)
    if ptype is None:
        if 'fdrive' in low_title:
            return 'fdrive'
        if 'memcard' in low_title:
            return 'memcard'
    else:
        ptype = ptype.group()
        if ptype == 'lte':
            ptype = 'phone'
        return ptype
    return ''

def x2_find_model_num(low_title):
    
    model_num = x2_model_num_reg.findall(low_title)
    if model_num is not None and len(model_num)>0:
        model_num = [x for x in model_num if len(x)>5]
        return " ".join(model_num)
    else:
        return ''
    
def x2_find_brand(low_title):
    
    brands = x2_brand_reg.findall(low_title)
    if brands is not None and len(brands)>0:
        return " ".join(sorted(list(set(brands))))
    else:
        return ''

def x2_find_models_and_type(low_title):
    
    models = x2_model_reg.findall(low_title)
    if models is not None and len(models)>0:
        return " ".join(sorted(list(set(models))))
    else:
        return ''
    
def x2_find_tv_phone(low_title):
    
    tvp_pattern = x2_tvp_reg.findall(low_title)
    if tvp_pattern is not None and len(tvp_pattern)>0:
        return " ".join(sorted(tvp_pattern))
    else:
        return ''
def x2_find_memcapacity(low_title):
    memCandidates = x2_mem_reg.findall(low_title)
    mem_capacity = ""
    if memCandidates is not None:
        mem_capacity = sorted(list(set([re.sub('[^0-9a-z]+','',i) for i in memCandidates])))
        mem_capacity = " ".join(mem_capacity)

    return mem_capacity

def x2_clean_data(name):
    name = unidecode.unidecode(str(name).lower())
    
    for key,val in x2_aliases.items():
        for word in val:
            name = name.replace(word,key)
            
    name = x2_clean_reg_remove.sub('',name)
    name = x2_clean_reg_class10.sub('class10',name)
    name = name.replace('class 4 ', 'class4 ')
    name = name.replace('class 3 ', 'class3 ')
    name = name.replace('  ', ' ')
    
    return name

def x2_find_single_occurence(pattern,name):
    res = re.search(pattern, name)
    if res is not None:
        return res.group()
    return ''
def x2_find_all_occurences_inorder(pattern, name):
    res = re.findall(pattern, name)
    if res is not None:
        return ' '.join(res)
    return ''

def x2_find_all_occurences_sorted_unique(pattern, name):
    res = re.findall(pattern, name)
    if res is not None and len(res)>0:
        return ' '.join(sorted(list(set(res))))
    return ''


def find_single_occurence_compreg(compiled_reg,name):
    '''
        Common - Find the result from compiled regex
    '''

    res = compiled_reg.search(name)
    if res is not None:
        return res.group()
    return ''

def find_all_occurences_inorder_compreg(compiled_reg, name):
    res = compiled_reg.findall(name)
    if res is not None:
        return ' '.join(res)
    return ''

def find_all_occurences_sorted_unique_compreg(compiled_reg, name):
    res = compiled_reg.findall(name)
    if res is not None and len(res)>0:
        return ' '.join(sorted(list(set(res))))
    return ''


def block_with_attr(X, attr):  # replace with your logic.
    '''
    This function performs blocking using attr
    :param X: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    '''

    # build index from patterns to tuples
    pattern2id = defaultdict(list)
        
    for i in tqdm(range(X.shape[0])):        
            attr_i = str(X[attr][i])
            cleaned_name = x2_clean_data(attr_i.lower())
            
            # Use various features as blocking keys
            brand = x2_find_brand(cleaned_name)
            model_type = x2_find_models_and_type(cleaned_name)
            mem_capacity = x2_find_memcapacity(cleaned_name)
            features = find_all_occurences_sorted_unique_compreg(x2_feature_reg, cleaned_name)
            
            if brand:
                pattern2id[brand].append(i)
            if model_type:
                pattern2id[model_type].append(i)
            if mem_capacity:
                pattern2id[mem_capacity].append(i)
            if features:
                pattern2id[features].append(i)
            if brand and mem_capacity:
                pattern2id[f"{brand}_{mem_capacity}"].append(i)
            if brand and model_type:
                pattern2id[f"{brand}_{model_type}"].append(i)
            if model_type and mem_capacity:
                pattern2id[f"{model_type}_{mem_capacity}"].append(i)
            
            # Add general token blocking for robustness
            for token in cleaned_name.split():
                if len(token) > 2 and token not in x2_aliases: # Avoid very short or aliased tokens
                    pattern2id[token].append(i)

    candidate_pairs = set()
    MAX_CANDIDATE_PAIRS = 2000000
    
    for pattern in tqdm(pattern2id):
        ids = list(sorted(pattern2id[pattern]))
        
        # Heuristic to limit pairs from large buckets
        # Instead of O(k^2), we can limit to a fixed number of pairs per bucket
        # Or use a smarter sampling/pair selection strategy for large buckets
        if len(ids) > 1 and len(candidate_pairs) < MAX_CANDIDATE_PAIRS:
            if len(ids) * (len(ids) - 1) / 2 > 10000: # If potential pairs in bucket is too large
                # Take a sample of pairs from large buckets
                # For simplicity, let's just take pairs between the first few and last few elements
                # A more sophisticated approach would be random sampling or LSH inside the bucket
                sample_size = min(len(ids), 100) # Limit to 100 records for pair generation
                
                for i in range(sample_size):
                    for j in range(i + 1, sample_size):
                        id1, id2 = ids[i], ids[j]
                        real_id1, real_id2 = X['id'][id1], X['id'][id2]
                        if real_id1 < real_id2:
                            candidate_pairs.add((real_id1, real_id2))
                        else:
                            candidate_pairs.add((real_id2, real_id1))
                        if len(candidate_pairs) >= MAX_CANDIDATE_PAIRS:
                            break
                    if len(candidate_pairs) >= MAX_CANDIDATE_PAIRS:
                        break
            else:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        id1, id2 = ids[i], ids[j]
                        real_id1, real_id2 = X['id'][id1], X['id'][id2]
                        if real_id1 < real_id2:
                            candidate_pairs.add((real_id1, real_id2))
                        else:
                            candidate_pairs.add((real_id2, real_id1))
                        if len(candidate_pairs) >= MAX_CANDIDATE_PAIRS:
                            break
                    if len(candidate_pairs) >= MAX_CANDIDATE_PAIRS:
                        break
        if len(candidate_pairs) >= MAX_CANDIDATE_PAIRS:
            break
            
    print("Generated raw candidate pairs: ", len(candidate_pairs))
    
    # Sort candidate pairs by Jaccard similarity and truncate
    jaccard_similarities = []
    candidate_pairs_list = list(candidate_pairs)

    for it in tqdm(candidate_pairs_list):
        id1, id2 = it
        name1 = str(X['name'][X[X['id'] == id1].index[0]])
        name2 = str(X['name'][X[X['id'] == id2].index[0]])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        jaccard_similarities.append(len(s1.intersection(s2)) / min(len(s1), len(s2)))

    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_list), reverse=True)]

    print("FINAL ", len(candidate_pairs_real_ids))
    return candidate_pairs_real_ids


def save_output(X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X2 = 2000000

    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    if len(X2_candidate_pairs) < expected_cand_size_X2:
        # Create dummy pairs to reach the expected size if not enough candidates are found
        # These will be (0,0) pairs as in the baseline
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    output_df = pd.DataFrame(X2_candidate_pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)

# read the datasets
X2 = pd.read_csv("../SIGMOD_full_data/X2.csv")

# perform blocking
X2_candidate_pairs = block_with_attr(X2, attr="name")

# save results
save_output(X2_candidate_pairs)
