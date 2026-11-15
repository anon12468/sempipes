from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re
import unidecode


# X2 PATTERNS ----> Regex patterns for extracting features from the X2 Dataset.
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
        model_num = [x for x in model_num if len(x)>5] # Filter short model numbers to reduce noise
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


def block_with_attr_x2(X, attr):  # optimized for X2 dataset
    all_record_features = defaultdict(list)
    inverted_index = defaultdict(list)
    
    # 1. Extract all relevant features and build an inverted index
    for i in tqdm(range(X.shape[0]), desc="Extracting features and building inverted index"):
        record_name = str(X[attr][i])
        cleaned_name = x2_clean_data(record_name)

        # Generate multiple blocking keys (features)
        features_for_record = []
        
        # Original name tokens, cleaned and sorted
        name_tokens = " ".join(sorted(cleaned_name.split()))
        if name_tokens:
            features_for_record.append(name_tokens)
        
        # Specific patterns from baseline's pattern2id_2
        pattern_2_matches = re.findall(r'\w+\s\w+\d+', record_name) # Example: "thinkpad x1"
        if len(pattern_2_matches) != 0:
            pattern_2_matches_cleaned = " ".join(sorted([str(it).lower() for it in pattern_2_matches]))
            if pattern_2_matches_cleaned:
                features_for_record.append(pattern_2_matches_cleaned)
        
        # Hyphenated patterns & model-number sequences
        hyphen_patterns = re.findall(r'[\d\w]+[-][\d\w]+[\d\w]+|[a-z]+[0-9]+', cleaned_name)
        if len(hyphen_patterns) != 0:
            hyphen_patterns_cleaned = " ".join(sorted([str(it).lower() for it in hyphen_patterns]))
            if hyphen_patterns_cleaned:
                features_for_record.append(hyphen_patterns_cleaned)

        # Features from specific x2_find functions
        brand = x2_find_brand(cleaned_name)
        if brand: features_for_record.append(f"brand_{brand}")

        model_num = x2_find_model_num(cleaned_name)
        if model_num: features_for_record.append(f"modelnum_{model_num}")

        model_type = x2_find_models_and_type(cleaned_name)
        if model_type: features_for_record.append(f"modeltype_{model_type}")

        mem_capacity = x2_find_memcapacity(cleaned_name)
        if mem_capacity: features_for_record.append(f"memcap_{mem_capacity}")

        x2_feat = find_all_occurences_sorted_unique_compreg(x2_feature_reg, cleaned_name)
        if x2_feat: features_for_record.append(f"x2feat_{x2_feat}")

        long_pattern = x2_find_single_occurence(x2_modelnum_long_pattern, cleaned_name)
        if long_pattern: features_for_record.append(f"longpat_{long_pattern}")
        
        short_pattern = x2_find_all_occurences_sorted_unique(x2_modelnum_short_pattern, cleaned_name)
        if short_pattern: features_for_record.append(f"shortpat_{short_pattern}")
        
        # Combine features to create more specific blocking keys
        if brand and mem_capacity: features_for_record.append(f"brand_mem_{brand}_{mem_capacity}")
        if brand and model_type: features_for_record.append(f"brand_model_{brand}_{model_type}")
        if brand and model_num: features_for_record.append(f"brand_modelnum_{brand}_{model_num}")
        if model_type and mem_capacity: features_for_record.append(f"model_mem_{model_type}_{mem_capacity}")
        if x2_feat and brand: features_for_record.append(f"brand_feature_{brand}_{x2_feat}")

        all_record_features[i] = features_for_record
        for feature in features_for_record:
            if feature: # Only add non-empty features to index
                inverted_index[feature].append(i)

    candidate_pairs_set = set()
    WINDOW_SIZE = 15 # Controls how many subsequent records to compare within a blocking key's list

    # 2. Generate candidate pairs with controlled comparisons
    for record_idx_i in tqdm(range(X.shape[0]), desc="Generating candidate pairs"):
        features_of_i = all_record_features[record_idx_i]
        
        for feature_key in features_of_i:
            if feature_key in inverted_index:
                matching_ids = sorted(inverted_index[feature_key])
                
                try:
                    current_pos_in_list = matching_ids.index(record_idx_i)
                except ValueError:
                    continue # Should not happen

                for j in range(current_pos_in_list + 1, min(current_pos_in_list + 1 + WINDOW_SIZE, len(matching_ids))):
                    record_idx_j = matching_ids[j]
                    if record_idx_i < record_idx_j: # Ensure left < right
                        candidate_pairs_set.add((record_idx_i, record_idx_j))
    
    candidate_pairs = list(candidate_pairs_set)
    print(f"Generated initial {len(candidate_pairs)} candidate pairs.")

    jaccard_similarities = []
    candidate_pairs_real_ids = []

    # 3. Calculate Jaccard similarity and prepare for sorting
    for idx_pair in tqdm(candidate_pairs, desc="Calculating Jaccard similarity"):
        id1_idx, id2_idx = idx_pair
        
        real_id1 = X['id'][id1_idx]
        real_id2 = X['id'][id2_idx]
        
        # Ensure left < right for real IDs
        if real_id1 > real_id2:
            real_id1, real_id2 = real_id2, real_id1
        
        # Calculate jaccard similarity using cleaned names
        name1_cleaned_tokens = set(x2_clean_data(str(X[attr][id1_idx])).split())
        name2_cleaned_tokens = set(x2_clean_data(str(X[attr][id2_idx])).split())
        
        denominator = min(len(name1_cleaned_tokens), len(name2_cleaned_tokens)) # Changed from max to min
        if denominator == 0:
            jaccard = 0.0
        else:
            jaccard = len(name1_cleaned_tokens.intersection(name2_cleaned_tokens)) / denominator
        
        jaccard_similarities.append(jaccard)
        candidate_pairs_real_ids.append((real_id1, real_id2))
    
    # Sort candidate pairs by jaccard similarity in descending order
    final_candidate_pairs_sorted = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    
    print(f"FINAL sorted candidate pairs count: {len(final_candidate_pairs_sorted)}")
    return final_candidate_pairs_sorted


def save_output(X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 2000000 pairs for dataset X2
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 2000000 pairs for dataset X2, pad with (0,0) if needed
    # The problem implies we should generate *enough* good candidates to reach 2M
    # Padding with (0,0) is usually for testing and shouldn't happen in a real solution
    # if the blocking is effective enough to generate enough good candidates.
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        print(f"Warning: Not enough candidate pairs generated ({len(X2_candidate_pairs)}). Padding with (0,0) to reach {expected_cand_size_X2}.")
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X2_candidate_pairs
    output_df = pd.DataFrame(all_cand_pairs, columns=["lid", "rid"])
    output_df.to_csv("output.csv", index=False)

# read the dataset
X2 = pd.read_csv("../SIGMOD_full_data/X2.csv") # Using sample_data for dev, will run on full X2 later.

# perform blocking
X2_candidate_pairs = block_with_attr_x2(X2, attr="name")

# save results
save_output(X2_candidate_pairs)
