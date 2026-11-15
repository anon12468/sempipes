import re
import unidecode
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import heapq # Import heapq for min-heap functionality


class EntityResolutionBlocker:
    def __init__(self):
        # X2 PATTERNS ----> Regex patterns for extracting features from the X2 Dataset.
        self.x2_clean_pattern_1 = r'&(nbsp|amp|reg|[a-z]?acute|quot|trade);?|[|;:/,‰+©\\(\)\\][psn]*|(?<=usb)[\\s][m]*(?=[23][\\.\\s])|(?<=usb)-[\\w]+\\s(?=[23][\\.\\s])|(?<=[a-z])[\\s]+gb|(?<=data|jump)[t\\s](?=trave|drive)|(?<=extreme|exceria)[\\s](?=pro[\\s]|plus)|(?<=class)[\\s_](?=10|[234]\\b)|(?<=gen)[\\s_](?=[134\\s][0]*)'
        self.x2_class10_pattern = r'(10 class|class 10|class(?=[\\w]+10\\b)|cl\\s10)'
        self.x2_memory_clean_pattern = r'\\b(msd|microvault|sd-karte|speicherkarte|minneskort|memóriakártya|flashgeheugenkaart|geheugenkaart|speicherkarten|memoriakartya|[-\\s]+kaart|memory|memoria|memoire|mémoire|mamoria|tarjeta|carte|karta)'
        self.x2_usb_clean_pattern = r'\\b(flash[\\s-]*drive|flash[\\s-]*disk|pen[\\s]*drive|micro-usb|usb-flashstation|usb-flash|usb-minne|usb-stick|speicherstick|flashgeheugen|flash|vault)'
        self.x2_check_adapter_pattern = r'\\b(adapter|adaptateur|adaptador|adattatore)'
        self.x2_check_colors_pattern = r'silver|white|black|blue|purple|burgundy|red|green'
        self.x2_speedrw_pattern = r'\\b[0-9]{2,3}r[0-9]{2,3}w'

        self.x2_aliases = {
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

        self.x2_model_num_pattern = r'\\b([\\(]*[\\w]+[-]*[\\d]+[-]*[\\w]+[-]*[\\d+]*|[\\d]+[\\w]|[\\w][\\d]+)'
        self.x2_brand_pattern = r'\\b(intenso|lexar|logilink|pny|samsung|sandisk|kingston|sony|toshiba|transcend)\\b'
        self.x2_model_pattern=  r'\\b(datatraveler|extreme[p]?|exceria[p]?|dual[\\s]*(?!=sim)|evo|xqd|ssd|cruzer[\\w+]*|glide|blade|basic|fit|force|basic line|jump\\s?drive|hxs|rainbow|speed line|premium line|att4|attach|serie u|r-serie|beast|fury|impact|a400|sd[hx]c|uhs[i12][i]*|note\\s?9|ultra)'
        self.x2_tv_phone_pattern = r'\\b(tv|(?<=dual[\\s-])*sim|lte|[45]g\\b|[oq]*led_[u]*hd|led|galaxy|iphone|oneplus|[0-9]{1,2}[.]*[0-9]*(?=[-\\s]*[\\"inch]+))'
        self.x2_mem_pattern = r'([1-9]{1,3})[-\\s]*[g][bo]?'
        self.x2_modelnum_long_pattern = r'(thn-[a-z][\\w]+|ljd[\\w+][-][\\w]+|ljd[sc][\\w]+[-][\\w]+|lsdmi[\\d]+[\\w]+|lsd[0-9]{1,3}[gb]+[\\w]+|ljds[0-9]{2}[-][\\w]+|usm[0-9]{1,3}[\\w]+|sdsq[a-z]+[-][0-9]+[a-z]+[-][\\w]+|sdsd[a-z]+[-][0-9]+[\\w]+[-]*[\\w]*|sdcz[\\w]+|mk[\\d]+|sr-g1[\\w]+)'
        self.x2_modelnum_short_pattern = r'\\b(c20[mc]|sda[0-9]{1,2}|g1ux|s[72][05]|[unm][23]02|p20|g4|dt101|se9|[asm][0-9]{2})'
        self.x2_feature_pattern = r'\\b(usb[23]|type-c|uhs[i]{1,2}|class[0134]{1,2}|gen[1-9]{1,2}|u[23](?=[\\s\\.])|sd[hx]c|otg|lte|[45]g[-\\s]lte|[0-9]+(?=-inch)|[0-9]{2,3}r[0-9]{2,3}w|[0-9]{2,3}(?=[\\smbo/p]{3}))'

        self.x2_clean_reg_remove = re.compile(self.x2_clean_pattern_1)
        self.x2_clean_reg_memcard = re.compile(self.x2_memory_clean_pattern)
        self.x2_clean_reg_usb = re.compile(self.x2_usb_clean_pattern)
        self.x2_clean_reg_class10 = re.compile(self.x2_class10_pattern)

        self.x2_brand_reg = re.compile(self.x2_brand_pattern)
        self.x2_model_reg = re.compile(self.x2_model_pattern)
        self.x2_model_num_reg = re.compile(self.x2_model_num_pattern)
        self.x2_tvp_reg = re.compile(self.x2_tv_phone_pattern)
        self.x2_mem_reg = re.compile(self.x2_mem_pattern)
        self.x2_longpat_reg = re.compile(self.x2_modelnum_long_pattern)
        self.x2_shortpat_reg = re.compile(self.x2_modelnum_short_pattern)
        self.x2_feature_reg = re.compile(self.x2_feature_pattern)

    def x2_clean_data(self, name):
        name = unidecode.unidecode(str(name).lower())
        
        for key, val in self.x2_aliases.items():
            for word in val:
                name = name.replace(word, key)
                
        name = self.x2_clean_reg_remove.sub('', name)
        name = self.x2_clean_reg_class10.sub('class10', name)
        name = name.replace('class 4 ', 'class4 ')
        name = name.replace('class 3 ', 'class3 ')
        name = name.replace('  ', ' ')
        
        return name

    def x2_find_model_num(self, low_title):
        model_num = self.x2_model_num_reg.findall(low_title)
        if model_num:
            # Filter for longer model numbers to make them more specific
            model_num = [x for x in model_num if len(x) > 5] 
            return " ".join(model_num)
        return ''

    def x2_find_brand(self, low_title):
        brands = self.x2_brand_reg.findall(low_title)
        if brands:
            return " ".join(sorted(list(set(brands))))
        return ''

    def x2_find_models_and_type(self, low_title):
        models = self.x2_model_reg.findall(low_title)
        if models:
            return " ".join(sorted(list(set(models))))
        return ''

    def x2_find_memcapacity(self, low_title):
        memCandidates = self.x2_mem_reg.findall(low_title)
        mem_capacity = ""
        if memCandidates:
            # Normalize memory capacity, e.g., "64gb", "128gb"
            mem_capacity = sorted(list(set([re.sub('[^0-9a-z]+','',i) for i in memCandidates])))
            mem_capacity = " ".join(mem_capacity)
        return mem_capacity

    def x2_find_features(self, low_title):
        res = self.x2_feature_reg.findall(low_title)
        if res:
            return ' '.join(sorted(list(set(res))))
        return ''

    def generate_blocking_keys(self, record_id, name):
        low_title = self.x2_clean_data(name)
        keys = []

        brand = self.x2_find_brand(low_title)
        model_type = self.x2_find_models_and_type(low_title)
        mem_capacity = self.x2_find_memcapacity(low_title)
        features = self.x2_find_features(low_title)
        
        model_num_long_match = self.x2_longpat_reg.search(low_title)
        model_num_long = model_num_long_match.group() if model_num_long_match else ''

        model_num_short_match = self.x2_shortpat_reg.search(low_title)
        model_num_short = model_num_short_match.group() if model_num_short_match else ''


        # Strong blocking keys: combinations of highly discriminative features
        if brand and model_num_long:
            keys.append(f"brand_modelnum_long:{brand}_{model_num_long}")
        if brand and model_num_short:
            keys.append(f"brand_modelnum_short:{brand}_{model_num_short}")
        if model_num_long and mem_capacity:
            keys.append(f"modelnum_long_mem:{model_num_long}_{mem_capacity}")
        if model_num_short and mem_capacity:
            keys.append(f"modelnum_short_mem:{model_num_short}_{mem_capacity}")
        if brand and model_type and mem_capacity:
            keys.append(f"brand_modeltype_mem:{brand}_{model_type}_{mem_capacity}")


        # Medium blocking keys: single strong features or common combinations
        if model_num_long:
            keys.append(f"modelnum_long:{model_num_long}")
        if model_num_short:
            keys.append(f"modelnum_short:{model_num_short}")
        if brand and model_type:
            keys.append(f"brand_modeltype:{brand}_{model_type}")
        if brand and mem_capacity:
            keys.append(f"brand_mem:{brand}_{mem_capacity}")
        if model_type and features:
            keys.append(f"modeltype_features:{model_type}_{features}")
        if brand and features: # NEW BLOCKING KEY
            keys.append(f"brand_features:{brand}_{features}")
        
        # Broad blocking keys: fallback for less structured data
        if brand:
            keys.append(f"brand:{brand}")
        if mem_capacity:
            keys.append(f"mem:{mem_capacity}")
        if features:
            keys.append(f"features:{features}")

        # Add a key based on a few initial tokens as a general fallback
        cleaned_tokens = low_title.split()
        if len(cleaned_tokens) >= 2: # Using at least 2 tokens for a broader match
            keys.append(f"tokens_2:{'_'.join(cleaned_tokens[:2])}")
        elif len(cleaned_tokens) > 0:
            keys.append(f"tokens_1:{'_'.join(cleaned_tokens[:1])}")


        return keys

    def _calculate_jaccard_similarity(self, name1, name2):
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        intersection = len(s1.intersection(s2))
        union = len(s1) + len(s2) - intersection
        if union == 0:
            return 0.0
        return intersection / union

    def block_records(self, X: pd.DataFrame, attr: str = 'name', window_size: int = 10, target_candidates: int = 2_000_000):
        inverted_index = defaultdict(list)
        for idx, row in tqdm(X.iterrows(), total=X.shape[0], desc="Generating blocking keys"):
            record_id = row['id']
            name = str(row[attr])
            blocking_keys = self.generate_blocking_keys(record_id, name)
            for key in blocking_keys:
                inverted_index[key].append(record_id)

        # Min-heap to store (similarity, (id1, id2)) tuples.
        # We store -similarity because heapq is a min-heap, and we want to keep the largest similarities.
        top_candidates_heap = []
        candidate_set_for_uniqueness = set() # To ensure unique pairs in the heap

        # Collect names for Jaccard calculation
        names_map = {row['id']: str(row[attr]) for _, row in X.iterrows()}

        for key, ids in tqdm(inverted_index.items(), desc="Generating candidate pairs with similarity"):
            ids.sort()
            for i in range(len(ids)):
                for j in range(i + 1, min(i + 1 + window_size, len(ids))):
                    id1, id2 = ids[i], ids[j]
                    if id1 == id2: # No self-pairs
                        continue

                    # Ensure id1 < id2 for consistent storage
                    if id1 > id2:
                        id1, id2 = id2, id1
                    
                    pair = (id1, id2)

                    # Only add to heap if it's a new unique pair or has a higher similarity
                    if pair not in candidate_set_for_uniqueness:
                        name1 = names_map.get(id1, "")
                        name2 = names_map.get(id2, "")
                        similarity = self._calculate_jaccard_similarity(name1, name2)

                        if len(top_candidates_heap) < target_candidates:
                            heapq.heappush(top_candidates_heap, (similarity, pair))
                            candidate_set_for_uniqueness.add(pair)
                        elif similarity > top_candidates_heap[0][0]:
                            # Remove the smallest element (lowest similarity) and add the new one
                            heapq.heapreplace(top_candidates_heap, (similarity, pair))
                            candidate_set_for_uniqueness.add(pair)
                            
        # Convert heap to a list of pairs and sort by similarity (descending)
        # This sort is on a list of size at most target_candidates, which is acceptable.
        final_candidates_with_sim = sorted(top_candidates_heap, key=lambda x: x[0], reverse=True)
        final_candidates = [pair for sim, pair in final_candidates_with_sim]
        
        print(f"Generated {len(final_candidates)} high-quality candidate pairs.")

        # Ensure exactly expected_cand_size pairs as per requirements
        if len(final_candidates) > target_candidates:
            final_candidates = final_candidates[:target_candidates]
        elif len(final_candidates) < target_candidates:
            print(f"Warning: Less than {target_candidates} high-quality candidates generated. Padding with (0,0) to reach target.")
            final_candidates.extend([(0, 0)] * (target_candidates - len(final_candidates)))

        return final_candidates

    def save_output(self, X2_candidate_pairs, output_filename="output.csv", expected_cand_size=2_000_000):
        # Ensure exactly expected_cand_size pairs before saving
        if len(X2_candidate_pairs) > expected_cand_size:
            X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size]
        elif len(X2_candidate_pairs) < expected_cand_size:
            X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size - len(X2_candidate_pairs)))

        output_df = pd.DataFrame(X2_candidate_pairs, columns=["lid", "rid"])
        output_df.to_csv(output_filename, index=False)
