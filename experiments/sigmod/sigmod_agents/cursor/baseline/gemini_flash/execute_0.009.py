from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re
import unidecode


#X1 PATTERNS ----> Regex patterns for extracting features from the X1 Dataset.
x1_clean_pattern_1 = r'quality|new|good|best|kids|product[s]*|(?<=\s)buy\s|computer[s]*|\s[-]|(?<=i[357])-|[|;:/,‰+©\(\)\\][psn]*|(?<=usb)[\s](?=[m23][.\s])|(?<=[a-z])[\s]+gb|(?<=gen)[\s_](?=[134\s][0]*)'
x1_aliases = {
    'panasonic': ['pansonic'],
    'notebook': ['notebooks'],
    'tablet': ['tablets'],
    'pavilion': ['pavillion'],
    'duo ': ['core2duo ', 'core 2 '],
    'hp': ['hewlett-packard'],
    'used ': ['use '],
    ' ': ['cheapest', 'cheap', 'portable', 'laptop', 'kids', ';']
}
 
core_list=['i5 3320m', 'i7 3667u', 'i7 2', 'i5 3427u', 'i5 2520m', 'i7 3520m', 'i7 620m', '2 duo', 'i3 2367m', 'i5 2540m', 'i5 3360m', 'i3 4010u', 'i3 2348m', 'i5 4210u', 'a8 5545m', 'i5 2 ', 'i5 3210m', 'i3 3110m', 'i5 4200u', 
'i7 4702mq', 'i7 3667u', 'i5 m520', 'i5 3380m', 'i5 2450m', 'i7 620m,', 'i5 4310u', 'i5 2467m', 'i7 720qm', 'i5 2410m', 'i5 560m', 'i5 3230m', 'i7 2620m', 'i3 3227u','i5 3437u','i7 4500u','i7 m640','i7 3630qm','i5 3317u','p8700','p8600','l9400']
cpu_map = {
    'i5': ['3320m', '3427u', '2540m', '3360m', '4210u', '3210m', '4200u', 'm520', '3380m', '2450m', '4310u', '2410m', '560m','3230m', '3437u', '3317u'],
    'i7': ['3667u', '3520m', '620m', '4702mq', '720qm', '2620m','4500u', '3630qm'],
    'i3' : ['2367m', '2348m', '3110m', '3227u']
}

intel_cpu_pattern = r'\b([i][357][-\s]*)|((?<=[\w+\d])*duo)|\b(pentium|atom|centrino|celeron|xeon|[-]*a8[-]*|radeon|athlon|turion|phenom|a6-?[0-9]*)'
intel_cpu_pattern_specific = r'([0-9]{4}[mqu]m?)'
intel_or_amd = r'\b(intel|amd)'
brands_pattern = r'\b(acer|panasonic|toshiba|hp|sony|lenovo|asus|dell|msi|xmg|targus|apple|macbook|aoson|gateway|microsoft|ibm|tandberg|tecra)\b'
models_pattern = r'\b((?!note|\snet|\smac)[\w]+book|(?!think)[\w]+pad|(?<=aspire )e\s|via8850|aspire|dominator-89|vostro|precision|compaq|raid|vaio|satellite|swift|envy|pavilion|voodoo|3000|flex|legion|miix|skylight|yoga|z60m|ultrabay|rog|xps|inspiron|adamo|latitude|e[67]240|[ns][0-9]{4}|g[\s]*15|[0-9]{4}p|studio|rog|zephyrus|aficio|carbon|precision|[0-9]{4}dx|x[12][23]0|x[12]|(?<=elitebook\s)[0-9]{4}[mp]|[et][54][3421]0[sp]*|alienware|travelmate|gateway|edge|thinkcentre|thinkserver|proone)\b'
nums_pattern = r'\b([\w]+[\d]+[-][\w]+[\d]+|[\d]+[\w]+[-][\d]+[\w]+|[\w]+[\d]+[-][\d]+[\w]+|[\d]+[\w]+[-][\w]+[\d]+|[\(][0-9a-z]+[\)])\b'
mems_pattern = r'[\s][1-9]{1,2}[\s]*?gb[\s]*?((sdram)|(ddr3)|(ram)|(memory))*'
seller_pattern = r'\b(firstshop|alibaba|ebay|mygofer|walmart|bestbuy|miniprice|softwarecity|thenerds|hoh.de|buy.net|topendelectronics|techbuy|schenker|overstock|tigerdirect|amazon|vology|paypal)'
loc_pattern = r'\b(china|johannes|india|russia|usa|uk|australia|japan|shenzhen)'
x1_longpattern = r'\b(g75[\w]+-[\w]+|l-[0-9]{6}|e[135]-[0-9]{3}|e[56][0-9]{3}|bx[0-9]+|nq[0-9]{4}|10a[a-z][0-9]+|9s7-[\w\d]+|[\w\d]+(?=#aba)|[sc]55[d]*-?[a-z][0-9]+|[a-z]{2,3}?[0-9]+[a-z]+|np[0-9]+[\w]|alw[0-9]{2}-[0-9]+slv|bx[\w]+|cf-[\w\d]+|20[abcd][\w\d]+|20[abc][0a][0-9a-z]+|ux[\w\d]+|v11h[0-9]+|v3-[0-9]+|mg[\w\d]+\s|[um][0-9]{4}\b)'
x1_check_colors_pattern = r'silver|white|black|blue|purple|red|green'
x1_features_pattern = r'(phenom[2]*|ssd|hdd|backlit|android|dvd|bluetooth|nvidia|refurbished|webcam|switching|used|reconditioned|wifi|camera|lcd|led|office|sata)'
x1_thinkpad_pat = r'\b(x1 carbon|x[12][023][0-3][e]?)'
lenovo_tp_modelnums = {
    'x230': r'(232[045]|343[0-9])', 
    'x130e': r'(062|233)(?=[2789])',
    'x201': r'3[06][0-9]{2}',
    'x1 carbon': r'(34[46][048])'   
}

aspire_cpu_map = {
    '6484': 'i34010u',
    '3401': 'i34010u',
    '2957': 'celeron2957u',
    '6458': 'i33110m',
    '6607': 'i32348m',
    '3234': 'i32348m',
    '5420': 'i54200u',
    '6870': 'i54200u',
    '767j': 'i74510u',
    '78s3': 'i7-4510U',
    '3311': 'i33110m',
    '6479': 'i54200u',
    '5842': 'i54200u',
    '30f1': 'i34005u',
    '38kj': 'i34005u',
    '6496': 'i53230m'    
}
x1_aspire_cpu_pat = r'i[357][-\s][0-9]{4}[mhqu]|a[0-9][-\s][0-9]{4}|celeron [0-9]{4}[mhqu]|pentium [0-9]{4}[mhqu]'
x1_aspire_model_pat = r'[a-z][0-9]-[0-9]{3}[a-z]*-?[0-9a-z]{4}'

x1_clean_reg = re.compile(x1_clean_pattern_1)
x1_cpu_reg_1 = re.compile(intel_cpu_pattern)
x1_cpu_reg_intamd = re.compile(intel_or_amd)
x1_cpu_reg_specific = re.compile(intel_cpu_pattern_specific)
x1_brands_reg = re.compile(brands_pattern)
x1_models_reg = re.compile(models_pattern)
x1_model_nums_reg = re.compile(nums_pattern)
x1_mems_reg = re.compile(mems_pattern)
x1_seller_reg = re.compile(seller_pattern)
x1_loc_reg = re.compile(loc_pattern)
x1_longpat_reg = re.compile(x1_longpattern)
x1_color_reg = re.compile(x1_check_colors_pattern)
x1_features_reg = re.compile(x1_features_pattern)
x1_thinkpad_reg = re.compile(x1_thinkpad_pat)
lenovo_tp_regs = {}
for key in lenovo_tp_modelnums:
    lenovo_tp_regs[key] = re.compile(lenovo_tp_modelnums[key])

#X2 PATTERNS ----> Regex patterns for extracting features from the X2 Dataset.
x2_clean_pattern_1 = r'&(nbsp|amp|reg|[a-z]?acute|quot|trade);?|[|;:/,‰+©\(\)\\][psn]*|(?<=usb)[\s]m*(?=[23][.\s])|(?<=usb)-[\w]+\s(?=[23][.\s])|(?<=[a-z])[\s]+gb|(?<=data|jump)[t\s](?=trave|drive)|(?<=extreme|exceria)[\s](?=pro[\s]|plus)|(?<=class)[\s_](?=10|[234]\b)|(?<=gen)[\s_](?=[134\s][0]*)'
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
x2_model_pattern=  r'\b(datatraveler|extreme[p]?|exceria[p]?|dual[\s]*(?!sim)|evo|xqd|ssd|cruzer[\w+]*|glide|blade|basic|fit|force|basic line|jump\s?drive|hxs|rainbow|speed line|premium line|att4|attach|serie u|r-serie|beast|fury|impact|a400|sd[hx]c|uhs[i12][i1]*|note\s?9|ultra)'
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

'''
    The following are matching functions for the X1 dataset
'''

def get_lenovo_key(pattern_1):
    model = find_single_occurence_compreg(x1_thinkpad_reg,pattern_1)
    model_num = ''    
    if model in lenovo_tp_modelnums:
        model_num = find_single_occurence_compreg(lenovo_tp_regs[model], pattern_1)
        if 'x230' in model and model_num == '2320':
            model_num = '3435'

    return model+model_num

def find_cpu(low_title):

    cpu_search = x1_cpu_reg_1.search(low_title)
    cpu_search_specific = x1_cpu_reg_specific.findall(low_title)
    
    if cpu_search is None:
        cpu_search = x1_cpu_reg_intamd.search(low_title)
    
    cpu_res = []
    if cpu_search is not None:
        cpu_res.append(cpu_search.group())
    if len(cpu_search_specific)>0:
        cpu_res.extend(list(set(cpu_search_specific)))

    return ("".join([re.sub('[^0-9a-z]','',i) for i in cpu_res])).strip()

def find_brands(low_title):
    brand_search = x1_brands_reg.findall(low_title)
    if brand_search is not None:
        return " ".join(sorted(list(brand_search)))
    else:
        return ''

def find_models(low_title):
    model_search = x1_models_reg.findall(low_title)
    if model_search is not None:
        return " ".join(sorted(list(model_search)))
    else:
        return ''
def find_modelnum(low_title):
    num_search = x1_model_nums_reg.findall(low_title)
    if num_search is not None:
        return " ".join(sorted(list(num_search)))
    else:
        return ''

def find_mems(low_title):
    ram_capacity = ''
    ramCandidates = x1_mems_reg.search(low_title)
    ram_digits = 1

    if ramCandidates is not None:
        ram_capacity = ramCandidates.group()[:5]
        ram_capacity = re.sub('[^0-9a-z]+','',ram_capacity)
    
    return ram_capacity

def find_mems_memory(low_title):
    ram_capacity = ''
    ramCandidates = re.search(r'[\s][1-9]{2,3}[\s]*?[GgTt][Bb][\s]*?(ssd|hdd|memory)*', low_title)
    ram_digits = 1

    if ramCandidates is not None:
        ram_capacity = ramCandidates.group()[:6]
        ram_capacity = re.sub('[^0-9]+','',ram_capacity)+'gb'
    else:
        ramCandidates = re.search(r'[\s][1-9][\s]*?[Tt][Bb][\s]*?(ssd|hdd|memory)*\b', low_title)
        if ramCandidates is not None:
            ram_capacity = ramCandidates.group()[:3]
            ram_capacity = re.sub('[^0-9]+','',ram_capacity)+'tb'
    return ram_capacity

def find_hd_type(low_title):
    hdtype = re.search(r'ssd|hdd',low_title)
    if hdtype is not None:
        return hdtype.group()
    return ''

def find_sellers(low_title):
    seller_search = x1_seller_reg.findall(low_title)
    if seller_search is not None and len(seller_search)>0:
        return " ".join(sorted(set(seller_search)))
    else:
        return ''

def find_location(low_title):
    loc_search =x1_loc_reg.search(low_title)
    if loc_search is not None:
        return loc_search.group()
    else:
        return ''

def x1_find_features(low_title):
    features = x1_features_reg.findall(low_title)
    if features is not None and len(features)>0:
        return " ".join(sorted(list(set(features))))
    return ''

def x1_clean_data(name):
    name = unidecode.unidecode(str(name).lower())
    
    for key,val in x1_aliases.items():
        for word in val:
            name = name.replace(word,key)
            
    name = x1_clean_reg.sub('',name)
    name = name.replace("  ", " ")
    
    return name

'''
    The following are matching functions for the x2 dataset
'''
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
    if res is not None:
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

    for i in tqdm(range(X.shape[0]), desc="Extracting features"):
        if attr == "title":
            attr_i = str(X[attr][i])
            cleaned_title = x1_clean_data(attr_i.lower())

            # Always add a simple token-based blocking for general recall
            tokens = cleaned_title.split()
            for token in tokens:
                if len(token) > 2: # Only block on tokens longer than 2 characters
                    pattern2id[token].append(i)

            # More specific blocking keys
            brands = find_brands(cleaned_title)
            models = find_models(cleaned_title)
            cpus = find_cpu(cleaned_title)
            mems = find_mems(cleaned_title)
            model_nums = find_modelnum(cleaned_title)
            features = x1_find_features(cleaned_title)

            # Combine features for more discriminative keys
            if brands and models:
                pattern2id[f"brand_model_{brands}_{models}"].append(i)
            if brands and cpus:
                pattern2id[f"brand_cpu_{brands}_{cpus}"].append(i)
            if models and mems:
                pattern2id[f"model_mem_{models}_{mems}"].append(i)
            if brands and model_nums:
                pattern2id[f"brand_modelnum_{brands}_{model_nums}"].append(i)
            if models and model_nums:
                pattern2id[f"model_modelnum_{models}_{model_nums}"].append(i)
            if cpus and mems:
                pattern2id[f"cpu_mem_{cpus}_{mems}"].append(i)
            if brands and models and cpus:
                pattern2id[f"brand_model_cpu_{brands}_{models}_{cpus}"].append(i)
            if brands and features:
                pattern2id[f"brand_feature_{brands}_{features}"].append(i)

            # Handle specific cases like Lenovo Thinkpad and Acer Aspire
            if 'lenovo' in brands or 'thinkpad' in cleaned_title:
                len_model = get_lenovo_key(cleaned_title)
                if len_model:
                    pattern2id[len_model].append(i)

            elif 'acer' in brands and 'aspire' in cleaned_title:
                cpu_or_model = x2_find_single_occurence(x1_aspire_cpu_pat, cleaned_title)
                cpu_or_model = re.sub(r'[^0-9a-z]', '', cpu_or_model)
                if not cpu_or_model:
                    pat = x2_find_single_occurence(x1_aspire_model_pat, cleaned_title)
                    if pat and pat[-4:] in aspire_cpu_map:
                        cpu_or_model = aspire_cpu_map[pat[-4:]]
                    elif pat:
                        cpu_or_model = pat
                if cpu_or_model:
                    pattern2id[f"acer_aspire_{cpu_or_model}"].append(i)


        elif attr == 'name':
            attr_i = str(X[attr][i])
            cleaned_name = x2_clean_data(attr_i.lower())

            # Always add a simple token-based blocking for general recall
            tokens = cleaned_name.split()
            for token in tokens:
                if len(token) > 2: # Only block on tokens longer than 2 characters
                    pattern2id[token].append(i)

            # More specific blocking keys for X2
            x2_model_nums = x2_find_model_num(cleaned_name)
            x2_brands = x2_find_brand(cleaned_name)
            x2_model_types = x2_find_models_and_type(cleaned_name)
            x2_mems = x2_find_memcapacity(cleaned_name)
            x2_features = find_all_occurences_sorted_unique_compreg(x2_feature_reg, cleaned_name)

            # Combine features for more discriminative keys
            if x2_brands and x2_features:
                pattern2id[f"brand_feature_{x2_brands}_{x2_features}"].append(i)
            if x2_brands and x2_mems:
                pattern2id[f"brand_mem_{x2_brands}_{x2_mems}"].append(i)
            if x2_brands and x2_model_types:
                pattern2id[f"brand_modeltype_{x2_brands}_{x2_model_types}"].append(i)
            if x2_brands and x2_model_nums:
                pattern2id[f"brand_modelnum_{x2_brands}_{x2_model_nums}"].append(i)
            if x2_mems and x2_model_types:
                pattern2id[f"mem_modeltype_{x2_mems}_{x2_model_types}"].append(i)
            if x2_model_nums and x2_features:
                pattern2id[f"modelnum_feature_{x2_model_nums}_{x2_features}"].append(i)

            x2_long_pattern = x2_find_single_occurence(x2_modelnum_long_pattern, cleaned_name)
            if x2_long_pattern:
                pattern2id[x2_long_pattern].append(i)

            x2_short_pattern = x2_find_all_occurences_sorted_unique(x2_modelnum_short_pattern, cleaned_name)
            if x2_short_pattern:
                pattern2id[x2_short_pattern].append(i)

    candidate_pairs = set()
    len_threshold = 200 # Increased threshold slightly, but rely more on combined keys
    for pattern, ids in tqdm(pattern2id.items(), desc="Generating candidate pairs"):
        if len(ids) > 1 and len(ids) < len_threshold:
            # Use sorted neighborhood method or similar for large buckets if needed,
            # but for now, rely on more discriminative keys keeping bucket sizes small.
            # If a bucket is still too large, this simple loop is still O(k^2).
            # For now, I'll rely on the more discriminative keys to keep `len(ids)` small.
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    id1, id2 = ids[i], ids[j]
                    if id1 < id2:
                        candidate_pairs.add((id1, id2))
                    else:
                        candidate_pairs.add((id2, id1))
        # If len(ids) >= len_threshold, skip to avoid O(k^2) as per prompt
        # This is the crucial part to enforce the feasibility constraint.
        # Very frequent terms that create large buckets will be ignored for pair generation.


    candidate_pairs = list(candidate_pairs)
    jaccard_similarities = []
    candidate_pairs_real_ids = []

    for it in tqdm(candidate_pairs, desc="Calculating Jaccard similarities"):
        id1, id2 = it
        real_id1 = X['id'][id1]
        real_id2 = X['id'][id2]

        if real_id1 < real_id2:
            candidate_pairs_real_ids.append((real_id1, real_id2))
        elif real_id2 < real_id1:
            candidate_pairs_real_ids.append((real_id2, real_id1))

        # Compute standard jaccard similarity
        name1 = str(X[attr][id1])
        name2 = str(X[attr][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        intersection_size = len(s1.intersection(s2))
        union_size = len(s1) + len(s2) - intersection_size
        if union_size > 0:
            jaccard_similarities.append(intersection_size / union_size)
        else:
            jaccard_similarities.append(0.0) # Handle empty sets gracefully

    # Sort candidate pairs by jaccard similarity
    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    print("FINAL candidate pairs before truncation:", len(candidate_pairs_real_ids))
    return candidate_pairs_real_ids


def save_output(X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dat
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)

# read the datasets
X2 = pd.read_csv("../SIGMOD_full_data/X2.csv")

# perform blocking
X2_candidate_pairs = block_with_attr(X2, attr="name")

# save results
save_output(X2_candidate_pairs)