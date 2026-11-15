import os
import sys
import time
import re
import pandas as pd
import numpy as np
import unidecode
from collections import defaultdict

def run_blocking(input_path=None, output_path="output.csv", max_output_pairs=2000000):
    start_time = time.time()
    
    # 1. Determine input file path
    if input_path is None:
        if os.path.exists("sample_data/X2.csv"):
            input_path = "sample_data/X2.csv"
        elif os.path.exists("X2.csv"):
            input_path = "X2.csv"
        elif os.path.exists("../sample_data/X2.csv"):
            input_path = "../sample_data/X2.csv"
        else:
            raise FileNotFoundError("Could not find X2.csv or sample_data/X2.csv")

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    n_rows = len(df)
    ids = df["id"].values
    print(f"Loaded {n_rows} records. Extracting features...")

    # 2. Domain-specific dictionaries & patterns
    clean_reg_remove = re.compile(r"&(nbsp|amp|reg|[a-z]?acute|quot|trade);?|[|;:/,‰+©()\\&]+")
    
    aliases = {
        "class10": ["10 class", "class 10", "cl 10", "classe 10", "clase 10", "klasse 10"],
        "class4": ["class 4", "cl 4", "classe 4", "clase 4", "klasse 4"],
        "class3": ["class 3", "cl 3", "classe 3", "clase 3", "klasse 3"],
        "uhsi": ["uhs1", "uhs-i", "ultra high-speed", "uhs-1"],
        "uhsii": ["uhs2", "uhs-ii", "uhs-2"],
        "type-c": ["typec", "type c", "usb-c", "usbc"],
        "datatraveler": ["data traveler", "datattraveller", "data-traveler"],
        "jumpdrive": ["jump drive", "jump-drive"],
        "transmemory": ["trans memory", "trans-memory"],
        "att4": ["attache 4", "attache4", "att4", "attache iv", "attache", "attach 4", "attach"],
        "g1ux": ["serie ux", "seria ux", "series ux"],
        "cz80": ["cz 80"],
        "dt101": ["dt 101", " 101 g2", "101g2", "dt101g2", "dtig4"],
        "u202": ["u 202", "tos-u202", "u202w", "u202-w", "u202k"],
        "s70": ["s 70"],
        "c20m": ["c 20m", "c20-m"],
        "c20c": ["c 20c", "c20-c"],
        "s25": ["s 25"],
        "p20": ["p 20"],
        "v30": ["v 30"],
        "v10": ["v 10"],
        "n101": ["n 101"],
        "n401": ["n 401"],
        "n302": ["n 302"],
        "m401": ["m 401"],
        "basic line": ["basicline"],
        "rainbow line": ["rainbowline"],
        "premium line": ["premiumline", "3534490", "3534460"],
        "speed line": ["speedline"],
        "r-serie": ["r-series", "r series", "serie r", "seria r", "micro vault r", "usm32gr", "usm16gr", "usm64gr", "usm8gr", "usm4gr"],
        "mach": ["machqx", "usm32gqx", "usm64gqx", "usm16gqx", "usm128gqx"],
        "extreme plus": ["ext+", "ext +", "extreme+"],
        "hdsl1": ["hd-sl1", "hd sl1"],
    }

    brands_list = ["sandisk", "kingston", "toshiba", "lexar", "sony", "samsung", "intenso", "pny", "transcend", "logilink"]
    valid_caps = {"1", "2", "4", "8", "16", "32", "64", "128", "200", "256", "400", "512", "1000", "2000"}
    mb_map = {
        32768: "32", 16384: "16", 65536: "64", 131072: "128", 262144: "256",
        32000: "32", 16000: "16", 64000: "64", 128000: "128", 256000: "256"
    }

    mpn_patterns = [
        re.compile(r"\b(thn-[a-z0-9]+)\b", re.I),
        re.compile(r"\b(sd[cszq][a-z0-9]+-[0-9]+[a-z0-9]+(?:-[a-z0-9]+)?)\b", re.I),
        re.compile(r"\b(ljd[a-z0-9]+-[0-9]+[a-z0-9]+)\b", re.I),
        re.compile(r"\b(lsd[a-z0-9]+-[0-9]+[a-z0-9]+|lsd[0-9]{1,3}[a-z0-9]+)\b", re.I),
        re.compile(r"\b(lxqd[a-z0-9]+-[0-9]+[a-z0-9]+|lxqd[a-z0-9]+)\b", re.I),
        re.compile(r"\b(dt[0-9a-z]+/[0-9]+[a-z0-9]+)\b", re.I),
        re.compile(r"\b(sda[0-9]+/[0-9]+[a-z0-9]+)\b", re.I),
        re.compile(r"\b(sdc[0-9a-z]*/[0-9]+[a-z0-9]+)\b", re.I),
        re.compile(r"\b(hxs[0-9a-z]*/[0-9]+[a-z0-9]+)\b", re.I),
        re.compile(r"\b(35[0-9]{5})\b"),
        re.compile(r"\b(mk[0-9]{7,12})\b", re.I),
        re.compile(r"\b(usm[0-9]{1,3}[a-z0-9]+)\b", re.I),
        re.compile(r"\b(sf-?[0-9]{1,3}[a-z0-9]+|sf[0-9]{1,3}[a-z0-9]+)\b", re.I),
        re.compile(r"\b(sr-?[a-z0-9]{3,8}|srg1[a-z0-9]+)\b", re.I),
        re.compile(r"\b(fd[0-9]+att[0-9]+[a-z0-9\-]*)\b", re.I),
        re.compile(r"\b(ts[0-9]{1,4}g[a-z0-9\-]+)\b", re.I),
        re.compile(r"\b(ua[0-9]{2}[a-z0-9]+)\b", re.I),
        re.compile(r"\b(cz[0-9]{2,3})\b", re.I),
    ]

    cleaned_titles = []
    brands = []
    caps = []
    model_words_list = []
    short_codes_list = []
    mpn_lists = []
    token_sets = []
    wwd_list = []

    # 3. Feature Extraction Loop
    for idx in range(n_rows):
        row_name = df["name"].iloc[idx]
        raw_name = str(row_name) if pd.notna(row_name) else ""
        row_brand = df["brand"].iloc[idx] if "brand" in df.columns else ""
        raw_brand = str(row_brand).lower() if pd.notna(row_brand) else ""

        # Normalize text
        name = unidecode.unidecode(raw_name.lower())
        for rep, targets in aliases.items():
            for t in targets:
                name = name.replace(t, rep)
        name = clean_reg_remove.sub(" ", name)
        name = re.sub(r"\s+", " ", name).strip()
        cleaned_titles.append(name)

        # Brand extraction (prioritize position in name, fallback to brand col, then implicit)
        b = ""
        earliest_pos = 999999
        for brand_item in brands_list:
            m = re.search(r"\b" + brand_item + r"\b", name)
            if m and m.start() < earliest_pos:
                earliest_pos = m.start()
                b = brand_item

        if not b:
            for brand_item in brands_list:
                if brand_item in raw_brand:
                    b = brand_item
                    break

        if not b:
            if "cruzer" in name or "sdisk" in name: b = "sandisk"
            elif "datatraveler" in name or "hyperx" in name or "canvas" in name or "sdca3" in name or "sdcs" in name: b = "kingston"
            elif "jumpdrive" in name: b = "lexar"
            elif "exceria" in name or "transmemory" in name: b = "toshiba"
            elif "micro vault" in name or "usm" in name or "g1ux" in name: b = "sony"
            elif "line" in name and any(x in name for x in ["basic", "rainbow", "premium", "speed", "alu"]): b = "intenso"
            elif "attach" in name or "att4" in name: b = "pny"
            elif "jetflash" in name: b = "transcend"
        brands.append(b)

        # Capacity extraction & normalization
        pats = re.findall(r"\b([1-9][0-9]{0,5})\s*[-_]?\s*(gb|go|tb|to|mb|mo|g\b)", name)
        mpn_cap = re.findall(r"[-/]([1-9][0-9]{0,3})g(?:b|o)?(?:[-/_]|\b)", raw_name.lower())
        ts_cap = re.findall(r"\bts([1-9][0-9]{0,3})g", raw_name.lower())
        caps_found = []
        for match_item in pats:
            c_val, unit = match_item[0], match_item[1].lower()
            v = int(c_val)
            if unit in ["tb", "to"]:
                v = v * 1000
            c_str = mb_map.get(v, str(v))
            if c_str in valid_caps and c_str not in caps_found:
                caps_found.append(c_str)
        for c_cand in mpn_cap + ts_cap:
            v = int(c_cand)
            c_str = mb_map.get(v, str(v))
            if c_str in valid_caps and c_str not in caps_found:
                caps_found.append(c_str)
        if not caps_found:
            caps_found = [""]
        caps.append(caps_found)

        # Model words
        m_words = re.findall(r"\b(datatraveler|extreme[p]?|exceria[p]?|dual|evo|xqd|ssd|cruzer|glide|blade|fit|force|basic line|jumpdrive|hxs|rainbow|speed line|premium line|att4|serie u|r-serie|beast|fury|impact|a400|note 9|ultra|canvas)\b", name)
        model_words_list.append(set(m_words))

        # Short codes & SKU prefixes
        sc = set(re.findall(r"\b(s[257][05]|c20[mc]|p20|v[13]0|dt101|dt100|dt50|se9|u20[23]|cz[0-9]{2,3}|n[134]0[12]|m401|sf[0-9]+[a-z0-9]*|sda[0-9]+|sdc[0-9]+|hxs[0-9]+|35[0-9]{5}|att4|r-serie|mach)\b", name))
        for m in re.findall(r"thn-([a-z][0-9]{3})", raw_name.lower()):
            sc.add(m)
        for m in re.findall(r"fd[0-9]+g?b?(att[0-9]*)", raw_name.lower()):
            sc.add(m)
        for m in re.findall(r"sd(cz[0-9]{2,3})", raw_name.lower()):
            sc.add(m)
        short_codes_list.append(sc)

        # MPNs
        m_list = []
        for pat in mpn_patterns:
            m = pat.findall(raw_name)
            if m:
                for match_str in m:
                    m_list.append(match_str.lower().replace("-", "").replace("/", ""))
        mpn_lists.append(m_list)

        # Word word digit pattern
        p2 = re.findall(r"\w+\s\w+\d+", raw_name)
        wwd_list.append(" ".join(sorted([str(it).lower() for it in p2])) if p2 else "")

        # Token set for Jaccard containment
        words = set(re.findall(r"[a-z0-9]+", name))
        token_sets.append(words)

    # 4. Multi-Pass Sorted Neighborhood Blocking
    # Pair scores mapping: pair -> max similarity
    pair_scores = {}

    def add_partition_pairs(partitions, window_size=40, min_s=0.07):
        for part_key, items in partitions.items():
            if len(items) < 2:
                continue
            # Sort partition items by sort key
            items.sort(key=lambda x: x[0])
            n_items = len(items)
            for i in range(n_items):
                _, idx_i = items[i]
                tokens_i = token_sets[idx_i]
                len_i = len(tokens_i)
                for offset in range(1, min(window_size + 1, n_items - i)):
                    _, idx_j = items[i + offset]
                    tokens_j = token_sets[idx_j]
                    len_j = len(tokens_j)

                    inter = len(tokens_i.intersection(tokens_j))
                    min_len = len_i if len_i < len_j else len_j
                    sim = inter / min_len if min_len > 0 else 0.0

                    if sim >= min_s:
                        id1 = ids[idx_i] if ids[idx_i] < ids[idx_j] else ids[idx_j]
                        id2 = ids[idx_j] if ids[idx_i] < ids[idx_j] else ids[idx_i]
                        if id1 != id2:
                            p = (id1, id2)
                            if p not in pair_scores or sim > pair_scores[p]:
                                pair_scores[p] = sim

    print("Running Pass 1: exact MPNs...")
    mpn_parts = defaultdict(list)
    for i, m_list in enumerate(mpn_lists):
        for m in m_list:
            mpn_parts[f"mpn_{m}"].append(("", i))
    add_partition_pairs(mpn_parts, window_size=50, min_s=0.0)

    print("Running Pass 2: Brand + Capacity (sorted by model + shortcode + tokens)...")
    p2_parts = defaultdict(list)
    for i in range(n_rows):
        b = brands[i]
        mw = " ".join(sorted(model_words_list[i]))
        sc = " ".join(sorted(short_codes_list[i]))
        sort_str = f"{sc}_{mw}_{' '.join(sorted(token_sets[i]))}"
        for c in caps[i][:2]:
            p2_parts[f"{b}_{c}"].append((sort_str, i))
    add_partition_pairs(p2_parts, window_size=40, min_s=0.07)

    print("Running Pass 3: Brand + Capacity (sorted by reverse title)...")
    p3_parts = defaultdict(list)
    for i in range(n_rows):
        b = brands[i]
        rev_title = " ".join(reversed(cleaned_titles[i].split()))
        for c in caps[i][:2]:
            p3_parts[f"{b}_{c}"].append((rev_title, i))
    add_partition_pairs(p3_parts, window_size=40, min_s=0.07)

    print("Running Pass 3b: Brand + Capacity (sorted by forward title)...")
    p3b_parts = defaultdict(list)
    for i in range(n_rows):
        b = brands[i]
        for c in caps[i][:2]:
            p3b_parts[f"{b}_{c}"].append((cleaned_titles[i], i))
    add_partition_pairs(p3b_parts, window_size=40, min_s=0.07)

    print("Running Pass 4: Brand + Short code...")
    p4_parts = defaultdict(list)
    for i in range(n_rows):
        b = brands[i]
        c = caps[i][0]
        if b:
            for sc in short_codes_list[i]:
                p4_parts[f"{b}_{sc}"].append((f"{c}_{cleaned_titles[i]}", i))
    add_partition_pairs(p4_parts, window_size=40, min_s=0.07)

    print("Running Pass 5: Brand + Model word...")
    p5_parts = defaultdict(list)
    for i in range(n_rows):
        b = brands[i]
        c = caps[i][0]
        if b:
            for mw in model_words_list[i]:
                p5_parts[f"{b}_{mw}"].append((f"{c}_{cleaned_titles[i]}", i))
                p5_parts[f"{b}_{mw}_nocap"].append((cleaned_titles[i], i))
    add_partition_pairs(p5_parts, window_size=40, min_s=0.07)

    print("Running Pass 6: Word word digit...")
    p6_parts = defaultdict(list)
    for i in range(n_rows):
        wwd = wwd_list[i]
        if wwd:
            p6_parts[wwd].append((cleaned_titles[i], i))
    add_partition_pairs(p6_parts, window_size=30, min_s=0.07)

    print("Running Pass 7: Brand only fallback (for uncapacitated items)...")
    p7_parts = defaultdict(list)
    for i in range(n_rows):
        if not caps[i][0] and brands[i]:
            mw = " ".join(sorted(model_words_list[i]))
            sc = " ".join(sorted(short_codes_list[i]))
            p7_parts[brands[i]].append((f"{mw}_{sc}_{cleaned_titles[i]}", i))
    add_partition_pairs(p7_parts, window_size=35, min_s=0.07)

    print("Running Pass 8: Capacity only fallback (for unbranded items)...")
    p8_parts = defaultdict(list)
    for i in range(n_rows):
        if not brands[i] and caps[i][0]:
            p8_parts[caps[i][0]].append((cleaned_titles[i], i))
    add_partition_pairs(p8_parts, window_size=30, min_s=0.07)

    print("Running Pass 9: Global sorted tokens pass...")
    p9_parts = defaultdict(list)
    for i in range(n_rows):
        sorted_tokens = " ".join(sorted(token_sets[i]))
        p9_parts["global"].append((sorted_tokens, i))
    add_partition_pairs(p9_parts, window_size=20, min_s=0.25)

    # 5. Sort candidates by similarity score descending
    print(f"Total candidate pairs found: {len(pair_scores)}")
    sorted_pairs = sorted(pair_scores.keys(), key=lambda p: pair_scores[p], reverse=True)

    # 6. Format and enforce exact 2,000,000 output constraints
    if len(sorted_pairs) > max_output_pairs:
        sorted_pairs = sorted_pairs[:max_output_pairs]

    final_pairs = list(sorted_pairs)
    existing_set = set(final_pairs)

    # Pad if dataset is smaller than max_output_pairs (e.g. sample_data subset)
    if len(final_pairs) < max_output_pairs:
        needed = max_output_pairs - len(final_pairs)
        print(f"Padding {needed} pairs to reach exactly {max_output_pairs} without self-pairs or duplicates...")
        curr = 1
        while len(final_pairs) < max_output_pairs:
            dummy_pair = (0, curr)
            if dummy_pair not in existing_set:
                final_pairs.append(dummy_pair)
                existing_set.add(dummy_pair)
            curr += 1

    print(f"Writing exactly {len(final_pairs)} candidate pairs to {output_path}...")
    # Fast chunked CSV writing
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("lid,rid\n")
        chunk_size = 100000
        for i in range(0, len(final_pairs), chunk_size):
            chunk = [f"{l},{r}\n" for l, r in final_pairs[i:i + chunk_size]]
            f.writelines(chunk)

    total_time = time.time() - start_time
    print(f"Blocking completed successfully in {total_time:.2f} seconds.")

    # 7. Evaluate against ground truth if available
    gt_path = None
    if os.path.exists("sample_data/Z2.csv"):
        gt_path = "sample_data/Z2.csv"
    elif os.path.exists("Z2.csv"):
        gt_path = "Z2.csv"
    elif os.path.exists("../sample_data/Z2.csv"):
        gt_path = "../sample_data/Z2.csv"

    if gt_path is not None:
        z2 = pd.read_csv(gt_path)
        z2_set = set(zip(z2["lid"], z2["rid"]))
        tp = len(existing_set.intersection(z2_set))
        print(f"Ground Truth Evaluation on {gt_path}:")
        print(f"True Positives: {tp} / {len(z2_set)} ({tp / len(z2_set):.2%})")
        print(f"Baseline comparison: 4,091 TP (93.15%) -> Our solution: {tp} TP ({tp / len(z2_set):.2%})")

if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else None
    out_file = sys.argv[2] if len(sys.argv) > 2 else "output.csv"
    run_blocking(input_path=in_file, output_path=out_file)
