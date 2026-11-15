"""
Blocking for X2 (electronic products).

Produces output.csv with up to 2,000,000 pairs (lid, rid), lid < rid.
Candidate generation is linear in the number of records:
- each record keeps only a few rare blocking keys
- keys whose document frequency is outside [2, MAX_DF] are ignored
- each record scores only a bounded number of neighbors
- sorted-neighborhood passes use a fixed window (at most 16 neighbors)

Ground truth (Z2 / Y2) is not read.
"""

import csv
import heapq
import math
import os
import re
import sys
import time

TARGET_PAIRS = 2_000_000
MAX_DF = 35
MAX_KEYS = 10
SCAN_BUDGET = 220
TOP_K = 20
MAX_WINDOW = 16

BRANDS = (
    "sandisk", "kingston", "samsung", "toshiba", "intenso", "lexar",
    "transcend", "gigastone", "verbatim", "patriot", "silicon", "adata",
    "sony", "pny", "corsair", "philips", "emtec", "hama",
)
CAP_OK = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
CAP_BIT = {c: 1 << i for i, c in enumerate(CAP_OK)}

STOP = set(
    """a an the and or for with von der die das den dem des ein eine mit und
    ou pour avec con per del della di da of to in on at by from zu zur zum
    au aux les des du""".split()
)
GENERIC = set(
    """usb stick flash drive drives pendrive chiavetta card carte cards
    memory memoire memoria class adapter sdhc sdxc microsd microsdhc
    microsdxc micro sd uhs high speed type gen cle lecteur tarjeta karta
    scheda speicherkarte""".split()
)

HTML_RE = re.compile(r"&[a-z#0-9]+;?")
TOK_RE = re.compile(r"[a-z0-9]+")
CAP_RE = re.compile(r"(\d{1,4})\s*(gb|go)\b")
CAP_TOK_RE = re.compile(r"(\d+)(gb|go)\Z")
SPEED_RE = re.compile(r"\d+(mb|mo|mbs|mbps)\Z")
HYPHEN_RE = re.compile(r"(?<=[a-z0-9])-(?=[a-z0-9])")
TRANS = str.maketrans(
    {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "á": "a", "à": "a", "ä": "a", "â": "a",
        "ö": "o", "ü": "u", "í": "i", "ó": "o",
        "ú": "u", "ñ": "n", "ß": "s", "?": " ",
    }
)


def normalize(text):
    text = HTML_RE.sub(" ", str(text).lower())
    text = HYPHEN_RE.sub("", text.translate(TRANS))
    return text


def find_input():
    for path in ["../SIGMOD_full_data/X2.csv"]:
        if os.path.exists(path):
            return path
    return None


def load_table(path):
    # csv is faster and lighter than pandas on a multi-million-row file
    ids = []
    names = []
    brands = []
    descriptions = []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_id = (row.get("id") or "").strip()
            if raw_id.endswith(".0"):
                raw_id = raw_id[:-2]
            ids.append(int(raw_id))
            names.append(row.get("name") or "")
            brands.append(row.get("brand") or "")
            descriptions.append(row.get("description") or "")
    return ids, names, brands, descriptions


class Vocabulary:
    def __init__(self):
        self.tok2id = {}
        self.id2tok = []
        self.df = []

    def intern(self, token):
        idx = self.tok2id.get(token)
        if idx is None:
            idx = len(self.id2tok)
            self.tok2id[token] = idx
            self.id2tok.append(token)
            self.df.append(0)
        return idx


def parse_records(names, brands, descriptions):
    vocab = Vocabulary()
    n = len(names)
    brand_ids = [0] * n  # 0 = unknown
    caps = [0] * n
    # token lists stored after the first pass so idf weights can be applied once
    word_tokens = [None] * n
    model_tokens = [None] * n
    brand_lookup = {b: i + 1 for i, b in enumerate(BRANDS)}

    for i, (name, brand, desc) in enumerate(zip(names, brands, descriptions)):
        raw_brand = normalize(brand)
        text = normalize(name)
        if raw_brand:
            text = text + " " + raw_brand
        if desc and desc not in ("nan", "None"):
            # model codes sometimes sit only in the description; skip long boilerplate
            text = text + " " + normalize(desc)[:240]

        raw = TOK_RE.findall(text)
        found_brand = 0
        for tok in raw:
            bid = brand_lookup.get(tok)
            if bid:
                found_brand = bid
                break
        if not found_brand:
            for tok in TOK_RE.findall(raw_brand):
                bid = brand_lookup.get(tok)
                if bid:
                    found_brand = bid
                    break
        brand_ids[i] = found_brand

        capmask = 0
        for amount, _unit in CAP_RE.findall(text):
            bit = CAP_BIT.get(int(amount))
            if bit:
                capmask |= bit
        words = []
        models = []
        seen_w = set()
        seen_m = set()
        for tok in raw:
            cap_tok = CAP_TOK_RE.fullmatch(tok)
            if cap_tok:
                bit = CAP_BIT.get(int(cap_tok.group(1)))
                if bit:
                    capmask |= bit
                continue
            if tok in STOP or tok in brand_lookup or tok.isdigit() and len(tok) < 6:
                continue
            if SPEED_RE.fullmatch(tok):
                continue
            has_digit = any(ch.isdigit() for ch in tok)
            if tok.isdigit() and len(tok) >= 6:
                if tok not in seen_m:
                    seen_m.add(tok)
                    models.append(vocab.intern("d" + tok))
                continue
            if has_digit and len(tok) >= 4:
                if tok not in seen_m:
                    seen_m.add(tok)
                    models.append(vocab.intern(tok))
                continue
            if len(tok) >= 3 and tok not in seen_w:
                seen_w.add(tok)
                words.append(vocab.intern(tok))
        for tok_id in words:
            vocab.df[tok_id] += 1
        for tok_id in models:
            vocab.df[tok_id] += 1
        word_tokens[i] = words
        model_tokens[i] = models
        caps[i] = capmask

    return vocab, brand_ids, caps, word_tokens, model_tokens


def build_vectors(vocab, word_tokens, model_tokens):
    n_docs = len(word_tokens)
    idf = [0.0] * len(vocab.id2tok)
    for tok_id, df in enumerate(vocab.df):
        idf[tok_id] = math.log((n_docs + 1) / (df + 1)) + 1.0

    generic_ids = set()
    for tok, tok_id in vocab.tok2id.items():
        if tok in GENERIC:
            generic_ids.add(tok_id)

    vecs = [None] * n_docs
    norms = [1.0] * n_docs
    specific_ids = [None] * n_docs
    for i, (words, models) in enumerate(zip(word_tokens, model_tokens)):
        vec = []
        spec = []
        for tok_id in words:
            weight = idf[tok_id] * (0.25 if tok_id in generic_ids else 1.0)
            specific = tok_id not in generic_ids
            vec.append((tok_id, weight, specific, False))
            if specific:
                spec.append(tok_id)
        for tok_id in models:
            vec.append((tok_id, idf[tok_id] * 2.2, False, True))
        vec.sort()
        # drop duplicate token ids (shouldn't happen) by keeping the last
        if vec:
            compact = [vec[0]]
            for item in vec[1:]:
                if item[0] == compact[-1][0]:
                    compact[-1] = item
                else:
                    compact.append(item)
            vec = compact
        vecs[i] = vec
        norms[i] = math.sqrt(sum(w * w for _t, w, _s, _m in vec)) or 1.0
        specific_ids[i] = spec
    return vecs, norms, specific_ids, generic_ids


def pair_score(i, j, vecs, norms, brand_ids, caps):
    a = vecs[i]
    b = vecs[j]
    ia = ib = 0
    na = len(a)
    nb = len(b)
    dot = 0.0
    inter = 0
    spec_inter = 0
    model_hit = False
    while ia < na and ib < nb:
        ta, wa, sa, ma = a[ia]
        tb, wb, sb, mb = b[ib]
        if ta == tb:
            dot += wa * wb
            inter += 1
            if sa and sb:
                spec_inter += 1
            if ma and mb:
                model_hit = True
            ia += 1
            ib += 1
        elif ta < tb:
            ia += 1
        else:
            ib += 1
    cosine = dot / (norms[i] * norms[j])
    shorter = na if na < nb else nb
    contain = (inter / shorter) if shorter else 0.0
    score = 0.65 * cosine + 0.35 * contain
    bi = brand_ids[i]
    bj = brand_ids[j]
    if bi and bj and bi != bj:
        score -= 0.35
    ci = caps[i]
    cj = caps[j]
    if ci and cj:
        score += 0.04 if (ci & cj) else -0.55
    if model_hit:
        score += 0.10
    if spec_inter:
        score += 0.04 * (spec_inter if spec_inter < 3 else 3)
    return score


def build_postings(vocab, brand_ids, caps, word_tokens, model_tokens, specific_ids):
    """Posting lists for keys with 2 <= df <= MAX_DF, plus each record's rarest keys."""
    n = len(word_tokens)
    # token_id -> posting, created lazily for eligible tokens
    token_posting = {}
    postings = []

    def posting_for_token(tok_id):
        slot = token_posting.get(tok_id)
        if slot is None:
            slot = len(postings)
            token_posting[tok_id] = slot
            postings.append([])
        return slot

    eligible_model = []
    eligible_word = []
    for tok_id, df in enumerate(vocab.df):
        eligible_model.append(2 <= df <= MAX_DF)
        eligible_word.append(2 <= df <= MAX_DF)

    rec_keys = [[] for _ in range(n)]
    for i, models in enumerate(model_tokens):
        keys = rec_keys[i]
        for tok_id in models:
            if eligible_model[tok_id]:
                slot = posting_for_token(tok_id)
                postings[slot].append(i)
                keys.append(slot)

    for i, words in enumerate(word_tokens):
        keys = rec_keys[i]
        for tok_id in words:
            tok = vocab.id2tok[tok_id]
            if tok in GENERIC:
                continue
            if eligible_word[tok_id]:
                slot = posting_for_token(tok_id)
                postings[slot].append(i)
                keys.append(slot)

    # brand + capacity + rare word. Count df first, keep only mid-frequency keys.
    bcw_count = {}
    bcw_of_record = [[] for _ in range(n)]
    for i, spec in enumerate(specific_ids):
        brand = brand_ids[i]
        capmask = caps[i]
        if not brand or not capmask or not spec:
            continue
        rare = sorted(spec, key=lambda tok_id: (vocab.df[tok_id], tok_id))[:4]
        cap_bits = [bit for bit in CAP_BIT.values() if capmask & bit]
        local = []
        for bit in cap_bits:
            for tok_id in rare:
                # hapax words cannot form a repeated key; very common words are too coarse
                df = vocab.df[tok_id]
                if df < 2 or df > 200:
                    continue
                key = (brand, bit, tok_id)
                bcw_count[key] = bcw_count.get(key, 0) + 1
                local.append(key)
        bcw_of_record[i] = local

    bcw_slot = {}
    for key, df in bcw_count.items():
        if 2 <= df <= MAX_DF:
            bcw_slot[key] = len(postings)
            postings.append([])
    del bcw_count

    for i, keys in enumerate(bcw_of_record):
        rec = rec_keys[i]
        seen = set()
        for key in keys:
            slot = bcw_slot.get(key)
            if slot is None or slot in seen:
                continue
            seen.add(slot)
            postings[slot].append(i)
            rec.append(slot)
    del bcw_of_record
    del bcw_slot

    # keep the rarest keys (shortest postings) for each record
    lengths = [len(p) for p in postings]
    for i, keys in enumerate(rec_keys):
        if len(keys) > MAX_KEYS:
            keys.sort(key=lambda slot: (lengths[slot], slot))
            del keys[MAX_KEYS:]
    return postings, rec_keys, lengths


class TopPairs:
    """Bounded unique pair buffer. Stores at most `limit` highest-scoring pairs."""

    def __init__(self, limit):
        self.limit = limit
        self.heap = []
        self.members = set()

    def __len__(self):
        return len(self.heap)

    def add(self, score, left_id, right_id):
        if left_id == right_id:
            return
        if left_id > right_id:
            left_id, right_id = right_id, left_id
        key = (left_id, right_id)
        if key in self.members:
            return
        heap = self.heap
        if len(heap) < self.limit:
            heapq.heappush(heap, (score, left_id, right_id))
            self.members.add(key)
            return
        if score <= heap[0][0]:
            return
        _old_score, old_left, old_right = heapq.heapreplace(
            heap, (score, left_id, right_id)
        )
        self.members.discard((old_left, old_right))
        self.members.add(key)

    def sorted_pairs(self):
        ordered = sorted(self.heap, reverse=True)
        return [(left, right) for _score, left, right in ordered]


def consider_neighbors(i, neighbors, best, vecs, norms, brand_ids, caps, real_ids):
    rid_i = real_ids[i]
    for j in neighbors:
        if j == i:
            continue
        best.add(
            pair_score(i, j, vecs, norms, brand_ids, caps),
            rid_i,
            real_ids[j],
        )


def index_pass(postings, rec_keys, lengths, best, vecs, norms, brand_ids, caps, real_ids):
    n = len(rec_keys)
    score_map = {}
    seen = []
    report = max(n // 10, 1)
    for i, keys in enumerate(rec_keys):
        if i and i % report == 0:
            print(f"  index scan {i}/{n}  kept {len(best)}", flush=True)
        if not keys:
            continue
        seen.clear()
        scanned = 0
        for slot in keys:
            weight = 1.0 / lengths[slot]
            for j in postings[slot]:
                if j == i:
                    continue
                prev = score_map.get(j)
                if prev is None:
                    score_map[j] = weight
                    seen.append(j)
                else:
                    score_map[j] = prev + weight
            scanned += lengths[slot]
            if scanned >= SCAN_BUDGET:
                break
        if not seen:
            continue
        if len(seen) > TOP_K:
            seen.sort(key=score_map.get, reverse=True)
            chosen = seen[:TOP_K]
        else:
            chosen = seen
        consider_neighbors(i, chosen, best, vecs, norms, brand_ids, caps, real_ids)
        for j in seen:
            del score_map[j]


def neighborhood_pass(groups, window, best, vecs, norms, brand_ids, caps, real_ids, sigs):
    for members in groups:
        size = len(members)
        if size < 2:
            continue
        members.sort(key=lambda row: sigs[row])
        span = size - 1 if size <= window else window
        for pos in range(size):
            left = members[pos]
            last = pos + 1 + span
            if last > size:
                last = size
            consider_neighbors(
                left,
                members[pos + 1:last],
                best,
                vecs,
                norms,
                brand_ids,
                caps,
                real_ids,
            )


def brand_cap_groups(brand_ids, caps):
    groups = {}
    for i, (brand, capmask) in enumerate(zip(brand_ids, caps)):
        if not brand or not capmask:
            continue
        for bit in CAP_BIT.values():
            if capmask & bit:
                key = (brand, bit)
                bucket = groups.get(key)
                if bucket is None:
                    groups[key] = [i]
                else:
                    bucket.append(i)
    return list(groups.values())


def block(ids, names, brands, descriptions, limit=TARGET_PAIRS):
    started = time.time()
    print(f"parsing {len(ids)} records", flush=True)
    vocab, brand_ids, caps, word_tokens, model_tokens = parse_records(
        names, brands, descriptions
    )
    vecs, norms, specific_ids, _generic = build_vectors(vocab, word_tokens, model_tokens)
    sigs = []
    for spec, models in zip(specific_ids, model_tokens):
        parts = [vocab.id2tok[tok_id] for tok_id in spec]
        parts.extend(vocab.id2tok[tok_id] for tok_id in models)
        parts.sort()
        sigs.append(" ".join(parts))
    print(f"building index  ({time.time() - started:.1f}s)", flush=True)
    postings, rec_keys, lengths = build_postings(
        vocab, brand_ids, caps, word_tokens, model_tokens, specific_ids
    )
    # model groups stay small because only codes with df <= MAX_DF are used
    model_buckets = {}
    for i, models in enumerate(model_tokens):
        for tok_id in models:
            df = vocab.df[tok_id]
            if 2 <= df <= MAX_DF:
                bucket = model_buckets.get(tok_id)
                if bucket is None:
                    model_buckets[tok_id] = [i]
                else:
                    bucket.append(i)
    model_groups = list(model_buckets.values())
    del model_buckets

    best = TopPairs(limit)
    print(f"scoring index neighbors  keys={len(postings)}  ({time.time() - started:.1f}s)", flush=True)
    index_pass(postings, rec_keys, lengths, best, vecs, norms, brand_ids, caps, ids)
    del postings
    del rec_keys

    print(f"sorted neighborhood  kept {len(best)}  ({time.time() - started:.1f}s)", flush=True)
    neighborhood_pass(
        brand_cap_groups(brand_ids, caps),
        MAX_WINDOW,
        best,
        vecs,
        norms,
        brand_ids,
        caps,
        ids,
        sigs,
    )
    neighborhood_pass(
        model_groups,
        12,
        best,
        vecs,
        norms,
        brand_ids,
        caps,
        ids,
        sigs,
    )
    print(f"done in {time.time() - started:.1f}s  pairs {len(best)}", flush=True)
    return best.sorted_pairs()


def write_output(pairs, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lid", "rid"])
        writer.writerows(pairs)


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else find_input()
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output.csv"
    if not in_path or not os.path.exists(in_path):
        print(
            "Could not find X2.csv "
            "(../SIGMOD_full_data/X2.csv, data/X2.csv, X2.csv, sample_data/X2.csv)"
        )
        sys.exit(1)
    print(f"reading {in_path}", flush=True)
    ids, names, brands, descriptions = load_table(in_path)
    pairs = block(ids, names, brands, descriptions, TARGET_PAIRS)
    if len(pairs) > TARGET_PAIRS:
        pairs = pairs[:TARGET_PAIRS]
    write_output(pairs, out_path)
    print(f"wrote {len(pairs)} pairs to {out_path}", flush=True)


if __name__ == "__main__":
    main()
