"""Scalable entity-resolution blocking for X2 (electronic products).

Candidate generation is multi-pass sorted neighborhood plus rare-token
neighborhoods. Every bucket is scanned with a bounded window, never all-pairs,
and only the best 2,000,000 pairs are retained.
"""

from collections import Counter, defaultdict
import heapq
import math
import os
import re

import pandas as pd


LIMIT = 2_000_000
WINDOW = 18
RARE_DF_MAX = 60

BRANDS = (
    "sandisk", "kingston", "lexar", "toshiba", "intenso", "samsung",
    "sony", "pny", "transcend", "hyperx", "logilink",
)
FAMILY_BRAND = {
    "datatraveler": "kingston",
    "jumpdrive": "lexar",
    "cruzer": "sandisk",
    "exceria": "toshiba",
    "transmemory": "toshiba",
    "hyperx": "kingston",
    "canvas": "kingston",
    "microvault": "sony",
}
FAMILIES = (
    "datatraveler", "jumpdrive", "transmemory", "microvault", "cruzer",
    "exceria", "canvas", "glide", "flair", "attache", "savage", "premium",
    "basic", "endurance", "hyperx", "select", "dual", "edge", "fit",
    "extreme", "ultra",
)
CAP_SET = {4, 8, 16, 32, 64, 128, 256, 512, 1024}
STOP = {
    "usb", "gb", "go", "gio", "sd", "sdhc", "sdxc", "micro", "microsd",
    "microsdhc", "microsdxc", "uhs", "uhs1", "uhs2", "class", "classe",
    "clase", "cl", "card", "carte", "karta", "tarjeta", "memoria", "memory",
    "memoire", "pami", "drive", "flash", "stick", "pendrive", "cle", "chave",
    "pro", "plus", "ultra", "extreme", "high", "speed", "class10", "c10",
    "u3", "u1", "a1", "v30", "type", "black", "red", "white", "silver",
    "nero", "de", "la", "le", "les", "et", "pour", "with", "and", "the",
    "to", "of", "in", "un", "une", "zu", "der", "die", "das", "mb", "s",
    "i", "adapter", "adaptateur", "adattatore", "adaptador", "blister",
    "pack", "new", "original", "tesco", "direct", "fnac", "amazon", "shop",
    "online", "html", "nbsp", "amp", "quot", "mbps", "mo", "read", "write",
    "vitesse", "foto", "home", "page", "unit", "unidad", "chiave",
    "speicherkarte", "minneskort", "geheugenkaart",
}

HTML_RE = re.compile(r"&(?:nbsp|amp|quot|eacute);")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
CAP_RE = re.compile(r"\b(\d{1,4})\s*(gb|go|gio)\b")
CAP_G_RE = re.compile(r"\b(4|8|16|32|64|128|256|512)g\b")
CAP_MB_RE = re.compile(r"\b(\d{4,6})\s*mb\b")
CAP_EMBED_RE = re.compile(r"(?:^|[^0-9])(4|8|16|32|64|128|256|512)g(?:b|o)?(?:[^a-z]|$)")
MODEL_RE = re.compile(r"\b[a-z]{1,8}\d[\w]*\b")
MODEL_STRIP_RE = re.compile(r"\d{2,3}g[bo]?.*$")
LONGNUM_RE = re.compile(r"\b\d{6,}\b")
SD_RE = re.compile(r"\bsd\b")


SYNONYM_RE = (
    (re.compile(r"datattravell?er|data[\s\-]*travell?er"), "datatraveler"),
    (re.compile(r"jump[\s\-]*drive"), "jumpdrive"),
    (re.compile(r"micro[\s\-]*vault"), "microvault"),
    (re.compile(r"cruzer[\s\-]*glide"), "cruzer glide"),
    (re.compile(r"trans[\s\-]*memory"), "transmemory"),
)


def _normalize(text):
    s = HTML_RE.sub(" ", str(text).lower())
    s = s.replace("é", "e").replace("è", "e").replace("ä", "a").replace("ö", "o")
    for pat, repl in SYNONYM_RE:
        s = pat.sub(repl, s)
    s = NON_ALNUM_RE.sub(" ", s)
    s = s.replace("microsdxc", "microsd").replace("microsdhc", "microsd")
    return s


def _caps(s):
    caps = set()
    for m in CAP_RE.finditer(s):
        v = int(m.group(1))
        if v in CAP_SET:
            caps.add(v)
    for m in CAP_G_RE.finditer(s):
        caps.add(int(m.group(1)))
    for m in CAP_MB_RE.finditer(s):
        gb = int(m.group(1)) // 1024
        if gb in CAP_SET:
            caps.add(gb)
    for m in CAP_EMBED_RE.finditer(s):
        caps.add(int(m.group(1)))
    return frozenset(caps)


def _brand(name_n, brand_raw, families):
    for b in BRANDS:
        if b in name_n:
            return b
    br = str(brand_raw).lower()
    for b in BRANDS:
        if b in br:
            return b
    for fam in families:
        inferred = FAMILY_BRAND.get(fam)
        if inferred:
            return inferred
    return "unk"


def _families(s):
    return tuple(f for f in FAMILIES if f in s)


def _tokens(s):
    return frozenset(t for t in s.split() if t not in STOP and len(t) >= 2)


def _models(s):
    out = set()
    for m in MODEL_RE.findall(s):
        if len(m) >= 3 and m not in STOP:
            out.add(m)
            stripped = MODEL_STRIP_RE.sub("", m)
            if len(stripped) >= 3:
                out.add(stripped)
    out.update(LONGNUM_RE.findall(s))
    if "datatraveler" in s or "dt101" in s or re.search(r"\bdt\b", s):
        if "101" in s or "g2" in s or "dt101" in s:
            out.add("dt101g2")
    return frozenset(out)


def _ptype(s):
    if "microsd" in s:
        return "microsd"
    if "sdhc" in s or "sdxc" in s or SD_RE.search(s):
        return "sd"
    if "usb" in s or "pendrive" in s or "stick" in s:
        return "usb"
    return "other"


def extract_features(X):
    n = len(X)
    names = X["name"].astype(str).tolist()
    brand_col = X["brand"].astype(str).tolist() if "brand" in X.columns else ["nan"] * n
    prices = X["price"].tolist() if "price" in X.columns else [float("nan")] * n
    rec_ids = X["id"].astype(int).tolist()

    norms = [_normalize(nm) for nm in names]
    fams = [_families(s) for s in norms]
    brands = [_brand(norms[i], brand_col[i], fams[i]) for i in range(n)]
    caps = [_caps(s) for s in norms]
    toks = [_tokens(s) for s in norms]
    models = [_models(s) for s in norms]
    types = [_ptype(s) for s in norms]

    df = Counter()
    for tset in toks:
        df.update(tset)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    idf_sum = [sum(idf[t] for t in tset) for tset in toks]
    rare_max = min(RARE_DF_MAX, max(20, n // 20))
    rare = [frozenset(t for t in tset if 2 <= df[t] <= rare_max) for tset in toks]

    return {
        "n": n,
        "ids": rec_ids,
        "prices": prices,
        "norms": norms,
        "brands": brands,
        "caps": caps,
        "fams": [frozenset(f) for f in fams],
        "toks": toks,
        "models": models,
        "types": types,
        "idf": idf,
        "idf_sum": idf_sum,
        "rare": rare,
        "df": df,
        "rare_max": rare_max,
    }


def _sort_key_lists(F):
    n = F["n"]
    keys = [[] for _ in range(5)]
    for i in range(n):
        cap = min(F["caps"][i]) if F["caps"][i] else 0
        capset = ",".join(map(str, sorted(F["caps"][i]))) if F["caps"][i] else "0"
        fam = next(iter(sorted(F["fams"][i])), "z")
        stoks = " ".join(sorted(F["toks"][i])[:12])
        mods = " ".join(sorted(F["models"][i])[:8]) or "z"
        keys[0].append((F["brands"][i], cap, fam, F["types"][i], mods, stoks))
        keys[1].append((F["brands"][i], capset, mods, F["norms"][i]))
        keys[2].append((cap, F["types"][i], F["brands"][i], fam, F["norms"][i]))
        keys[3].append((F["brands"][i], cap, fam, F["norms"][i][::-1]))
        keys[4].append((mods, F["brands"][i], cap, F["types"][i], stoks))
    return keys


def _score(F, i, j):
    a, b = F["toks"][i], F["toks"][j]
    if len(a) > len(b):
        a, b = b, a
    wi = 0.0
    ov = 0
    idf = F["idf"]
    for t in a:
        if t in b:
            ov += 1
            wi += idf[t]
    union = F["idf_sum"][i] + F["idf_sum"][j] - wi
    wjac = wi / union if union else 0.0
    ni, nj = len(F["toks"][i]), len(F["toks"][j])
    jac = ov / (ni + nj - ov) if (ni + nj - ov) else 0.0

    bi, bj = F["brands"][i], F["brands"][j]
    if bi == bj and bi != "unk":
        brand = 1.0
    elif bi != bj and bi != "unk" and bj != "unk":
        brand = -0.8
    else:
        brand = 0.25

    ci, cj = F["caps"][i], F["caps"][j]
    if ci and cj:
        cap_s = 1.2 if ci & cj else -1.2
    else:
        cap_s = 0.15

    fi, fj = F["fams"][i], F["fams"][j]
    if fi & fj:
        fam_s = 1.3
    elif fi and fj:
        fam_s = -0.6
    else:
        fam_s = 0.0

    sku_s = 1.5 if (F["models"][i] & F["models"][j]) else 0.0
    rare_s = 0.45 * len(F["rare"][i] & F["rare"][j])

    ti, tj = F["types"][i], F["types"][j]
    if ti == tj and ti != "other":
        type_s = 0.5
    elif ti != "other" and tj != "other" and ti != tj:
        type_s = -0.3
    else:
        type_s = 0.0

    p1, p2 = F["prices"][i], F["prices"][j]
    if p1 == p1 and p2 == p2 and p1 and p2:
        ratio = max(p1, p2) / max(min(p1, p2), 1e-6)
        price_s = 0.5 if ratio < 1.8 else (0.15 if ratio < 4 else -0.2)
    else:
        price_s = 0.15

    return (
        3.2 * wjac + 1.5 * jac + 1.4 * brand + 1.6 * cap_s
        + 1.3 * fam_s + 2.2 * sku_s + 0.5 * type_s + 0.35 * price_s + rare_s
    )


class TopPairs:
    """Keep at most LIMIT unique pairs with the highest scores."""

    def __init__(self, limit):
        self.limit = limit
        self.best = {}
        self.heap = []

    def _cleanup_min(self):
        heap = self.heap
        best = self.best
        while heap:
            s, i, j = heap[0]
            if best.get((i, j)) == s:
                return
            heapq.heappop(heap)

    def consider(self, i, j, score):
        if i == j:
            return
        if i > j:
            i, j = j, i
        key = (i, j)
        prev = self.best.get(key)
        if prev is not None:
            if score > prev:
                self.best[key] = score
                heapq.heappush(self.heap, (score, i, j))
            return
        if len(self.best) < self.limit:
            self.best[key] = score
            heapq.heappush(self.heap, (score, i, j))
            return
        self._cleanup_min()
        if not self.heap:
            return
        min_s, oi, oj = self.heap[0]
        if score <= min_s:
            return
        heapq.heappop(self.heap)
        if self.best.get((oi, oj)) == min_s:
            del self.best[(oi, oj)]
        self.best[key] = score
        heapq.heappush(self.heap, (score, i, j))
        if len(self.heap) > self.limit * 3:
            self._rebuild()

    def _rebuild(self):
        self.heap = [(s, i, j) for (i, j), s in self.best.items()]
        heapq.heapify(self.heap)

    def ranked_pairs(self, rec_ids):
        items = sorted(self.best.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for (i, j), _ in items:
            a, b = rec_ids[i], rec_ids[j]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            out.append((a, b))
        return out


def _scan_sorted_neighborhood(F, key_list, window, top):
    order = sorted(range(F["n"]), key=lambda i: key_list[i])
    n = F["n"]
    for a in range(n):
        ia = order[a]
        last = min(n, a + 1 + window)
        for b in range(a + 1, last):
            ib = order[b]
            top.consider(ia, ib, _score(F, ia, ib))


def _scan_rare_tokens(F, window, top):
    inv = defaultdict(list)
    for i, rset in enumerate(F["rare"]):
        for t in rset:
            inv[t].append(i)
    brands, caps, norms = F["brands"], F["caps"], F["norms"]
    for posting in inv.values():
        posting.sort(key=lambda i: (brands[i], min(caps[i]) if caps[i] else 0, norms[i]))
        k = len(posting)
        w = min(window, k)
        for a in range(k):
            ia = posting[a]
            last = min(k, a + 1 + w)
            for b in range(a + 1, last):
                ib = posting[b]
                top.consider(ia, ib, _score(F, ia, ib))


def block_with_attr(X, attr="name"):
    F = extract_features(X)
    top = TopPairs(LIMIT)
    for keys in _sort_key_lists(F):
        _scan_sorted_neighborhood(F, keys, WINDOW, top)
    _scan_rare_tokens(F, WINDOW, top)
    return top.ranked_pairs(F["ids"])


def save_output(X2_candidate_pairs):
    expected = LIMIT
    # Drop accidental duplicates while preserving score order.
    seen = set()
    cleaned = []
    for a, b in X2_candidate_pairs:
        if a == b:
            continue
        if a > b:
            a, b = b, a
        if (a, b) in seen:
            continue
        seen.add((a, b))
        cleaned.append((a, b))
    if len(cleaned) > expected:
        cleaned = cleaned[:expected]
    if len(cleaned) < expected:
        cleaned.extend([(0, 0)] * (expected - len(cleaned)))
    out = pd.DataFrame(cleaned, columns=["left_instance_id", "right_instance_id"])
    out.to_csv("output.csv", index=False)


def _input_path():
    return os.path.join("../SIGMOD_full_data", "X2.csv")

X2 = pd.read_csv(_input_path())
X2_candidate_pairs = block_with_attr(X2, attr="name")
save_output(X2_candidate_pairs)
