"""Scalable blocking for X2 product entity resolution.

Designed for ~2M records in < 30 minutes: never enumerates all pairs in a
large bucket (no unbounded O(k^2)), never materializes an unbounded pair
set, and keeps at most TARGET scored candidates.
"""

from collections import defaultdict, Counter
from pathlib import Path
import re
import heapq

import pandas as pd
from tqdm import tqdm

TARGET = 2_000_000

BRANDS = (
    "sandisk", "kingston", "toshiba", "samsung", "intenso",
    "transcend", "lexar", "sony", "pny", "verbatim", "emtec",
)
BRAND_SET = set(BRANDS)
# Longer names first so they match before prefixes.
SERIES = (
    "datatraveler", "datattraveller", "jumpdrive", "transmemory",
    "exceria", "cruzer", "cruizer", "hyperx", "savage", "attache",
    "attach", "extreme", "ultra", "canvas", "premium", "basic",
    "rainbow", "glide", "blade", "flair", "platinum", "professional",
    "dual", "fit",
)
TYPES = ("microsdxc", "microsdhc", "microsd", "sdxc", "sdhc", "xqd")
STOP = {
    "the", "a", "an", "of", "and", "or", "to", "for", "with", "in", "on", "by",
    "at", "from", "as", "is", "de", "la", "le", "les", "el", "los", "die",
    "der", "das", "und", "des", "du", "et", "en", "un", "une", "von", "zu",
    "im", "pour", "avec", "sur", "usb", "gb", "go", "mb", "sd", "card",
    "carte", "karta", "memoria", "memory", "flash", "drive", "stick", "cle",
    "pendrive", "speicherkarte", "class", "classe", "clase", "uhs", "adapter",
    "adaptateur", "ebay", "fnac", "tesco", "direct", "product", "color",
    "colour", "noir", "black", "silber", "silver", "type", "gen", "high",
    "speed", "line", "serie", "series", "pro", "plus", "micro", "scheda",
    "tarjeta", "chiavetta", "unidad", "lecteur", "memoire", "pami", "amazon",
    "shop", "online", "pack", "new", "free", "inc",
}
SKIP_MODELS = {
    "uhs1", "uhsi", "uhs2", "uhsii", "class10", "class4", "class6", "class2",
    "gen1", "gen2", "usb2", "usb3", "usb30", "usb20", "cl10", "cl4",
}
MB_TO_GB = {
    "4096": "4", "8192": "8", "16384": "16", "32768": "32",
    "65536": "64", "131072": "128", "262144": "256", "524288": "512",
}
CAP_EXPLICIT = re.compile(r"\b(4|8|16|32|64|128|256|512)\s*(gb|go|gi(?:ga)?|g)\b")
CAP_GLUED = re.compile(r"\b(4|8|16|32|64|128|256|512)g[bo]?\b")
CAP_STANDALONE = re.compile(r"\b(8|16|32|64|128|256|512)\b")
CAP_MB = re.compile(r"\b(\d{4,6})\s*mb\b")
TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
MODEL_RE = re.compile(r"[a-z]*\d[a-z0-9\-]*")
SKU_RE = re.compile(r"\b\d{6,10}\b")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
CAPLIKE_MODEL = re.compile(r"\d+[gmb]b?")


def _params(n):
    """Scale pairing work with dataset size; all per-bucket work is O(k)."""
    if n >= 200_000:
        return dict(max_all=10, window=4, partners=5, max_bucket=8_000, rare_k=2, sorts=2, local_cap=1500)
    if n >= 20_000:
        return dict(max_all=16, window=8, partners=10, max_bucket=8_000, rare_k=3, sorts=2, local_cap=4000)
    return dict(max_all=24, window=16, partners=48, max_bucket=2_000, rare_k=3, sorts=3, local_cap=2000)


def _extract(name, brand_field):
    text = (str(name) + " " + str(brand_field)).lower()
    text = (text.replace("&amp;", " ").replace("&quot;", " ")
                .replace("\n", " ").replace("\\n", " ")
                .replace("cruizer", "cruzer").replace("attaché", "attache"))
    compact = NON_ALNUM.sub("", text)

    brand = None
    for b in BRANDS:
        if b in compact:
            brand = b
            break

    caps = set()
    for m in CAP_EXPLICIT.finditer(text):
        caps.add(m.group(1))
    for m in CAP_GLUED.finditer(text):
        caps.add(m.group(1))
    for m in CAP_MB.finditer(text):
        mapped = MB_TO_GB.get(m.group(1))
        if mapped:
            caps.add(mapped)
    if not caps:
        caps.update(CAP_STANDALONE.findall(text))

    ptype = next((t for t in TYPES if t in compact), None)
    if ptype is None and "usb" in compact:
        ptype = "usb"

    series = frozenset(s for s in SERIES if s in compact or s in text)

    models = set()
    for t in MODEL_RE.findall(text):
        t2 = t.replace("-", "")
        if not (3 <= len(t2) <= 20):
            continue
        if t2 in SKIP_MODELS or t2.isdigit():
            continue
        if CAPLIKE_MODEL.fullmatch(t2):
            continue
        if not any(ch.isalpha() for ch in t2):
            continue
        models.add(t2)
        if len(t2) >= 8:
            models.add(t2[:8])
    for t in SKU_RE.findall(text):
        models.add(t)

    toks = set()
    for t in TOKEN_RE.findall(text):
        if t in STOP or t.isdigit() or t in BRAND_SET:
            continue
        toks.add(t)

    sig = 0
    for t in toks:
        sig ^= hash(t)
    return brand, caps, ptype, series, models, toks, sig, text


def _score(a, b, recs):
    ra, rb = recs[a], recs[b]
    inter = len(ra["toks"] & rb["toks"])
    sc = 0.28 * inter
    shared_m = ra["models"] & rb["models"]
    if shared_m:
        sc += 2.6 + 0.15 * min(3, len(shared_m))
    if ra["series"] & rb["series"]:
        sc += 1.2
    if ra["brand"] and ra["brand"] == rb["brand"]:
        sc += 0.35
    if ra["caps"] and rb["caps"] and (ra["caps"] & rb["caps"]):
        sc += 0.6
    elif ra["caps"] and rb["caps"]:
        sc -= 0.4
    if ra["ptype"] and ra["ptype"] == rb["ptype"]:
        sc += 0.2
    return sc


def _compatible(a, b, recs):
    ra, rb = recs[a], recs[b]
    if ra["brand"] and rb["brand"] and ra["brand"] != rb["brand"]:
        return False
    return True


def _strong_edge(a, b, recs, score):
    """Edges conservative enough to union into a duplicate cluster."""
    if not _compatible(a, b, recs):
        return False
    ra, rb = recs[a], recs[b]
    if ra["models"] & rb["models"]:
        return True
    if ra["series"] & rb["series"] and score >= 2.0:
        return True
    return score >= 3.2


class PairStore:
    """Bounded candidate store: dict on small n, per-record top-P heaps on large n."""

    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.max_dict = min(4 * TARGET, max(TARGET, n * max(p, 8)))
        # Dict is safe while it cannot grow past a few million entries.
        self.use_dict = n < 30_000
        self.best = {} if self.use_dict else None
        self.heaps = None if self.use_dict else [[] for _ in range(n)]

    def add(self, i, j, score):
        if i == j:
            return
        if i > j:
            i, j = j, i
        if self.best is not None:
            old = self.best.get((i, j))
            if old is None:
                if len(self.best) >= self.max_dict and score < 1.2:
                    return
                self.best[(i, j)] = score
            elif score > old:
                self.best[(i, j)] = score
            return
        self._push(i, j, score)
        self._push(j, i, score)

    def _push(self, i, j, score):
        h = self.heaps[i]
        if len(h) < self.p:
            heapq.heappush(h, (score, j))
        elif score > h[0][0]:
            heapq.heapreplace(h, (score, j))

    def _edge_map(self, recs):
        if self.best is not None:
            return dict(self.best)
        best = {}
        for i, h in enumerate(self.heaps):
            for score, j in h:
                a, b = (i, j) if i < j else (j, i)
                prev = best.get((a, b))
                if prev is None or score > prev:
                    best[(a, b)] = score
        return best

    def complete_small_clusters(self, recs):
        """If a tight neighborhood forms a tiny CC, emit all pairs inside it."""
        edges = self._edge_map(recs)
        parent = list(range(self.n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (i, j), score in edges.items():
            if _strong_edge(i, j, recs, score):
                union(i, j)
        comps = defaultdict(list)
        for i in range(self.n):
            comps[find(i)].append(i)
        max_cc = 18
        for mem in comps.values():
            if not (2 <= len(mem) <= max_cc):
                continue
            mem.sort()
            for a in range(len(mem)):
                ia = mem[a]
                for b in range(a + 1, len(mem)):
                    ib = mem[b]
                    if _compatible(ia, ib, recs):
                        self.add(ia, ib, max(_score(ia, ib, recs), 1.0))

    def unique_pairs(self, recs, real_ids):
        self.complete_small_clusters(recs)
        ranked = sorted(self._edge_map(recs).items(), key=lambda kv: kv[1], reverse=True)
        out = []
        seen = set()
        for (i, j), _ in ranked:
            if not _compatible(i, j, recs):
                continue
            lid, rid = int(real_ids[i]), int(real_ids[j])
            if lid == rid:
                continue
            if lid > rid:
                lid, rid = rid, lid
            if (lid, rid) in seen:
                continue
            seen.add((lid, rid))
            out.append((lid, rid))
            if len(out) >= TARGET:
                break
        return out


def _add_bucket_pairs(ids, recs, store, max_all, window, nsorts=3, local_cap=2000):
    """Pair a bucket in O(k * window) or O(k) for tiny buckets (k <= max_all)."""
    seen = set()
    uniq = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    k = len(uniq)
    if k < 2:
        return
    if k <= max_all:
        for a in range(k):
            ia = uniq[a]
            for b in range(a + 1, k):
                ib = uniq[b]
                if _compatible(ia, ib, recs):
                    store.add(ia, ib, _score(ia, ib, recs))
        return
    sort_fns = (
        lambda i: recs[i]["sig"],
        lambda i: recs[i]["text"],
        lambda i: recs[i]["text"][::-1],
    )[:nsorts]
    for sfn in sort_fns:
        order = sorted(uniq, key=sfn)
        for a in range(k):
            ia = order[a]
            lim = a + 1 + window
            if lim > k:
                lim = k
            for b in range(a + 1, lim):
                ib = order[b]
                if _compatible(ia, ib, recs):
                    store.add(ia, ib, _score(ia, ib, recs))
    if k > local_cap:
        return
    local_df = Counter()
    for i in uniq:
        local_df.update(recs[i]["toks"])
        local_df.update(recs[i]["models"])
    inv = defaultdict(list)
    for i in uniq:
        cands = [t for t in (recs[i]["toks"] | recs[i]["models"])
                 if 2 <= local_df[t] <= max_all]
        cands.sort(key=lambda t: local_df[t])
        for t in cands[:4]:
            inv[t].append(i)
    for lst in inv.values():
        seen = set()
        u = []
        for i in lst:
            if i not in seen:
                seen.add(i)
                u.append(i)
        L = len(u)
        if L < 2:
            continue
        if L > max_all:
            u.sort(key=lambda i: recs[i]["sig"])
            for a in range(L):
                lim = min(L, a + 1 + window)
                for b in range(a + 1, lim):
                    if _compatible(u[a], u[b], recs):
                        store.add(u[a], u[b], _score(u[a], u[b], recs))
        else:
            for a in range(L):
                for b in range(a + 1, L):
                    if _compatible(u[a], u[b], recs):
                        store.add(u[a], u[b], _score(u[a], u[b], recs))


def block_records(X):
    n = len(X)
    p = _params(n)
    names = X["name"].astype(str).tolist()
    brand_col = X["brand"].fillna("").astype(str).tolist() if "brand" in X.columns else [""] * n
    real_ids = X["id"].tolist()

    recs = []
    df = Counter()
    for i in tqdm(range(n), desc="features"):
        brand, caps, ptype, series, models, toks, sig, text = _extract(names[i], brand_col[i])
        recs.append({
            "brand": brand,
            "caps": frozenset(caps),
            "ptype": ptype,
            "series": series,
            "models": frozenset(models),
            "toks": frozenset(toks),
            "sig": sig,
            "text": text,
        })
        df.update(toks)
        df.update(models)

    def keys_of(i):
        r = recs[i]
        keys = []
        for m in r["models"]:
            if df[m] <= p["max_bucket"]:
                keys.append("m:" + m)
        ss = r["series"] if r["series"] else ("*",)
        caps = r["caps"] if r["caps"] else ("?",)
        brand = r["brand"] or "?"
        for c in caps:
            for s in ss:
                keys.append("bcs:%s|%s|%s" % (brand, c, s))
            keys.append("bct:%s|%s|%s" % (brand, c, r["ptype"]))
            if r["series"]:
                for s in r["series"]:
                    keys.append("sc:%s|%s" % (s, c))
        rare = sorted(r["toks"], key=lambda t: (df[t], t))[:p["rare_k"]]
        for c in caps:
            for t in rare:
                if 2 <= df[t] <= p["max_bucket"]:
                    keys.append("r:%s|%s|%s" % (brand, c, t))
        return keys

    buckets = defaultdict(list)
    for i in tqdm(range(n), desc="index"):
        for k in keys_of(i):
            buckets[k].append(i)

    store = PairStore(n, p["partners"])
    for key, ids in tqdm(buckets.items(), desc="buckets"):
        if len(ids) > p["max_bucket"] * 2:
            # Extremely generic key: cheap SNM only, tiny window.
            seen = set()
            uniq = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            uniq.sort(key=lambda i: recs[i]["sig"])
            w = 3
            for a in range(len(uniq)):
                lim = min(len(uniq), a + 1 + w)
                for b in range(a + 1, lim):
                    if _compatible(uniq[a], uniq[b], recs):
                        store.add(uniq[a], uniq[b], _score(uniq[a], uniq[b], recs))
            continue
        _add_bucket_pairs(
            ids, recs, store, p["max_all"], p["window"],
            nsorts=p["sorts"], local_cap=p["local_cap"],
        )

    # Global sorted-neighborhood on (brand, capacity, token-signature).
    def snm_key(i):
        r = recs[i]
        cap = min(r["caps"]) if r["caps"] else ""
        return (r["brand"] or "", cap, r["sig"])

    order = sorted(range(n), key=snm_key)
    w = p["window"]
    for a in tqdm(range(n), desc="snm"):
        ia = order[a]
        lim = min(n, a + 1 + w)
        for b in range(a + 1, lim):
            ib = order[b]
            if _compatible(ia, ib, recs):
                store.add(ia, ib, _score(ia, ib, recs))

    return store.unique_pairs(recs, real_ids)


def save_output(pairs):
    pairs = list(pairs[:TARGET])
    if len(pairs) < TARGET:
        dummy = 10 ** 12
        need = TARGET - len(pairs)
        pairs.extend((dummy + 2 * i, dummy + 2 * i + 1) for i in range(need))
    pd.DataFrame(pairs, columns=["lid", "rid"]).to_csv("output.csv", index=False)


def _load_x2():
    path = Path("../SIGMOD_full_data/X2.csv")
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError("../SIGMOD_full_data/X2.csv not found")


if __name__ == "__main__":
    X2 = _load_x2()
    pairs = block_records(X2)
    save_output(pairs)
