#!/usr/bin/env python3
"""Bounded candidate generation for the X2 product data.

The blocker never expands a posting list quadratically.  Each blocking key
keeps a fixed-size sample, each input row contributes only its best few
candidates, and the global heap never exceeds the requested output size.
Z2/Y2 is deliberately not referenced anywhere in this program.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import html
import math
import re
import sys
import unicodedata
import zlib
from array import array
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_LIMIT = 2_000_000
POSTING_CAP = 16
LOCAL_CANDIDATES = 12
MASK64 = (1 << 64) - 1

BRANDS = {
    "sandisk": 1,
    "kingston": 2,
    "sony": 3,
    "toshiba": 4,
    "lexar": 5,
    "intenso": 6,
    "transcend": 7,
    "samsung": 8,
    "pny": 9,
    "logilink": 10,
}
BRAND_ALIASES = {
    "sandisc": "sandisk",
    "scan disk": "sandisk",
    "kingstone": "kingston",
}

STOPWORDS = {
    # Storefront and product-page noise in the multilingual X2 titles.
    "a", "an", "and", "avec", "black", "blau", "blue", "buy", "card",
    "carte", "color", "de", "del", "der", "die", "direct", "drive", "en",
    "et", "flash", "for", "gb", "geheugenkaart", "go", "in", "inkl", "le",
    "les", "memory", "memoria", "memoire", "minneskort", "mit", "noir", "of",
    "online", "pack", "pendrive", "pour", "product", "red", "rouge", "sd",
    "silber", "silver", "speicherkarte", "stick", "tarjeta", "the", "to",
    "usb", "white", "with", "zu",
}
GENERIC_CODES = {
    "class10", "class4", "cl10", "sdhc", "sdxc", "microsd", "microsdhc",
    "microsdxc", "uhs1", "uhsi", "uhs2", "uhsii", "usb20", "usb30", "usb31",
    "usb32", "typec", "4k", "1080p",
}
SERIES = {
    "a400", "attach", "basic", "beast", "blade", "canvas", "cruzer",
    "datatraveler", "dual", "evo", "exceria", "extreme", "fit", "flair",
    "force", "fury", "glide", "hyperx", "impact", "jumpdrive", "premium",
    "professional", "rainbow", "savage", "speed", "ultra", "vault",
}

WORD_RE = re.compile(r"[a-z]+[0-9]+[a-z0-9]*|[0-9]+[a-z]+[a-z0-9]*|[a-z]{2,}|[0-9]+")
CAPACITY_RE = re.compile(r"\b(1|2|4|8|16|32|64|128|256|512|1024|2048)\s*(tb|gb|go)\b")
CODE_RE = re.compile(r"(?=[a-z0-9-]{3,32}\b)(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*[0-9])[a-z0-9]+(?:-[a-z0-9]+)*")


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def hash_text(text: str) -> int:
    raw = text.encode("utf-8", "ignore")
    return (zlib.crc32(raw) << 32) | zlib.crc32(raw, 0xA5A5A5A5)


def combine(kind: int, *values: int) -> int:
    result = mix64(kind)
    for value in values:
        result = mix64(result ^ value)
    return result


def ascii_text(value: str) -> str:
    value = html.unescape(value).lower()
    for old, new in BRAND_ALIASES.items():
        value = value.replace(old, new)
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()


def product_type(words: set[str], text: str) -> int:
    if "galaxy" in words or "iphone" in words or "oneplus" in words or "lte" in words:
        return 5
    if "tv" in words or "television" in words or "led" in words and "inch" in words:
        return 4
    if "ssd" in words:
        return 3
    if "microsd" in text or words.intersection({"sdhc", "sdxc", "xqd"}):
        return 2
    if words.intersection({"datatraveler", "jumpdrive", "cruzer", "pendrive"}):
        return 1
    if "usb" in words and not words.intersection({"adapter", "cable"}):
        return 1
    return 0


def capacity_mask(text: str) -> int:
    result = 0
    for number, unit in CAPACITY_RE.findall(text):
        value = int(number) * (1024 if unit == "tb" else 1)
        result |= 1 << int(math.log2(value))
    return result


def model_codes(text: str) -> list[int]:
    codes: list[tuple[int, int]] = []
    seen: set[str] = set()
    for match in CODE_RE.finditer(text):
        raw = match.group()
        compact = raw.replace("-", "")
        if compact in GENERIC_CODES or compact in seen:
            continue
        letters = sum(c.isalpha() for c in compact)
        digits = sum(c.isdigit() for c in compact)
        if letters < 2 or digits < 1:
            continue
        # Speeds/capacities such as 95mb and generic standards are poor IDs.
        if re.fullmatch(r"\d+(?:mb|gb|go|tb|mhz|mbs)", compact):
            continue
        specificity = min(len(compact), 20) + min(digits, 5) * 3
        if compact.startswith(("sdcz", "sdsq", "sdsd", "thn", "ljd", "lsd", "usm", "sr", "dt", "sda")):
            specificity += 12
        seen.add(compact)
        codes.append((specificity, hash_text(compact)))
    codes.sort(reverse=True)
    return [code for _, code in codes[:3]]


def features(name: str, brand_field: str) -> tuple[list[tuple[int, int]], int, int, int, int, int, int]:
    text = ascii_text(f"{brand_field} {name}")
    tokens = WORD_RE.findall(text)
    words = set(tokens)

    brand = 0
    for token in tokens:
        if token in BRANDS:
            brand = BRANDS[token]
            break
    cap = capacity_mask(text)
    ptype = product_type(words, text)
    codes = model_codes(text)

    informative = {
        token for token in tokens
        if token not in STOPWORDS and len(token) >= 3 and not token.isdigit()
    }
    # Long descriptions otherwise dominate both runtime and the Bloom score.
    if len(informative) > 48:
        informative = set(sorted(informative, key=lambda x: (zlib.crc32(x.encode()), x))[:48])

    token_hashes = [zlib.crc32(token.encode()) for token in informative]
    bloom_lo = 0
    bloom_hi = 0
    for token_hash in token_hashes:
        bit = token_hash & 127
        if bit < 64:
            bloom_lo |= 1 << bit
        else:
            bloom_hi |= 1 << (bit - 64)

    series_hashes = sorted(hash_text(word) for word in words.intersection(SERIES))
    capacity_key = mix64(cap) if cap else 0
    signatures: list[tuple[int, int]] = []

    # Exact normalized token multisets are very precise even when word order differs.
    canonical = " ".join(sorted(informative))
    if canonical:
        signatures.append((combine(1, hash_text(canonical), capacity_key), 950))

    for code in codes:
        signatures.append((combine(2, code), 900))
        if brand:
            signatures.append((combine(3, brand, code), 980))

    for series_hash in series_hashes[:2]:
        if brand and cap:
            signatures.append((combine(4, brand, capacity_key, series_hash), 760))
        elif brand:
            signatures.append((combine(5, brand, series_hash, ptype), 570))

    # Three one-permutation MinHash blocks provide fuzzy recall with constant work.
    if token_hashes:
        minimums = [
            min(token_hashes),
            min(((h * 0x9E3779B1 + 0x85EBCA77) & 0xFFFFFFFF) for h in token_hashes),
            min(((h * 0xC2B2AE3D + 0x27D4EB2F) & 0xFFFFFFFF) for h in token_hashes),
        ]
        context = brand or (32 + ptype)
        for band, minimum in enumerate(minimums):
            signatures.append((combine(10 + band, context, capacity_key, minimum), 390))

    # A broad but still contextual fallback for heavily corrupted short titles.
    if brand and cap and ptype:
        signatures.append((combine(20, brand, capacity_key, ptype), 330))

    return signatures, bloom_lo, bloom_hi, brand, cap, ptype, min(len(informative), 255)


def compatibility_score(
    base: int,
    current: tuple[int, int, int, int, int],
    previous_index: int,
    blooms_lo: array,
    blooms_hi: array,
    brands: bytearray,
    capacities: array,
    types: bytearray,
    token_counts: bytearray,
) -> int:
    bloom_lo, bloom_hi, brand, cap, ptype = current
    overlap = ((bloom_lo & blooms_lo[previous_index]).bit_count()
               + (bloom_hi & blooms_hi[previous_index]).bit_count())
    denominator = max(1, min(token_counts[previous_index],
                             bloom_lo.bit_count() + bloom_hi.bit_count()))
    score = base + min(240, overlap * 240 // denominator)

    other_brand = brands[previous_index]
    if brand and other_brand:
        score += 90 if brand == other_brand else -180
    other_cap = capacities[previous_index]
    if cap and other_cap:
        score += 100 if cap & other_cap else -150
    other_type = types[previous_index]
    if ptype and other_type:
        score += 55 if ptype == other_type else -100
    return score


def add_to_posting(postings: dict[int, int | list[int]], key: int, index: int) -> None:
    old = postings.get(key)
    if old is None:
        postings[key] = index
    elif isinstance(old, int):
        postings[key] = [old, index]
    elif len(old) < POSTING_CAP:
        old.append(index)
    else:
        # A deterministic ring sample avoids keeping only the earliest rows.
        old[index % POSTING_CAP] = index


def posting_values(value: int | list[int]) -> Iterable[int]:
    if isinstance(value, int):
        yield value
    else:
        yield from value


def encode_pair(left: int, right: int) -> int:
    if left > right:
        left, right = right, left
    if left < 0 or right > 0xFFFFFFFF:
        raise ValueError("ids must be non-negative 32-bit integers")
    return (left << 32) | right


def decode_pair(code: int) -> tuple[int, int]:
    return code >> 32, code & 0xFFFFFFFF


def generate(input_path: Path, limit: int) -> tuple[list[tuple[int, int, int]], array]:
    postings: dict[int, int | list[int]] = {}
    ids = array("I")
    blooms_lo = array("Q")
    blooms_hi = array("Q")
    capacities = array("Q")
    brands = bytearray()
    types = bytearray()
    token_counts = bytearray()
    heap: list[tuple[int, int, int]] = []

    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "id" not in reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError("X2 must contain id and name columns")

        for index, row in enumerate(reader):
            record_id = int(row["id"])
            if not 0 <= record_id <= 0xFFFFFFFF:
                raise ValueError(f"id {record_id} is outside the supported 32-bit range")

            signatures, bloom_lo, bloom_hi, brand, cap, ptype, token_count = features(
                row.get("name", ""), row.get("brand", "")
            )
            local: dict[int, int] = {}
            for key, base in signatures:
                old = postings.get(key)
                if old is not None:
                    for previous_index in posting_values(old):
                        if base > local.get(previous_index, -1):
                            local[previous_index] = base

            current = (bloom_lo, bloom_hi, brand, cap, ptype)
            scored: list[tuple[int, int]] = []
            for previous_index, base in local.items():
                score = compatibility_score(
                    base, current, previous_index, blooms_lo, blooms_hi,
                    brands, capacities, types, token_counts,
                )
                if score >= 360:
                    scored.append((score, previous_index))
            scored.sort(reverse=True)

            for score, previous_index in scored[:LOCAL_CANDIDATES]:
                code = encode_pair(ids[previous_index], record_id)
                tie_break = mix64(code)
                candidate = (score, tie_break, code)
                if len(heap) < limit:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    heapq.heapreplace(heap, candidate)

            for key, _ in signatures:
                add_to_posting(postings, key, index)
            ids.append(record_id)
            blooms_lo.append(bloom_lo)
            blooms_hi.append(bloom_hi)
            capacities.append(cap)
            brands.append(brand)
            types.append(ptype)
            token_counts.append(token_count)

            if index and index % 250_000 == 0:
                print(
                    f"processed {index:,} rows; retained {len(heap):,} candidates",
                    file=sys.stderr,
                )

    return heap, ids


def fill_candidates(
    candidates: list[tuple[int, int, int]], ids: Sequence[int], limit: int
) -> list[tuple[int, int, int]]:
    if len(ids) * (len(ids) - 1) // 2 < limit:
        raise ValueError(
            f"{len(ids):,} records permit fewer than {limit:,} unique non-self pairs; "
            "use --allow-fewer only for development samples"
        )
    if len(candidates) >= limit:
        return candidates

    used = {candidate[2] for candidate in candidates}
    count = len(ids)
    # Deterministic long-distance edges are cheap filler and cannot repeat within
    # an offset.  They are only used when blocking produced too few pairs.
    for offset in range(count // 2, 0, -1):
        for index, left in enumerate(ids):
            right = ids[(index + offset) % count]
            if left == right:
                continue
            code = encode_pair(left, right)
            if code in used:
                continue
            used.add(code)
            candidates.append((0, mix64(code), code))
            if len(candidates) == limit:
                return candidates
    raise RuntimeError("could not construct the requested number of unique pairs")


def write_output(path: Path, candidates: list[tuple[int, int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("lid", "rid"))
        for _, _, code in candidates:
            writer.writerow(decode_pair(code))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("../SIGMOD_full_data/X2.csv"))
    parser.add_argument("--output", type=Path, default=Path("output.csv"))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="development only: write fewer than --limit pairs if the sample is too small",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates, ids = generate(args.input, args.limit)
    if args.allow_fewer and len(ids) * (len(ids) - 1) // 2 < args.limit:
        target = min(args.limit, len(candidates))
        candidates = heapq.nlargest(target, candidates)
    else:
        candidates = fill_candidates(candidates, ids, args.limit)
    if len(candidates) != args.limit and not args.allow_fewer:
        raise RuntimeError(f"expected {args.limit:,} pairs, got {len(candidates):,}")
    write_output(args.output, candidates)
    print(f"wrote {len(candidates):,} pairs to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
