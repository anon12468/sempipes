#!/usr/bin/env python3
"""Scalable blocking for the X2 electronic-products data set.

The blocker uses several product-specific keys and bounded sorted
neighbourhoods.  For every key bucket it examines only fixed offsets in a
small number of orderings, so a bucket containing k records costs O(k), not
O(k**2).  The global candidate store is pruned to the requested output size
after every ordering; all generated pairs are never retained at once.
"""

from __future__ import annotations

import argparse
from array import array
import csv
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import unidecode


TARGET_PAIRS = 2_000_000
MASK64 = (1 << 64) - 1
RANK_TIE_MASK = (1 << 44) - 1

BRANDS = (
    "intenso",
    "lexar",
    "logilink",
    "pny",
    "samsung",
    "sandisk",
    "kingston",
    "sony",
    "toshiba",
    "transcend",
)

ALIASES = (
    ("classe", "class"),
    ("clase", "class"),
    ("klasse", "class"),
    ("ultra high-speed", "uhsi"),
    ("uhs-i", "uhsi"),
    ("uhs1", "uhsi"),
    ("type c", "type-c"),
    ("typec", "type-c"),
    ("usb-c", "type-c"),
    ("usbc", "type-c"),
    ("data traveler", "datatraveler"),
    ("jump drive", "jumpdrive"),
    ("micro sd", "microsd"),
    ("extrem ", "extreme "),
)

STOP_WORDS = frozenset(
    """
    a an and the for with from in of to on at by or is this product buy new
    de la le les en et du des dans avec pour sur au aux un une
    mit fur von im auf und der die das den dem
    con da di del el los las para por do dos
    memoria memory memoire carte card karta tarjeta speicherkarte minneskort
    geheugenkaart usb flash drive disk stick adapter adaptateur adaptador
    adattatore black white red blue silver silber noir schwarz
    """.split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
PAIR_TOKEN_RE = re.compile(r"\w+\s\w+\d+")
CAPACITY_RE = re.compile(r"(?<!\d)(\d{1,4})\s*(?:g|gb|go|gbo)(?![a-z])")
MODEL_NUM_RE = re.compile(
    r"\b(?:\(?[\w]+-?\d+-?[\w]*-?\d*|\d+[a-z]|[a-z]\d+)\b"
)
LONG_MODEL_RE = re.compile(
    r"(?:thn-[a-z][\w-]+|ljd[\w+]+-[\w]+|lsdmi\d+\w+|"
    r"lsd\d{1,3}gb\w+|usm\d{1,3}\w+|sdsq[a-z]+-\d+[a-z]+-\w+|"
    r"sdsd[a-z]+-\d+\w+(?:-\w+)?|sdcz\w+|mk\d+|sr-g1\w+)"
)
SHORT_MODEL_RE = re.compile(
    r"\b(?:c20[mc]|sda\d{1,2}|g1ux|s[72][05]|[unm][23]02|"
    r"p20|g4|dt101|se9|[asm]\d{2})\b"
)
MODEL_RE = re.compile(
    r"\b(?:datatraveler|extremep?|exceriap?|dual|evo|xqd|ssd|"
    r"cruzer\w*|glide|blade|basic|fit|force|jumpdrive|hxs|rainbow|"
    r"speedline|premiumline|att4|attach|serieu|beast|fury|impact|"
    r"a400|sd[hx]c|uhs[i12]+|ultra)\b"
)
FEATURE_RE = re.compile(
    r"\b(?:usb[23]|type-c|uhsii?|class(?:10|[0134])|gen\d{1,2}|"
    r"u[23]|sd[hx]c|otg|lte|[45]g|(?:\d{2,3}r)?\d{2,3}w)\b"
)

HTML_RE = re.compile(r"&(?:nbsp|amp|reg|[a-z]?acute|quot|trade);?")
PUNCT_RE = re.compile(r"[^a-z0-9 -]+")
SPACES_RE = re.compile(r"\s+")


def hash64(text: str) -> int:
    """Fast deterministic 64-bit hash made from two independent CRC32s."""
    data = text.encode("utf-8", errors="ignore")
    value = (zlib.crc32(data) << 32) | zlib.crc32(data, 0xA5A5A5A5)
    return value or 1


def token_hash(token: str) -> int:
    return zlib.crc32(token.encode("utf-8", errors="ignore"))


def lexical_prefix(text: str) -> int:
    data = text.encode("ascii", errors="ignore")[:8].ljust(8, b"\0")
    return int.from_bytes(data, "big")


def clean_name(value: str) -> str:
    name = unidecode.unidecode(value.lower())
    name = HTML_RE.sub(" ", name)
    for old, new in ALIASES:
        name = name.replace(old, new)
    name = PUNCT_RE.sub(" ", name)
    return SPACES_RE.sub(" ", name).strip()


def bitset128(hashes: Iterable[int]) -> tuple[int, int]:
    lo = 0
    hi = 0
    for value in hashes:
        bit = value & 127
        if bit < 64:
            lo |= 1 << bit
        else:
            hi |= 1 << (bit - 64)
    return lo, hi


def capacity_mask(capacities: set[int]) -> int:
    common = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 1024, 2000, 2048)
    result = 0
    for capacity in capacities:
        try:
            result |= 1 << common.index(capacity)
        except ValueError:
            pass
    return result


def minhashes(hashes: list[int]) -> tuple[int, int, int, int]:
    if not hashes:
        return 0, 0, 0, 0
    constants = (
        (0x9E3779B1, 0x85EBCA77),
        (0xC2B2AE3D, 0x27D4EB2F),
        (0x165667B1, 0xD3A2646C),
        (0xA24BAED5, 0x9FB21C65),
    )
    result = []
    for multiplier, addend in constants:
        result.append(min(((value * multiplier + addend) & 0xFFFFFFFF) for value in hashes))
    return tuple(result)  # type: ignore[return-value]


class FeatureStore:
    """Compact append-only arrays used by candidate generation and scoring."""

    def __init__(self) -> None:
        self.ids = array("Q")
        self.keys = [array("Q") for _ in range(8)]
        self.secondary = [array("Q") for _ in range(6)]
        self.raw_lo = array("Q")
        self.raw_hi = array("Q")
        self.info_lo = array("Q")
        self.info_hi = array("Q")
        self.model_bits = array("Q")
        self.capacities = array("H")
        self.brands = array("H")
        self.raw_bit_count = array("B")
        self.info_bit_count = array("B")

    def append(self, row: dict[str, str]) -> None:
        record_id = int(row["id"])
        raw_name = row.get("name", "") or ""
        cleaned = clean_name(raw_name)
        tokens = set(TOKEN_RE.findall(cleaned))

        brand_mask = 0
        brand_text = unidecode.unidecode((row.get("brand", "") or "").lower())
        brand_names = []
        for index, brand in enumerate(BRANDS):
            if brand in tokens or brand in brand_text:
                brand_mask |= 1 << index
                brand_names.append(brand)
        brand_key = " ".join(brand_names)

        capacities = {int(value) for value in CAPACITY_RE.findall(cleaned)}
        memory_key = " ".join(str(value) for value in sorted(capacities))
        models = sorted(set(MODEL_RE.findall(cleaned)))
        model_key = " ".join(models)
        features = sorted(set(FEATURE_RE.findall(cleaned)))
        feature_key = " ".join(features)

        model_numbers = []
        for value in MODEL_NUM_RE.findall(cleaned):
            compact = re.sub(r"[^a-z0-9]", "", value)
            if len(compact) > 5:
                model_numbers.append(compact)
        model_numbers = sorted(set(model_numbers))
        model_number_key = " ".join(model_numbers)
        score_models = set(model_numbers)
        for token in tokens:
            if (
                len(token) >= 3
                and any(character.isalpha() for character in token)
                and any(character.isdigit() for character in token)
                and token not in {"usb2", "usb3", "uhs1", "uhs2"}
            ):
                score_models.add(token)

        long_model = " ".join(sorted(set(LONG_MODEL_RE.findall(cleaned))))
        short_model = " ".join(sorted(set(SHORT_MODEL_RE.findall(cleaned))))

        raw_signature = " ".join(sorted(tokens))
        informative = sorted(
            token
            for token in tokens
            if len(token) > 1
            and token not in STOP_WORDS
        )
        info_signature = " ".join(informative)

        phrase_matches = PAIR_TOKEN_RE.findall(cleaned)
        phrase_key = " ".join(sorted(match.lower() for match in phrase_matches))

        key_texts = (
            raw_signature,
            phrase_key,
            " ".join(part for part in (brand_key, feature_key) if part),
            long_model,
            short_model,
            " ".join(part for part in (brand_key, memory_key) if part),
            " ".join(part for part in (brand_key, memory_key, model_key) if part),
            " ".join(part for part in (brand_key, model_number_key) if part),
        )

        raw_hashes = [token_hash(token) for token in tokens]
        info_hashes = [token_hash(token) for token in informative]
        raw_lo, raw_hi = bitset128(raw_hashes)
        info_lo, info_hi = bitset128(info_hashes)
        model_lo, model_hi = bitset128(token_hash(value) for value in score_models)
        minima = minhashes(raw_hashes)

        semantic = "|".join((brand_key, memory_key, model_key, feature_key))
        secondary = (
            lexical_prefix(cleaned),
            minima[0],
            minima[1],
            minima[2],
            minima[3],
            hash64(semantic) if semantic else 0,
        )

        self.ids.append(record_id)
        for values, text in zip(self.keys, key_texts):
            values.append(hash64(text) if text else 0)
        for values, value in zip(self.secondary, secondary):
            values.append(value)
        self.raw_lo.append(raw_lo)
        self.raw_hi.append(raw_hi)
        self.info_lo.append(info_lo)
        self.info_hi.append(info_hi)
        self.model_bits.append(model_lo | ((model_hi << 1) & MASK64))
        self.capacities.append(capacity_mask(capacities))
        self.brands.append(brand_mask)
        self.raw_bit_count.append(raw_lo.bit_count() + raw_hi.bit_count())
        self.info_bit_count.append(info_lo.bit_count() + info_hi.bit_count())

    def numpy_views(self) -> dict[str, object]:
        return {
            "ids": np.frombuffer(self.ids, dtype=np.uint64),
            "keys": [np.frombuffer(values, dtype=np.uint64) for values in self.keys],
            "secondary": [
                np.frombuffer(values, dtype=np.uint64) for values in self.secondary
            ],
            "raw_lo": np.frombuffer(self.raw_lo, dtype=np.uint64),
            "raw_hi": np.frombuffer(self.raw_hi, dtype=np.uint64),
            "info_lo": np.frombuffer(self.info_lo, dtype=np.uint64),
            "info_hi": np.frombuffer(self.info_hi, dtype=np.uint64),
            "model_bits": np.frombuffer(self.model_bits, dtype=np.uint64),
            "capacities": np.frombuffer(self.capacities, dtype=np.uint16),
            "brands": np.frombuffer(self.brands, dtype=np.uint16),
            "raw_bit_count": np.frombuffer(self.raw_bit_count, dtype=np.uint8),
            "info_bit_count": np.frombuffer(self.info_bit_count, dtype=np.uint8),
        }


POPCOUNT8 = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def popcount64(values: np.ndarray) -> np.ndarray:
    bytes_view = np.ascontiguousarray(values).view(np.uint8).reshape(-1, 8)
    return POPCOUNT8[bytes_view].sum(axis=1, dtype=np.uint16)


def score_pairs(
    left: np.ndarray, right: np.ndarray, data: dict[str, object]
) -> np.ndarray:
    raw_lo = data["raw_lo"]
    raw_hi = data["raw_hi"]
    info_lo = data["info_lo"]
    info_hi = data["info_hi"]
    raw_count = data["raw_bit_count"]
    info_count = data["info_bit_count"]
    model_bits = data["model_bits"]
    capacities = data["capacities"]
    brands = data["brands"]
    keys = data["keys"]

    raw_intersection = popcount64(raw_lo[left] & raw_lo[right])
    raw_intersection += popcount64(raw_hi[left] & raw_hi[right])
    raw_denominator = np.minimum(raw_count[left], raw_count[right]).astype(np.uint16)
    raw_containment = (
        raw_intersection.astype(np.uint32) * 256
        // np.maximum(raw_denominator, 1)
    )

    info_intersection = popcount64(info_lo[left] & info_lo[right])
    info_intersection += popcount64(info_hi[left] & info_hi[right])
    info_union = (
        info_count[left].astype(np.uint16)
        + info_count[right].astype(np.uint16)
        - info_intersection
    )
    info_jaccard = (
        info_intersection.astype(np.uint32) * 256 // np.maximum(info_union, 1)
    )

    model_matches = np.minimum(
        popcount64(model_bits[left] & model_bits[right]).astype(np.uint32), 2
    )
    left_caps = capacities[left]
    right_caps = capacities[right]
    cap_match = ((left_caps & right_caps) != 0).astype(np.int32)
    cap_conflict = (
        (left_caps != 0) & (right_caps != 0) & ((left_caps & right_caps) == 0)
    ).astype(np.int32)
    left_brands = brands[left]
    right_brands = brands[right]
    brand_match = ((left_brands & right_brands) != 0).astype(np.int32)
    brand_conflict = (
        (left_brands != 0)
        & (right_brands != 0)
        & ((left_brands & right_brands) == 0)
    ).astype(np.int32)
    exact_match = (keys[0][left] == keys[0][right]).astype(np.int32)

    scores = (
        3 * raw_containment.astype(np.int32)
        + 7 * info_jaccard.astype(np.int32)
        + 384 * model_matches.astype(np.int32)
        + 256 * cap_match
        + 128 * brand_match
        + 512 * exact_match
        - 256 * cap_conflict
        - 256 * brand_conflict
    )
    return np.maximum(scores, 1).astype(np.uint32)


def mix64(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.uint64, copy=True)
    result ^= result >> np.uint64(30)
    result *= np.uint64(0xBF58476D1CE4E5B9)
    result ^= result >> np.uint64(27)
    result *= np.uint64(0x94D049BB133111EB)
    result ^= result >> np.uint64(31)
    return result


def candidate_rank(codes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return (scores.astype(np.uint64) << np.uint64(44)) | (
        mix64(codes) & np.uint64(RANK_TIE_MASK)
    )


def keep_best(
    top_codes: np.ndarray,
    top_ranks: np.ndarray,
    codes: np.ndarray,
    scores: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray]:
    if codes.size == 0:
        return top_codes, top_ranks

    ranks = candidate_rank(codes, scores)
    if codes.size > target:
        selected = np.argpartition(ranks, -target)[-target:]
        codes = codes[selected]
        ranks = ranks[selected]

    if top_codes.size:
        codes = np.concatenate((top_codes, codes))
        ranks = np.concatenate((top_ranks, ranks))

    # A pair can be proposed by several blocking keys.  Deduplication happens
    # at every merge while the retained collection remains capped.
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    unique = np.empty(sorted_codes.size, dtype=bool)
    unique[0] = True
    unique[1:] = sorted_codes[1:] != sorted_codes[:-1]
    selected = order[unique]
    codes = codes[selected]
    ranks = ranks[selected]

    if codes.size > target:
        selected = np.argpartition(ranks, -target)[-target:]
        codes = codes[selected]
        ranks = ranks[selected]
    return codes, ranks


def generate_candidates(data: dict[str, object], target: int) -> np.ndarray:
    ids = data["ids"]
    keys = data["keys"]
    secondary_keys = data["secondary"]
    n_records = ids.size
    top_codes = np.empty(0, dtype=np.uint64)
    top_ranks = np.empty(0, dtype=np.uint64)
    offsets = (1, 2, 4)

    total_orderings = len(keys) * len(secondary_keys)
    ordering_number = 0
    for primary in keys:
        active = np.flatnonzero(primary)
        if active.size < 2:
            ordering_number += len(secondary_keys)
            continue

        primary_active = primary[active]
        exact_active = keys[0][active]
        for secondary in secondary_keys:
            ordering_number += 1
            order = np.lexsort(
                (exact_active, secondary[active], primary_active)
            )
            rows = active[order]
            sorted_primary = primary[rows]

            left_parts = []
            right_parts = []
            for offset in offsets:
                if rows.size <= offset:
                    continue
                same_bucket = (
                    sorted_primary[:-offset] == sorted_primary[offset:]
                )
                left_parts.append(rows[:-offset][same_bucket])
                right_parts.append(rows[offset:][same_bucket])
            if not left_parts:
                continue

            left = np.concatenate(left_parts).astype(np.uint64, copy=False)
            right = np.concatenate(right_parts).astype(np.uint64, copy=False)
            lo = np.minimum(left, right)
            hi = np.maximum(left, right)
            codes = lo * np.uint64(n_records) + hi
            scores = score_pairs(
                lo.astype(np.intp), hi.astype(np.intp), data
            )
            top_codes, top_ranks = keep_best(
                top_codes, top_ranks, codes, scores, target
            )
            print(
                f"\rBlocking {ordering_number}/{total_orderings}; "
                f"retained {top_codes.size:,}",
                end="",
                file=sys.stderr,
                flush=True,
            )

    print(file=sys.stderr)
    return fill_candidates(top_codes, top_ranks, n_records, target)


def fill_candidates(
    top_codes: np.ndarray,
    top_ranks: np.ndarray,
    n_records: int,
    target: int,
) -> np.ndarray:
    """Fill a short candidate set with O(target) deterministic valid pairs."""
    distance = 1
    base = np.arange(n_records, dtype=np.uint64)
    while top_codes.size < target:
        right = (base + np.uint64(distance)) % np.uint64(n_records)
        lo = np.minimum(base, right)
        hi = np.maximum(base, right)
        valid = lo != hi
        codes = lo[valid] * np.uint64(n_records) + hi[valid]
        scores = np.zeros(codes.size, dtype=np.uint32)
        top_codes, top_ranks = keep_best(
            top_codes, top_ranks, codes, scores, target
        )
        distance += 1
        if distance >= n_records:
            raise RuntimeError("Could not construct enough unique candidate pairs")

    # Highest-scoring pairs first; output order does not affect correctness.
    order = np.argsort(top_ranks, kind="stable")[::-1]
    return top_codes[order]


def read_features(path: Path) -> FeatureStore:
    features = FeatureStore()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: id, name")
        for number, row in enumerate(reader, 1):
            features.append(row)
            if number % 100_000 == 0:
                print(f"\rRead {number:,} records", end="", file=sys.stderr, flush=True)
    print(f"\rRead {len(features.ids):,} records", file=sys.stderr)
    return features


def write_output(
    path: Path, codes: np.ndarray, ids: np.ndarray, n_records: int
) -> None:
    left_rows = (codes // np.uint64(n_records)).astype(np.intp)
    right_rows = (codes % np.uint64(n_records)).astype(np.intp)
    left_ids = ids[left_rows]
    right_ids = ids[right_rows]
    lids = np.minimum(left_ids, right_ids)
    rids = np.maximum(left_ids, right_ids)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("lid", "rid"))
        writer.writerows(zip(lids.tolist(), rids.tolist()))


def default_input() -> Path:
    root = Path("../SIGMOD_full_data/X2.csv")
    return root if root.exists() else Path("../SIGMOD_full_data/X2.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input())
    parser.add_argument("--output", type=Path, default=Path("output.csv"))
    parser.add_argument("--target-pairs", type=int, default=TARGET_PAIRS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    features = read_features(args.input)
    data = features.numpy_views()
    n_records = data["ids"].size
    possible_pairs = n_records * (n_records - 1) // 2
    if possible_pairs < args.target_pairs:
        raise ValueError(
            f"{n_records:,} records have only {possible_pairs:,} valid pairs; "
            f"cannot output {args.target_pairs:,} unique non-self pairs. "
            "Use --target-pairs for development subsets."
        )

    if np.unique(data["ids"]).size != n_records:
        raise ValueError("X2 contains duplicate record IDs")

    codes = generate_candidates(data, args.target_pairs)
    if codes.size != args.target_pairs or np.unique(codes).size != args.target_pairs:
        raise AssertionError("candidate generation did not produce the requested size")
    write_output(args.output, codes, data["ids"], n_records)
    elapsed = time.monotonic() - started
    print(
        f"Wrote {codes.size:,} pairs to {args.output} in {elapsed:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
