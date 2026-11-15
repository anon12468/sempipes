"""Scalable duplicate-candidate blocking for X2.

The important bounds are:

* a blocking key remembers at most ``POSTING_CAP`` previous records;
* a record therefore does constant work per blocking pass;
* ``CandidatePool`` never contains more than ``TARGET_PAIRS`` pairs.

No ground-truth file is read. The weak final fill exists only to satisfy the
exact output-size contract when an input has too few useful key collisions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import math
import os
import re
import unicodedata
import zlib
from array import array
from collections import Counter
from typing import Callable, Hashable, Iterable, Iterator


TARGET_PAIRS = 2_000_000
POSTING_CAP = 8
TOKEN_RE = re.compile(r"[a-z0-9]+")

# These mostly distinguish a language/store rather than a product.
STOPWORDS = {
    "a", "ab", "and", "avec", "black", "bleu", "card", "carte", "class",
    "color", "de", "der", "di", "direct", "drive", "en", "et", "for",
    "from", "gb", "go", "grau", "high", "in", "la", "le", "les", "memory",
    "memoria", "memoire", "micro", "mit", "noir", "of", "online", "para",
    "pour", "pro", "product", "red", "scheda", "sd", "shop", "silber",
    "speicherkarte", "stick", "tarjeta", "the", "to", "ultra", "und", "unit",
    "usb", "with", "zu",
}

KNOWN_BRANDS = {
    "adata", "corsair", "emtec", "hama", "hp", "intenso", "kingston",
    "lexar", "pny", "samsung", "sandisk", "sony", "toshiba", "transcend",
    "verbatim",
}

TYPE_BITS = {
    "usb": 1,
    "microsd": 2,
    "sdhc": 4,
    "sdxc": 8,
    "flash": 16,
    "pendrive": 32,
    "jumpdrive": 64,
    "datatraveler": 128,
}


def normalize(value: str) -> str:
    value = html.unescape(value or "").lower()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(TOKEN_RE.findall(value))


def hash32(value: str) -> int:
    return zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF


def useful_tokens(tokens: list[str]) -> tuple[str, ...]:
    """Return de-duplicated product-bearing tokens in stable order."""
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen or token in STOPWORDS or len(token) < 2:
            continue
        # Capacities and other bare numbers are represented separately.
        if token.isdigit():
            continue
        seen.add(token)
        result.append(token)
    return tuple(result)


def token_bloom(tokens: Iterable[str]) -> int:
    mask = 0
    for token in tokens:
        h = hash32(token)
        mask |= 1 << (h & 63)
        mask |= 1 << ((h >> 10) & 63)
    return mask


def capacity_mask(tokens: list[str]) -> int:
    """Encode storage capacities (8 GB, 128GB, 1 TB, ...) in a small mask."""
    result = 0
    for i, token in enumerate(tokens):
        match = re.fullmatch(r"(\d{1,4})(gb|go|tb)", token)
        if match:
            amount = int(match.group(1))
            if match.group(2) == "tb":
                amount *= 1024
        elif (
            token.isdigit()
            and i + 1 < len(tokens)
            and tokens[i + 1] in {"gb", "go", "tb"}
        ):
            amount = int(token)
            if tokens[i + 1] == "tb":
                amount *= 1024
        else:
            continue
        bit = (
            int(math.log2(amount))
            if amount > 0 and amount & (amount - 1) == 0
            else amount % 31
        )
        result |= 1 << min(bit, 31)
    return result


def type_mask(tokens: list[str]) -> int:
    joined = "".join(tokens)
    result = 0
    for product_type, bit in TYPE_BITS.items():
        if product_type in tokens or product_type in joined:
            result |= bit
    return result


def brand_code(brand: str, tokens: list[str]) -> int:
    normalized_brand = normalize(brand)
    if normalized_brand:
        first = normalized_brand.split()[0]
        if first in KNOWN_BRANDS:
            return hash32(first)
    for token in tokens:
        if token in KNOWN_BRANDS:
            return hash32(token)
    return 0


class CandidatePool:
    """A bounded, de-duplicated bucket queue retaining the best scores."""

    def __init__(self, capacity: int, row_bits: int) -> None:
        self.capacity = capacity
        self.row_bits = row_bits
        self.row_mask = (1 << row_bits) - 1
        self.buckets = [array("Q") for _ in range(256)]
        self.members: set[int] = set()
        self.minimum_score = 0

    def encode(self, left: int, right: int) -> int:
        if left > right:
            left, right = right, left
        return (left << self.row_bits) | right

    def add(self, left: int, right: int, score: int) -> None:
        if left == right:
            return
        code = self.encode(left, right)
        if code in self.members:
            return
        score = max(0, min(255, score))

        if len(self.members) >= self.capacity:
            while (
                self.minimum_score < 255
                and not self.buckets[self.minimum_score]
            ):
                self.minimum_score += 1
            if score <= self.minimum_score:
                return
            removed = self.buckets[self.minimum_score].pop()
            self.members.remove(removed)

        self.buckets[score].append(code)
        self.members.add(code)
        if score < self.minimum_score or len(self.members) == 1:
            self.minimum_score = score

    def pairs(self) -> Iterator[tuple[int, int]]:
        emitted = 0
        for score in range(255, -1, -1):
            for code in self.buckets[score]:
                yield code >> self.row_bits, code & self.row_mask
                emitted += 1
                if emitted == self.capacity:
                    return


def add_to_posting(
    index: dict[Hashable, int | tuple[int, ...]],
    key: Hashable,
    row: int,
) -> tuple[int, ...]:
    """Return prior rows and retain only a constant-size posting list."""
    old = index.get(key)
    if old is None:
        index[key] = row
        return ()
    if isinstance(old, int):
        index[key] = (old, row)
        return (old,)
    previous = old
    if len(old) < POSTING_CAP:
        index[key] = old + (row,)
    else:
        index[key] = old[1:] + (row,)
    return previous


def similarity_score(
    left: int,
    right: int,
    blooms: array,
    capacities: array,
    brands: array,
    types: array,
    source_bonus: int,
) -> int:
    intersection = (blooms[left] & blooms[right]).bit_count()
    union = (blooms[left] | blooms[right]).bit_count()
    score = source_bonus + (100 * intersection // union if union else 0)
    if brands[left] and brands[left] == brands[right]:
        score += 18
    if capacities[left] and capacities[left] & capacities[right]:
        score += 24
    if types[left] and types[left] & types[right]:
        score += 8
    return min(score, 255)


def blocking_pass(
    label: str,
    keys: Iterable[Hashable | None],
    source_bonus: int,
    pool: CandidatePool,
    blooms: array,
    capacities: array,
    brands: array,
    types: array,
) -> None:
    index: dict[Hashable, int | tuple[int, ...]] = {}
    for row, key in enumerate(keys):
        if key is None:
            continue
        previous = add_to_posting(index, key, row)
        for other in previous:
            score = similarity_score(
                other, row, blooms, capacities, brands, types, source_bonus
            )
            pool.add(other, row, score)
    print(f"{label}: retained pool size {len(pool.members):,}")
    del index
    gc.collect()


def blocking_multikey_pass(
    label: str,
    keys_per_row: Iterable[Iterable[Hashable]],
    source_bonus: int,
    pool: CandidatePool,
    blooms: array,
    capacities: array,
    brands: array,
    types: array,
) -> None:
    """Block on several keys per row while keeping work strictly bounded."""
    index: dict[Hashable, int | tuple[int, ...]] = {}
    for row, keys in enumerate(keys_per_row):
        candidates: set[int] = set()
        for key in keys:
            candidates.update(add_to_posting(index, key, row))
        for other in candidates:
            score = similarity_score(
                other, row, blooms, capacities, brands, types, source_bonus
            )
            pool.add(other, row, score)
    print(f"{label}: retained pool size {len(pool.members):,}")
    del index
    gc.collect()


def read_records(
    input_path: str,
) -> tuple[list[str], array, array, array, array, array, Counter[str]]:
    ids = array("q")
    names: list[str] = []
    blooms = array("Q")
    capacities = array("I")
    brands = array("I")
    types = array("H")
    frequencies: Counter[str] = Counter()

    with open(input_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"id", "name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{input_path} must contain columns: id, name")
        for record in reader:
            ids.append(int(record["id"]))
            name = normalize(record.get("name", ""))
            names.append(name)
            tokens = name.split()
            useful = useful_tokens(tokens)
            blooms.append(token_bloom(useful))
            capacities.append(capacity_mask(tokens))
            brands.append(brand_code(record.get("brand", ""), tokens))
            types.append(type_mask(tokens))
            frequencies.update(useful)

    if len(ids) != len(set(ids)):
        raise ValueError("Input ids must be unique")
    print(f"Loaded {len(ids):,} records")
    return names, ids, blooms, capacities, brands, types, frequencies


def sorted_token_key(name: str) -> str | None:
    tokens = useful_tokens(name.split())
    return "\x1f".join(sorted(tokens)) if tokens else None


def rare_tokens(name: str, frequencies: Counter[str]) -> list[str]:
    tokens = useful_tokens(name.split())
    return sorted(
        tokens,
        key=lambda token: (frequencies[token], -len(token), token),
    )


def rare_pair_key(name: str, frequencies: Counter[str]) -> int | None:
    tokens = rare_tokens(name, frequencies)
    if len(tokens) < 2:
        return None
    first, second = sorted(tokens[:2])
    return (hash32(first) << 32) | hash32(second)


def rare_single_key(name: str, frequencies: Counter[str]) -> int | None:
    tokens = rare_tokens(name, frequencies)
    return hash32(tokens[0]) if tokens else None


def rare_token_keys(
    name: str,
    frequencies: Counter[str],
    limit: int = 8,
) -> tuple[int, ...]:
    return tuple(hash32(token) for token in rare_tokens(name, frequencies)[:limit])


def model_key(name: str, frequencies: Counter[str]) -> int | None:
    models = [
        token
        for token in useful_tokens(name.split())
        if len(token) >= 4
        and any(char.isalpha() for char in token)
        and any(char.isdigit() for char in token)
    ]
    if not models:
        return None
    token = min(
        models,
        key=lambda value: (frequencies[value], -len(value), value),
    )
    return hash32(token)


def spec_key(
    row: int,
    brands: array,
    capacities: array,
    types: array,
) -> int | None:
    # Require capacity and one other attribute to avoid giant capacity buckets.
    if not capacities[row] or not (brands[row] or types[row]):
        return None
    return (
        (int(brands[row]) << 48)
        ^ (int(capacities[row]) << 16)
        ^ int(types[row])
    )


def fill_to_target(pool: CandidatePool, record_count: int) -> None:
    """Add bounded deterministic fallback pairs."""
    total_possible = record_count * (record_count - 1) // 2
    if total_possible < pool.capacity:
        raise ValueError(
            f"{record_count:,} records provide only {total_possible:,} "
            f"distinct pairs; cannot write {pool.capacity:,}"
        )

    # This walk stops as soon as the bounded output pool is full. It never
    # materializes a Cartesian product or an unbounded bucket.
    distance = 1
    while len(pool.members) < pool.capacity and distance < record_count:
        for left in range(record_count - distance):
            pool.add(left, left + distance, 0)
            if len(pool.members) == pool.capacity:
                return
        distance += 1


def generate_candidates(
    input_path: str,
    target: int,
) -> tuple[CandidatePool, array]:
    (
        names,
        ids,
        blooms,
        capacities,
        brands,
        types,
        frequencies,
    ) = read_records(input_path)
    row_bits = max(1, (len(ids) - 1).bit_length())
    pool = CandidatePool(target, row_bits)

    passes: list[
        tuple[str, int, Callable[[], Iterable[Hashable | None]]]
    ] = [
        (
            "exact normalized name",
            125,
            lambda: (name or None for name in names),
        ),
        (
            "token-order invariant",
            105,
            lambda: (sorted_token_key(name) for name in names),
        ),
        (
            "rare token pair",
            78,
            lambda: (rare_pair_key(name, frequencies) for name in names),
        ),
        (
            "model identifier",
            68,
            lambda: (model_key(name, frequencies) for name in names),
        ),
    ]

    for label, bonus, key_factory in passes:
        blocking_pass(
            label,
            key_factory(),
            bonus,
            pool,
            blooms,
            capacities,
            brands,
            types,
        )

    blocking_multikey_pass(
        "rare-token ensemble",
        (rare_token_keys(name, frequencies) for name in names),
        42,
        pool,
        blooms,
        capacities,
        brands,
        types,
    )

    blocking_pass(
        "brand/capacity/type",
        (
            spec_key(row, brands, capacities, types)
            for row in range(len(names))
        ),
        25,
        pool,
        blooms,
        capacities,
        brands,
        types,
    )

    fill_to_target(pool, len(ids))
    print(f"Final candidate count: {len(pool.members):,}")
    return pool, ids


def write_output(output_path: str, pool: CandidatePool, ids: array) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("lid", "rid"))
        count = 0
        for left_row, right_row in pool.pairs():
            left_id = ids[left_row]
            right_id = ids[right_row]
            if left_id > right_id:
                left_id, right_id = right_id, left_id
            writer.writerow((left_id, right_id))
            count += 1
    if count != pool.capacity:
        raise RuntimeError(f"Wrote {count:,} rows instead of {pool.capacity:,}")
    print(f"Wrote {count:,} pairs to {output_path}")


def default_input() -> str:
    if os.path.exists("../SIGMOD_full_data/X2.csv"):
        return "../SIGMOD_full_data/X2.csv"
    return "../SIGMOD_full_data/X2.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=default_input(), help="X2 CSV path")
    parser.add_argument("--output", default="output.csv", help="candidate CSV path")
    parser.add_argument("--target", type=int, default=TARGET_PAIRS)
    args = parser.parse_args()

    pool, ids = generate_candidates(args.input, args.target)
    write_output(args.output, pool, ids)


if __name__ == "__main__":
    main()
