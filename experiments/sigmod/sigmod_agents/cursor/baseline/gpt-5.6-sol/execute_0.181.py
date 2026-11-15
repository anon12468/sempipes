#!/usr/bin/env python3
"""Scalable candidate blocking for the X2 electronic-products data set.

The generator never reads ground truth.  Its work is linear in the number of
records (apart from hash-table operations), and candidate memory is bounded by
the requested output size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import heapq
import math
import re
import sys
import time
import unicodedata
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


MASK64 = (1 << 64) - 1
HASH_MULTIPLIER = 0x9E3779B97F4A7C15

KNOWN_BRANDS = {
    "adata",
    "apple",
    "corsair",
    "crucial",
    "emtec",
    "freecom",
    "hama",
    "hp",
    "intenso",
    "kingston",
    "lexar",
    "logilink",
    "pny",
    "samsung",
    "sandisk",
    "sony",
    "toshiba",
    "transcend",
    "verbatim",
}

FAMILIES = {
    "a400",
    "attach",
    "attache",
    "basic",
    "blade",
    "canvas",
    "cruzer",
    "datatraveler",
    "dual",
    "elite",
    "evo",
    "exceria",
    "extreme",
    "fit",
    "fury",
    "glide",
    "hyperx",
    "impact",
    "jumpdrive",
    "premium",
    "savage",
    "ultra",
    "vault",
}

STOPWORDS = {
    "adapter",
    "adaptador",
    "adaptateur",
    "and",
    "avec",
    "black",
    "blue",
    "card",
    "carte",
    "class",
    "drive",
    "flash",
    "for",
    "geheugenkaart",
    "green",
    "karte",
    "karta",
    "memory",
    "memoria",
    "memoire",
    "micro",
    "minne",
    "product",
    "red",
    "scheda",
    "silver",
    "speicherkarte",
    "stick",
    "tarjeta",
    "und",
    "usb",
    "white",
    "with",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
CAPACITY_RE = re.compile(r"^([1-9][0-9]{0,3})(gb|tb)$")
SPEED_RE = re.compile(r"^[0-9]{2,4}(mb|r[0-9]{2,4}w?)$")
GENERIC_CODE_RE = re.compile(
    r"^(usb[0-9]*|uhs[iv0-9]*|sdhc|sdxc|microsd[a-z]*|class[0-9]+|"
    r"[0-9]+gb|[0-9]+tb|[0-9]+mbs?|[0-9]+r[0-9]+w?)$"
)

# Key kinds are ordered by precision. Their maximum document frequencies keep
# broad keys from turning into unhelpful blocks.
EXACT, CODE_CAP, CODE, STRUCTURED, ANCHOR = range(5)
BASE_SCORE = {
    EXACT: 220,
    CODE_CAP: 200,
    CODE: 160,
    STRUCTURED: 110,
    ANCHOR: 45,
}
MAX_FREQUENCY = {
    EXACT: 96,
    CODE_CAP: 192,
    CODE: 192,
    STRUCTURED: 192,
    ANCHOR: 96,
}
KIND_SALTS = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
    0x452821E638D01377,
)


@dataclass(frozen=True)
class Key:
    value: int
    kind: int


def mix64(value: int) -> int:
    """SplitMix64 finalizer."""
    value &= MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & MASK64
    value ^= value >> 31
    return value


def stable_hash(value: object) -> int:
    if isinstance(value, tuple):
        text = "\x1f".join(str(part) for part in value)
    else:
        text = str(value)
    digest = hashlib.blake2b(
        text.encode("utf-8"), digest_size=8, person=b"X2block"
    ).digest()
    return int.from_bytes(digest, "little")


def key_hash(value: object, kind: int) -> int:
    return mix64(stable_hash(value) ^ KIND_SALTS[kind])


def normalize(value: str) -> list[str]:
    value = html.unescape(value).lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.replace("&", " and ")
    tokens = TOKEN_RE.findall(value)

    result: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if index + 1 < len(tokens):
            following = tokens[index + 1]
            if token.isdigit() and following in {"g", "gb", "go"}:
                result.append(f"{int(token)}gb")
                index += 2
                continue
            if token.isdigit() and following in {"t", "tb", "to"}:
                result.append(f"{int(token)}tb")
                index += 2
                continue
        token = re.sub(r"(?<=\d)(go)$", "gb", token)
        result.append(token)
        index += 1
    return result


def is_model_code(token: str) -> bool:
    if len(token) < 3 or len(token) > 30:
        return False
    if GENERIC_CODE_RE.match(token) or SPEED_RE.match(token):
        return False
    return any(c.isalpha() for c in token) and any(c.isdigit() for c in token)


def record_features(name: str, brand_column: str) -> tuple[tuple[Key, ...], int, int]:
    tokens = normalize(name)
    if not tokens:
        return (), 0, 0

    token_set = set(tokens)
    signature = 0
    for token in token_set:
        if len(token) >= 2 and token not in STOPWORDS:
            signature |= 1 << (stable_hash(token) & 127)
    brands = sorted((token_set & KNOWN_BRANDS) | (set(normalize(brand_column)) & KNOWN_BRANDS))
    brand = brands[0] if brands else ""
    capacities = sorted({token for token in tokens if CAPACITY_RE.match(token)})
    families = sorted(token_set & FAMILIES)

    codes = {token for token in tokens if is_model_code(token)}
    for left, right in zip(tokens, tokens[1:]):
        joined = left + right
        if (
            len(left) >= 2
            and len(right) >= 2
            and len(joined) <= 30
            and is_model_code(joined)
        ):
            codes.add(joined)
    ranked_codes = sorted(codes, key=lambda item: (-len(item), item))[:3]

    useful = [
        token
        for token in tokens
        if len(token) >= 3
        and token not in STOPWORDS
        and token not in KNOWN_BRANDS
        and not SPEED_RE.match(token)
    ]

    keys: list[Key] = []
    canonical = tuple(sorted(set(useful)))
    if len(canonical) >= 2:
        keys.append(Key(key_hash(canonical, EXACT), EXACT))

    for code in ranked_codes:
        keys.append(Key(key_hash(code, CODE), CODE))
        for capacity in capacities[:2]:
            keys.append(Key(key_hash((code, capacity), CODE_CAP), CODE_CAP))

    for family in families[:2]:
        if brand and capacities:
            keys.append(
                Key(key_hash((brand, family, capacities[0]), STRUCTURED), STRUCTURED)
            )
        elif brand:
            keys.append(Key(key_hash((brand, family), STRUCTURED), STRUCTURED))
        elif capacities:
            keys.append(Key(key_hash((family, capacities[0]), STRUCTURED), STRUCTURED))

    anchors = sorted(
        {
            token
            for token in useful
            if len(token) >= 5
            and token not in FAMILIES
            and not CAPACITY_RE.match(token)
            and not token.isdigit()
        },
        key=lambda item: (-len(item), item),
    )
    for anchor in anchors[:2]:
        keys.append(Key(key_hash(anchor, ANCHOR), ANCHOR))

    # Preserve kind because two feature families may intentionally hash the
    # same textual value differently.
    unique: dict[int, Key] = {}
    for key in keys:
        unique[key.value] = key
    return tuple(unique.values()), signature, signature.bit_count()


def record_keys(name: str, brand_column: str) -> tuple[Key, ...]:
    return record_features(name, brand_column)[0]


def rows(path: Path) -> Iterator[tuple[int, str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "id" not in reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError(f"{path} must contain id and name columns")
        for row in reader:
            yield int(row["id"]), row.get("name", ""), row.get("brand", "")


class CountMinSketch:
    """A saturating count-min sketch with fixed, predictable memory."""

    def __init__(self, width_power: int = 24) -> None:
        self.width = 1 << width_power
        self.mask = self.width - 1
        self.tables = (bytearray(self.width), bytearray(self.width), bytearray(self.width))

    def _positions(self, value: int) -> tuple[int, int, int]:
        first = value & self.mask
        second = mix64(value ^ 0xD6E8FEB86659FD93) & self.mask
        third = mix64(value ^ 0xA5A3564E27F8862F) & self.mask
        return first, second, third

    def add(self, value: int) -> None:
        for table, position in zip(self.tables, self._positions(value)):
            count = table[position]
            if count < 255:
                table[position] = count + 1

    def count(self, value: int) -> int:
        return min(
            table[position]
            for table, position in zip(self.tables, self._positions(value))
        )


def count_features(input_path: Path, sketch_power: int) -> tuple[CountMinSketch, int]:
    sketch = CountMinSketch(sketch_power)
    count = 0
    for count, (_, name, brand) in enumerate(rows(input_path), start=1):
        keys, _, _ = record_features(name, brand)
        for key in keys:
            sketch.add(key.value)
        if count % 250_000 == 0:
            print(f"frequency pass: {count:,} records", file=sys.stderr)
    return sketch, count


def pack_pair(left: int, right: int) -> int:
    if left > right:
        left, right = right, left
    if left < 0 or right < 0 or left >= 1 << 32 or right >= 1 << 32:
        raise ValueError("record ids must be unsigned 32-bit integers")
    return (left << 32) | right


def unpack_pair(pair: int) -> tuple[int, int]:
    return pair >> 32, pair & 0xFFFFFFFF


class PairCollector:
    """Keep the best K pairs without ever materializing all candidates."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.heap: list[tuple[int, int]] = []
        self.pairs: set[int] = set()

    def offer(self, pair: int, score: int) -> None:
        if pair in self.pairs:
            return
        # Similarity score is primary. The mixed pair id supplies a stable,
        # well-distributed tie breaker without favoring input order.
        rank = (score << 64) | mix64(pair)
        item = (rank, pair)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, item)
            self.pairs.add(pair)
            return
        if item <= self.heap[0]:
            return
        _, removed = heapq.heapreplace(self.heap, item)
        self.pairs.remove(removed)
        self.pairs.add(pair)


def candidate_pass(
    input_path: Path,
    sketch: CountMinSketch,
    collector: PairCollector,
    pass_number: int,
) -> int:
    # One representative per key makes every block O(k), not O(k^2).
    # The value packs a 64-bit priority and a 32-bit record id.
    representatives: dict[int, int] = {}
    record_ids = array("I")
    signatures_low = array("Q")
    signatures_high = array("Q")
    signature_sizes = bytearray()
    offered = 0
    salt = mix64(0xC6BC279692B5CC83 ^ pass_number)

    for row_number, (record_id, name, brand) in enumerate(rows(input_path)):
        keys, signature, signature_size = record_features(name, brand)
        scores: dict[int, int] = defaultdict(int)

        for key in keys:
            frequency = sketch.count(key.value)
            if frequency < 2 or frequency > MAX_FREQUENCY[key.kind]:
                continue
            packed = representatives.get(key.value)
            previous = None if packed is None else packed & 0xFFFFFFFF
            if previous is not None and record_ids[previous] != record_id:
                rarity_bonus = max(0, 28 - 4 * int(math.log2(frequency)))
                scores[previous] += BASE_SCORE[key.kind] + rarity_bonus

        if scores:
            low = signature & MASK64
            high = signature >> 64
            for other in scores:
                overlap = (
                    (low & signatures_low[other]).bit_count()
                    + (high & signatures_high[other]).bit_count()
                )
                denominator = min(signature_size, signature_sizes[other])
                if denominator:
                    scores[other] += 400 * overlap // denominator
            ordered = sorted(
                scores,
                key=lambda other: (
                    -scores[other],
                    mix64(pack_pair(record_id, other) ^ salt),
                ),
            )
            # A fixed per-record limit preserves linear work even if a record
            # has many noisy features.
            for other in ordered[:6]:
                pair = pack_pair(record_id, record_ids[other])
                collector.offer(pair, scores[other])
                offered += 1

        # Keep the minimum-priority representative. Different pass salts yield
        # different stars for repeated keys while retaining constant fanout.
        for key in keys:
            frequency = sketch.count(key.value)
            if frequency < 2 or frequency > MAX_FREQUENCY[key.kind]:
                continue
            priority = mix64(record_id ^ key.value ^ salt)
            packed = representatives.get(key.value)
            if packed is None or priority < packed >> 32:
                representatives[key.value] = (priority << 32) | row_number

        record_ids.append(record_id)
        signatures_low.append(signature & MASK64)
        signatures_high.append(signature >> 64)
        signature_sizes.append(min(signature_size, 255))

        if (row_number + 1) % 250_000 == 0:
            print(
                f"candidate pass {pass_number + 1}: {row_number + 1:,} records, "
                f"{len(collector.pairs):,}/{collector.capacity:,} retained pairs",
                file=sys.stderr,
            )
    return offered


def fill_pairs(ids: array, selected: set[int], target: int) -> None:
    """Fill a shortfall in O(target) work using deterministic cyclic offsets."""
    count = len(ids)
    if count * (count - 1) // 2 < target:
        raise ValueError(
            f"{count:,} records provide fewer than {target:,} distinct valid pairs"
        )
    offset = 1
    while len(selected) < target and offset < count:
        upper = count - offset
        for index in range(upper):
            pair = pack_pair(ids[index], ids[index + offset])
            if pair not in selected:
                selected.add(pair)
                if len(selected) == target:
                    return
        offset += 1
    if len(selected) != target:
        raise RuntimeError("could not construct enough unique candidate pairs")


def read_ids(input_path: Path) -> array:
    result = array("I")
    seen: set[int] = set()
    for record_id, _, _ in rows(input_path):
        if record_id in seen:
            raise ValueError(f"duplicate record id: {record_id}")
        seen.add(record_id)
        result.append(record_id)
    result = array("I", sorted(result))
    return result


def write_output(output_path: Path, selected: Iterable[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("lid", "rid"))
        for pair in selected:
            writer.writerow(unpack_pair(pair))


def default_input() -> Path:
    full = Path("../SIGMOD_full_data/X2.csv")
    return full if full.exists() else Path("../SIGMOD_full_data/X2.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input())
    parser.add_argument("--output", type=Path, default=Path("output.csv"))
    parser.add_argument("--target", type=int, default=2_000_000)
    parser.add_argument(
        "--sketch-power",
        type=int,
        default=24,
        help="count-min width as a power of two (24 uses about 48 MiB)",
    )
    parser.add_argument("--passes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target <= 0:
        raise ValueError("--target must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    started = time.monotonic()
    sketch, row_count = count_features(args.input, args.sketch_power)
    if row_count < 2:
        raise ValueError("at least two input records are required")

    collector = PairCollector(args.target)
    for pass_number in range(args.passes):
        offered = candidate_pass(args.input, sketch, collector, pass_number)
        print(
            f"candidate pass {pass_number + 1} offered {offered:,} pairs; "
            f"retained {len(collector.pairs):,}",
            file=sys.stderr,
        )
        if offered == 0:
            break

    ids = read_ids(args.input)
    fill_pairs(ids, collector.pairs, args.target)
    if len(collector.pairs) != args.target:
        raise AssertionError("internal error: incorrect output size")
    write_output(args.output, collector.pairs)
    elapsed = time.monotonic() - started
    print(
        f"wrote {len(collector.pairs):,} pairs to {args.output} in {elapsed:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
