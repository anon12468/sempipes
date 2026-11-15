#!/usr/bin/env python3
"""Scalable blocking for the X2 electronic-products data set.

The blocker never reads a ground-truth file.  Every inverted-index posting is
bounded by a small constant, so a common key cannot cause quadratic work.
Candidates are added in confidence-ordered passes and the candidate set never
grows beyond the requested output size.
"""

from __future__ import annotations

import argparse
from array import array
import csv
import gc
import heapq
import html
from pathlib import Path
import re
import struct
import sys
import tempfile
import unicodedata


TARGET_SIZE = 2_000_000
MASK64 = (1 << 64) - 1
FEATURE_STRUCT = struct.Struct("<17Q")

ALIASES = {
    "attache": "attach",
    "cl": "class",
    "clas": "class",
    "clase": "class",
    "classe": "class",
    "datattraveler": "datatraveler",
    "datattraveller": "datatraveler",
    "datatraveller": "datatraveler",
    "extrem": "extreme",
    "gbyte": "gb",
    "gigabyte": "gb",
    "klasse": "class",
    "sandisc": "sandisk",
    "traveller": "traveler",
    "uhs1": "uhsi",
}

# Words which carry little product identity.  Product families, capacities,
# speeds, and part numbers are intentionally retained.
STOP_WORDS = {
    "a",
    "an",
    "and",
    "avec",
    "card",
    "carte",
    "con",
    "da",
    "de",
    "del",
    "des",
    "direct",
    "disk",
    "drive",
    "du",
    "en",
    "et",
    "flash",
    "for",
    "geheugenkaart",
    "in",
    "karta",
    "la",
    "le",
    "les",
    "memoria",
    "memoire",
    "memory",
    "mit",
    "new",
    "online",
    "pami",
    "pendrive",
    "pour",
    "product",
    "scheda",
    "shop",
    "speicherkarte",
    "stick",
    "tarjeta",
    "the",
    "to",
    "und",
    "von",
    "with",
    "zu",
}

TOKEN_RE = re.compile(
    r"[a-z0-9]+(?:[-_/][a-z0-9]+)+|[a-z]+\d+[a-z0-9]*|"
    r"\d+[a-z]+[a-z0-9]*|[a-z]+|\d+"
)

# Competitive domain-specific signatures from the supplied baseline, retained
# here only as feature extraction (the baseline's quadratic expansion is not
# used).
BASELINE_CLEAN_RE = re.compile(
    r"&(nbsp|amp|reg|[a-z]?acute|quot|trade);?|[|;:/,‰+©()\\][psn]*|"
    r"(?<=usb)[\s][m]*(?=[23][.\s])|(?<=usb)-[\w]+\s(?=[23][.\s])|"
    r"(?<=[a-z])[\s]+gb|(?<=data|jump)[t\s](?=trave|drive)|"
    r"(?<=extreme|exceria)[\s](?=pro[\s]|plus)|"
    r"(?<=class)[\s_](?=10|[234]\b)|(?<=gen)[\s_](?=[134\s][0]*)"
)
BASELINE_CLASS10_RE = re.compile(r"(10 class|class 10|class(?=[\w]+10\b)|cl\s10)")
BASELINE_MODEL_NUM_RE = re.compile(
    r"\b([\(]*[\w]+[-]*[\d]+[-]*[\w]+[-]*[\d+]*|[\d]+[\w]|[\w][\d]+)"
)
BASELINE_BRAND_RE = re.compile(
    r"\b(intenso|lexar|logilink|pny|samsung|sandisk|kingston|sony|toshiba|transcend)\b"
)
BASELINE_MODEL_RE = re.compile(
    r"\b(datatraveler|extreme[p]?|exceria[p]?|dual[\s]*(?!=sim)|evo|xqd|"
    r"ssd|cruzer[\w+]*|glide|blade|basic|fit|force|basic line|jump\s?drive|"
    r"hxs|rainbow|speed line|premium line|att4|attach|serie u|r-serie|"
    r"beast|fury|impact|a400|sd[hx]c|uhs[i12][i1]*|note\s?9|ultra)"
)
BASELINE_MEM_RE = re.compile(r"([1-9]{1,3})[-\s]*[g][bo]?")
BASELINE_LONG_RE = re.compile(
    r"(thn-[a-z][\w]+|ljd[\w+]-[\w]+|ljd[sc][\w]+-[\w]+|"
    r"lsdmi[\d]+[\w]+|lsd[0-9]{1,3}[gb]+[\w]+|ljds[0-9]{2}-[\w]+|"
    r"usm[0-9]{1,3}[\w]+|sdsq[a-z]+-[0-9]+[a-z]+-[\w]+|"
    r"sdsd[a-z]+-[0-9]+[\w]+-?[\w]*|sdcz[\w]+|mk[\d]+|sr-g1[\w]+)"
)
BASELINE_SHORT_RE = re.compile(
    r"\b(c20[mc]|sda[0-9]{1,2}|g1ux|s[72][05]|[unm][23]02|p20|g4|dt101|se9|[asm][0-9]{2})"
)
BASELINE_FEATURE_RE = re.compile(
    r"\b(usb[23]|type-c|uhs[i]{1,2}|class[0134]{1,2}|gen[1-9]{1,2}|"
    r"u[23](?=[\s.])|sd[hx]c|otg|lte|[45]g[-\s]lte|"
    r"[0-9]+(?=-inch)|[0-9]{2,3}r[0-9]{2,3}w|[0-9]{2,3}(?=[\smbo/p]{3}))"
)
BASELINE_ALIASES = {
    "class": ("classe", "clase", "clas ", "klasse", "cl "),
    "uhsi": ("uhs1", "uhs-i", "ultra high-speed"),
    "type-c": ("typec", "type c", "usb-c", "usbc"),
    "intenso premium line": ("3534490",),
    "kingston hxs ": ("hyperx", "savage"),
    "sony g1ux": ("serie ux",),
    " kingston dt101 ": ("dtig4", " 101 ", "dt101g2"),
    " kingston ultimate ": ("sda10", "sda3"),
    "extreme ": ("extrem ",),
    "att4": ("attach",),
}


def stable_runtime_hash(value: str) -> int:
    """Return a nonzero 64-bit hash (persistence across runs is unnecessary)."""
    result = hash(value) & MASK64
    return result or 1


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def ascii_text(value: object) -> str:
    text = html.unescape("" if value is None else str(value)).lower()
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char) and ord(char) < 128
    )


def normalise_tokens(name: str) -> list[str]:
    raw_tokens = TOKEN_RE.findall(ascii_text(name))
    tokens: list[str] = []
    index = 0
    while index < len(raw_tokens):
        token = ALIASES.get(raw_tokens[index].replace("_", "-"), raw_tokens[index])
        compact = re.sub(r"[-_/]", "", token)

        # Canonicalise "64 GB", "64 Go", and similar capacity spellings.
        if (
            compact.isdigit()
            and index + 1 < len(raw_tokens)
            and raw_tokens[index + 1] in {"g", "gb", "go", "gbyte", "tb", "to"}
        ):
            unit = "tb" if raw_tokens[index + 1] in {"tb", "to"} else "gb"
            tokens.append(compact + unit)
            index += 2
            continue

        if compact.endswith("go") and compact[:-2].isdigit():
            compact = compact[:-2] + "gb"
        elif compact.endswith("g") and compact[:-1].isdigit():
            compact = compact[:-1] + "gb"
        compact = ALIASES.get(compact, compact)
        if compact and compact not in STOP_WORDS:
            tokens.append(compact)
        index += 1
    return tokens


def token_bloom(tokens: set[str]) -> tuple[int, int, int, int]:
    blooms = [0, 0, 0, 0]
    for token in tokens:
        token_hash = stable_runtime_hash(token)
        for bit_number in (token_hash & 255, (token_hash >> 17) & 255):
            blooms[bit_number >> 6] |= 1 << (bit_number & 63)
    return blooms[0], blooms[1], blooms[2], blooms[3]


def baseline_signatures(name: str) -> tuple[int, ...]:
    raw_name = ascii_text(name)
    raw_exact = stable_runtime_hash(" ".join(sorted(raw_name.split())))
    phrases = sorted(re.findall(r"\w+\s\w+\d+", raw_name))
    phrase_key = stable_runtime_hash("\x1f".join(phrases)) if phrases else 0

    cleaned = raw_name
    for canonical, variants in BASELINE_ALIASES.items():
        for variant in variants:
            cleaned = cleaned.replace(variant, canonical)
    cleaned = BASELINE_CLEAN_RE.sub("", cleaned)
    cleaned = BASELINE_CLASS10_RE.sub("class10", cleaned)
    cleaned = cleaned.replace("class 4 ", "class4 ").replace("class 3 ", "class3 ")
    cleaned = " ".join(cleaned.split())
    clean_exact = stable_runtime_hash("\x1f".join(sorted(cleaned.split())))

    brands = " ".join(sorted(set(BASELINE_BRAND_RE.findall(cleaned))))
    models = " ".join(sorted(set(BASELINE_MODEL_RE.findall(cleaned))))
    memories = " ".join(sorted(set(BASELINE_MEM_RE.findall(cleaned))))
    features = " ".join(sorted(set(BASELINE_FEATURE_RE.findall(cleaned))))
    model_numbers = " ".join(
        value for value in BASELINE_MODEL_NUM_RE.findall(cleaned) if len(value) > 5
    )
    long_match = BASELINE_LONG_RE.search(cleaned)
    long_code = long_match.group() if long_match else ""
    short_codes = " ".join(sorted(set(BASELINE_SHORT_RE.findall(cleaned))))

    def make_key(tag: str, value: str) -> int:
        value = value.strip()
        return stable_runtime_hash(f"{tag}:{value}") if value else 0

    return (
        raw_exact,
        phrase_key,
        clean_exact,
        make_key("long", long_code),
        make_key("short", short_codes),
        make_key("bf", f"{brands} {features}"),
        make_key("bm", f"{brands} {memories}"),
        make_key("bmt", f"{brands} {memories} {models}"),
        make_key("bn", f"{brands} {model_numbers}"),
    )


def extract_features(name: str) -> tuple[int, ...]:
    token_list = normalise_tokens(name)
    tokens = set(token_list)
    signatures = baseline_signatures(name)

    minhash_tokens = tokens - STOP_WORDS
    minhashes: list[int] = []
    for seed in (
        0x243F6A8885A308D3,
        0x13198A2E03707344,
        0xA4093822299F31D0,
        0x082EFA98EC4E6C89,
    ):
        minimum = MASK64
        for token in minhash_tokens:
            value = splitmix64(stable_runtime_hash(token) ^ seed)
            if value < minimum:
                minimum = value
        minhashes.append(0 if minimum == MASK64 else minimum or 1)

    blooms = token_bloom(tokens)
    return (
        blooms[0],
        blooms[1],
        blooms[2],
        blooms[3],
        signatures[0],
        signatures[1],
        signatures[2],
        signatures[3],
        signatures[4],
        signatures[5],
        signatures[6],
        signatures[7],
        signatures[8],
        minhashes[0],
        minhashes[1],
        minhashes[2],
        minhashes[3],
    )


def locate_column(fieldnames: list[str], choices: tuple[str, ...]) -> str:
    lowered = {field.lower(): field for field in fieldnames}
    for choice in choices:
        if choice in lowered:
            return lowered[choice]
    raise ValueError(f"Missing required column; expected one of {choices}")


def build_feature_cache(
    input_path: Path, cache_path: Path
) -> tuple[array, tuple[array, array, array, array]]:
    ids = array("q")
    blooms = (array("Q"), array("Q"), array("Q"), array("Q"))
    seen_ids: set[int] = set()

    with input_path.open("r", encoding="utf-8-sig", newline="") as source, cache_path.open(
        "wb"
    ) as cache:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("The input CSV has no header")
        id_column = locate_column(reader.fieldnames, ("id", "instance_id"))
        name_column = locate_column(reader.fieldnames, ("name", "title"))
        for row_number, row in enumerate(reader, start=2):
            try:
                real_id = int(row[id_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid id on CSV row {row_number}") from error
            if real_id in seen_ids:
                raise ValueError(f"Duplicate input id: {real_id}")
            seen_ids.add(real_id)

            features = extract_features(row.get(name_column, ""))
            ids.append(real_id)
            for bloom_index in range(4):
                blooms[bloom_index].append(features[bloom_index])
            cache.write(FEATURE_STRUCT.pack(*features))

    return ids, blooms


def feature_records(cache_path: Path):
    chunk_size = FEATURE_STRUCT.size * 16_384
    with cache_path.open("rb") as cache:
        remainder = b""
        while True:
            chunk = cache.read(chunk_size)
            if not chunk:
                break
            chunk = remainder + chunk
            usable = len(chunk) - (len(chunk) % FEATURE_STRUCT.size)
            yield from struct.iter_unpack(FEATURE_STRUCT.format, chunk[:usable])
            remainder = chunk[usable:]
        if remainder:
            raise ValueError("Corrupt feature cache")


def bloom_similarity(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    denominator = min(
        sum(value.bit_count() for value in left),
        sum(value.bit_count() for value in right),
    )
    intersection = sum((a & b).bit_count() for a, b in zip(left, right))
    return intersection / denominator if denominator else 0.0


class TopCandidates:
    """Keep only the best requested candidates, never the full stream."""

    def __init__(self, limit: int):
        self.limit = limit
        self.scores: dict[int, int] = {}
        self.heap: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return len(self.scores)

    def offer(self, packed_pair: int, score: int) -> None:
        if packed_pair in self.scores:
            return
        if len(self.scores) < self.limit:
            self.scores[packed_pair] = score
            heapq.heappush(self.heap, (score, packed_pair))
            return
        if score <= self.heap[0][0]:
            return
        _, removed_pair = heapq.heapreplace(self.heap, (score, packed_pair))
        del self.scores[removed_pair]
        self.scores[packed_pair] = score


def gather_candidates(
    index: dict[int, int | list[int]],
    keys: tuple[int, ...],
    position: int,
    posting_cap: int,
) -> dict[int, int]:
    votes: dict[int, int] = {}
    for key in set(keys):
        if not key:
            continue
        posting = index.get(key)
        if posting == -2:  # key was too frequent in the counting pass
            continue
        if posting == -1:  # first occurrence of an eligible key
            index[key] = position
            continue
        if posting is None:
            index[key] = position
            continue
        if isinstance(posting, int):
            votes[posting] = votes.get(posting, 0) + 1
            index[key] = [posting, position]
            continue
        for previous in posting:
            votes[previous] = votes.get(previous, 0) + 1
        if len(posting) >= posting_cap:
            del posting[0]
        posting.append(position)
    return votes


def run_phase(
    cache_path: Path,
    key_indexes: tuple[int, ...],
    posting_cap: int,
    candidates: TopCandidates,
    blooms: tuple[array, array, array, array],
    phase_score: int,
    minimum_votes: int = 1,
    per_record_limit: int = 32,
    frequency_cap: int = 256,
) -> None:
    """Generate at most posting_cap candidates per key and record."""
    counts: dict[int, int] = {}
    for features in feature_records(cache_path):
        for key in {
            features[key_index] for key_index in key_indexes if features[key_index]
        }:
            count = counts.get(key, 0)
            if count <= frequency_cap:
                counts[key] = count + 1

    # -1 means eligible but unseen in this pass; -2 means intentionally
    # skipped.  This reuses the count dictionary as the posting index.
    index: dict[int, int | list[int]] = {
        key: (-1 if count <= frequency_cap else -2) for key, count in counts.items()
    }
    del counts
    for position, features in enumerate(feature_records(cache_path)):
        votes = gather_candidates(
            index,
            tuple(features[key_index] for key_index in key_indexes),
            position,
            posting_cap,
        )
        ranked = sorted(
            (
                (
                    vote,
                    bloom_similarity(
                        (features[0], features[1], features[2], features[3]),
                        (
                            blooms[0][previous],
                            blooms[1][previous],
                            blooms[2][previous],
                            blooms[3][previous],
                        ),
                    ),
                    previous,
                )
                for previous, vote in votes.items()
                if vote >= minimum_votes
            ),
            reverse=True,
        )
        for vote, similarity, previous in ranked[:per_record_limit]:
            score = phase_score + vote * 100 + int(similarity * 99)
            candidates.offer((previous << 32) | position, score)


def add_fallback_pairs(
    candidates: TopCandidates, record_count: int, target: int
) -> None:
    """Fill a shortfall in O(target) time without self-pairs or duplicates."""
    offset = 1
    while len(candidates) < target and offset < record_count:
        for left in range(record_count - offset):
            candidates.offer((left << 32) | (left + offset), 0)
            if len(candidates) >= target:
                return
        offset += 1


def write_output(
    output_path: Path, candidates: TopCandidates, ids: array, target: int
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("lid", "rid"))
        written = 0
        for packed_pair in candidates.scores:
            left_position = packed_pair >> 32
            right_position = packed_pair & 0xFFFFFFFF
            left_id = ids[left_position]
            right_id = ids[right_position]
            if left_id > right_id:
                left_id, right_id = right_id, left_id
            writer.writerow((left_id, right_id))
            written += 1
        if written != target:
            raise AssertionError(f"Expected {target} pairs, wrote {written}")


def block(input_path: Path, output_path: Path, target: int = TARGET_SIZE) -> None:
    if target <= 0:
        raise ValueError("target must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_handle = tempfile.NamedTemporaryFile(
        prefix=".x2_features_", suffix=".bin", dir=output_path.parent, delete=False
    )
    cache_path = Path(cache_handle.name)
    cache_handle.close()
    try:
        ids, blooms = build_feature_cache(input_path, cache_path)
        record_count = len(ids)
        if record_count * (record_count - 1) // 2 < target:
            raise ValueError(
                f"{record_count} records allow only "
                f"{record_count * (record_count - 1) // 2:,} unique pairs; "
                f"cannot write {target:,}. Use --target for development data."
            )

        candidates = TopCandidates(target)
        # Passes are ordered from precise to broad.  All posting caps are
        # constants, independent of bucket size.
        phases = (
            ("exact title", (4, 5, 6), 32, 1, 32, 6_000, 128),
            ("domain keys", (7, 8, 9, 10, 11, 12), 64, 1, 64, 5_000, 149),
            ("minhash-2", (13, 14, 15, 16), 16, 2, 16, 3_000, 128),
            ("minhash-1", (13, 14, 15, 16), 8, 1, 16, 1_000, 256),
        )
        for (
            phase_name,
            key_indexes,
            posting_cap,
            minimum_votes,
            per_record_limit,
            phase_score,
            frequency_cap,
        ) in phases:
            run_phase(
                cache_path,
                key_indexes,
                posting_cap,
                candidates,
                blooms,
                phase_score,
                minimum_votes,
                per_record_limit,
                frequency_cap,
            )
            print(f"{phase_name}: {len(candidates):,} candidates", file=sys.stderr)
            gc.collect()

        add_fallback_pairs(candidates, record_count, target)
        write_output(output_path, candidates, ids, target)
    finally:
        try:
            cache_path.unlink()
        except FileNotFoundError:
            pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_input = Path("X2.csv")
    if not default_input.exists() and Path("../SIGMOD_full_data/X2.csv").exists():
        default_input = Path("../SIGMOD_full_data/X2.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=default_input)
    parser.add_argument("output", nargs="?", type=Path, default=Path("output.csv"))
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_SIZE,
        help="candidate count (default: 2,000,000; lower values are for dev data)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args(sys.argv[1:])
    block(arguments.input, arguments.output, arguments.target)
