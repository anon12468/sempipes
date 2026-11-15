"""Scalable blocking for the X2 electronic-products data.

The blocker uses only X2. It never reads the ground-truth file. Every
blocking window has a constant width, so even a key shared by millions of
rows takes linear rather than quadratic work.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import html
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TARGET = 2_000_000
UINT32_MAX = (1 << 32) - 1
UINT64_MASK = (1 << 64) - 1

# Column positions in the compact uint64 feature matrix.
EXACT, FINGERPRINT, BRAND, CAP1, CAP2 = range(5)
MODEL1, MODEL2, MODEL3 = range(5, 8)
TERM1, TERM2, TERM3, TERM4 = range(8, 12)
MINHASH_START = 12
MINHASH_COUNT = 8

MINHASH_SEEDS = (
    0,
    0x9E3779B97F4A7C15,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0x85EBCA77C2B2AE63,
    0x27D4EB2F165667C5,
    0xD6E8FEB86659FD93,
    0xA5A3564E27F8862F,
)

KNOWN_BRANDS = {
    "adata",
    "corsair",
    "emtec",
    "hama",
    "integral",
    "intenso",
    "kingston",
    "lexar",
    "maxell",
    "philips",
    "pny",
    "samsung",
    "sandisk",
    "sony",
    "toshiba",
    "transcend",
    "verbatim",
}

# Product-family words such as "cruzer", "datatraveler", and "exceria" are
# intentionally absent: unlike these generic/multilingual words, they help.
STOP_WORDS = {
    "a",
    "amazon",
    "avec",
    "adapter",
    "adaptador",
    "adaptateur",
    "black",
    "blue",
    "card",
    "carte",
    "class",
    "classe",
    "clase",
    "cle",
    "da",
    "de",
    "di",
    "direct",
    "do",
    "drive",
    "du",
    "en",
    "et",
    "flash",
    "fnac",
    "gb",
    "go",
    "high",
    "inch",
    "jusqu",
    "karta",
    "la",
    "le",
    "les",
    "mb",
    "memoria",
    "memoire",
    "memory",
    "micro",
    "microsd",
    "microsdhc",
    "microsdxc",
    "mit",
    "nero",
    "noir",
    "online",
    "pamieci",
    "pendrive",
    "pour",
    "product",
    "read",
    "red",
    "scheda",
    "schwarz",
    "sd",
    "sdhc",
    "sdxc",
    "series",
    "shop",
    "silver",
    "speed",
    "speicherkarte",
    "stick",
    "tarjeta",
    "tb",
    "tesco",
    "the",
    "uhs",
    "und",
    "up",
    "usb",
    "weiss",
    "white",
    "with",
    "write",
    "zu",
}

CAPACITIES = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}
TOKEN_RE = re.compile(r"[a-z0-9]+")
CAPACITY_RE = re.compile(r"(\d{1,4})(tb|gb|go)")
MODEL_RE = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z0-9]{3,}")
NON_MODEL_RE = re.compile(r"\d+(?:mb|gb|go|tb|mbps)")


def normalize(value: object) -> str:
    """Return a language-independent alphanumeric representation."""
    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value)).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(TOKEN_RE.findall(text))


def nonzero_hash(value: object) -> int:
    """Return a fast 64-bit hash, reserving zero for a missing feature."""
    result = hash(value) & UINT64_MASK
    return result or 1


def extract_capacities(tokens: list[str]) -> list[int]:
    capacities: list[int] = []
    for position, token in enumerate(tokens):
        match = CAPACITY_RE.fullmatch(token)
        if match:
            number, unit = match.groups()
        elif (
            token.isdigit()
            and position + 1 < len(tokens)
            and tokens[position + 1] in {"tb", "gb", "go"}
        ):
            number, unit = token, tokens[position + 1]
        else:
            continue

        capacity = int(number) * (1024 if unit == "tb" else 1)
        if capacity in CAPACITIES and capacity not in capacities:
            capacities.append(capacity)
    return capacities[:2]


def extract_features(names: np.ndarray, brands: np.ndarray) -> np.ndarray:
    """Extract fixed-size signatures without retaining posting lists."""
    features = np.zeros(
        (len(names), MINHASH_START + MINHASH_COUNT), dtype=np.uint64
    )

    for row_number, (raw_name, raw_brand) in enumerate(zip(names, brands)):
        normalized_name = normalize(raw_name)
        tokens = normalized_name.split()

        brand = next((token for token in tokens if token in KNOWN_BRANDS), "")
        if not brand:
            normalized_brand = normalize(raw_brand)
            brand = normalized_brand.split()[0] if normalized_brand else ""

        capacities = extract_capacities(tokens)
        informative = {
            token
            for token in tokens
            if len(token) >= 2
            and not token.isdigit()
            and token not in STOP_WORDS
            and token not in KNOWN_BRANDS
        }
        models = sorted(
            (
                token
                for token in informative
                if MODEL_RE.fullmatch(token) and not NON_MODEL_RE.fullmatch(token)
            ),
            key=lambda token: (-len(token), token),
        )
        terms = sorted(
            informative,
            key=lambda token: (token not in models, -len(token), token),
        )

        if normalized_name:
            features[row_number, EXACT] = nonzero_hash(normalized_name)
        if informative:
            features[row_number, FINGERPRINT] = nonzero_hash(
                tuple(sorted(informative))
            )
        if brand:
            features[row_number, BRAND] = nonzero_hash(brand)
        for offset, capacity in enumerate(capacities):
            features[row_number, CAP1 + offset] = capacity
        for offset, model in enumerate(models[:3]):
            features[row_number, MODEL1 + offset] = nonzero_hash(model)
        for offset, term in enumerate(terms[:4]):
            features[row_number, TERM1 + offset] = nonzero_hash(term)

        token_hashes = [nonzero_hash(token) for token in informative]
        if token_hashes:
            for offset, seed in enumerate(MINHASH_SEEDS):
                features[row_number, MINHASH_START + offset] = min(
                    token_hash ^ seed for token_hash in token_hashes
                )

        if row_number and row_number % 100_000 == 0:
            print(f"extracted features for {row_number:,} rows", file=sys.stderr)

    return features


def any_cross_match(
    features: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_columns: range,
    right_columns: range,
) -> np.ndarray:
    result = np.zeros(len(left), dtype=bool)
    for left_column in left_columns:
        left_values = features[left, left_column]
        for right_column in right_columns:
            right_values = features[right, right_column]
            result |= (left_values != 0) & (left_values == right_values)
    return result


def score_pairs(
    features: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Score pairs using only compact, precomputed signatures."""
    exact = features[left, EXACT]
    fingerprint = features[left, FINGERPRINT]
    brand = features[left, BRAND]

    scores = (
        30 * ((exact != 0) & (exact == features[right, EXACT])).astype(np.int16)
        + 20
        * (
            (fingerprint != 0)
            & (fingerprint == features[right, FINGERPRINT])
        ).astype(np.int16)
        + 6
        * ((brand != 0) & (brand == features[right, BRAND])).astype(np.int16)
    )
    capacity_match = any_cross_match(
        features, left, right, range(CAP1, CAP2 + 1), range(CAP1, CAP2 + 1)
    )
    model_match = any_cross_match(
        features,
        left,
        right,
        range(MODEL1, MODEL3 + 1),
        range(MODEL1, MODEL3 + 1),
    )
    term_match = any_cross_match(
        features,
        left,
        right,
        range(TERM1, TERM4 + 1),
        range(TERM1, TERM4 + 1),
    )
    scores += 8 * capacity_match.astype(np.int16)
    scores += 12 * model_match.astype(np.int16)
    scores += 4 * term_match.astype(np.int16)

    for column in range(MINHASH_START, MINHASH_START + MINHASH_COUNT):
        values = features[left, column]
        scores += (
            3
            * ((values != 0) & (values == features[right, column])).astype(
                np.int16
            )
        )
    return scores


class OnlineTopPairs:
    """A deduplicated top-K heap whose memory is bounded by K."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.heap: list[tuple[int, int]] = []
        self.selected: set[int] = set()

    def add(self, pair_code: int, score: int) -> None:
        if pair_code in self.selected:
            return

        item = (score, pair_code)
        if len(self.heap) < self.target:
            self.heap.append(item)
            self.selected.add(pair_code)
            if len(self.heap) == self.target:
                heapq.heapify(self.heap)
            return

        if item <= self.heap[0]:
            return
        removed = heapq.heapreplace(self.heap, item)
        self.selected.remove(removed[1])
        self.selected.add(pair_code)

    def add_arrays(self, pair_codes: np.ndarray, scores: np.ndarray) -> None:
        for pair_code, score in zip(pair_codes.tolist(), scores.tolist()):
            self.add(pair_code, score)


def encode_pairs(ids: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_ids = ids[left]
    right_ids = ids[right]
    low = np.minimum(left_ids, right_ids)
    high = np.maximum(left_ids, right_ids)
    return (low << np.uint64(32)) | high


def process_sorted_neighborhood(
    label: str,
    features: np.ndarray,
    ids: np.ndarray,
    top_pairs: OnlineTopPairs,
    primary_columns: tuple[int, ...],
    secondary_columns: tuple[int, ...],
    window: int,
) -> None:
    """Compare at most ``window`` successors per row in each key group."""
    columns = primary_columns + secondary_columns
    # np.lexsort treats its last key as primary.
    sort_keys = tuple(features[:, column] for column in reversed(columns))
    order = np.lexsort(sort_keys)
    emitted = 0

    for distance in range(1, window + 1):
        left_all = order[:-distance]
        right_all = order[distance:]
        same_group = np.ones(len(left_all), dtype=bool)
        for column in primary_columns:
            left_values = features[left_all, column]
            same_group &= (left_values != 0) & (
                left_values == features[right_all, column]
            )

        matching_positions = np.flatnonzero(same_group)
        emitted += len(matching_positions)
        for start in range(0, len(matching_positions), 200_000):
            positions = matching_positions[start : start + 200_000]
            left = left_all[positions]
            right = right_all[positions]
            pair_codes = encode_pairs(ids, left, right)
            scores = score_pairs(features, left, right)
            top_pairs.add_arrays(pair_codes, scores)

    print(
        f"{label}: scored {emitted:,} local pairs; "
        f"retained {len(top_pairs.heap):,}",
        file=sys.stderr,
    )


def add_filler_pairs(ids: np.ndarray, top_pairs: OnlineTopPairs) -> None:
    """Fill a shortfall with unique valid pairs in O(target) work."""
    sorted_ids = np.sort(ids)
    for distance in range(1, len(sorted_ids)):
        left = sorted_ids[:-distance]
        right = sorted_ids[distance:]
        codes = (left << np.uint64(32)) | right
        for code in codes.tolist():
            top_pairs.add(code, -1)
            if len(top_pairs.heap) == top_pairs.target:
                return
    raise RuntimeError("could not construct the requested number of pairs")


def generate_pairs(
    input_path: Path, target: int
) -> tuple[list[tuple[int, int]], int]:
    frame = pd.read_csv(
        input_path,
        usecols=["id", "name", "brand"],
        dtype={"name": "string", "brand": "string"},
    )
    numeric_ids = pd.to_numeric(frame["id"], errors="raise")
    if numeric_ids.isna().any() or not np.equal(
        numeric_ids, np.floor(numeric_ids)
    ).all():
        raise ValueError("all ids must be non-null integers")
    ids = numeric_ids.to_numpy(dtype=np.uint64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("ids must be unique")
    if len(ids) and int(ids.max()) > UINT32_MAX:
        raise ValueError("ids must fit in an unsigned 32-bit integer")
    maximum_pairs = len(ids) * (len(ids) - 1) // 2
    if target > maximum_pairs:
        raise ValueError(
            f"{len(ids):,} rows allow only {maximum_pairs:,} unique pairs, "
            f"fewer than the requested {target:,}"
        )

    names = frame["name"].to_numpy(dtype=object)
    brands = frame["brand"].to_numpy(dtype=object)
    del frame
    print(f"loaded {len(ids):,} records", file=sys.stderr)
    features = extract_features(names, brands)
    del names, brands

    top_pairs = OnlineTopPairs(target)
    passes = [
        ("exact normalized name", (EXACT,), (FINGERPRINT,), 8),
        ("exact token fingerprint", (FINGERPRINT,), (EXACT,), 8),
        ("brand/capacity/model-1", (BRAND, CAP1, MODEL1), (FINGERPRINT,), 4),
        ("brand/model-1", (BRAND, MODEL1), (CAP1, FINGERPRINT), 3),
        ("brand/capacity/model-2", (BRAND, CAP1, MODEL2), (FINGERPRINT,), 4),
        ("brand/model-2", (BRAND, MODEL2), (CAP1, FINGERPRINT), 3),
        ("brand/capacity/model-3", (BRAND, CAP1, MODEL3), (FINGERPRINT,), 3),
        ("brand/capacity/term-1", (BRAND, CAP1, TERM1), (FINGERPRINT,), 3),
        ("brand/capacity/term-2", (BRAND, CAP1, TERM2), (FINGERPRINT,), 3),
        ("brand/capacity/term-3", (BRAND, CAP1, TERM3), (FINGERPRINT,), 3),
    ]
    for offset in range(MINHASH_COUNT):
        passes.append(
            (
                f"brand/capacity/minhash-{offset + 1}",
                (BRAND, CAP1, MINHASH_START + offset),
                (FINGERPRINT,),
                2,
            )
        )

    for label, primary, secondary, window in passes:
        process_sorted_neighborhood(
            label,
            features,
            ids,
            top_pairs,
            primary,
            secondary,
            window,
        )

    if len(top_pairs.heap) < target:
        print(
            f"adding {target - len(top_pairs.heap):,} deterministic filler pairs",
            file=sys.stderr,
        )
        add_filler_pairs(ids, top_pairs)

    return top_pairs.heap, len(ids)


def write_output(output_path: Path, pairs: list[tuple[int, int]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["lid", "rid"])
        for _, pair_code in pairs:
            writer.writerow([pair_code >> 32, pair_code & UINT32_MAX])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../SIGMOD_full_data/X2.csv"),
        help="X2 input CSV (default: X2.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.csv"),
        help="candidate CSV (default: output.csv)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"number of output pairs (default: {DEFAULT_TARGET})",
    )
    args = parser.parse_args()
    if args.target <= 0:
        parser.error("--target must be positive")
    return args


def main() -> None:
    args = parse_args()
    pairs, row_count = generate_pairs(args.input, args.target)
    write_output(args.output, pairs)
    print(
        f"wrote {len(pairs):,} unique pairs from {row_count:,} records "
        f"to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
