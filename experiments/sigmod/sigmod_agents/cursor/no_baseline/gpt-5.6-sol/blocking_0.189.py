"""Scalable blocking for the X2 electronic-products data.

Candidate generation never expands a blocking bucket into all of its pairs.
Each posting list is sorted by a secondary signature and then linked with a
constant-size neighbour window, so a bucket of size k costs O(k log k) to
order and O(k) to emit. The in-memory pair set is capped at the output size.
"""

from __future__ import annotations

import argparse
from array import array
import csv
from itertools import groupby
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import unicodedata


TARGET_PAIRS = 2_000_000
PARTITIONS = 32  # Must be a power of two.
MAX_BUCKET = 256
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
RAW_RECORD = struct.Struct("<QQI")  # feature hash, signature, row number
UINT32 = struct.Struct("<I")
NUM_MINHASH = 24
BAND_SIZE = 2
NUM_BANDS = NUM_MINHASH // BAND_SIZE

STOP_WORDS = frozenset(
    """
    a an and are avec by con das de del der des die du ein eine el en et for
    from in la le les mit new of on or para per pour the to und van von with
    amazon best brand computer computers ebay electronic electronics item
    jetzt laptop notebook product shop online tesco fnac direct achat rayon
    egypt media technology digital color black white negro nero schwarz
    silber nbsp amp eacute jusqu jusqua
    """.split()
)
GENERIC_WORDS = frozenset(
    """
    adapter adaptador adattatore adaptateur card carte class drive extreme
    flash gb hd hdd led line mb memory memoria micro microsd mhz pendrive pro
    ram scheda sd sdhc sdxc speed ssd stick tb uhs ultra usb wifi
    speicherkarte karta pamici lecteur unidad unita blister high performance
    series serie gen type class10 cl10 usb3
    """.split()
)
BRANDS = frozenset(
    """
    sandisk sony kingston lexar intenso samsung toshiba pny transcend
    verbatim corsair patriot crucial kingstontechnology
    """.split()
)
PRODUCT_LINES = frozenset(
    """
    rainbow exceria cruzer glide jumpdrive canvas datatraveler attache flair
    premium basic edge fit blade dual rescue expert traveler microvault
    transmemory hyperx savage evo vault select otg extreme ultra plus
    professional
    """.split()
)
TOKEN_RE = re.compile(r"[a-z]+\d+[a-z0-9]*|\d+[a-z]+[a-z0-9]*|[a-z]{2,}|\d+")
HYPHEN_CODE_RE = re.compile(r"[a-z]{1,8}-[a-z0-9]{2,24}")
CAPACITY_RE = re.compile(r"^(\d+)(gb|go|gib|g|tb|to|tib|mb|mo|mib)$")
GENERIC_CODE_RE = re.compile(
    r"^(?:usb|sd|sdhc|sdxc|microsd|ram|hdd|ssd|class|uhs|wifi|hd|led|"
    r"c|cl|u|v|a|gen|type|g|x|w|r)\d+[a-z]*$"
)
JOIN_PAIRS = {
    ("micro", "sd"): "microsd",
    ("micro", "sdhc"): "microsdhc",
    ("micro", "sdxc"): "microsdxc",
    ("usb", "stick"): "usbstick",
}

# Bucket-size bands let selective keys consume the output budget first. No
# band changes the O(k) processing bound for a bucket of size k: the window
# is a constant stored with the band.
SIZE_BANDS = (
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, 128),
    (129, MAX_BUCKET),
)

# Feature types:
#   0 canonical names, 1 model/SKU/token-triple identifiers,
#   2 selective composites, 3 rare-token pairs, 4 minhash and brand+capacity.
# Priority is size-band first so the tiniest buckets consume the budget.
# Each tuple is (feature type, size-band index, neighbour window).
PRIORITY = (
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 1),
    (3, 0, 1),
    (0, 1, 2),
    (1, 1, 2),
    (2, 1, 2),
    (3, 1, 2),
    (0, 2, 3),
    (1, 2, 3),
    (2, 2, 3),
    (0, 3, 7),
    (1, 3, 7),
    (2, 3, 7),
    (1, 4, 15),
    (2, 4, 15),
    (0, 4, 8),
    (-1, 0, 31),
    (3, 2, 3),
    (4, 0, 1),
    (4, 1, 2),
    (3, 3, 3),
    (4, 2, 2),
    (1, 5, 8),
    (2, 5, 6),
    (3, 4, 3),
    (4, 3, 2),
    (0, 5, 4),
    (1, 6, 3),
    (2, 6, 3),
    (3, 5, 2),
    (4, 4, 2),
    (4, 5, 2),
    (0, 6, 3),
    (1, 7, 2),
    (2, 7, 2),
    (3, 6, 2),
    (4, 6, 2),
    (0, 7, 2),
    (1, 8, 2),
    (2, 8, 2),
    (3, 7, 1),
    (4, 7, 1),
    (0, 8, 2),
    (3, 8, 1),
    (4, 8, 1),
)

MINHASH_SEEDS = tuple(
    (GOLDEN * (index + 1)) & MASK64 for index in range(NUM_MINHASH)
)


def mix64(value: int) -> int:
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & MASK64
    return value ^ (value >> 31)


def canonical_capacity(token: str) -> str | None:
    match = CAPACITY_RE.fullmatch(token)
    if match is None:
        return None
    number, unit = match.groups()
    if unit in {"gb", "go", "gib", "g"}:
        unit = "gb"
    elif unit in {"tb", "to", "tib"}:
        unit = "tb"
    else:
        unit = "mb"
    return number + unit


def normalize_tokens(value: str) -> list[str]:
    """Return inexpensive, language-agnostic product-name tokens."""
    ascii_value = (
        unicodedata.normalize("NFKD", value.lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    fused_codes = [
        token.replace("-", "")
        for token in HYPHEN_CODE_RE.findall(ascii_value)
        if not token.startswith(("uhs-", "usb-", "micro-", "sd-", "x-"))
    ]
    ascii_value = ascii_value.replace("-", " ").replace("_", " ").replace("/", " ")
    raw = TOKEN_RE.findall(ascii_value)
    result: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if token.isdigit() and index + 1 < len(raw):
            joined = canonical_capacity(token + raw[index + 1])
            if joined is not None:
                result.append(joined)
                index += 2
                continue
        if index + 1 < len(raw):
            fused = JOIN_PAIRS.get((token, raw[index + 1]))
            if fused is not None:
                result.append(fused)
                index += 2
                continue
        result.append(canonical_capacity(token) or token)
        index += 1
    for code in fused_codes:
        if code not in result:
            result.append(code)
    return result


def is_model_code(token: str) -> bool:
    if not 3 <= len(token) <= 32:
        return False
    if canonical_capacity(token) is not None or GENERIC_CODE_RE.fullmatch(token):
        return False
    if token in STOP_WORDS or token in GENERIC_WORDS:
        return False
    return any(char.isalpha() for char in token) and any(
        char.isdigit() for char in token
    )


def intern_token(
    vocab: dict[str, int], token_strings: list[str], token: str
) -> int:
    token_id = vocab.get(token)
    if token_id is None:
        token_id = len(vocab)
        vocab[token] = token_id
        token_strings.append(token)
    return token_id


def load_records(
    input_path: Path,
) -> tuple[array, list[str], array, array, array, array]:
    """Load ids and interned token sequences. One pass over the CSV."""
    vocab: dict[str, int] = {}
    ids = array("q")
    token_strings: list[str] = []
    seq_flat = array("I")
    seq_ptr = array("I")
    seq_ptr.append(0)
    uniq_flat = array("I")
    uniq_ptr = array("I")
    uniq_ptr.append(0)

    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"id", "name"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{input_path} must contain columns {sorted(required)}"
            )

        for row_number, row in enumerate(reader):
            if row_number >= (1 << 32):
                raise ValueError("The blocker supports at most 2^32 input rows")
            try:
                ids.append(int(row["id"]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Non-integer id at data row {row_number + 2}"
                ) from error

            name = row.get("name") or ""
            brand = row.get("brand") or ""
            description = row.get("description") or ""
            name_tokens = normalize_tokens(name)
            extra_tokens: list[str] = []
            if brand:
                for token in normalize_tokens(brand):
                    if token not in name_tokens and token not in extra_tokens:
                        extra_tokens.append(token)
            if description:
                for token in normalize_tokens(description):
                    if (
                        is_model_code(token)
                        or (token.isdigit() and len(token) >= 6)
                    ) and token not in name_tokens and token not in extra_tokens:
                        extra_tokens.append(token)

            seq_ids = [
                intern_token(vocab, token_strings, token)
                for token in name_tokens
            ]
            extra_ids = [
                intern_token(vocab, token_strings, token)
                for token in extra_tokens
            ]

            seq_flat.extend(seq_ids)
            seq_ptr.append(len(seq_flat))
            seen: set[int] = set()
            for token_id in seq_ids:
                if token_id not in seen:
                    uniq_flat.append(token_id)
                    seen.add(token_id)
            for token_id in extra_ids:
                if token_id not in seen:
                    uniq_flat.append(token_id)
                    seen.add(token_id)
            uniq_ptr.append(len(uniq_flat))

            if (row_number + 1) % 250_000 == 0:
                print(
                    f"Read {row_number + 1:,} rows, {len(vocab):,} tokens",
                    file=sys.stderr,
                )

    return ids, token_strings, seq_flat, seq_ptr, uniq_flat, uniq_ptr


def document_frequency(n_rows: int, uniq_flat: array, uniq_ptr: array) -> array:
    freq = array("I", [0]) * (max(uniq_flat) + 1 if uniq_flat else 0)
    for row in range(n_rows):
        start, stop = uniq_ptr[row], uniq_ptr[row + 1]
        for token_id in uniq_flat[start:stop]:
            freq[token_id] += 1
    return freq


def row_slice(flat: array, ptr: array, row: int) -> array:
    return flat[ptr[row] : ptr[row + 1]]


def detect_brand(token_strings: list[str], token_ids: array) -> str:
    for token_id in token_ids:
        token = token_strings[token_id]
        if token in BRANDS:
            return token
    return ""


def selected_tokens(
    token_strings: list[str],
    token_ids: array,
    freq: array,
) -> list[str]:
    """Pick the rarest distinctive tokens; omit brands and capacities."""
    models: list[tuple[int, str]] = []
    rare: list[tuple[int, int, str]] = []
    for token_id in token_ids:
        token = token_strings[token_id]
        if token in BRANDS or token in STOP_WORDS or token in GENERIC_WORDS:
            continue
        if canonical_capacity(token) is not None:
            continue
        document_freq = freq[token_id]
        if document_freq < 2 or document_freq > MAX_BUCKET:
            continue
        if is_model_code(token):
            models.append((document_freq, token))
            continue
        if len(token) >= 3 and not token.isdigit():
            rare.append((document_freq, -len(token), token))

    models.sort()
    rare.sort()
    chosen: list[str] = []
    seen: set[str] = set()
    for _, token in models[:4]:
        if token not in seen:
            seen.add(token)
            chosen.append(token)
    for _, _, token in rare:
        if token in seen:
            continue
        seen.add(token)
        chosen.append(token)
        if len(chosen) >= 6:
            break
    return chosen[:6]


def minhash_bands(token_ids: array) -> list[int]:
    if not token_ids:
        return []
    hashes = [MASK64] * NUM_MINHASH
    for token_id in token_ids:
        seeded = mix64(token_id + GOLDEN)
        for index, seed in enumerate(MINHASH_SEEDS):
            value = mix64(seeded ^ seed)
            if value < hashes[index]:
                hashes[index] = value
    bands: list[int] = []
    for band in range(NUM_BANDS):
        start = band * BAND_SIZE
        mixed = mix64(hashes[start] ^ GOLDEN)
        for offset in range(1, BAND_SIZE):
            mixed = mix64(mixed ^ hashes[start + offset] ^ (offset * GOLDEN))
        bands.append((band << 56) | (mixed & ((1 << 56) - 1)))
    return bands


def record_signature(
    token_strings: list[str], token_ids: array, freq: array
) -> int:
    ranked = sorted(
        (
            (freq[token_id], token_strings[token_id], token_id)
            for token_id in token_ids
            if token_strings[token_id] not in STOP_WORDS
        )
    )[:8]
    mixed = GOLDEN ^ (len(ranked) * 0x100000001B3)
    for document_freq, _, token_id in ranked:
        mixed = mix64(mixed ^ mix64((document_freq << 32) ^ token_id))
    return mixed


def emit_feature(
    handles: list[list],
    partitions: int,
    feature_type: int,
    value: int,
    signature: int,
    row_number: int,
) -> None:
    partition = value & (partitions - 1)
    handles[feature_type][partition].write(
        RAW_RECORD.pack(value, signature, row_number)
    )


def hash_text(prefix: int, text: str) -> int:
    mixed = mix64(prefix * GOLDEN)
    for char in text:
        mixed = mix64(mixed ^ (ord(char) + GOLDEN))
    return mixed


def spool_features(
    token_strings: list[str],
    seq_flat: array,
    seq_ptr: array,
    uniq_flat: array,
    uniq_ptr: array,
    freq: array,
    temporary: Path,
    partitions: int,
) -> tuple[array, int]:
    n_rows = len(uniq_ptr) - 1
    signatures = array("Q")
    paths = [
        [
            temporary / f"raw-{feature_type}-{partition:02d}.bin"
            for partition in range(partitions)
        ]
        for feature_type in range(5)
    ]
    handles = [
        [path.open("wb", buffering=64 * 1024) for path in feature_paths]
        for feature_paths in paths
    ]
    feature_count = 0
    try:
        for row in range(n_rows):
            uniq_ids = row_slice(uniq_flat, uniq_ptr, row)
            seq_ids = row_slice(seq_flat, seq_ptr, row)
            signature = record_signature(token_strings, uniq_ids, freq)
            signatures.append(signature)

            seq_tokens = [token_strings[token_id] for token_id in seq_ids]
            uniq_tokens = [token_strings[token_id] for token_id in uniq_ids]
            exact = " ".join(seq_tokens)
            sorted_name = " ".join(sorted(set(seq_tokens)))
            models = [token for token in uniq_tokens if is_model_code(token)]
            models.sort(key=lambda token: (-len(token), token))
            skus = [
                token
                for token in seq_tokens
                if token.isdigit() and len(token) >= 6
            ]
            brand = detect_brand(token_strings, uniq_ids)
            capacities = [
                token
                for token in uniq_tokens
                if canonical_capacity(token) is not None
            ]
            lines = [token for token in uniq_tokens if token in PRODUCT_LINES]
            chosen = selected_tokens(token_strings, uniq_ids, freq)

            names = []
            if len(sorted_name) >= 5:
                names.append(hash_text(1, sorted_name))
            if len(exact) >= 5:
                names.append(hash_text(2, exact))
            info = " ".join(
                sorted(
                    token
                    for token in set(uniq_tokens)
                    if token not in STOP_WORDS
                    and token not in GENERIC_WORDS
                    and len(token) >= 3
                )
            )
            if len(info) >= 5:
                names.append(hash_text(3, info))

            identifiers = [hash_text(10, token) for token in models[:5]]
            identifiers.extend(hash_text(11, token) for token in skus[:3])
            chosen_sorted = sorted(chosen)
            if len(chosen_sorted) >= 3:
                identifiers.append(
                    hash_text(12, "\x1f".join(chosen_sorted[:3]))
                )

            composites = []
            if brand and lines and capacities:
                composites.append(
                    hash_text(20, brand + "|" + lines[0] + "|" + capacities[0])
                )
            if brand and models:
                composites.append(hash_text(21, brand + "|" + models[0]))
            if lines and capacities:
                composites.append(
                    hash_text(23, lines[0] + "|" + capacities[0])
                )

            pair_values = [hash_text(30, token) for token in chosen[:4]]
            for left in range(len(chosen_sorted)):
                for right in range(left + 1, len(chosen_sorted)):
                    pair_values.append(
                        hash_text(
                            31,
                            chosen_sorted[left] + "\x1f" + chosen_sorted[right],
                        )
                    )

            informative_ids = array(
                "I",
                [
                    token_id
                    for token_id in uniq_ids
                    if token_strings[token_id] not in STOP_WORDS
                    and token_strings[token_id] not in GENERIC_WORDS
                ],
            )
            bands = minhash_bands(informative_ids or uniq_ids)
            if brand and capacities:
                bands.append(hash_text(22, brand + "|" + capacities[0]))

            packed = (names, identifiers, composites, pair_values, bands)
            for feature_type, values in enumerate(packed):
                for value in values:
                    emit_feature(
                        handles, partitions, feature_type, value, signature, row
                    )
                    feature_count += 1

            if (row + 1) % 250_000 == 0:
                print(
                    f"Wrote {feature_count:,} blocking features "
                    f"from {row + 1:,} rows",
                    file=sys.stderr,
                )
    finally:
        for feature_handles in handles:
            for handle in feature_handles:
                handle.close()

    return signatures, feature_count


def size_band(size: int) -> int | None:
    for band, (lower, upper) in enumerate(SIZE_BANDS):
        if lower <= size <= upper:
            return band
    return None


def sort_and_group_features(
    temporary: Path, partitions: int
) -> dict[tuple[int, int], Path]:
    grouped_paths = {
        (feature_type, band): temporary / f"group-{feature_type}-{band}.bin"
        for feature_type in range(5)
        for band in range(len(SIZE_BANDS))
    }
    grouped_handles = {
        key: path.open("wb", buffering=256 * 1024)
        for key, path in grouped_paths.items()
    }

    try:
        completed = 0
        total = 5 * partitions
        for feature_type in range(5):
            for partition in range(partitions):
                raw_path = temporary / f"raw-{feature_type}-{partition:02d}.bin"
                blob = raw_path.read_bytes()
                if len(blob) % RAW_RECORD.size:
                    raise RuntimeError(f"Corrupt temporary file: {raw_path}")
                postings = list(RAW_RECORD.iter_unpack(blob))
                postings.sort()

                for _, group in groupby(postings, key=lambda item: item[0]):
                    members = [(item[1], item[2]) for item in group]
                    # One row may emit the same hashed feature twice.
                    seen_rows: dict[int, int] = {}
                    for signature, row in members:
                        previous = seen_rows.get(row)
                        if previous is None or signature < previous:
                            seen_rows[row] = signature
                    if len(seen_rows) < 2:
                        continue
                    band = size_band(len(seen_rows))
                    if band is None:
                        continue
                    ordered = sorted(
                        seen_rows.items(), key=lambda item: (item[1], item[0])
                    )
                    rows = [row for row, _ in ordered]
                    destination = grouped_handles[(feature_type, band)]
                    destination.write(UINT32.pack(len(rows)))
                    destination.write(array("I", rows).tobytes())

                del postings
                raw_path.unlink()
                completed += 1
                if completed % 8 == 0 or completed == total:
                    print(
                        f"Grouped {completed}/{total} feature partitions",
                        file=sys.stderr,
                    )
    finally:
        for handle in grouped_handles.values():
            handle.close()

    return grouped_paths


def add_group_neighbours(
    rows: tuple[int, ...],
    window: int,
    pairs: set[int],
    target: int,
) -> bool:
    """Add O(len(rows) * window) links; return True at the target."""
    limit = min(window, len(rows) - 1)
    for distance in range(1, limit + 1):
        for left, right in zip(rows, rows[distance:]):
            if left > right:
                left, right = right, left
            encoded = (left << 32) | right
            if encoded not in pairs:
                pairs.add(encoded)
                if len(pairs) == target:
                    return True
    return False


def close_small_components(
    pairs: set[int],
    signatures: array,
    target: int,
    max_size: int,
    window: int,
) -> bool:
    """Link neighbours inside tiny connected components. Component size is capped."""
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for encoded in pairs:
        left = encoded >> 32
        right = encoded & 0xFFFFFFFF
        parent.setdefault(left, left)
        parent.setdefault(right, right)
        union(left, right)

    groups: dict[int, list[int]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)

    for members in groups.values():
        if not 3 <= len(members) <= max_size:
            continue
        members.sort(key=lambda row: (signatures[row], row))
        if add_group_neighbours(tuple(members), window, pairs, target):
            return True
    return False


def read_prioritized_candidates(
    grouped_paths: dict[tuple[int, int], Path],
    signatures: array,
    target: int,
) -> set[int]:
    pairs: set[int] = set()
    for feature_type, band, window in PRIORITY:
        if feature_type < 0:
            if close_small_components(
                pairs, signatures, target, 64, window
            ):
                return pairs
            print(
                f"Candidates after component closing: {len(pairs):,}",
                file=sys.stderr,
            )
            continue
        path = grouped_paths[(feature_type, band)]
        with path.open("rb", buffering=256 * 1024) as source:
            while True:
                header = source.read(UINT32.size)
                if not header:
                    break
                if len(header) != UINT32.size:
                    raise RuntimeError(f"Corrupt grouped file: {path}")
                (size,) = UINT32.unpack(header)
                payload = source.read(size * UINT32.size)
                if len(payload) != size * UINT32.size:
                    raise RuntimeError(f"Corrupt grouped file: {path}")
                rows = struct.unpack(f"<{size}I", payload)
                if add_group_neighbours(rows, window, pairs, target):
                    return pairs
        print(
            f"Candidates after feature {feature_type}, "
            f"bucket {SIZE_BANDS[band]}: {len(pairs):,}",
            file=sys.stderr,
        )
    return pairs


def fill_to_target(
    pairs: set[int], signatures: array, row_count: int, target: int
) -> None:
    """Fill leftover budget with sorted-neighbourhood links, O(target)."""
    order = sorted(range(row_count), key=lambda row: (signatures[row], row))
    for window in range(1, row_count):
        for index in range(row_count - window):
            left = order[index]
            right = order[index + window]
            if left > right:
                left, right = right, left
            encoded = (left << 32) | right
            if encoded not in pairs:
                pairs.add(encoded)
                if len(pairs) == target:
                    return
    raise RuntimeError("Could not create the requested number of unique pairs")


def write_output(output_path: Path, ids: array, pairs: set[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")
    with temporary_output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("lid", "rid"))
        for encoded in pairs:
            first = ids[encoded >> 32]
            second = ids[encoded & 0xFFFFFFFF]
            if first < second:
                writer.writerow((first, second))
            else:
                writer.writerow((second, first))
    os.replace(temporary_output, output_path)


def run(
    input_path: Path,
    output_path: Path,
    target: int = TARGET_PAIRS,
    partitions: int = PARTITIONS,
) -> None:
    if target < 0:
        raise ValueError("--target must be non-negative")
    if partitions <= 0 or partitions & (partitions - 1):
        raise ValueError("--partitions must be a positive power of two")

    workspace = output_path.resolve().parent
    workspace.mkdir(parents=True, exist_ok=True)

    ids, token_strings, seq_flat, seq_ptr, uniq_flat, uniq_ptr = load_records(
        input_path
    )
    row_count = len(ids)
    possible_pairs = row_count * (row_count - 1) // 2
    print(
        f"Loaded {row_count:,} records and {len(token_strings):,} distinct tokens",
        file=sys.stderr,
    )
    if target > possible_pairs:
        print(
            f"{row_count:,} records permit only {possible_pairs:,} unique "
            f"pairs; clamping --target from {target:,}",
            file=sys.stderr,
        )
        target = possible_pairs

    freq = document_frequency(row_count, uniq_flat, uniq_ptr)
    with tempfile.TemporaryDirectory(
        prefix=".blocking-", dir=workspace
    ) as temporary_name:
        temporary = Path(temporary_name)
        signatures, feature_count = spool_features(
            token_strings,
            seq_flat,
            seq_ptr,
            uniq_flat,
            uniq_ptr,
            freq,
            temporary,
            partitions,
        )
        print(f"Wrote {feature_count:,} blocking features", file=sys.stderr)
        grouped_paths = sort_and_group_features(temporary, partitions)
        pairs = read_prioritized_candidates(grouped_paths, signatures, target)
        if len(pairs) < target:
            print(
                f"Filling {target - len(pairs):,} remaining candidate slots",
                file=sys.stderr,
            )
            fill_to_target(pairs, signatures, row_count, target)
        write_output(output_path, ids, pairs)
        print(f"Wrote {len(pairs):,} pairs to {output_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("../SIGMOD_full_data/X2.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("output.csv"))
    parser.add_argument("--target", type=int, default=TARGET_PAIRS)
    parser.add_argument("--partitions", type=int, default=PARTITIONS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.input,
        arguments.output,
        target=arguments.target,
        partitions=arguments.partitions,
    )
