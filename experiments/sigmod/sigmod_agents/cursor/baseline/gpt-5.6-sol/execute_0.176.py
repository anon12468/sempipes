#!/usr/bin/env python3
"""Scalable blocking for the X2 electronic-products data set.

The generator deliberately does not know about Z2/Y2.  It extracts a fixed
number of blocking signatures, sorts each signature, and compares records only
inside a fixed-width sorted neighbourhood.  Thus, even a signature shared by
millions of rows produces O(n * window) work rather than O(n**2) work.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
import time
import unicodedata
import zlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


TARGET_PAIRS = 2_000_000

BRANDS = {
    "adata",
    "corsair",
    "emtec",
    "hama",
    "intenso",
    "integral",
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

# These words are common descriptions, sellers, units, or translated variants.
# They are useful for parsing but poor identity evidence.
STOP_WORDS = {
    "a",
    "about",
    "achat",
    "adaptador",
    "adapter",
    "adaptateur",
    "amazon",
    "and",
    "avec",
    "black",
    "blanc",
    "blanco",
    "blue",
    "buy",
    "card",
    "carte",
    "class",
    "cle",
    "color",
    "computer",
    "con",
    "dans",
    "de",
    "des",
    "di",
    "disk",
    "drive",
    "en",
    "et",
    "flash",
    "for",
    "gb",
    "geheugenkaart",
    "go",
    "high",
    "in",
    "karte",
    "memory",
    "memoria",
    "memoire",
    "minne",
    "new",
    "noir",
    "of",
    "on",
    "para",
    "pendrive",
    "product",
    "red",
    "retail",
    "scheda",
    "silver",
    "speicherkarte",
    "stick",
    "storage",
    "tarjeta",
    "the",
    "to",
    "usb",
    "und",
    "unit",
    "with",
    "white",
    "zu",
}

FAMILY_RULES: tuple[tuple[str, str], ...] = (
    (r"\bdata\s*travell?er\b|\bdatatravell?er\b", "datatraveler"),
    (r"\bdt\s*101\b|\b101\s*g2\b", "dt101"),
    (r"\bjump\s*drive\b|\bjumpdrive\b", "jumpdrive"),
    (r"\bc20\s*c\b|\bc20c", "c20c"),
    (r"\bc20\s*m\b|\bc20m", "c20m"),
    (r"\battach[e]?\s*4?\b|\batt\s*4\b|\batt4\b", "attache"),
    (r"\bcru[iz]er\s*glide\b", "cruzerglide"),
    (r"\bcru[iz]er\s*blade\b", "cruzerblade"),
    (r"\bcru[iz]er\s*fit\b", "cruzerfit"),
    (r"\bcru[iz]er\b", "cruzer"),
    (r"\bexceria\s*pro\b", "exceriapro"),
    (r"\bexceria\b", "exceria"),
    (r"\bextreme\s*pro\b", "extremepro"),
    (r"\bextreme\s*go\b", "extremego"),
    (r"\bextreme\b", "extreme"),
    (r"\bultra\s*plus\b", "ultraplus"),
    (r"\bultra\s*dual\b", "ultradual"),
    (r"\bultra\b", "ultra"),
    (r"\bbasic\s*line\b", "basicline"),
    (r"\brainbow\s*line\b", "rainbowline"),
    (r"\bpremium\s*line\b", "premiumline"),
    (r"\bspeed\s*line\b", "speedline"),
    (r"\btrans\s*memory\b|\btransmemory\b", "transmemory"),
    (r"\bu\s*202\b|\bu202\b|\bthn[-\s]*u202", "u202"),
    (r"\bcanvas\s*go\b", "canvasgo"),
    (r"\bindustrial(?:\s*temperature)?\b", "industrial"),
    (r"\bsavage\b|\bhyperx\b|\bhxs\b", "savage"),
    (r"\bserie\s*u\b|\bseries\s*u\b|\bsf\s*\d+\s*u\b|\bsf\d+u\b", "serieu"),
    (r"\bserie\s*n\b|\bseries\s*n\b|\bsf\s*\d+\s*n\b|\bsf\d+n", "serien"),
    (r"\bmicro\s*vault\b|\busm\d+gr\b|\br[\s-]*series\b", "microvaultr"),
    (r"\b3534490\b", "premiumline"),
    (r"\bultimate\b|\bsda\s*(?:3|10)\b", "ultimate"),
    (r"\bevo\b", "evo"),
    (r"\bpro\s*plus\b", "proplus"),
)
FAMILY_REGEX = tuple((re.compile(pattern), value) for pattern, value in FAMILY_RULES)

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
TOKEN_RE = re.compile(r"[a-z0-9]+")
CAPACITY_RE = re.compile(r"\b(1|2|4|8|16|32|64|128|256|512|1024|2048)\s*(?:gb|go|g)\b")
JOINED_CODE_RE = re.compile(r"\b[a-z0-9]+(?:[-_/][a-z0-9]+)+\b")
SPEED_RE = re.compile(r"^(?:\d{2,4}(?:mb|mo|r|w|x)|r\d+|u[123]|v\d+)$")


def stable_hash(value: str) -> int:
    """Fast deterministic 64-bit hash; zero is reserved as a missing key."""
    encoded = value.encode("utf-8")
    result = (zlib.crc32(encoded) << 32) | zlib.adler32(encoded)
    return result or 1


def ascii_text(value: object) -> str:
    text = html.unescape("" if value is None else str(value)).lower()
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def normalize_text(name: str) -> str:
    text = ascii_text(name)
    text = text.replace("data traveller", "datatraveler")
    text = text.replace("data traveler", "datatraveler")
    text = text.replace("datattraveller", "datatraveler")
    text = text.replace("micro sd", "microsd")
    text = text.replace("sd xc", "sdxc").replace("sd hc", "sdhc")
    text = re.sub(r"\bclasse?\s*(\d+)\b", r"class\1", text)
    text = re.sub(r"\buhs[\s-]*(?:i|1)\b", "uhs1", text)
    text = re.sub(r"\buhs[\s-]*(?:ii|2)\b", "uhs2", text)
    text = re.sub(r"\busb[\s-]*3(?:[.,]0|[.,]1)?\b", "usb3", text)
    text = re.sub(r"\busb[\s-]*2(?:[.,]0)?\b", "usb2", text)
    text = re.sub(r"\b(\d+)\s*(?:gbytes?|gbyte|giga|go)\b", r"\1gb", text)
    return " ".join(TOKEN_RE.findall(text))


def product_type(tokens: set[str], text: str) -> str:
    if "ssd" in tokens:
        return "ssd"
    if "xqd" in tokens:
        return "xqd"
    if "microsd" in tokens or "microsdhc" in tokens or "microsdxc" in tokens:
        return "microsd"
    if "sdhc" in tokens or "sdxc" in tokens or "sdhc" in text or "sdxc" in text:
        return "sdcard"
    if any(token.startswith("usb") for token in tokens) or "pendrive" in tokens:
        return "usbdrive"
    if "phone" in tokens or "galaxy" in tokens or "iphone" in tokens:
        return "phone"
    return ""


def extract_codes(raw_text: str, tokens: Sequence[str]) -> list[str]:
    codes: set[str] = set()
    for match in JOINED_CODE_RE.findall(raw_text):
        code = NON_ALNUM_RE.sub("", match)
        if len(code) >= 4 and any(c.isalpha() for c in code) and any(c.isdigit() for c in code):
            codes.add(code)

    for token in tokens:
        if not (3 <= len(token) <= 32):
            continue
        if not (any(c.isalpha() for c in token) and any(c.isdigit() for c in token)):
            continue
        if re.fullmatch(r"\d+(?:gb|go|mb|mo|tb)", token):
            continue
        if token in {"usb2", "usb3", "uhs1", "uhs2", "class4", "class10", "class3"}:
            continue
        if SPEED_RE.fullmatch(token):
            continue
        codes.add(token)

    # Long mixed identifiers are much more discriminative than short features.
    return sorted(codes, key=lambda value: (-len(value), value))[:3]


def bottom_hashes(tokens: Iterable[str], count: int = 4) -> list[int]:
    values = sorted({stable_hash("t:" + token) for token in tokens})
    return values[:count]


def make_keys(name: str, brand_field: str) -> tuple[int, ...]:
    raw = ascii_text(name)
    normalized = normalize_text(raw)
    tokens = normalized.split()
    token_set = set(tokens)

    brands = sorted((token_set | set(TOKEN_RE.findall(ascii_text(brand_field)))) & BRANDS)
    brand = brands[0] if brands else ""
    capacities = sorted(set(CAPACITY_RE.findall(normalized)), key=int)
    capacity = capacities[-1] if capacities else ""
    capacity2 = capacities[-2] if len(capacities) > 1 else ""
    ptype = product_type(token_set, normalized)

    families = [value for regex, value in FAMILY_REGEX if regex.search(normalized)]
    family = families[0] if families else ""
    family2 = families[1] if len(families) > 1 else ""
    codes = extract_codes(raw, tokens)

    core = sorted(
        {
            token
            for token in tokens
            if token not in STOP_WORDS
            and token not in BRANDS
            and len(token) > 1
            and not re.fullmatch(r"\d+", token)
        }
    )
    bag = " ".join(core)

    # Prefer product-bearing tokens for approximate signatures.
    informative = [
        token
        for token in core
        if len(token) >= 3 and token not in {"class10", "class4", "uhs1", "uhs2", "usb2", "usb3"}
    ]
    minima = bottom_hashes(informative, 4)
    while len(minima) < 4:
        minima.append(0)

    compact = "".join(tokens)[:320]
    trigrams = {compact[i : i + 3] for i in range(max(0, len(compact) - 2))}
    tri_minima = bottom_hashes(trigrams, 2)
    while len(tri_minima) < 2:
        tri_minima.append(0)

    code1 = codes[0] if codes else ""
    code2 = codes[1] if len(codes) > 1 else ""
    code3 = codes[2] if len(codes) > 2 else ""

    def key(*parts: object) -> int:
        values = [str(part) for part in parts]
        return stable_hash("\x1f".join(values)) if all(values) else 0

    # Column order is also candidate priority: precise signatures come first.
    return (
        key("exact", normalized),
        key("bag", bag),
        key("code", code1),
        key("code", code2),
        key("code", code3),
        key("bcf", brand, capacity, family),
        key("bcf", brand, capacity2, family),
        key("bcap", brand, capacity),
        key("fcap", family, capacity),
        key("cfp", capacity, family, ptype),
        key("bc", brand, capacity, ptype),
        key("bf", brand, family),
        key("ff", family, family2, capacity),
        key("m0", brand or ptype, capacity or ptype, minima[0]),
        key("m1", brand or ptype, capacity or ptype, minima[1]),
        key("m2", brand or ptype, minima[2]),
        key("m3", capacity or ptype, minima[3]),
        key("g0", brand or ptype, tri_minima[0]),
        key("g1", capacity or ptype, tri_minima[1]),
    )


SIGNATURE_COUNT = len(make_keys("probe", ""))

# Precise equalities can safely use a wider constant neighbourhood. Approximate
# keys use a narrow one to protect the pair budget.
WINDOWS = (32, 32, 32, 24, 16, 32, 16, 64, 48, 32, 24, 16, 12, 6, 6, 4, 4, 4, 4)


def count_records(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def load_signatures(path: Path) -> tuple[np.ndarray, np.ndarray]:
    count = count_records(path)
    if count < 2:
        raise ValueError("X2 must contain at least two records")

    ids = np.empty(count, dtype=np.int64)
    signatures = np.zeros((count, SIGNATURE_COUNT), dtype=np.uint64)

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: id, name")

        actual = 0
        for actual, row in enumerate(reader, start=1):
            ids[actual - 1] = int(row["id"])
            signatures[actual - 1, :] = make_keys(row.get("name", ""), row.get("brand", ""))

    if actual != count:
        ids = ids[:actual]
        signatures = signatures[:actual, :]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("X2 contains duplicate record ids")
    return ids, signatures


def pack_pair(left: int, right: int) -> int:
    if left > right:
        left, right = right, left
    if left < 0 or right > 0xFFFFFFFF:
        raise ValueError("record ids must be in the unsigned 32-bit range")
    return (left << 32) | right


def add_signature_pairs(
    ids: np.ndarray,
    keys: np.ndarray,
    window: int,
    pairs: set[int],
    target: int,
) -> None:
    """Add fixed-window pairs for equal, nonzero keys."""
    order = np.argsort(keys, kind="quicksort")
    size = len(order)
    start = 0
    while start < size and len(pairs) < target:
        key_value = int(keys[order[start]])
        end = start + 1
        while end < size and keys[order[end]] == key_value:
            end += 1

        if key_value:
            for pos in range(start + 1, end):
                row = int(order[pos])
                lower = max(start, pos - window)
                for previous in range(pos - 1, lower - 1, -1):
                    other = int(order[previous])
                    left, right = int(ids[row]), int(ids[other])
                    if left != right:
                        pairs.add(pack_pair(left, right))
                        if len(pairs) >= target:
                            return
        start = end


def fill_pairs(ids: np.ndarray, pairs: set[int], target: int) -> None:
    """Deterministic O(target) fallback used only to reach the exact row count."""
    ordered = np.sort(ids)
    n = len(ordered)
    stride = 1
    while len(pairs) < target and stride < n:
        for index in range(n - stride):
            left = int(ordered[index])
            right = int(ordered[index + stride])
            pairs.add(pack_pair(left, right))
            if len(pairs) >= target:
                return
        stride += 1
    if len(pairs) != target:
        raise RuntimeError(f"only {len(pairs):,} distinct valid pairs exist; requested {target:,}")


def write_output(path: Path, pairs: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("lid", "rid"))
        for packed in pairs:
            writer.writerow((packed >> 32, packed & 0xFFFFFFFF))
    os.replace(temporary, path)


def generate(input_path: Path, output_path: Path, target: int, dev_mode: bool) -> None:
    started = time.monotonic()
    ids, signatures = load_signatures(input_path)
    n = len(ids)
    possible = n * (n - 1) // 2

    if target > possible:
        if not dev_mode:
            raise ValueError(
                f"{n:,} records have only {possible:,} possible pairs, fewer than the "
                f"requested {target:,}. Use the full X2.csv or pass --dev."
            )
        target = min(possible, max(1, n * 64))
        print(f"development input detected: writing {target:,} pairs", file=sys.stderr)

    pairs: set[int] = set()
    for column, window in enumerate(WINDOWS):
        add_signature_pairs(ids, signatures[:, column], window, pairs, target)
        print(
            f"signature {column + 1:02d}/{SIGNATURE_COUNT}: {len(pairs):,} pairs",
            file=sys.stderr,
        )
        if len(pairs) >= target:
            break

    if len(pairs) < target:
        fill_pairs(ids, pairs, target)

    write_output(output_path, pairs)
    elapsed = time.monotonic() - started
    print(f"wrote {len(pairs):,} pairs to {output_path} in {elapsed:.1f}s", file=sys.stderr)


def default_input() -> Path:
    full = Path("../SIGMOD_full_data/X2.csv")
    return full if full.exists() else Path("../SIGMOD_full_data/X2.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input(), help="X2 CSV path")
    parser.add_argument("--output", type=Path, default=Path("output.csv"), help="output CSV path")
    parser.add_argument("--target", type=int, default=TARGET_PAIRS, help="number of candidate pairs")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="allow a scaled-down output when the input has fewer than 2,001 records",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments.input, arguments.output, arguments.target, arguments.dev)
