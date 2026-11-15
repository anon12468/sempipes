from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from skrub import TableVectorizer

import sempipes  # pylint: disable=unused-import

from experiments.size_sweep.tracking import attach_feature_dump

DATA_DIR = Path("experiments/processing_modes/data/bc5cdr")
TRAIN_PATH = DATA_DIR / "CDR_TrainingSet.PubTator.txt"
DEV_PATH = DATA_DIR / "CDR_DevelopmentSet.PubTator.txt"
TEST_PATH = DATA_DIR / "CDR_TestSet.PubTator.txt"
SPLIT_SEED = 42
TRAIN_SIZES = (300, 1800, 10557)
TEST_SIZE = 1000
OUTPUT_COLUMNS = {
    "chemical_surface": (
        "Copy every mention of the given chemical as written in the abstract, separated by ' | '. "
        "If none, respond with 'none'."
    ),
    "disease_surface": (
        "Copy every mention of the given disease as written in the abstract, separated by ' | '. "
        "If none, respond with 'none'."
    ),
    "linking_sentence": (
        "Copy the sentence that mentions both the given chemical and the given disease. "
        "If no such sentence, respond with 'none'."
    ),
}
PROMPT = (
    "Extract the surface forms of the given chemical and the given disease from the PubMed abstract, "
    "and copy the sentence that mentions both. "
    "Keep formulas and abbreviations as written. "
    "The given names may appear as abbreviations, formulas, or other surface forms. "
    "The sentence that mentions both is the feature that separates a described relationship from co-occurrence. "
    "Do not say whether the chemical induces the disease."
)


def _mention_names(block: str) -> tuple[str, str, dict[tuple[str, str], list[str]], set[tuple[str, str]]]:
    title = ""
    abstract = ""
    mentions: dict[tuple[str, str], list[str]] = {}
    relations: set[tuple[str, str]] = set()
    for line in block.splitlines():
        if "|t|" in line:
            _, _, title = line.split("|", 2)
        elif "|a|" in line:
            _, _, abstract = line.split("|", 2)
        else:
            parts = line.split("\t")
            if len(parts) >= 6 and parts[4] in {"Chemical", "Disease"} and parts[5] != "-1":
                mentions.setdefault((parts[4], parts[5]), [])
                if parts[3] not in mentions[(parts[4], parts[5])]:
                    mentions[(parts[4], parts[5])].append(parts[3])
            elif len(parts) >= 4 and parts[1] == "CID":
                relations.add((parts[2], parts[3]))
    return title, abstract, mentions, relations


def _pairs_from_pubtator(path: Path) -> pd.DataFrame:
    rows = []
    for block in path.read_text(encoding="utf-8", errors="replace").split("\n\n"):
        if not block.strip():
            continue
        title, abstract, mentions, relations = _mention_names(block)
        text = f"{title} {abstract}".strip()
        chemicals = {mesh: names for (kind, mesh), names in mentions.items() if kind == "Chemical"}
        diseases = {mesh: names for (kind, mesh), names in mentions.items() if kind == "Disease"}
        for chemical_mesh, chemical_names in chemicals.items():
            for disease_mesh, disease_names in diseases.items():
                rows.append(
                    {
                        "text": text,
                        "chemical_name": " | ".join(chemical_names),
                        "disease_name": " | ".join(disease_names),
                        "cid": int((chemical_mesh, disease_mesh) in relations),
                    }
                )
    return pd.DataFrame(rows)


def sempipes_pipeline(mode: str, artifact_dir: Path | None = None):
    papers = skrub.var("data")
    labels = skrub.var("labels")
    papers = papers.skb.mark_as_X()
    labels = labels.skb.mark_as_y()
    prompt = PROMPT
    input_columns = ["text", "chemical_name", "disease_name"]
    extracted = papers.sem_extract_features(
        nl_prompt=prompt,
        input_columns=input_columns,
        name="extract_pubmed_features",
        output_columns=OUTPUT_COLUMNS,
        generate_via_code=mode != "llm",
    )
    if artifact_dir is not None:
        extracted = attach_feature_dump(extracted, artifact_dir, "extract_pubmed_features", input_columns)
    encoded = extracted.skb.apply(TableVectorizer())
    return encoded.skb.apply(RandomForestClassifier(random_state=0, n_jobs=1), y=labels)


def _stratified_order(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SPLIT_SEED)
    groups = []
    ordered = frame.sort_values(["text", "chemical_name", "disease_name"])
    for _, group in ordered.groupby("cid", sort=True):
        index = group.index.to_numpy().copy()
        rng.shuffle(index)
        groups.append(index)
    chosen = []
    cursor = 0
    while any(cursor < len(group) for group in groups):
        for group in groups:
            if cursor < len(group):
                chosen.append(group[cursor])
        cursor += 1
    return frame.loc[chosen]


def _test_pairs() -> pd.DataFrame:
    return _stratified_order(_pairs_from_pubtator(TEST_PATH)).iloc[:TEST_SIZE]


def _natural_pool() -> pd.DataFrame:
    official = _pairs_from_pubtator(TRAIN_PATH)
    development = _pairs_from_pubtator(DEV_PATH)
    return pd.concat([official, development], ignore_index=True)


def _natural_train(pool: pd.DataFrame, train_size: int) -> pd.DataFrame:
    """Nested subset of the full pool at the pool's own positive rate.

    Positives and negatives are shuffled once. A smaller size is a prefix of
    each class, so 300 is contained in 1,800 and both are contained in the
    10,557 pool. This keeps the label mix of the full pool instead of the
    balanced round-robin used for the other 300 and 1,800 sets.
    """
    rate = float(pool["cid"].mean())
    rng = np.random.default_rng(SPLIT_SEED)
    chosen = []
    n_pos = int(round(train_size * rate))
    for label, group in pool.groupby("cid", sort=True):
        index = group.index.to_numpy().copy()
        rng.shuffle(index)
        count = n_pos if int(label) == 1 else train_size - n_pos
        chosen.append(index[:count])
    selected = np.concatenate(chosen)
    np.random.default_rng(SPLIT_SEED).shuffle(selected)
    return pool.loc[selected]


def load_split(train_size: int) -> tuple[dict, dict, pd.Series, pd.Series]:
    if train_size not in TRAIN_SIZES:
        raise ValueError(f"BC5CDR train size must be one of {TRAIN_SIZES}")
    official_train = _stratified_order(_pairs_from_pubtator(TRAIN_PATH))
    development = _pairs_from_pubtator(DEV_PATH)
    test = _test_pairs()
    if train_size == TRAIN_SIZES[-1]:
        train = pd.concat([official_train, development], ignore_index=True)
    else:
        train = official_train.iloc[:train_size]
    return _pack(train, test)


def load_split_natural(train_size: int) -> tuple[dict, dict, pd.Series, pd.Series]:
    if train_size not in (300, 1800):
        raise ValueError("Natural-rate BC5CDR sizes are 300 and 1800")
    pool = _natural_pool()
    return _pack(_natural_train(pool, train_size), _test_pairs())


def _pack(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict, pd.Series, pd.Series]:
    def _env(rows: pd.DataFrame) -> dict:
        rows = rows.reset_index(drop=True)
        return {"data": rows.drop(columns=["cid"]), "labels": rows["cid"]}

    train_env = _env(train)
    test_env = _env(test)
    return train_env, test_env, train_env["labels"], test_env["labels"]
