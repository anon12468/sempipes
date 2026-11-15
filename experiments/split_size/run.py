"""Sweep train size and validation size for the SemPipes tasks.

A fixed 25% test set is held out. The rest is split into train and validation.
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from pathlib import Path

import numpy as np

import sempipes
from experiments.split_size.pipelines import REPO_ROOT, Task, get_task
from sempipes.logging import get_logger
from sempipes.optimisers import EvolutionarySearch, optimise_colopro

warnings.filterwarnings("ignore")

logger = get_logger()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FRACTIONS = (0.1, 0.25, 0.5, 1.0)
TEST_FRACTION = 0.25
CSV_FIELDNAMES = [
    "dataset",
    "subexperiment",
    "fraction",
    "n_train",
    "n_val",
    "n_test",
    "rep",
    "rep_seed",
    "val_score",
    "test_score",
    "metric",
    "chosen_trial",
    "record_type",
]


class OneSplit:
    """One train/validation split over rows ordered train-then-validation."""

    def __init__(self, n_train: int):
        self.n_train = n_train

    def get_n_splits(self, X=None, y=None, groups=None):  # pylint: disable=unused-argument
        return 1

    def split(self, X, y=None, groups=None):  # pylint: disable=unused-argument
        n_samples = len(X)
        if not 0 < self.n_train < n_samples:
            raise ValueError(f"n_train={self.n_train} is outside 1..{n_samples - 1} for {n_samples} rows")
        yield np.arange(self.n_train), np.arange(self.n_train, n_samples)

    def __repr__(self) -> str:
        return f"OneSplit(n_train={self.n_train})"


def _split_pools(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    permutation = np.random.default_rng(seed).permutation(n_rows)
    n_test = int(round(n_rows * TEST_FRACTION))
    n_test = min(max(n_test, 1), n_rows - 2)
    test_idx = permutation[:n_test]
    rest = permutation[n_test:]
    n_train = len(rest) // 2
    if n_train < 1 or n_train >= len(rest):
        raise ValueError(f"Not enough rows to form train and validation reservoirs from {n_rows} rows")
    return test_idx, rest[:n_train], rest[n_train:]


def _prefix(indices: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return indices
    n_keep = max(1, int(round(len(indices) * fraction)))
    return indices[:n_keep]


def _condition_indices(subexperiment: str, fraction: float, train_reservoir, val_reservoir):
    if subexperiment == "train":
        return _prefix(train_reservoir, fraction), val_reservoir
    if subexperiment == "val":
        return train_reservoir, _prefix(val_reservoir, fraction)
    raise ValueError(f"Unknown subexperiment {subexperiment!r}. Choose from 'train' or 'val'.")


def _configure_llm() -> None:
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        )
    )


def _evaluation_env(frames: dict, outcome) -> dict:
    """Bind data plus the operator inputs the semantic estimators require.

    Skrub does not use a variable's default during fit unless the name is present
    in the environment. Prefitted evaluation still evaluates those inputs.
    """
    env = dict(frames)
    for operator_name in outcome.states.operators():
        env[f"sempipes_memory__{operator_name}"] = None
        env[f"sempipes_pipeline_summary__{operator_name}"] = None
        env[f"sempipes_prefitted_state__{operator_name}"] = outcome.states.get(operator_name)
        env[f"sempipes_inspirations__{operator_name}"] = []
    return env


def _evaluate_test(task: Task, pipeline, train_frames: dict, test_frames: dict, outcome) -> float:
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    learner.fit(_evaluation_env(train_frames, outcome))
    return task.score(test_frames, learner.predict(_evaluation_env(test_frames, outcome)))


def _run_condition(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    task: Task,
    subexperiment: str,
    fraction: float,
    rep: int,
    rep_seed: int,
    num_trials: int,
    learner: str,
    frames,
    test_idx,
    train_idx,
    val_idx,
) -> dict:
    pipeline = task.build(rep_seed, learner)
    search_frames = task.concat(task.take(frames, train_idx), task.take(frames, val_idx))
    outcomes = optimise_colopro(
        pipeline,
        num_trials=num_trials,
        scoring=task.scoring,
        search=EvolutionarySearch(population_size=6),
        cv=OneSplit(len(train_idx)),
        run_name=f"split_size_{task.name}_{subexperiment}_frac{fraction}_seed{rep_seed}",
        n_jobs_for_evaluation=1,
        additional_env_variables=search_frames,
        only_optimize_operator=task.operator_name,
    )
    best = max(outcomes, key=lambda outcome: (outcome.score, -outcome.search_node.trial))
    test_score = _evaluate_test(task, pipeline, task.take(frames, train_idx), task.take(frames, test_idx), best)
    return {
        "dataset": task.name,
        "subexperiment": subexperiment,
        "fraction": fraction,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "rep": rep + 1,
        "rep_seed": rep_seed,
        "val_score": best.score,
        "test_score": test_score,
        "metric": task.metric,
        "chosen_trial": best.search_node.trial,
        "record_type": "rep",
    }


def _summary_rows(rep_rows: list[dict]) -> list[dict]:
    summaries = []
    fractions = sorted({row["fraction"] for row in rep_rows})
    for fraction in fractions:
        scores = [row["test_score"] for row in rep_rows if row["fraction"] == fraction and row["test_score"] != ""]
        scores = [float(score) for score in scores if np.isfinite(float(score))]
        template = next(row for row in rep_rows if row["fraction"] == fraction)
        base = {
            "dataset": template["dataset"],
            "subexperiment": template["subexperiment"],
            "fraction": fraction,
            "n_train": template["n_train"],
            "n_val": template["n_val"],
            "n_test": template["n_test"],
            "rep": "",
            "rep_seed": "",
            "val_score": "",
            "metric": template["metric"],
            "chosen_trial": "",
        }
        mean_score = float(np.mean(scores)) if scores else ""
        std_score = float(np.std(scores)) if scores else ""
        summaries.append({**base, "test_score": mean_score, "record_type": "mean"})
        summaries.append({**base, "test_score": std_score, "record_type": "std"})
    return summaries


def _write_csv(path: Path, rep_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rep_rows)
        if rep_rows:
            writer.writerows(_summary_rows(rep_rows))


def run(
    dataset: str,
    subexperiment: str,
    seed: int = 42,
    nreps: int = 3,
    num_trials: int = 36,
    learner: str = "fast",
    fractions: tuple[float, ...] = FRACTIONS,
    results_csv: Path | None = None,
) -> Path:
    os.chdir(REPO_ROOT)
    _configure_llm()
    task = get_task(dataset)
    frames = task.load()
    csv_path = results_csv or (
        RESULTS_DIR / f"{dataset}_{subexperiment}_seed{seed}_nreps{nreps}.csv"
    )
    rep_rows: list[dict] = []

    for rep in range(nreps):
        rep_seed = seed + rep
        np.random.seed(rep_seed)
        test_idx, train_reservoir, val_reservoir = _split_pools(task.n_rows(frames), rep_seed)
        for fraction in fractions:
            train_idx, val_idx = _condition_indices(subexperiment, fraction, train_reservoir, val_reservoir)
            logger.info(
                "SPLIT_SIZE> %s %s fraction=%s rep=%s n_train=%s n_val=%s n_test=%s",
                dataset,
                subexperiment,
                fraction,
                rep + 1,
                len(train_idx),
                len(val_idx),
                len(test_idx),
            )
            try:
                row = _run_condition(
                    task,
                    subexperiment,
                    fraction,
                    rep,
                    rep_seed,
                    num_trials,
                    learner,
                    frames,
                    test_idx,
                    train_idx,
                    val_idx,
                )
            except Exception as error:  # pylint: disable=broad-except
                logger.error("SPLIT_SIZE> failed fraction=%s rep=%s: %s", fraction, rep + 1, error, exc_info=True)
                row = {
                    "dataset": dataset,
                    "subexperiment": subexperiment,
                    "fraction": fraction,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                    "rep": rep + 1,
                    "rep_seed": rep_seed,
                    "val_score": "",
                    "test_score": "",
                    "metric": task.metric,
                    "chosen_trial": "",
                    "record_type": "rep",
                }
            rep_rows.append(row)
            print(
                f"[{dataset}/{subexperiment}] fraction={fraction} rep={rep + 1}/{nreps} "
                f"val={row['val_score']} test={row['test_score']} ({task.metric})"
            )
            _write_csv(csv_path, rep_rows)

    print(f"Wrote {csv_path}")
    return csv_path


def _parse_fractions(text: str) -> tuple[float, ...]:
    fractions = tuple(float(part) for part in text.split(",") if part.strip())
    if not fractions:
        raise argparse.ArgumentTypeError("at least one fraction is required")
    return fractions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validation size sweep")
    parser.add_argument("dataset", choices=["tmdb", "house", "midwest"])
    parser.add_argument("subexperiment", choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nreps", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=36)
    parser.add_argument("--learner", choices=["fast", "full"], default="fast")
    parser.add_argument("--fractions", type=_parse_fractions, default=FRACTIONS)
    parser.add_argument("--results-csv", type=Path, default=None)
    args = parser.parse_args()
    run(
        dataset=args.dataset,
        subexperiment=args.subexperiment,
        seed=args.seed,
        nreps=args.nreps,
        num_trials=args.num_trials,
        learner=args.learner,
        fractions=args.fractions,
        results_csv=args.results_csv,
    )


if __name__ == "__main__":
    main()
