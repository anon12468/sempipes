from __future__ import annotations

import argparse
import csv
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_log_error
from sklearn.model_selection import train_test_split

import sempipes
from experiments.house_prices_advanced_regression_techniques._sempipes_impl import rmsle
from experiments.prompt_samples.house_pipeline import sempipes_pipeline as house_pipeline
from experiments.prompt_samples.midwest_pipeline import OPERATOR_NAME as MIDWEST_OPERATOR
from experiments.prompt_samples.midwest_pipeline import TEST_SIZE
from experiments.prompt_samples.midwest_pipeline import load_frames as load_midwest_frames
from experiments.prompt_samples.midwest_pipeline import sempipes_pipeline as midwest_pipeline
from experiments.prompt_samples.tmdb_pipeline import sempipes_pipeline as tmdb_pipeline
from sempipes.logging import get_logger
from sempipes.optimisers import EvolutionarySearch, optimise_colopro
from sempipes.optimisers.search_policy import Outcome
from sempipes.optimisers.trajectory import load_trajectory_from_json

warnings.filterwarnings("ignore")

logger = get_logger()

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
HOUSE_DATA = Path("experiments/house_prices_advanced_regression_techniques/data.csv")
TMDB_DATA = Path("experiments/tmdb_box_office_prediction/data.csv")
PROMPT_SAMPLE_COUNTS = (5, 10, 25, 50, 100, 200)
# The copied TMDB graph subsamples to 100 rows before sem_gen_features.
TMDB_FIT_ROWS = 100
CSV_FIELDNAMES = [
    "dataset",
    "n_prompt_samples",
    "n_prompt_samples_actual",
    "n_search_rows",
    "n_train",
    "n_test",
    "rep",
    "rep_seed",
    "val_score",
    "test_score",
    "metric",
    "chosen_trial",
    "record_type",
]


def select_best(outcomes: list[Outcome]) -> Outcome:
    """Highest validation score. Ties keep the earlier trial."""
    if not outcomes:
        raise ValueError("Trajectory has no outcomes")
    return max(outcomes, key=lambda outcome: (outcome.score, -outcome.search_node.trial))


def actual_prompt_samples(dataset: str, n_prompt_samples: int, n_search_rows: int) -> int:
    available = n_search_rows
    if dataset == "tmdb":
        available = min(available, TMDB_FIT_ROWS)
    return min(n_prompt_samples, max(available, 0))


def _configure_llm() -> None:
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        ),
        prefer_empty_state_in_preview=True,
    )


def _load_house(seed: int, train_only: bool) -> dict:
    data = pd.read_csv(HOUSE_DATA)
    if train_only:
        data, _ = train_test_split(data, train_size=TEST_SIZE, random_state=seed)
    return {"data": data}


def _load_tmdb(seed: int, train_only: bool) -> dict:
    data = pd.read_csv(TMDB_DATA)
    movie_stats = data.drop(columns=["revenue"])
    revenue = data["revenue"]
    if train_only:
        movie_stats, _, revenue, _ = train_test_split(
            movie_stats, revenue, train_size=TEST_SIZE, random_state=seed
        )
    return {"movie_stats": movie_stats, "revenue": revenue}


def _load_search_env(dataset: str, seed: int) -> dict:
    if dataset == "midwest":
        frames = load_midwest_frames(seed, train_only=True)
        return {"data": frames["data"], "labels": frames["labels"], "x_description": frames["x_description"], "y_description": frames["y_description"]}
    if dataset == "house":
        return _load_house(seed, train_only=True)
    if dataset == "tmdb":
        return _load_tmdb(seed, train_only=True)
    raise ValueError(dataset)


def _load_full_env(dataset: str, seed: int) -> dict:
    if dataset == "midwest":
        frames = load_midwest_frames(seed, train_only=False)
        return {"data": frames["data"], "labels": frames["labels"], "x_description": frames["x_description"], "y_description": frames["y_description"]}
    if dataset == "house":
        return _load_house(seed, train_only=False)
    if dataset == "tmdb":
        return _load_tmdb(seed, train_only=False)
    raise ValueError(dataset)


def _build(dataset: str, seed: int, learner: str, n_prompt_samples: int, env: dict):
    if dataset == "midwest":
        return midwest_pipeline(env["x_description"], env["y_description"], n_prompt_samples=n_prompt_samples)
    if dataset == "house":
        kind = "fast" if learner == "fast" else "ensemble"
        return house_pipeline(learner=kind, n_prompt_samples=n_prompt_samples)
    if dataset == "tmdb":
        kind = "fast" if learner == "fast" else "default"
        return tmdb_pipeline(pipeline_seed=seed, learner=kind, n_prompt_samples=n_prompt_samples)
    raise ValueError(dataset)


def _metric(dataset: str) -> str:
    return "accuracy" if dataset == "midwest" else "rmsle"


def _scoring(dataset: str) -> str:
    if dataset == "midwest":
        return "accuracy"
    if dataset == "house":
        return "neg_root_mean_squared_error"
    return "neg_root_mean_squared_log_error"


def _operator_name(dataset: str) -> str:
    if dataset == "midwest":
        return MIDWEST_OPERATOR
    if dataset == "house":
        return "house_features"
    return "additional_movie_features"


def _search_row_count(dataset: str, env: dict) -> int:
    if dataset == "midwest":
        return len(env["data"])
    if dataset == "house":
        return len(env["data"])
    return len(env["movie_stats"])


def _apply_states(env: dict, outcome: Outcome) -> None:
    for operator_name in outcome.states.operators():
        env[f"sempipes_prefitted_state__{operator_name}"] = outcome.states.get(operator_name)


def _score_test(dataset: str, seed: int, pipeline, full_env: dict, outcome: Outcome) -> tuple[float, int, int]:
    """One held-out score of this node, using the prompt-levels 75/25 split."""
    np.random.seed(seed)
    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    if dataset == "midwest":
        train_data, test_data, train_labels, test_labels = train_test_split(
            full_env["data"], full_env["labels"], test_size=TEST_SIZE, random_state=seed
        )
        train_env = pipeline.skb.get_data()
        test_env = pipeline.skb.get_data()
        train_env["data"] = train_data
        train_env["labels"] = train_labels
        test_env["data"] = test_data
        _apply_states(train_env, outcome)
        _apply_states(test_env, outcome)
        learner.fit(train_env)
        score = float(accuracy_score(test_labels, learner.predict(test_env)))
        return score, len(train_data), len(test_data)
    if dataset == "house":
        train_data, test_data = train_test_split(full_env["data"], test_size=TEST_SIZE, random_state=seed)
        train_env = pipeline.skb.get_data()
        test_env = pipeline.skb.get_data()
        train_env["data"] = train_data
        test_env["data"] = test_data
        _apply_states(train_env, outcome)
        _apply_states(test_env, outcome)
        learner.fit(train_env)
        score = float(rmsle(np.log1p(test_data["SalePrice"]), learner.predict(test_env)))
        return score, len(train_data), len(test_data)
    train_ms, test_ms, train_rev, test_rev = train_test_split(
        full_env["movie_stats"], full_env["revenue"], test_size=TEST_SIZE, random_state=seed
    )
    train_env = pipeline.skb.get_data()
    test_env = pipeline.skb.get_data()
    train_env["movie_stats"] = train_ms
    train_env["revenue"] = train_rev
    test_env["movie_stats"] = test_ms
    test_env["revenue"] = test_rev
    _apply_states(train_env, outcome)
    _apply_states(test_env, outcome)
    learner.fit(train_env)
    score = float(np.sqrt(mean_squared_log_error(test_rev.values, learner.predict(test_env))))
    return score, len(train_ms), len(test_ms)


def _print_best(row: dict, rep: int | None, nreps: int | None) -> None:
    rep_label = f"rep={row['rep']}" if nreps is None else f"rep={rep}/{nreps}"
    print(
        f"[{row['dataset']}] n_prompt_samples={row['n_prompt_samples']} "
        f"actual={row['n_prompt_samples_actual']} {rep_label} seed={row['rep_seed']} "
        f"trial={row['chosen_trial']} val={row['val_score']} test={row['test_score']} ({row['metric']})"
    )


def _summary_rows(rep_rows: list[dict]) -> list[dict]:
    if not rep_rows:
        return []
    template = rep_rows[0]
    scores = [
        float(row["test_score"])
        for row in rep_rows
        if row["test_score"] != "" and np.isfinite(float(row["test_score"]))
    ]
    base = {
        "dataset": template["dataset"],
        "n_prompt_samples": template["n_prompt_samples"],
        "n_prompt_samples_actual": template["n_prompt_samples_actual"],
        "n_search_rows": template["n_search_rows"],
        "n_train": template["n_train"],
        "n_test": template["n_test"],
        "rep": "",
        "rep_seed": "",
        "val_score": "",
        "metric": template["metric"],
        "chosen_trial": "",
    }
    mean_score = float(np.mean(scores)) if scores else ""
    std_score = float(np.std(scores)) if scores else ""
    return [
        {**base, "test_score": mean_score, "record_type": "mean"},
        {**base, "test_score": std_score, "record_type": "std"},
    ]


def _write_csv(path: Path, rep_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rep_rows)
        if rep_rows:
            writer.writerows(_summary_rows(rep_rows))


def _empty_row(dataset: str, n_prompt_samples: int, n_actual: int, n_search: int, rep: int, rep_seed: int) -> dict:
    return {
        "dataset": dataset,
        "n_prompt_samples": n_prompt_samples,
        "n_prompt_samples_actual": n_actual,
        "n_search_rows": n_search,
        "n_train": "",
        "n_test": "",
        "rep": rep,
        "rep_seed": rep_seed,
        "val_score": "",
        "test_score": "",
        "metric": _metric(dataset),
        "chosen_trial": "",
        "record_type": "rep",
    }


def _score_best(
    dataset: str,
    n_prompt_samples: int,
    rep: int,
    rep_seed: int,
    num_trials: int,
    learner: str,
    cv: int,
    trajectory_path: Path | None = None,
) -> dict:
    os.chdir(REPO_ROOT)
    np.random.seed(rep_seed)
    search_env = _load_search_env(dataset, rep_seed)
    full_env = _load_full_env(dataset, rep_seed)
    n_search = _search_row_count(dataset, search_env)
    n_actual = actual_prompt_samples(dataset, n_prompt_samples, n_search)
    pipeline = _build(dataset, rep_seed, learner, n_prompt_samples, search_env)
    only_operator = _operator_name(dataset) if dataset != "midwest" else None
    if trajectory_path is None:
        outcomes = optimise_colopro(
            pipeline,
            num_trials=num_trials,
            scoring=_scoring(dataset),
            search=EvolutionarySearch(population_size=6),
            cv=cv,
            run_name=f"prompt_samples_{dataset}_n{n_prompt_samples}_seed{rep_seed}",
            n_jobs_for_evaluation=-1,
            additional_env_variables={key: value for key, value in search_env.items() if key not in {"x_description", "y_description"}},
            only_optimize_operator=only_operator,
        )
    else:
        outcomes = load_trajectory_from_json(trajectory_path).outcomes
    best = select_best(outcomes)
    eval_pipeline = _build(dataset, rep_seed, learner, n_prompt_samples, full_env)
    test_score, n_train, n_test = _score_test(dataset, rep_seed, eval_pipeline, full_env, best)
    return {
        "dataset": dataset,
        "n_prompt_samples": n_prompt_samples,
        "n_prompt_samples_actual": n_actual,
        "n_search_rows": n_search,
        "n_train": n_train,
        "n_test": n_test,
        "rep": rep,
        "rep_seed": rep_seed,
        "val_score": best.score,
        "test_score": test_score,
        "metric": _metric(dataset),
        "chosen_trial": best.search_node.trial,
        "record_type": "rep",
    }


def run(
    dataset: str,
    n_prompt_samples: int,
    seed: int = 42,
    nreps: int = 5,
    num_trials: int = 36,
    learner: str = "prompt_levels",
    cv: int = 5,
    results_csv: Path | None = None,
    trajectory_path: Path | None = None,
) -> Path | None:
    os.chdir(REPO_ROOT)
    _configure_llm()
    resolved_learner = _default_learner(dataset) if learner == "prompt_levels" else learner
    if trajectory_path is not None:
        row = _score_best(
            dataset, n_prompt_samples, 1, seed, num_trials, resolved_learner, cv, trajectory_path
        )
        _print_best(row, rep=None, nreps=None)
        return None

    csv_path = results_csv or (RESULTS_DIR / f"{dataset}_n{n_prompt_samples}_seed{seed}_nreps{nreps}.csv")
    rep_rows: list[dict] = []
    for rep in range(nreps):
        rep_seed = seed + rep
        search_env = _load_search_env(dataset, rep_seed)
        n_search = _search_row_count(dataset, search_env)
        n_actual = actual_prompt_samples(dataset, n_prompt_samples, n_search)
        logger.info(
            "PROMPT_SAMPLES> %s n_prompt_samples=%s actual=%s rep=%s n_search_rows=%s learner=%s cv=%s",
            dataset,
            n_prompt_samples,
            n_actual,
            rep + 1,
            n_search,
            resolved_learner,
            cv,
        )
        try:
            row = _score_best(dataset, n_prompt_samples, rep + 1, rep_seed, num_trials, resolved_learner, cv)
        except Exception as error:  # pylint: disable=broad-except
            logger.error(
                "PROMPT_SAMPLES> failed n_prompt_samples=%s rep=%s: %s",
                n_prompt_samples,
                rep + 1,
                error,
                exc_info=True,
            )
            row = _empty_row(dataset, n_prompt_samples, n_actual, n_search, rep + 1, rep_seed)
        rep_rows.append(row)
        _print_best(row, rep + 1, nreps)
        _write_csv(csv_path, rep_rows)
    print(f"Wrote {csv_path}")
    return csv_path


def _default_learner(dataset: str) -> str:
    if dataset == "house":
        return "ensemble"
    if dataset == "tmdb":
        return "default"
    return "rf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate prompt sample count using the prompt-levels medium setup")
    parser.add_argument("dataset", choices=["tmdb", "house", "midwest"])
    parser.add_argument("--n-prompt-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nreps", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=36)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--learner",
        choices=["prompt_levels", "fast", "ensemble", "default"],
        default="prompt_levels",
        help="prompt_levels uses ensemble for house, default for TMDB, and the random forest for Midwest",
    )
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Score only the best node of this trajectory. Does not run a new search.",
    )
    args = parser.parse_args()
    if args.n_prompt_samples < 1:
        parser.error("--n-prompt-samples must be at least 1")
    run(
        dataset=args.dataset,
        n_prompt_samples=args.n_prompt_samples,
        seed=args.seed,
        nreps=args.nreps,
        num_trials=args.num_trials,
        learner=args.learner,
        cv=args.cv,
        results_csv=args.results_csv,
        trajectory_path=args.trajectory,
    )


if __name__ == "__main__":
    main()
