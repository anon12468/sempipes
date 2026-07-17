from __future__ import annotations

import sys
import warnings

import numpy as np

import sempipes
from experiments.beaver_enrollment.enrollment_pipeline import EnrollmentPipeline
from experiments.optimize_multiple import Setup
from sempipes.logging import get_logger
from sempipes.optimisers import MonteCarloTreeSearch

warnings.filterwarnings("ignore")

NUM_TRIALS = 36
CV = 5


def main() -> None:
    logger = get_logger()

    if len(sys.argv) < 6:
        print(
            "Usage: minibench enrollment <model> <search> <seed> <operator_selection_strategy> "
            "[exploration_param]\n"
            "  model: gemini (gemini-2.5-flash) | gemini_pro (gemini-2.5-pro)\n"
            "  search: mct_search\n"
            "  operator_selection_strategy: round_robin | ucb | epsilon_greedy | ...\n"
            "  exploration_param: epsilon for epsilon_greedy, alpha for ucb (default 0.1 / 0.2)",
            file=sys.stderr,
        )
        sys.exit(1)

    if sys.argv[1] != "enrollment":
        print(f"Unknown pipeline: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[2]
    search_name = sys.argv[3]
    seed = int(sys.argv[4])
    operator_selection_strategy = sys.argv[5]
    ucb_alpha = 0.2
    epsilon = 0.1
    if len(sys.argv) >= 7:
        exploration_param = float(sys.argv[6])
        if operator_selection_strategy == "ucb":
            ucb_alpha = exploration_param
        elif operator_selection_strategy == "epsilon_greedy":
            epsilon = exploration_param

    if model_name == "gemini":
        llm_name = "gemini/gemini-2.5-flash"
    elif model_name in {"gemini_pro", "gemini-pro", "pro"}:
        llm_name = "gemini/gemini-2.5-pro"
    else:
        print(f"Unsupported model: {model_name} (use 'gemini' or 'gemini_pro')", file=sys.stderr)
        sys.exit(1)

    if search_name != "mct_search":
        print(f"Unsupported search: {search_name} (only 'mct_search' is configured)", file=sys.stderr)
        sys.exit(1)

    llm = sempipes.LLM(
        name=llm_name,
        parameters={"temperature": 1.0},
    )

    sempipes.update_config(
        llm_for_code_generation=llm,
        prefer_empty_state_in_preview=True,
    )

    setup = Setup(
        search=MonteCarloTreeSearch(c=0.5),
        num_trials=NUM_TRIALS,
        cv=CV,
        llm_for_code_generation=llm,
        operator_selection_strategy=operator_selection_strategy,
        optimize_all_operators=False,
        ucb_alpha=ucb_alpha,
        epsilon=epsilon,
    )

    pipeline = EnrollmentPipeline()
    np.random.seed(seed)

    logger.info(
        "Starting enrollment minibench: model=%s temp=1.0 "
        "search=mct_search steps=%s strategy=%s seed=%s ucb_alpha=%s epsilon=%s",
        llm_name,
        NUM_TRIALS,
        operator_selection_strategy,
        seed,
        ucb_alpha,
        epsilon,
    )

    scores_over_time = pipeline.optimize(seed, setup)

    print("#" * 120)
    for max_trial, chosen_trial, train_score, test_score in scores_over_time:
        line = (
            f"MINIBENCH_RESULT> {llm_name},1.0,mct_search,enrollment,"
            f"{seed},{max_trial},{chosen_trial},{train_score},{test_score}"
        )
        print(line)


if __name__ == "__main__":
    main()
