from __future__ import annotations

import sys
import warnings

import numpy as np

import sempipes
from experiments.nyc_penalties.nyc_pipeline import NYCPenaltiesPipeline
from experiments.nyc_penalties.sempipes import ENABLE_OPTIMIZED_SEMGEN, OPERATOR_NAMES, resolve_llm_name
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
            "Usage: minibench nyc_penalties <model> <search> <seed> <operator_selection_strategy> "
            "[exploration_param]\n"
            "  model: gemini | gemini_pro | any LiteLLM model name\n"
            "  search: mct_search\n"
            "  operator_selection_strategy: round_robin | ucb | epsilon_greedy | ...",
            file=sys.stderr,
        )
        sys.exit(1)

    if sys.argv[1] != "nyc_penalties":
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

    if search_name != "mct_search":
        print(f"Unsupported search: {search_name} (only 'mct_search' is configured)", file=sys.stderr)
        sys.exit(1)

    llm = sempipes.LLM(
        name=resolve_llm_name(model_name),
        parameters={"temperature": 1.0},
    )
    sempipes.update_config(llm_for_code_generation=llm)

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

    pipeline = NYCPenaltiesPipeline()
    np.random.seed(seed)

    logger.info(
        "Starting NYC penalties minibench: model=%s temp=1.0 search=mct_search "
        "steps=%s strategy=%s seed=%s ucb_alpha=%s epsilon=%s semgen=%s operators=%s",
        llm.name,
        NUM_TRIALS,
        operator_selection_strategy,
        seed,
        ucb_alpha,
        epsilon,
        ENABLE_OPTIMIZED_SEMGEN,
        OPERATOR_NAMES,
    )

    scores_over_time = pipeline.optimize(seed, setup)

    print("#" * 120)
    for max_trial, chosen_trial, train_score, test_score in scores_over_time:
        print(
            "MINIBENCH_RESULT> "
            f"{llm.name},1.0,mct_search,nyc_penalties,"
            f"{seed},{max_trial},{chosen_trial},{train_score},{test_score}"
        )


if __name__ == "__main__":
    main()
