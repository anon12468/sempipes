from dataclasses import dataclass

from sempipes import LLM
from sempipes.optimisers.search_policy import SearchPolicy
from experiments.colopro.test_pipeline import TestPipeline


@dataclass(frozen=True)
class Setup:
    search: SearchPolicy
    num_trials: int
    cv: int
    llm_for_code_generation: LLM
    operator_selection_strategy: str = "ucb"
    optimize_all_operators: bool = False
    ucb_alpha: float = 0.2
    ucb_c: float = 1.0
    epsilon: float = 0.1


__all__ = ["Setup", "TestPipeline"]
