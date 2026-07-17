"""BEAVER enrollment forecasting pipeline for MCTS / operator-selection optimization."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from skrub import DataOp
from skrub._data_ops._evaluation import find_node_by_name

from experiments.beaver_enrollment.config import (
    GOLD_TRAIN_PATH,
    TRAIN_DATA_DIR,
    VAL_TERM_RANK_THRESHOLD,
)
from experiments.beaver_enrollment.pipeline import (
    OP_AGG,
    OP_GEN,
    _fit_threshold,
    build_labeled_frame,
    build_pipeline,
    load_sections,
)
from experiments.optimize_multiple import TestPipeline
from sempipes.optimisers.search_policy import OperatorStates

warnings.filterwarnings("ignore")


class EnrollmentPipeline(TestPipeline):
    OPERATOR_NAMES = [OP_AGG, OP_GEN]

    @property
    def name(self) -> str:
        return "enrollment"

    @property
    def scoring(self) -> str:
        return "f1_macro"

    def score(self, y_true, y_pred) -> float:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    def pipeline_with_all_data(self, seed: int) -> tuple[DataOp, dict]:
        x_train, y_train, _meta = _load_train_frame()
        env = _make_env(x_train, y_train, seed)
        return build_pipeline(seed=seed), env

    def pipeline_with_train_data(self, seed: int) -> tuple[DataOp, dict]:
        x_train, y_train, meta = _load_train_frame()
        fit_mask = meta["term_rank"] < VAL_TERM_RANK_THRESHOLD
        x_fit = x_train.loc[fit_mask].reset_index(drop=True)
        y_fit = y_train.loc[fit_mask].reset_index(drop=True)
        env = _make_env(x_fit, y_fit, seed)
        return build_pipeline(seed=seed), env

    def _evaluate(
        self,
        seed: int,
        pipeline: DataOp,
        additional_env_variables: dict,
        operator_states: OperatorStates | None = None,
    ) -> float:
        if operator_states is None:
            operator_states = OperatorStates()
            for name in self.OPERATOR_NAMES:
                data_op = find_node_by_name(pipeline, name)
                operator_states.set(name, data_op._skrub_impl.estimator.empty_state())

        np.random.seed(seed)

        data = additional_env_variables["data"]
        labels = pd.Series(additional_env_variables["labels"], index=data.index)
        sections = additional_env_variables["sections"]

        fit_mask = data["term_rank"] < VAL_TERM_RANK_THRESHOLD
        train_data = data.loc[fit_mask]
        test_data = data.loc[~fit_mask]
        train_labels = labels.loc[fit_mask]
        test_labels = labels.loc[~fit_mask]

        train_env = pipeline.skb.get_data()
        test_env = pipeline.skb.get_data()
        train_env["data"] = train_data
        train_env["labels"] = train_labels.to_numpy()
        train_env["sections"] = sections
        test_env["data"] = test_data
        test_env["labels"] = test_labels.to_numpy()
        test_env["sections"] = sections

        for name in operator_states.operators():
            state = operator_states.get(name)
            train_env[f"sempipes_prefitted_state__{name}"] = state
            test_env[f"sempipes_prefitted_state__{name}"] = state

        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        learner.fit(train_env)
        y_proba = np.asarray(learner.predict_proba(test_env))
        y_proba = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] > 1 else y_proba.ravel()
        threshold = _fit_threshold(train_labels, pd.Series(y_proba))
        y_pred = (y_proba >= threshold).astype(int)
        return self.score(test_labels, y_pred)


def _load_train_frame() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    from experiments.beaver_enrollment.pipeline import _read_gold

    gold_train = _read_gold(GOLD_TRAIN_PATH)
    return build_labeled_frame(TRAIN_DATA_DIR, gold_train)


def _make_env(x: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    del seed
    return {
        "data": x,
        "labels": y.to_numpy(),
        "sections": load_sections(TRAIN_DATA_DIR),
    }
