from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skrub import DataOp
from skrub._data_ops._evaluation import find_node_by_name

from experiments.nyc_penalties.sempipes import (
    CAMERA_GROWTH,
    CAMERA_VTYPES,
    CORPUS_DIR,
    GROWTH,
    OPERATOR_NAMES,
    TEST_PATH,
    TRAIN_PATH,
    add_base_features,
    build_grid,
    build_pipeline,
    load_corpus_tables,
    prefix_key,
    read_keyed_csv,
    rmse,
)
from experiments.nyc_penalties.baseline import build_corpus_features
from experiments.optimize_multiple import TestPipeline
from sempipes.optimisers.search_policy import OperatorStates

warnings.filterwarnings("ignore")

BASELINE_CORPUS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "corpus_features_baseline.parquet")


class NYCPenaltiesPipeline(TestPipeline):
    OPERATOR_NAMES = OPERATOR_NAMES

    @property
    def name(self) -> str:
        return "nyc_penalties"

    @property
    def scoring(self) -> str:
        return "neg_root_mean_squared_error"

    def score(self, y_true, y_pred) -> float:
        return -rmse(y_true, y_pred)

    def pipeline_with_all_data(self, seed: int) -> tuple[DataOp, dict]:
        data, labels, corpus_tables, final_eval = _load_final_eval_env()
        return build_pipeline(seed=seed), _make_env(data, labels, corpus_tables, **final_eval)

    def pipeline_with_train_data(self, seed: int) -> tuple[DataOp, dict]:
        return self.pipeline_with_all_data(seed)

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

        if "test_data" in additional_env_variables:
            return _evaluate_final_2023(seed, pipeline, additional_env_variables, operator_states)

        train_data, test_data, train_labels, test_labels = train_test_split(
            data,
            labels,
            test_size=self.TEST_SIZE,
            random_state=seed,
        )

        train_env = pipeline.skb.get_data()
        test_env = pipeline.skb.get_data()
        for key, value in additional_env_variables.items():
            if key not in {"data", "labels"}:
                train_env[key] = value
                test_env[key] = value

        for name in operator_states.operators():
            state = operator_states.get(name)
            train_env[f"sempipes_prefitted_state__{name}"] = state
            test_env[f"sempipes_prefitted_state__{name}"] = state

        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        train_env["data"] = train_data.reset_index(drop=True)
        train_env["labels"] = train_labels.reset_index(drop=True).to_numpy()
        learner.fit(train_env)

        test_env["data"] = test_data.reset_index(drop=True)
        test_env["labels"] = test_labels.reset_index(drop=True).to_numpy()
        y_pred = learner.predict(test_env)
        return self.score(test_labels, y_pred)


def _load_train_env() -> tuple[pd.DataFrame, pd.Series, dict[str, pd.DataFrame]]:
    train, _ = read_keyed_csv(TRAIN_PATH)
    vtypes = sorted(train["vtype"].unique())
    grid = build_grid(train, vtypes)
    labels = grid["count"].copy()
    data = add_base_features(grid.drop(columns=["count"]), vtypes)
    baseline_corpus_features = _load_baseline_corpus_features()
    if baseline_corpus_features is not None and not baseline_corpus_features.empty:
        data = data.merge(baseline_corpus_features, left_on="k", right_index=True, how="left")
    return data.reset_index(drop=True), labels.reset_index(drop=True), load_corpus_tables(CORPUS_DIR)


def _load_final_eval_env() -> tuple[pd.DataFrame, pd.Series, dict[str, pd.DataFrame], dict]:
    train, _ = read_keyed_csv(TRAIN_PATH)
    test, has_target = read_keyed_csv(TEST_PATH)
    if not has_target:
        data, labels, corpus_tables = _load_train_env()
        return data, labels, corpus_tables, {}

    vtypes = sorted(train["vtype"].unique())
    grid = build_grid(train, vtypes)
    labels = grid["count"].copy()
    baseline_corpus_features = _load_baseline_corpus_features()

    train_data = add_base_features(grid.drop(columns=["count"]), vtypes)
    test_data = add_base_features(test[["street", "vtype"]], vtypes)
    if baseline_corpus_features is not None and not baseline_corpus_features.empty:
        train_data = train_data.merge(baseline_corpus_features, left_on="k", right_index=True, how="left")
        test_data = test_data.merge(baseline_corpus_features, left_on="k", right_index=True, how="left")

    return (
        train_data.reset_index(drop=True),
        labels.reset_index(drop=True),
        load_corpus_tables(CORPUS_DIR),
        {
            "test_data": test_data.reset_index(drop=True),
            "test_labels": test["count"].reset_index(drop=True).to_numpy(),
            "train_raw": train.reset_index(drop=True),
            "test_raw": test.reset_index(drop=True),
        },
    )


def _evaluate_final_2023(
    seed: int,
    pipeline: DataOp,
    additional_env_variables: dict,
    operator_states: OperatorStates,
) -> float:
    np.random.seed(seed)

    train_env = pipeline.skb.get_data()
    test_env = pipeline.skb.get_data()
    for key, value in additional_env_variables.items():
        if key not in {"data", "labels", "test_data", "test_labels", "train_raw", "test_raw"}:
            train_env[key] = value
            test_env[key] = value

    for name in operator_states.operators():
        state = operator_states.get(name)
        train_env[f"sempipes_prefitted_state__{name}"] = state
        test_env[f"sempipes_prefitted_state__{name}"] = state

    learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
    train_env["data"] = additional_env_variables["data"].reset_index(drop=True)
    train_env["labels"] = np.asarray(additional_env_variables["labels"])
    learner.fit(train_env)

    test_env["data"] = additional_env_variables["test_data"].reset_index(drop=True)
    test_env["labels"] = np.asarray(additional_env_variables["test_labels"])
    fallback_pred = learner.predict(test_env)
    y_pred = _apply_baseline_resolution(
        additional_env_variables["test_raw"],
        additional_env_variables["train_raw"],
        fallback_pred,
    )
    return -rmse(additional_env_variables["test_labels"], y_pred)


def _apply_baseline_resolution(te: pd.DataFrame, tr: pd.DataFrame, fallback_pred: np.ndarray) -> np.ndarray:
    pred = np.full(len(te), np.nan)

    merged = te.merge(tr.rename(columns={"count": "_prior"}), on=["street", "vtype"], how="left")
    matched = ~pd.isna(merged["_prior"].values)
    pred[matched] = np.nan_to_num(merged["_prior"].values)[matched]

    tr_pk = tr.assign(pk=tr["street"].map(prefix_key))
    agg = tr_pk.groupby(["pk", "vtype"])["count"].sum().rename("pk_count").reset_index()
    te_pk = te[["street", "vtype"]].assign(pk=te["street"].map(prefix_key))
    te_pk = te_pk.merge(agg, on=["pk", "vtype"], how="left")
    siblings = te_pk.groupby(["pk", "vtype"])["street"].transform("size")
    pk_share = (te_pk["pk_count"] / siblings).values
    use_prefix = np.isnan(pred) & ~np.isnan(pk_share)
    pred[use_prefix] = pk_share[use_prefix]

    use_model = np.isnan(pred)
    pred[use_model] = np.asarray(fallback_pred)[use_model]

    growth = np.where(te["vtype"].isin(CAMERA_VTYPES).values, CAMERA_GROWTH, GROWTH)
    is_persistence = matched | use_prefix
    pred[is_persistence] *= growth[is_persistence]
    return np.clip(pred, 0, None)


def _load_baseline_corpus_features() -> pd.DataFrame:
    rebuild = os.environ.get("NYC_REBUILD_BASELINE_FEATURES") == "1"
    if os.path.exists(BASELINE_CORPUS_CACHE_PATH) and not rebuild:
        return pd.read_parquet(BASELINE_CORPUS_CACHE_PATH)

    features = build_corpus_features(CORPUS_DIR)
    tmp_path = f"{BASELINE_CORPUS_CACHE_PATH}.{os.getpid()}.tmp"
    features.to_parquet(tmp_path)
    os.replace(tmp_path, BASELINE_CORPUS_CACHE_PATH)
    return features


def _make_env(data: pd.DataFrame, labels: pd.Series, corpus_tables: dict[str, pd.DataFrame], **extra) -> dict:
    return {
        "data": data,
        "labels": labels.to_numpy(),
        **corpus_tables,
        **extra,
    }
