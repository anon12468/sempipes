from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_squared_log_error
from skrub import DataOp, TableVectorizer

import sempipes  # pylint: disable=unused-import
from experiments.prompt_levels.house._pipeline import sempipes_pipeline as house_pipeline
from experiments.prompt_levels.tmdb._pipeline import sempipes_pipeline as tmdb_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
TMDB_DATA = REPO_ROOT / "experiments" / "tmdb_box_office_prediction" / "data.csv"
HOUSE_DATA = REPO_ROOT / "experiments" / "house_prices_advanced_regression_techniques" / "data.csv"

MIDWEST_PROMPT = """
            Compute additional features which help predict the census region of a respondent based on their demographics. Use your intrinsic knowledge about the US to come up with the features. Pay special attention to the zip code of the person.
        """
MIDWEST_OPERATOR = "op_to_evolve"


def _as_1d(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def _take(frames: dict[str, pd.DataFrame | pd.Series], indices: np.ndarray) -> dict[str, pd.DataFrame | pd.Series]:
    return {key: value.iloc[indices].reset_index(drop=True) for key, value in frames.items()}


def _concat(
    left: dict[str, pd.DataFrame | pd.Series], right: dict[str, pd.DataFrame | pd.Series]
) -> dict[str, pd.DataFrame | pd.Series]:
    return {key: pd.concat([left[key], right[key]], ignore_index=True) for key in left}


class Task(ABC):
    name: str
    operator_name: str
    scoring: str
    metric: str

    @abstractmethod
    def load(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Return row-aligned frames for the full dataset."""

    @abstractmethod
    def build(self, seed: int, learner: str) -> DataOp:
        """Build the medium pipeline. ``learner`` is ``fast`` or ``full``."""

    @abstractmethod
    def score(self, test_frames: dict[str, pd.DataFrame | pd.Series], y_pred) -> float:
        """Score predictions on the held-out test frames."""

    def n_rows(self, frames: dict[str, pd.DataFrame | pd.Series]) -> int:
        return len(next(iter(frames.values())))

    def take(self, frames: dict[str, pd.DataFrame | pd.Series], indices: np.ndarray):
        return _take(frames, indices)

    def concat(self, left, right):
        return _concat(left, right)


class TmdbTask(Task):
    name = "tmdb"
    operator_name = "additional_movie_features"
    scoring = "neg_root_mean_squared_log_error"
    metric = "rmsle"

    def load(self) -> dict[str, pd.DataFrame | pd.Series]:
        data = pd.read_csv(TMDB_DATA)
        return {
            "movie_stats": data.drop(columns=["revenue"]).reset_index(drop=True),
            "revenue": data["revenue"].reset_index(drop=True),
        }

    def build(self, seed: int, learner: str, n_prompt_samples: int = 10) -> DataOp:
        learner_kind = "fast" if learner == "fast" else "default"
        return tmdb_pipeline(pipeline_seed=seed, learner=learner_kind, n_prompt_samples=n_prompt_samples)

    def score(self, test_frames: dict[str, pd.DataFrame | pd.Series], y_pred) -> float:
        return float(np.sqrt(mean_squared_log_error(_as_1d(test_frames["revenue"]), _as_1d(y_pred))))


class HouseTask(Task):
    name = "house"
    operator_name = "house_features"
    scoring = "neg_root_mean_squared_error"
    metric = "rmsle"

    def load(self) -> dict[str, pd.DataFrame | pd.Series]:
        return {"data": pd.read_csv(HOUSE_DATA)}

    def build(self, seed: int, learner: str, n_prompt_samples: int = 10) -> DataOp:
        del seed
        learner_kind = "fast" if learner == "fast" else "ensemble"
        return house_pipeline(learner=learner_kind, n_prompt_samples=n_prompt_samples)

    def score(self, test_frames: dict[str, pd.DataFrame | pd.Series], y_pred) -> float:
        y_true = np.log1p(_as_1d(test_frames["data"]["SalePrice"]))
        return float(np.sqrt(mean_squared_error(y_true, _as_1d(y_pred))))


class MidwestTask(Task):
    name = "midwest"
    operator_name = MIDWEST_OPERATOR
    scoring = "accuracy"
    metric = "accuracy"

    def load(self) -> dict[str, pd.DataFrame | pd.Series]:
        dataset = skrub.datasets.fetch_midwest_survey()
        features = dataset.X.iloc[:2000]
        mask = ~features["In_what_ZIP_code_is_your_home_located"].str.contains(r"[^0-9.]")
        features = features[mask].reset_index(drop=True)
        labels = dataset.y.iloc[:2000][mask].reset_index(drop=True)
        self._x_description = dataset.metadata["description"]
        self._y_description = dataset.metadata["target"]
        return {"data": features, "labels": labels}

    def build(self, seed: int, learner: str, n_prompt_samples: int = 10) -> DataOp:
        del seed, learner
        if not hasattr(self, "_x_description"):
            self.load()

        responses = skrub.var("data").skb.set_description(self._x_description)
        labels = skrub.var("labels").skb.set_description(self._y_description)
        responses = responses.skb.mark_as_X()
        labels = labels.skb.mark_as_y()
        responses = responses.sem_gen_features(
            nl_prompt=MIDWEST_PROMPT,
            name=MIDWEST_OPERATOR,
            how_many=5,
            n_prompt_samples=n_prompt_samples,
        )
        encoded = responses.skb.apply(TableVectorizer())
        return encoded.skb.apply(RandomForestClassifier(random_state=0), y=labels)

    def score(self, test_frames: dict[str, pd.DataFrame | pd.Series], y_pred) -> float:
        return float(accuracy_score(_as_1d(test_frames["labels"]), _as_1d(y_pred)))


def get_task(name: str) -> Task:
    tasks: dict[str, Task] = {"tmdb": TmdbTask(), "house": HouseTask(), "midwest": MidwestTask()}
    if name not in tasks:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {', '.join(tasks)}.")
    return tasks[name]
