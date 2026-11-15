"""TMDB box office revenue regression for the size sweep.

Copied from experiments/processing_modes/tmdb_pipeline.py. Code generation
asks for 34 features. The train table is every labeled movie that is not in
validation.csv. The test set is that validation file at every train size.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub

import sempipes  # pylint: disable=unused-import

from experiments.size_sweep.tracking import attach_feature_dump

warnings.filterwarnings("ignore", category=UserWarning)

TRAIN_PATH = Path("experiments/size_sweep/data_new.csv")
VALIDATION_PATH = Path("experiments/tmdb_box_office_prediction/validation.csv")
SPLIT_SEED = 42
TRAIN_SIZES = (300, 800, 2001)
HOW_MANY = 34
TARGET = "revenue"
# Per-row mode would read *.jpg paths as images, and the poster files are not in the data.
NON_INPUT_COLUMNS = {TARGET, "poster_path"}
DESCRIPTION = """
    In a world… where movies made an estimated $41.7 billion in 2018, the film industry is more popular than ever. But what movies make the most money at the box office? How much does a director matter? Or the budget? For some movies, it's "You had me at 'Hello.'" For others, the trailer falls short of expectations and you think "What we have here is a failure to communicate." In this competition, you're presented with metadata on several thousand past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. You can collect other publicly available data to use in your model predictions, but in the spirit of this competition, use only data that would have been available before a movie's release. It is your job to predict the international box office revenue for each movie. For each id in the test set, you must predict the value of the revenue variable. Submissions are evaluated on Root-Mean-Squared-Logarithmic-Error (RMSLE) between the predicted value and the actual revenue. Logs are taken to not overweight blockbuster revenue movies.
    """
PROMPT = """
            Create additional features that could help predict the box office revenue of a movie.
            Consider aspects like genre, production details, cast, crew, and any other relevant information
            that could influence a movie's financial success. Some of the attributes are in JSON format,
            so you might need to parse them to extract useful information.
        """
# Recorded tmdb rows used PROMPT. This rerun keeps those rows and writes tmdb_names.
PROMPT_NAMES = """
Create additional features that could help predict the box office revenue of a movie.
Keep the identity of the film as its own columns: the collection it belongs to, the director,
the lead actor, the main production company, and the main genre. Also add counts and other
simple summaries of genre, production, cast, crew, keywords, languages, budget, and release
date where those would help. Some of the attributes are in JSON format that contain a lot of information, so you might need to
parse them to extract useful information.
"""
JSON_COLUMNS = [
    "belongs_to_collection",
    "genres",
    "production_companies",
    "production_countries",
    "spoken_languages",
    "Keywords",
    "cast",
    "crew",
]
LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "rmse",
    "n_estimators": 3000,
    "learning_rate": 0.003,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": 1,
    "seed": 42,
    "boosting_type": "gbdt",
}


def _input_columns() -> list[str]:
    header = pd.read_csv(TRAIN_PATH, nrows=0).columns
    return [column for column in header if column not in NON_INPUT_COLUMNS]


def _cleanup_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
    return df


def _exp_if_transform(outputs, mode):
    if mode in {"transform", "predict"}:
        return np.expm1(outputs)
    return outputs


def sempipes_pipeline(mode: str, artifact_dir: Path | None = None, nl_prompt: str = PROMPT):
    movie_stats = skrub.var("data")
    revenue = skrub.var("labels")
    movie_stats = movie_stats.skb.mark_as_X().skb.set_description(DESCRIPTION)
    revenue = revenue.skb.mark_as_y().skb.set_description("the international box office revenue for a movie")
    input_columns = _input_columns()

    if mode == "llm":
        movie_stats = movie_stats.sem_extract_features(
            nl_prompt=nl_prompt,
            input_columns=input_columns,
            name="additional_movie_features",
            output_columns=None,
            generate_via_code=False,
        )
    else:
        movie_stats = movie_stats.sem_gen_features(
            nl_prompt=nl_prompt,
            name="additional_movie_features",
            how_many=HOW_MANY,
        )
    if artifact_dir is not None:
        movie_stats = attach_feature_dump(
            movie_stats, artifact_dir, "additional_movie_features", input_columns
        )

    movie_stats = movie_stats.drop(columns=JSON_COLUMNS)
    X = movie_stats.skb.apply(skrub.TableVectorizer())
    X = X.fillna(-999)
    X = X.skb.apply_func(_cleanup_column_names)

    y_log = revenue.skb.apply_func(np.log1p)
    predictions = X.skb.apply(lgb.LGBMRegressor(**LGBM_PARAMS), y=y_log)
    return predictions.skb.apply_func(_exp_if_transform, mode=skrub.eval_mode())


def _stratified_order(frame: pd.DataFrame) -> pd.DataFrame:
    """One permutation whose prefixes are stratified on revenue quantile."""
    revenue = pd.qcut(frame[TARGET], q=10, duplicates="drop")
    rng = np.random.default_rng(SPLIT_SEED)
    groups = []
    for _, group in frame.groupby(revenue, observed=True, sort=True):
        order = group.index.to_numpy().copy()
        rng.shuffle(order)
        groups.append(order)
    chosen = []
    cursor = 0
    while any(cursor < len(group) for group in groups):
        for group in groups:
            if cursor < len(group):
                chosen.append(group[cursor])
        cursor += 1
    return frame.loc[chosen]


def load_split(train_size: int) -> tuple[dict, dict, pd.Series, pd.Series]:
    if train_size not in TRAIN_SIZES:
        raise ValueError(f"Movie train size must be one of {TRAIN_SIZES}")
    train_pool = _stratified_order(pd.read_csv(TRAIN_PATH))
    test = pd.read_csv(VALIDATION_PATH)
    if set(train_pool["id"]) & set(test["id"]):
        raise RuntimeError("data_new.csv and validation.csv share movie ids")
    train = train_pool.iloc[:train_size]

    def _env(rows: pd.DataFrame) -> dict:
        rows = rows.reset_index(drop=True)
        return {"data": rows.drop(columns=[TARGET]), "labels": rows[TARGET]}

    train_env = _env(train)
    test_env = _env(test)
    return train_env, test_env, train_env["labels"], test_env["labels"]
