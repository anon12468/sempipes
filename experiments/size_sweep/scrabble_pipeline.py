from __future__ import annotations

import numpy as np
import pandas as pd
import skrub
from sklearn.preprocessing import OneHotEncoder
from skrub import selectors as s
from xgboost import XGBRegressor

import sempipes  # pylint: disable=unused-import

from experiments.size_sweep.tracking import attach_feature_dump

DATA_DIR = "experiments/scrabble_player_rating"
DATA_PATH = f"{DATA_DIR}/data.csv"
GAMES_PATH = f"{DATA_DIR}/games.csv"
TURNS_PATH = f"{DATA_DIR}/turns.csv.gz"
SPLIT_SEED = 42
TRAIN_SIZES = (300, 2000, 10000)
TEST_ROWS = 1000
TARGET = "rating"
BOT_NAMES = {"BetterBot", "HastyBot", "MasterBot", "STEEBot"}
PROMPT = """
        Create additional features that could help predict the rating of a player. Such features could relate to how
        the player scores when they go first in a round and to which games they won, how often they won games, etc.

        Furthermore, consider how much time a player typically uses, how many points per turn they make, how many points per second, etc.
        """
# Columns of the merged player-game table, apart from the rating.
INPUT_COLUMNS = [
    "game_id",
    "nickname",
    "score",
    "total_turns",
    "first_five_turns_points",
    "max_points_turn",
    "min_points_turn",
    "max_min_difference",
    "first",
    "time_control_name",
    "game_end_reason",
    "winner",
    "created_at",
    "lexicon",
    "initial_time_seconds",
    "increment_seconds",
    "rating_mode",
    "max_overtime_minutes",
    "game_duration_seconds",
]


def _sum_first_five(series):
    return sum(series.values[::-1][:5])


def _drop_if_exists(df):
    cols_to_drop = [col for col in ("created_at", "nickname", "game_id") if col in df.columns]
    return df.drop(columns=cols_to_drop)


def _replace_inf(frame):
    return frame.replace([np.inf, -np.inf], np.nan)


def _features(frame, mode: str, artifact_dir):
    if mode == "llm":
        featured = frame.sem_extract_features(
            nl_prompt=PROMPT,
            input_columns=INPUT_COLUMNS,
            name="player_features",
            output_columns=None,
            generate_via_code=False,
        )
    else:
        featured = frame.sem_gen_features(nl_prompt=PROMPT, name="player_features", how_many=15)
    if artifact_dir is None:
        return featured
    return attach_feature_dump(featured, artifact_dir, "player_features", INPUT_COLUMNS)


def sempipes_pipeline(mode: str, artifact_dir=None):
    games_df = pd.read_csv(GAMES_PATH)
    turns_df = pd.read_csv(TURNS_PATH)
    games = skrub.var("games", games_df)
    turns = skrub.var("turns", turns_df)
    data = skrub.var("data").skb.mark_as_X()
    labels = (
        skrub.var("labels").skb.mark_as_y().skb.set_description("the player's Scrabble rating")
    )

    total_turns = turns.groupby(["game_id", "nickname"]).turn_number.count()
    max_points = turns.groupby(["game_id", "nickname"]).points.max()
    min_points = turns.groupby(["game_id", "nickname"]).points.min()
    first_five_turn_point_sum = turns.groupby(["game_id", "nickname"]).points.agg(_sum_first_five)

    game_player_data = total_turns.reset_index()
    game_player_data = game_player_data.rename(columns={"turn_number": "total_turns"})
    game_player_data = game_player_data.assign(
        first_five_turns_points=first_five_turn_point_sum.reset_index()["points"],
        max_points_turn=max_points.reset_index()["points"],
        min_points_turn=min_points.reset_index()["points"],
    )
    game_player_data = game_player_data.assign(
        max_min_difference=game_player_data.max_points_turn - game_player_data.min_points_turn
    )
    game_player_data = game_player_data.join(games.set_index("game_id"), how="left", on="game_id")
    merged = data.merge(game_player_data, how="left", left_on=["game_id", "nickname"], right_on=["game_id", "nickname"])

    featured = _features(merged, mode, artifact_dir)
    encoded = featured.skb.apply(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        cols=(s.categorical() - s.cols("lexicon")),
    )
    encoded = encoded.skb.apply(
        OneHotEncoder(categories=[["NWL20", "CSW21", "ECWL", "NSWL20"]], sparse_output=False),
        cols=s.cols("lexicon"),
    )
    encoded = encoded.skb.apply_func(_drop_if_exists)
    encoded = encoded.skb.select(s.numeric() | s.boolean())
    encoded = encoded.skb.apply_func(_replace_inf)

    model = XGBRegressor(
        colsample_bytree=0.4,
        gamma=0,
        learning_rate=0.07,
        max_depth=3,
        min_child_weight=1.5,
        n_estimators=8000,
        reg_alpha=0.75,
        reg_lambda=0.45,
        subsample=0.6,
        seed=42,
        n_jobs=1,
        eval_metric="rmse",
    )
    return encoded.skb.apply(model, y=labels)


def _player_order(players: list[str]) -> list[str]:
    rng = np.random.default_rng(SPLIT_SEED)
    order = np.array(sorted(players))
    rng.shuffle(order)
    return order.tolist()


def _game_index(frame: pd.DataFrame) -> tuple[dict[str, list[int]], dict[int, list[str]]]:
    humans = frame.loc[~frame["nickname"].isin(BOT_NAMES), ["game_id", "nickname"]]
    games: dict[int, list[str]] = {}
    player_games: dict[str, list[int]] = {}
    for game_id, nickname in humans.itertuples(index=False):
        games.setdefault(int(game_id), []).append(nickname)
        player_games.setdefault(nickname, []).append(int(game_id))
    return player_games, games


def _kept_games(player: str, blocked: set[str], player_games: dict[str, list[int]], games: dict[int, list[str]]) -> list[int]:
    kept = []
    for game_id in player_games.get(player, []):
        if any(name in blocked and name != player for name in games[game_id]):
            continue
        kept.append(game_id)
    return kept


def _rows_for(
    frame: pd.DataFrame,
    players: list[str],
    blocked: set[str],
    player_games: dict[str, list[int]],
    games: dict[int, list[str]],
) -> pd.DataFrame:
    game_ids = []
    for player in players:
        game_ids.extend(_kept_games(player, blocked, player_games, games))
    rows = frame.loc[frame["nickname"].isin(players) & frame["game_id"].isin(set(game_ids))]
    return rows.reset_index(drop=True)


def _closest_end(counts: list[int], target: int, start: int) -> int:
    """Prefix length, at least `start`, whose row count is closest to `target`."""
    best = start
    best_distance = abs(counts[start - 1] - target)
    for end in range(start, len(counts) + 1):
        distance = abs(counts[end - 1] - target)
        if distance < best_distance:
            best = end
            best_distance = distance
        if counts[end - 1] > target and distance > best_distance:
            break
    return best


def load_split(train_size: int) -> tuple[dict, dict, pd.Series, pd.Series]:
    if train_size not in TRAIN_SIZES:
        raise ValueError(f"Scrabble train size must be one of {TRAIN_SIZES}")
    frame = pd.read_csv(DATA_PATH)
    player_games, games = _game_index(frame)
    ordered = _player_order(list(player_games))
    blocked = set(ordered)
    already: set[str] = set()
    test_counts = []
    running = 0
    for player in ordered:
        blocked.remove(player)
        running += len(_kept_games(player, blocked, player_games, games))
        for game_id in player_games.get(player, []):
            running += sum(other in already for other in games[game_id])
        already.add(player)
        test_counts.append(running)
    test_end = _closest_end(test_counts, TEST_ROWS, 1)
    test_players = ordered[:test_end]
    train_players = ordered[test_end:]
    train_counts = []
    running = 0
    blocked = set(test_players)
    for player in train_players:
        running += len(_kept_games(player, blocked, player_games, games))
        train_counts.append(running)
    train_end = _closest_end(train_counts, train_size, 1)
    test_rows = _rows_for(frame, test_players, set(train_players), player_games, games)
    train_rows = _rows_for(frame, train_players[:train_end], set(test_players), player_games, games)

    def _env(rows: pd.DataFrame) -> dict:
        rows = rows.reset_index(drop=True)
        return {"data": rows.drop(columns=[TARGET]), "labels": rows[TARGET]}

    train_env = _env(train_rows)
    test_env = _env(test_rows)
    return train_env, test_env, train_env["labels"], test_env["labels"]
