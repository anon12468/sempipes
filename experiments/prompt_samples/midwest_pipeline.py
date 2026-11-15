from __future__ import annotations

import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skrub import DataOp, TableVectorizer

import sempipes  # pylint: disable=unused-import

OPERATOR_NAME = "op_to_evolve"
TEST_SIZE = 0.25
PROMPT = """
            Compute additional features which help predict the census region of a respondent based on their demographics. Use your intrinsic knowledge about the US to come up with the features. Pay special attention to the zip code of the person.
        """


def load_frames(seed: int, train_only: bool) -> dict:
    dataset = skrub.datasets.fetch_midwest_survey()
    features = dataset.X.iloc[:2000]
    mask = ~features["In_what_ZIP_code_is_your_home_located"].str.contains(r"[^0-9.]")
    features = features[mask]
    labels = dataset.y.iloc[:2000][mask]
    if train_only:
        features, _, labels, _ = train_test_split(features, labels, train_size=TEST_SIZE, random_state=seed)
    return {
        "data": features,
        "labels": labels,
        "x_description": dataset.metadata["description"],
        "y_description": dataset.metadata["target"],
    }


def sempipes_pipeline(x_description: str, y_description: str, n_prompt_samples: int = 10) -> DataOp:
    responses = skrub.var("data").skb.set_description(x_description)
    labels = skrub.var("labels").skb.set_description(y_description)
    responses = responses.skb.mark_as_X()
    labels = labels.skb.mark_as_y()
    responses = responses.sem_gen_features(
        nl_prompt=PROMPT,
        name=OPERATOR_NAME,
        how_many=5,
        n_prompt_samples=n_prompt_samples,
    )
    encoded = responses.skb.apply(TableVectorizer())
    return encoded.skb.apply(RandomForestClassifier(random_state=0), y=labels)
