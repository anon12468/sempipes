import warnings

import numpy as np
import pandas as pd
from autogluon.features.generators import ArithmeticFeatureGenerator, AutoMLPipelineFeatureGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from experiments.caafe import scoring

warnings.filterwarnings("ignore")

datasets = ["airlines", "balance-scale", "tic-tac-toe"]


def _clean_features(X):
    if isinstance(X, pd.DataFrame):
        return X.replace([np.inf, -np.inf, np.nan], 0)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _target_type(y):
    return "binary" if pd.Series(y).nunique() == 2 else "multiclass"


def _autogluon_features(X_train, X_test, y_train):
    pipeline_gen = AutoMLPipelineFeatureGenerator(verbosity=0)
    X_train_fe = pipeline_gen.fit_transform(X_train)
    X_test_fe = pipeline_gen.transform(X_test)

    arith_gen = ArithmeticFeatureGenerator(target_type=_target_type(y_train), verbosity=0)
    X_train_fe = arith_gen.fit_transform(X_train_fe, y_train)
    X_test_fe = arith_gen.transform(X_test_fe)

    return _clean_features(X_train_fe), _clean_features(X_test_fe)


for dataset_name in datasets:
    baselines_scores = []
    autogluon_scores = []

    failures = 0
    for seed in range(0, 5):
        data = pd.read_csv(f"experiments/caafe/data/{dataset_name}/data.csv")
        target_column = data.columns[-1]

        train, test = train_test_split(data, test_size=0.25, random_state=seed)

        y_train = train[target_column]
        X_train = train.drop(columns=[target_column])
        X_train = X_train.fillna(0)

        y_test = test[target_column]
        X_test = test.drop(columns=[target_column])
        X_test = X_test.fillna(0)

        X_train_baseline = X_train.copy(deep=True)
        X_test_baseline = X_test.copy(deep=True)

        baseline_clf = RandomForestClassifier(random_state=seed)
        baseline_clf.fit(X_train_baseline, y_train)

        baseline_y_pred = baseline_clf.predict_proba(X_test_baseline)
        baseline_score = scoring(y_test, baseline_y_pred)
        baselines_scores.append(baseline_score)

        try:
            X_train_fe, X_test_fe = _autogluon_features(X_train, X_test, y_train)

            autogluon_clf = RandomForestClassifier(random_state=seed)
            autogluon_clf.fit(X_train_fe, y_train)

            autogluon_y_pred = autogluon_clf.predict_proba(X_test_fe)
            autogluon_score = scoring(y_test, autogluon_y_pred)
            autogluon_scores.append(autogluon_score)

        except Exception:
            failures += 1

    baseline_mean = np.mean(baselines_scores)
    baseline_std = np.std(baselines_scores)
    autogluon_mean = np.mean(autogluon_scores) if autogluon_scores else float("nan")
    autogluon_std = np.std(autogluon_scores) if autogluon_scores else float("nan")

    print(
        f"{dataset_name},{autogluon_mean - baseline_mean},{baseline_mean},{baseline_std},"
        f"{np.round(autogluon_mean, 3)},{np.round(autogluon_std, 3)},{failures}"
    )
