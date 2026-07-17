"""Paths for the BEAVER enrollment-forecasting experiment."""
from __future__ import annotations

import os
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent

_DEFAULT_BEAVER_ROOT = Path(__file__).resolve().parents[4] / "BEAVER"
BEAVER_ROOT = Path(os.environ.get("BEAVER_DATA_DIR", _DEFAULT_BEAVER_ROOT))

TRAIN_DATA_DIR = BEAVER_ROOT / "table_splits" / "train"
TEST_DATA_DIR = BEAVER_ROOT / "test"
GOLD_TRAIN_PATH = BEAVER_ROOT / "eval" / "gold_enrollment_train.csv"
GOLD_TEST_PATH = BEAVER_ROOT / "eval" / "gold_enrollment_test.csv"
SCRIPTS_DIR = BEAVER_ROOT / "scripts"

VAL_TERM_RANK_THRESHOLD = 2015  # terms before 2015 = fit; 2015+ within train = validation
