from __future__ import annotations

from pathlib import Path

import pandas as pd


FLOAT_INPUT_COLUMNS = ("year",)
DATA_PATH = Path(__file__).resolve().parent / "data" / "penguins.csv"
TRAIN_SIZE = 0.7
RANDOM_STATE = 42


def load_penguins_frame() -> pd.DataFrame:
    """Load the versioned Palmer Penguins dataset from the repository."""
    return pd.read_csv(DATA_PATH)


def normalize_model_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce raw model inputs to stable dtypes before training, logging, or inference.

    MLflow signatures inferred from integer columns can be brittle when future requests
    contain missing values, because pandas promotes those columns to float. Casting any
    nullable-prone integer inputs to float up front keeps the logged signature stable.
    """
    normalized = df.copy()
    for column in FLOAT_INPUT_COLUMNS:
        if column in normalized.columns:
            normalized[column] = normalized[column].astype("float64")
    return normalized
