from __future__ import annotations

import warnings

import pandas as pd


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    from palmerpenguins import load_penguins as _load_penguins


FLOAT_INPUT_COLUMNS = ("year",)


def load_penguins_frame() -> pd.DataFrame:
    """Load the Palmer Penguins dataset while hiding an upstream deprecation warning."""
    return _load_penguins()


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
