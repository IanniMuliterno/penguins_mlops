import pandas as pd
from palmerpenguins import load_penguins

from src.features import PenguinFeatureEngineer


def test_feature_engineer_adds_expected_columns():
    df = load_penguins().dropna().head(5)

    engineered = PenguinFeatureEngineer().fit_transform(df)

    assert "bmi" in engineered.columns
    assert "bill_ratio" in engineered.columns
    pd.testing.assert_index_equal(df.index, engineered.index)  # preserves row order
    assert not engineered[["bmi", "bill_ratio"]].isna().any().any()
