import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_penguins_frame, normalize_model_input_schema
from src.features import PenguinFeatureEngineer, preprocessor
from src.inference import _align_features, load_feature_names, predict


def _make_pipeline():
    return Pipeline([
        ("add_features", PenguinFeatureEngineer()),
        ("preprocess", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=0, max_depth=2)),
    ])


def _feature_file(tmp_path: Path, features):
    path = tmp_path / "feature_names.json"
    with open(path, "w") as f:
        json.dump({"features": features}, f)
    return path


def test_predict_aligns_columns(tmp_path):
    df = load_penguins_frame().dropna().head(10)
    X = normalize_model_input_schema(df.drop("species", axis=1))
    y = df["species"]
    feature_order = list(X.columns)  # training-time order
    feature_path = _feature_file(tmp_path, feature_order)

    # Shuffle columns in input to simulate messy inference data
    shuffled = X.sample(frac=1, axis=1, random_state=1)

    model = _make_pipeline().fit(X, y)
    preds = predict(shuffled, model=model, feature_names_path=feature_path)
    assert len(preds) == len(shuffled)


def test_align_features_raises_on_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    try:
        _align_features(df, ["a", "b", "c"])
    except ValueError as exc:
        assert "Missing required feature columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing features")
