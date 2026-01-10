import joblib
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features import PenguinFeatureEngineer, preprocessor
from src.inference import predict


def _small_dataset():
    return load_penguins().dropna().head(20)


def _make_pipeline():
    return Pipeline([
        ("add_features", PenguinFeatureEngineer()),
        ("preprocess", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=0, max_depth=3)),
    ])


def test_pipeline_trains_and_predicts():
    df = _small_dataset()
    X, y = df.drop("species", axis=1), df["species"]

    pipeline = _make_pipeline()
    pipeline.fit(X, y)

    preds = pipeline.predict(X.head(3))
    assert len(preds) == 3


def test_inference_predict_handles_mapping():
    df = _small_dataset()
    X, y = df.drop("species", axis=1), df["species"]

    pipeline = _make_pipeline()
    pipeline.fit(X, y)

    # single row inference from mapping
    single_row = X.iloc[0].to_dict()
    preds = predict(single_row, model=pipeline)
    assert len(preds) == 1

    # batch inference from iterable of mappings
    batch_rows = X.head(2).to_dict(orient="records")
    preds_batch = predict(batch_rows, model=pipeline)
    assert len(preds_batch) == 2
