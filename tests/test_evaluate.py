import json
from pathlib import Path

import skops.io as sio
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_penguins_frame, normalize_model_input_schema
from src.evaluate import evaluate
from src.features import PenguinFeatureEngineer, preprocessor


def _train_and_save(tmp_path: Path):
    df = load_penguins_frame().dropna()
    X = normalize_model_input_schema(df.drop("species", axis=1))
    y = df["species"]

    pipeline = Pipeline([
        ("add_features", PenguinFeatureEngineer()),
        ("preprocess", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=0, max_depth=3)),
    ])
    pipeline.fit(X, y)

    model_path = tmp_path / "model.skops"
    sio.dump(pipeline, model_path)
    return model_path


def test_evaluate_writes_metrics(tmp_path):
    model_path = _train_and_save(tmp_path)
    metrics_path = tmp_path / "metrics.json"

    metrics = evaluate(model_path=model_path, metrics_path=metrics_path)

    assert metrics_path.exists()
    with open(metrics_path) as f:
        saved = json.load(f)

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert saved == metrics  # ensure persisted matches return
