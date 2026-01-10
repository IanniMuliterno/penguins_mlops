import json
from pathlib import Path
from typing import Tuple

import joblib
from palmerpenguins import load_penguins
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


ARTIFACTS_DIR = Path("model_artifacts")
MODEL_PATH = ARTIFACTS_DIR / "penguin_classifier_model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def load_data(test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Load and split the Palmer Penguins dataset."""
    df = load_penguins()
    df.dropna(inplace=True)

    X = df.drop("species", axis=1)
    y = df["species"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def evaluate(model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH):
    """Evaluate the trained model and persist metrics to metrics.json."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the model first.")

    model = joblib.load(model_path)
    _, X_test, _, y_test = load_data()

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    results = evaluate()
    print(json.dumps(results, indent=2))
