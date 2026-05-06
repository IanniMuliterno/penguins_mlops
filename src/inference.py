import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import joblib
import pandas as pd

from src import logger
from src.data_loader import normalize_model_input_schema

DEFAULT_MODEL_PATH = Path("model_artifacts/penguin_classifier_model.pkl")
DEFAULT_FEATURES_PATH = Path("model_artifacts/feature_names.json")


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """Load a trained sklearn Pipeline model from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
    return joblib.load(model_path)


def _to_dataframe(input_data: Union[pd.DataFrame, Mapping, Iterable]) -> pd.DataFrame:
    """
    Convert common input formats to a pandas DataFrame expected by the model.

    Supported formats:
      - pandas DataFrame
      - mapping/dict of feature -> value (single row)
      - iterable/list of mappings for batch predictions
    """
    if isinstance(input_data, pd.DataFrame):
        return input_data

    if isinstance(input_data, Mapping):
        return pd.DataFrame([input_data])

    if isinstance(input_data, Iterable):
        return pd.DataFrame(list(input_data))

    raise ValueError("Unsupported input type for prediction. Provide a DataFrame, mapping, or iterable of mappings.")


def load_feature_names(feature_names_path: Union[str, Path] = DEFAULT_FEATURES_PATH) -> Sequence[str]:
    """Load the ordered list of feature names used during training."""
    feature_names_path = Path(feature_names_path)
    if not feature_names_path.exists():
        raise FileNotFoundError(
            f"Feature names file not found at {feature_names_path}. "
            "Run training to generate model_artifacts/feature_names.json."
        )

    with open(feature_names_path) as f:
        data = json.load(f)

    features = data.get("features")
    if not features:
        raise ValueError(f"No 'features' key found in {feature_names_path}")
    return features


def _align_features(df: pd.DataFrame, feature_order: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for inference: {missing}")

    # Drop any unexpected columns and reorder to training-time order
    aligned = df[feature_order]
    return normalize_model_input_schema(aligned)


def predict(
    input_data: Union[pd.DataFrame, Mapping, Iterable],
    model: Any = None,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    feature_names_path: Union[str, Path] = DEFAULT_FEATURES_PATH,
):
    """Generate predictions given input data and a trained model."""
    if model is None:
        model = load_model(model_path)

    data_frame = _to_dataframe(input_data)
    feature_order = load_feature_names(feature_names_path)
    aligned = _align_features(data_frame, feature_order)
    return model.predict(aligned)


def _load_from_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {csv_path}")
    return pd.read_csv(csv_path,sep='\t')


def main():
    parser = argparse.ArgumentParser(description="Run inference using the trained penguin classifier.")
    parser.add_argument(
        "--input-csv",
        type=str,
        help="Path to a CSV file containing feature columns for prediction.",
        required=True,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained model artifact.",
    )
    parser.add_argument(
        "--feature-names-path",
        type=str,
        default=str(DEFAULT_FEATURES_PATH),
        help="Path to feature_names.json generated during training.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save predictions as JSON.",
    )
    args = parser.parse_args()

    model = load_model(args.model_path)
    df = _load_from_csv(args.input_csv)
    preds = predict(df, model=model, feature_names_path=args.feature_names_path)

    predictions = preds.tolist()
    logger.info(f"Predictions: {predictions}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"predictions": predictions}, f, indent=2)
        logger.info(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
