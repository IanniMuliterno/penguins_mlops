import mlflow.sklearn
from palmerpenguins import load_penguins
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient, log_metrics, log_params
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from features import PenguinFeatureEngineer,preprocessor
import joblib
import json
from datetime import datetime
import os

from src import logger
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "penguin_classification_experiment"
MODEL_NAME = "penguin_classifier"
# Load data
penguins_df = load_penguins()

# setup mlflow
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)
experiment = mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(max_tuning_results=None)  
# Simple feature cleaning due to low percentage of NA and the purpose of this project
penguins_df.dropna(inplace=True)

with mlflow.start_run(run_name="decision_tree_classifier") as run:
    mlflow.set_tags({"stage": "training",
                     "model": "DecisionTreeClassifier"})
    
    X = penguins_df.drop(['species'], axis=1)
    y = penguins_df['species']
    train_size = 0.7
    # Train-test split with fixed random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )
    mlflow.log_param("train_size", train_size)
    # Create full pipeline including model
    # Pipeline order: add engineered features first, then preprocess, then model
    rnd_state = 42
    full_pipeline = Pipeline([
        ("add_features", PenguinFeatureEngineer()),
        ("preprocess", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=rnd_state))
    ])
    mlflow.log_param("random_state",rnd_state)
    # Hyperparameter tuning with cross-validation
    param_grid = {
        'classifier__max_depth': [3, 5, 7, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    logger.info("Performing Grid Search with 5-Fold Cross-Validation...")
    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',  # multiclass friendly metric
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    # Get best model
    best_model = grid_search.best_estimator_

    # Cross-validation scores on best model
    cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=5, scoring='f1_macro'
    )
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Evaluate
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    mlflow.register_model(best_model, MODEL_NAME)
    mlflow.log_metrics({ "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "cv_scores_macro": [float(score) for score in cv_scores],
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "classification_report": classification_report(y_test, y_test_pred, output_dict=True)})

    logger.info("\n" + "="*60)
    logger.info("MODEL PERFORMANCE")
    logger.info("="*60)
    logger.info(f"\nBest Parameters: {grid_search.best_params_}")
    logger.info(f"\nCross-Validation F1 (macro): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    logger.info(f"Training Accuracy: {train_accuracy:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    logger.info("\n" + "-"*60)
    logger.info("CLASSIFICATION REPORT (Test Set)")
    logger.info("-"*60)
    logger.info(classification_report(y_test, y_test_pred))

    logger.info("\nConfusion Matrix (Test Set):")
    logger.info(confusion_matrix(y_test, y_test_pred))

    # Create metadata
    metadata = {
        "model_info": {
            "model_type": "DecisionTreeClassifier",
            "pipeline_steps": [step[0] for step in best_model.steps],
            "best_params": grid_search.best_params_,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "data_info": {
            "total_samples": len(penguins_df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": list(X.columns),
            "target_classes": list(y.unique()),
            "class_distribution": y.value_counts().to_dict()
        },
        "performance_metrics": {
            "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "cv_scores_macro": [float(score) for score in cv_scores],
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "classification_report": classification_report(y_test, y_test_pred, output_dict=True)
        },
        "hyperparameter_search": {
            "param_grid": param_grid,
            "cv_folds": 5,
            "best_cv_score_macro": float(grid_search.best_score_)
        }
    }

    # Save artifacts
    artifacts_dir = "model_artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(artifacts_dir, "penguin_classifier_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"\n✓ Model saved to: {model_path}")

    # Save metadata
    metadata_path = os.path.join(artifacts_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"✓ Metadata saved to: {metadata_path}")

    # Save feature names (for inference)
    feature_names_path = os.path.join(artifacts_dir, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump({"features": list(X.columns)}, f, indent=4)
    logger.info(f"✓ Feature names saved to: {feature_names_path}")

    # Save predictions for analysis
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_test_pred,
        'correct': y_test == y_test_pred
    })
    predictions_path = os.path.join(artifacts_dir, "test_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"✓ Test predictions saved to: {predictions_path}")

    logger.info("\n" + "="*60)
    logger.info("All artifacts saved successfully!")
    logger.info("="*60)

    # Example of loading the model for inference
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE: Loading Model for Inference")
    logger.info("="*60)
    loaded_model = joblib.load(model_path)
    sample_prediction = loaded_model.predict(X_test.iloc[:3])
    logger.info(f"Sample predictions: {sample_prediction}")
    logger.info(f"Actual values: {y_test.iloc[:3].values}")
