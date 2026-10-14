"""
Stage 2: Model Engineering
Feature engineering, training, evaluation and packaging of the model.
Metrics and the model are logged to MLflow (local file mlflow.db).

Run from the project root:
    python code/models/train_model.py
"""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- paths ---
ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
TEST_FILE = ROOT / "data" / "processed" / "test.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "model.pkl"
METRICS_FILE = MODEL_DIR / "metrics.json"
# MLflow stores runs in a local SQLite file (the folder backend is deprecated)
MLFLOW_DB_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()

TARGET = "Outcome"
PARAMS = {"C": 1.0, "max_iter": 1000, "random_state": 42}


def load_data():
    """Read train and test files created by Stage 1"""
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def build_pipeline():
    """Feature engineering + model in one object.
    StandardScaler transforms every feature to mean 0 and std 1
    (logistic regression works better with scaled features).
    Because the scaler is inside the pipeline, the API can pass raw values
    and the same transformation will be applied automatically."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**PARAMS)),
        ]
    )


def evaluate(pipeline, X_test, y_test):
    """Calculate metrics on the test data"""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    X_train, y_train, X_test, y_test = load_data()

    # MLflow writes runs to the local file mlflow.db (model files go to mlruns/)
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_experiment("diabetes_prediction")

    with mlflow.start_run():
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)

        # log to MLflow
        mlflow.log_params(PARAMS)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print("Test metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.3f}")

    # package the model into a file (this file is used by the API in Stage 3)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
    