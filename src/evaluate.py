import argparse
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import load_config


def evaluate(config_path="configs/config.yaml"):
    config = load_config(config_path)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-exp")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load test data
    test_path = "data/processed/test.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}. Please run train.py first.")

    test_df = pd.read_csv(test_path)
    target_col = config["data"]["target"]
    feature_cols = config["features"]["numeric"] + config["features"]["categorical"]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Map target if needed
    unique_vals = set(y_test.dropna().unique())
    if unique_vals in ({"Yes", "No"}, {"yes", "no"}):
        mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
        y_test = y_test.map(mapping).astype(int)

    # Find run ID from .last_run_id or latest run
    run_id = None
    if os.path.exists(".last_run_id"):
        with open(".last_run_id", "r", encoding="utf-8") as f:
            run_id = f.read().strip()

    if run_id:
        model_uri = f"runs:/{run_id}/model"
    else:
        model_uri = "models:/ChurnClassifier/latest"

    print(f"Loading model from {model_uri}...")
    model = mlflow.sklearn.load_model(model_uri)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # If y_pred is string ('Yes'/'No'), map to int
    if set(pd.Series(y_pred).unique()).issubset({"Yes", "No", "yes", "no"}):
        mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
        y_pred = pd.Series(y_pred).map(mapping).values

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("--- Test Evaluation Results ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # Generate figures
    os.makedirs("artifacts", exist_ok=True)

    # Confusion Matrix
    fig_cm, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix (Test Set)")
    fig_cm.tight_layout()
    fig_cm.savefig("artifacts/confusion_matrix.png", dpi=150)
    plt.close(fig_cm)

    # ROC Curve
    fig_roc, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="ChurnClassifier")
    ax.set_title("ROC Curve (Test Set)")
    fig_roc.tight_layout()
    fig_roc.savefig("artifacts/roc_curve.png", dpi=150)
    plt.close(fig_roc)

    # Precision-Recall Curve
    fig_pr, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name="ChurnClassifier")
    ax.set_title("Precision-Recall Curve (Test Set)")
    fig_pr.tight_layout()
    fig_pr.savefig("artifacts/pr_curve.png", dpi=150)
    plt.close(fig_pr)

    # Save sample predictions for error analysis
    preds_df = X_test.copy()
    preds_df["actual"] = y_test
    preds_df["predicted"] = y_pred
    preds_df["probability"] = y_proba
    preds_df.head(100).to_csv("artifacts/predictions_sample.csv", index=False)

    # Log to MLflow
    eval_run_id = run_id if run_id else None
    with mlflow.start_run(run_id=eval_run_id):
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.log_artifacts("artifacts", artifact_path="evaluation_artifacts")

    print("Evaluation artifacts successfully logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    evaluate(args.config)
