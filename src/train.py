import argparse
import os

import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.pipeline import build_pipeline
from src.utils import load_config, load_data, split_data


def train(config_path="configs/config.yaml"):
    config = load_config(config_path)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-exp")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"Loading data from {config['data']['csv_path']}...")
    X, y = load_data(config)
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    # Save test set for reproducible evaluation
    os.makedirs("data/processed", exist_ok=True)
    test_df = X_test.copy()
    test_df[config["data"]["target"]] = y_test
    test_df.to_csv("data/processed/test.csv", index=False)

    numeric_cols = config["features"]["numeric"]
    categorical_cols = config["features"]["categorical"]
    model_type = config["model"]["type"]

    pipeline = build_pipeline(numeric_cols, categorical_cols, model_type=model_type)

    # Param grid for GridSearchCV (model__ prefix for pipeline)
    raw_params = config["model"].get("params", {})
    param_grid = {f"model__{k}": v for k, v in raw_params.items()}

    cv_splits = config["cv"].get("n_splits", 5)
    scoring = config["cv"].get("scoring", "roc_auc")
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=config["data"]["random_state"]
    )

    with mlflow.start_run(run_name=f"{model_type}-training-tuning") as run:
        print(f"Starting GridSearchCV with {cv_splits}-fold CV...")
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True
        )

        grid.fit(X_train, y_train)

        best_score = grid.best_score_
        best_params = grid.best_params_

        print(f"Best CV {scoring}: {best_score:.4f}")
        print(f"Best parameters: {best_params}")

        # Log params & metrics
        for p_name, p_val in best_params.items():
            mlflow.log_param(p_name, p_val)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric(f"best_cv_{scoring}", float(best_score))

        # Log and register the best model using cloudpickle
        mlflow.sklearn.log_model(
            sk_model=grid.best_estimator_,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name="ChurnClassifier"
        )

        # Save run ID to file for evaluate.py
        with open(".last_run_id", "w", encoding="utf-8") as f:
            f.write(run.info.run_id)

        print(f"Model saved and logged to MLflow Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    train(args.config)
