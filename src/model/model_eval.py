import json
import pickle

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# MLflow / DagsHub Configuration
# ============================================================

# MLFLOW_TRACKING_URI = (
#     "https://dagshub.com/"
#     "muhammedhm/end-to-end-water-potability-prediction.mlflow"
# )

# EXPERIMENT_NAME = "DVC PIPELINE 1"

MODEL_NAME = "Best_Model"


# dagshub.init(
#     repo_owner="muhammedhm",
#     repo_name="end-to-end-water-potability-prediction",
#     mlflow=True,
# )

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment(EXPERIMENT_NAME)

import os

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_TOKEN environment variable is not set. ")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "muhammedhm"
repo_name = "end-to-end-water-potability-prediction"
mlflow_tracking_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
mlflow.set_experiment("DVC PIPELINE 1")


# ============================================================
# Data Loading
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)

    except Exception as e:
        raise Exception(
            f"Error loading data from {filepath}: {e}"
        )


def prepare_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:

    try:
        X = data.drop(columns=["Potability"])
        y = data["Potability"]

        return X, y

    except Exception as e:
        raise Exception(
            f"Error preparing data: {e}"
        )


# ============================================================
# Model Loading
# ============================================================

def load_model(filepath: str):

    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)

        return model

    except Exception as e:
        raise Exception(
            f"Error loading model from {filepath}: {e}"
        )


# ============================================================
# Model Evaluation
# ============================================================

def evaluation_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:

    try:
        with open("params.yaml", "r") as file:
            params = yaml.safe_load(file)

        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = model.predict(X_test)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        acc = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "test_size",
            test_size,
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators,
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            acc,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred,
        )

        plt.figure(figsize=(5, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(
            f"Confusion Matrix for {model_name}"
        )

        cm_path = (
            f"confusion_matrix_"
            f"{model_name.replace(' ', '_')}.png"
        )

        plt.savefig(
            cm_path,
            bbox_inches="tight",
        )

        plt.close()

        mlflow.log_artifact(cm_path)

        # ----------------------------------------------------
        # Metrics dictionary
        # ----------------------------------------------------

        metrics_dict = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        return metrics_dict

    except Exception as e:
        raise Exception(
            f"Error evaluating model: {e}"
        )


# ============================================================
# Save Metrics
# ============================================================

def save_metrics(
    metrics: dict,
    metrics_path: str,
) -> None:

    try:
        with open(metrics_path, "w") as file:
            json.dump(
                metrics,
                file,
                indent=4,
            )

    except Exception as e:
        raise Exception(
            f"Error saving metrics to {metrics_path}: {e}"
        )


# ============================================================
# Main
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        test_data_path = (
            "./data/processed/test_processed.csv"
        )

        model_path = "models/model.pkl"

        metrics_path = "reports/metrics.json"

        # ----------------------------------------------------
        # Load test data
        # ----------------------------------------------------

        test_data = load_data(
            test_data_path
        )

        X_test, y_test = prepare_data(
            test_data
        )

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        model = load_model(
            model_path
        )

        # ----------------------------------------------------
        # Start MLflow run
        # ----------------------------------------------------

        with mlflow.start_run() as run:

            print(
                f"Run ID: {run.info.run_id}"
            )

            # ------------------------------------------------
            # Evaluate model
            # ------------------------------------------------

            metrics = evaluation_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name=MODEL_NAME,
            )

            # ------------------------------------------------
            # Save metrics locally
            # ------------------------------------------------

            save_metrics(
                metrics,
                metrics_path,
            )

            # ------------------------------------------------
            # Log local artifacts
            # ------------------------------------------------

            mlflow.log_artifact(
                model_path
            )

            mlflow.log_artifact(
                metrics_path
            )

            mlflow.log_artifact(
                __file__
            )

            # ------------------------------------------------
            # Create model signature
            # ------------------------------------------------

            predictions = model.predict(
                X_test
            )

            signature = infer_signature(
                X_test,
                predictions,
            )

            # ------------------------------------------------
            # Log MLflow model
            #
            # IMPORTANT:
            # Use `name`, not artifact_path.
            # MLflow 3.x recommends the model URI returned
            # by this call.
            # ------------------------------------------------

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name=MODEL_NAME,
                signature=signature,
                input_example=X_test.head(5),
            )

            print(
                f"Logged model ID: "
                f"{model_info.model_id}"
            )

            print(
                f"Logged model URI: "
                f"{model_info.model_uri}"
            )

            # ------------------------------------------------
            # Save model information for registration stage
            # ------------------------------------------------

            run_info = {
                "run_id": run.info.run_id,
                "model_name": MODEL_NAME,
                "model_id": model_info.model_id,
                "model_uri": model_info.model_uri,
            }

            reports_path = (
                "reports/run_info.json"
            )

            with open(
                reports_path,
                "w",
            ) as file:

                json.dump(
                    run_info,
                    file,
                    indent=4,
                )

            print(
                f"Run information saved to "
                f"{reports_path}"
            )

            print(
                "\nEvaluation completed successfully."
            )

            print(
                f"Accuracy:  {metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: {metrics['precision']:.4f}"
            )

            print(
                f"Recall:    {metrics['recall']:.4f}"
            )

            print(
                f"F1 Score:  {metrics['f1_score']:.4f}"
            )

    except Exception as e:

        raise Exception(
            f"An error occurred: {e}"
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()