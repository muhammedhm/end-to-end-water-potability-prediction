import json
from mlflow.tracking import MlflowClient
import dagshub
import mlflow


# ============================================================
# MLflow / DagsHub Configuration
# ============================================================

MLFLOW_TRACKING_URI = (
    "https://dagshub.com/"
    "muhammedhm/end-to-end-water-potability-prediction.mlflow"
)

REGISTERED_MODEL_NAME = "water_potability_model"


dagshub.init(
    repo_owner="muhammedhm",
    repo_name="end-to-end-water-potability-prediction",
    mlflow=True,
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


client = MlflowClient()

# ============================================================
# Load Model Information
# ============================================================

with open(
    "reports/run_info.json",
    "r",
) as file:

    run_info = json.load(file)


run_id = run_info["run_id"]

model_uri = run_info["model_uri"]


print(
    f"Run ID: {run_id}"
)

print(
    f"Model URI: {model_uri}"
)

print(
    f"Registering as: "
    f"{REGISTERED_MODEL_NAME}"
)


# ============================================================
# Register Model
# ============================================================

try:

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    print(
        "\nSuccessfully registered model!"
    )

    print(
        f"Model name: "
        f"{registered_model.name}"
    )

    print(
        f"Model version: "
        f"{registered_model.version}"
    )


    new_stage = "Production"

    client.transition_model_version_stage(
        name=registered_model.name,
        version=registered_model.version,
        stage=new_stage,
        archive_existing_versions=True
    )

except Exception as e:

    raise Exception(
        f"Model registration failed: {e}"
    )