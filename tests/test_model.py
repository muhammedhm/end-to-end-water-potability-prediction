
import os
import mlflow
import pandas as pd


# ============================================================
# MLflow / DagsHub Configuration
# ============================================================

REGISTERED_MODEL_NAME = "water_potability_model"

dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_TOKEN environment variable is not set."
    )

# DagsHub authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# MLflow tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = "muhammedhm"
repo_name = "end-to-end-water-potability-prediction"

mlflow_tracking_uri = (
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)

mlflow.set_tracking_uri(mlflow_tracking_uri)


# ============================================================
# Load Production Model
# ============================================================

model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"

print("=" * 60)
print("Loading MLflow model")
print("=" * 60)

print(f"Tracking URI : {mlflow_tracking_uri}")
print(f"Model URI    : {model_uri}")


try:
    model = mlflow.pyfunc.load_model(model_uri)

    print("\nModel loaded successfully!")

except Exception as e:
    raise Exception(
        f"Error loading model from MLflow: {e}"
    )


# ============================================================
# Test Data
# ============================================================

# Water quality sample
test_data = pd.DataFrame(
    [
        {
            "ph": 7.0,
            "Hardness": 200.0,
            "Solids": 20000.0,
            "Chloramines": 7.0,
            "Sulfate": 350.0,
            "Conductivity": 400.0,
            "Organic_carbon": 10.0,
            "Trihalomethanes": 70.0,
            "Turbidity": 4.0,
        }
    ]
)


print("\nTest input:")
print(test_data)


# ============================================================
# Prediction
# ============================================================

try:

    prediction = model.predict(test_data)

    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)

    print(f"Prediction: {prediction}")

except Exception as e:

    raise Exception(
        f"Prediction failed: {e}"
    )


# ============================================================
# Result
# ============================================================

if hasattr(prediction, "__len__") and len(prediction) > 0:

    result = prediction[0]

    if result == 1:
        print("\nWater is predicted to be POTABLE.")

    else:
        print("\nWater is predicted to be NOT POTABLE.")


print("\nModel test completed successfully.")

