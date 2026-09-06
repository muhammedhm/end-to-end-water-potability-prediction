import mlflow
import pandas as pd


# ============================================================
# MLflow Configuration
# ============================================================

mlflow.set_tracking_uri(
    "https://dagshub.com/muhammedhm/"
    "end-to-end-water-potability-prediction.mlflow"
)


# Registered model name
model_name = "water_potability_model"


try:

    # Create MLflow client
    client = mlflow.tracking.MlflowClient()


    # Get latest Production model
    versions = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )


    if versions:

        latest_version = versions[0].version

        print(
            f"Latest version in Production: "
            f"{latest_version}"
        )


        # ====================================================
        # CORRECT MODEL REGISTRY URI
        # ====================================================

        model_uri = (
            f"models:/{model_name}/{latest_version}"
        )

        print("Model URI:", model_uri)


        # Load model
        loaded_model = mlflow.pyfunc.load_model(
            model_uri
        )

        print(
            f"Model successfully loaded from "
            f"{model_uri}"
        )


        # ====================================================
        # Input Data
        # ====================================================

        data = pd.DataFrame({
            "ph": [3.71608],
            "Hardness": [204.89045],
            "Solids": [20791.318981],
            "Chloramines": [7.300212],
            "Sulfate": [368.516441],
            "Conductivity": [564.308654],
            "Organic_carbon": [10.379783],
            "Trihalomethanes": [86.99097],
            "Turbidity": [2.963135],
        })


        # ====================================================
        # Prediction
        # ====================================================

        prediction = loaded_model.predict(data)

        print("Prediction:", prediction)


    else:

        print(
            "No model found in the "
            "'Production' stage."
        )


except Exception as e:

    print(
        f"Error fetching model: {e}"
    )