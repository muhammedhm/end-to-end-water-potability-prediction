import mlflow
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel


app = FastAPI(
    title="Water Potability Prediction API",
    description="API for predicting water potability using a trained machine learning model.",
)


# MLflow tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = "muhammedhm"
repo_name = "end-to-end-water-potability-prediction"

mlflow_tracking_uri = (
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)

mlflow.set_tracking_uri(mlflow_tracking_uri)

# load model
def load_model():
    model_uri = "models:/water_potability_model/Production"

    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise Exception(
            f"Error loading latest model from MLflow: {e}"
        )

model = load_model()


# Input schema for the API
class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
def home():
    return {"message": "Welcome to the Water Potability Prediction API!"}


@app.post("/predict")
def predict(sample: WaterSample):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame({
            "ph": [sample.ph],
            "Hardness": [sample.Hardness],
            "Solids": [sample.Solids],
            "Chloramines": [sample.Chloramines],
            "Sulfate": [sample.Sulfate],
            "Conductivity": [sample.Conductivity],
            "Organic_carbon": [sample.Organic_carbon],
            "Trihalomethanes": [sample.Trihalomethanes],
            "Turbidity": [sample.Turbidity],
        })

        # Make prediction
        prediction = model.predict(input_data)

        # Return the prediction result
        {"potability_prediction": int(prediction[0])}

        if prediction[0] == 1:
            return {"potability_prediction": "Potable"}
        else:
            return {"potability_prediction": "Not Potable"}

    except Exception as e:
        raise Exception(f"Error during prediction: {e}")