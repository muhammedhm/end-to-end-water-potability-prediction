import mlflow
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import threading


# ============================================================
# MLflow Configuration
# ============================================================

mlflow.set_tracking_uri(
    "https://dagshub.com/muhammedhm/"
    "end-to-end-water-potability-prediction.mlflow"
)

MODEL_NAME = "water_potability_model"


class PredictionApp(tk.Tk):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Window Configuration
        # ----------------------------------------------------

        self.title("Water Quality Prediction")
        self.geometry("450x600")
        self.configure(bg="#eaeaea")

        # ----------------------------------------------------
        # Input Frame
        # ----------------------------------------------------

        self.input_frame = tk.Frame(
            self,
            bg="#ffffff",
            padx=20,
            pady=20
        )

        self.input_frame.pack(
            pady=20,
            padx=20,
            fill="both",
            expand=True
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title_label = tk.Label(
            self.input_frame,
            text="Water Quality Prediction",
            font=("Helvetica", 16, "bold"),
            bg="#ffffff"
        )

        title_label.grid(
            row=0,
            column=0,
            columnspan=2,
            pady=10
        )

        # ----------------------------------------------------
        # Input Fields
        # ----------------------------------------------------

        self.create_input_fields()

        # ----------------------------------------------------
        # Predict Button
        # ----------------------------------------------------

        self.predict_button = tk.Button(
            self,
            text="Predict",
            command=self.run_prediction_thread,
            bg="#4CAF50",
            fg="white",
            font=("Helvetica", 12, "bold"),
            width=15
        )

        self.predict_button.pack(
            pady=15
        )

        # ----------------------------------------------------
        # Status Label
        # ----------------------------------------------------

        self.status_label = tk.Label(
            self,
            text="Loading model...",
            bg="#eaeaea",
            font=("Helvetica", 10)
        )

        self.status_label.pack(
            pady=5
        )

        # ----------------------------------------------------
        # Load Model
        # ----------------------------------------------------

        self.loaded_model = self.load_model()

        if self.loaded_model:
            self.status_label.config(
                text="Model loaded successfully"
            )
        else:
            self.status_label.config(
                text="Model loading failed"
            )


    # ========================================================
    # Create Input Fields
    # ========================================================

    def create_input_fields(self):

        self.inputs = {}

        labels = [
            "pH",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity"
        ]

        for idx, label in enumerate(labels):

            lbl = tk.Label(
                self.input_frame,
                text=label,
                bg="#ffffff",
                font=("Helvetica", 11)
            )

            lbl.grid(
                row=idx + 1,
                column=0,
                sticky="e",
                pady=5,
                padx=10
            )

            entry = tk.Entry(
                self.input_frame,
                width=25,
                font=("Helvetica", 11)
            )

            entry.grid(
                row=idx + 1,
                column=1,
                pady=5
            )

            self.inputs[label] = entry


    # ========================================================
    # Load MLflow Model
    # ========================================================

    def load_model(self):

        try:

            client = mlflow.tracking.MlflowClient()

            versions = client.get_latest_versions(
                MODEL_NAME,
                stages=["Production"]
            )

            if not versions:

                messagebox.showerror(
                    "Model Error",
                    "No model found in Production stage."
                )

                return None


            latest_version = versions[0].version

            # Correct Model Registry URI
            model_uri = (
                f"models:/{MODEL_NAME}/{latest_version}"
            )

            print(
                f"Loading model from: {model_uri}"
            )

            loaded_model = mlflow.pyfunc.load_model(
                model_uri
            )

            print(
                "Model loaded successfully."
            )

            return loaded_model


        except Exception as e:

            print(
                f"Model loading error: {e}"
            )

            messagebox.showerror(
                "Model Error",
                f"Error loading model:\n{e}"
            )

            return None


    # ========================================================
    # Run Prediction Thread
    # ========================================================

    def run_prediction_thread(self):

        if self.loaded_model is None:

            messagebox.showerror(
                "Error",
                "Model is not loaded."
            )

            return

        self.predict_button.config(
            state="disabled",
            text="Predicting..."
        )

        thread = threading.Thread(
            target=self.make_prediction,
            daemon=True
        )

        thread.start()


    # ========================================================
    # Make Prediction
    # ========================================================

    def make_prediction(self):

        try:

            # ------------------------------------------------
            # Collect Input Data
            # ------------------------------------------------

            input_data = {
                "ph": [
                    float(self.inputs["pH"].get())
                ],

                "Hardness": [
                    float(self.inputs["Hardness"].get())
                ],

                "Solids": [
                    float(self.inputs["Solids"].get())
                ],

                "Chloramines": [
                    float(
                        self.inputs["Chloramines"].get()
                    )
                ],

                "Sulfate": [
                    float(self.inputs["Sulfate"].get())
                ],

                "Conductivity": [
                    float(
                        self.inputs["Conductivity"].get()
                    )
                ],

                "Organic_carbon": [
                    float(
                        self.inputs["Organic_carbon"].get()
                    )
                ],

                "Trihalomethanes": [
                    float(
                        self.inputs["Trihalomethanes"].get()
                    )
                ],

                "Turbidity": [
                    float(
                        self.inputs["Turbidity"].get()
                    )
                ]
            }


            # ------------------------------------------------
            # Convert to DataFrame
            # ------------------------------------------------

            data = pd.DataFrame(
                input_data
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = self.loaded_model.predict(
                data
            )

            result = prediction[0]

            print(
                f"Prediction result: {result}"
            )


            # ------------------------------------------------
            # Show Result
            # ------------------------------------------------

            if result == 1:

                self.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Prediction Result",
                        "Water is Potable."
                    )
                )

            else:

                self.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Prediction Result",
                        "Water is Not Potable."
                    )
                )


        except ValueError:

            self.after(
                0,
                lambda: messagebox.showerror(
                    "Input Error",
                    "Please enter valid numeric values "
                    "for all fields."
                )
            )


        except Exception as e:

            print(
                f"Prediction error: {e}"
            )

            self.after(
                0,
                lambda: messagebox.showerror(
                    "Prediction Error",
                    f"Error during prediction:\n{e}"
                )
            )


        finally:

            self.after(
                0,
                lambda: self.predict_button.config(
                    state="normal",
                    text="Predict"
                )
            )


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    app = PredictionApp()

    app.mainloop()