# Import necessary libraries for data manipulation, ML models, metrics, visualization, logging, and tracking
import pandas as pd
import numpy as np
import mlflow
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import infer_signature
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import dagshub
from sklearn.model_selection import train_test_split

# Initialize DagsHub integration and set the experiment for MLflow tracking
mlflow.set_tracking_uri("https://dagshub.com/muhammedhm/end-to-end-water-potability-prediction.mlflow")

dagshub.init(repo_owner='muhammedhm', repo_name='end-to-end-water-potability-prediction', mlflow=True)
mlflow.set_experiment("Experiment 4")  # Set the experiment name
# Registry URI for model management

# Load the dataset from CSV file and split into training and testing sets
data = pd.read_csv(r"data/raw/water_potability.csv")

X = data.drop(columns=["Potability"])
y = data["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define a function to fill missing values with the median value for each column
# Handle missing values
imputer = SimpleImputer(strategy="mean")

# Keep original indices
train_index = X_train.index
test_index = X_test.index

# Transform and convert back to DataFrames
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns,
    index=train_index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns,
    index=test_index
)


# Verify NaNs are removed
print("NaNs in training data:", np.isnan(X_train).sum())
print("NaNs in testing data:", np.isnan(X_test).sum())



# Define the Random Forest Classifier model and the parameter distribution for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],  # Different values of n_estimators to try
    'max_depth': [None, 4, 5, 6, 10],  # Different max_depth values to explore
}

# Perform RandomizedSearchCV to find the best hyperparameters for the Random Forest model
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)

# Start a parent MLflow run to track the overall experiment
with mlflow.start_run(run_name="Water Potability Models Experiment"):
            random_search.fit(X_train, y_train)

            # Log the parameters and mean test scores for each combination tried
            for i in range(len(random_search.cv_results_['params'])):
                with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
                    mlflow.log_params(random_search.cv_results_['params'][i])  # Log the parameters
                    mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])  # Log the mean test score

            # Print the best hyperparameters found by RandomizedSearchCV
            print("Best parameters found: ", random_search.best_params_)

            # Log the best parameters in MLflow
            mlflow.log_params(random_search.best_params_)

            # Train the model using the best parameters identified by RandomizedSearchCV
            best_rf = random_search.best_estimator_
            best_rf.fit(X_train, y_train)

            # Save the trained model to a file for later use
            model = pickle.load(open('model.pkl', "rb"))

            # Make predictions on the test set using the loaded model
            y_pred = model.predict(X_test)

            # Calculate and print performance metrics: accuracy, precision, recall, and F1-score
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log performance metrics into MLflow for tracking
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1 score", f1)

            # Log the training and testing data as inputs in MLflow
            train_df = mlflow.data.from_pandas(
                pd.concat(
                    [
                        X_train,
                        pd.Series(y_train, index=train_index, name="Potability")
                    ],
                    axis=1
                )
            )
            test_df = mlflow.data.from_pandas(
                pd.concat(
                    [
                        X_test,
                        pd.Series(y_test, index=test_index, name="Potability")
                    ],
                    axis=1
                )
            )

            mlflow.log_input(train_df, "train")  # Log training data
            mlflow.log_input(test_df, "test")  # Log test data

            # Log the current script file as an artifact in MLflow
            mlflow.log_artifact(__file__)

            # Infer the model signature using the test features and predictions
            sign = infer_signature(X_test, random_search.best_estimator_.predict(X_test))
            
            # Log the trained model in MLflow with its signature
            mlflow.sklearn.log_model(random_search.best_estimator_, "Best Model", signature=sign)

            # Print the calculated performance metrics to the console for review
            print("Accuracy: ", acc)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1-score: ", f1)