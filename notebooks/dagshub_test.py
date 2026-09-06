import mlflow
import dagshub



mlflow.set_tracking_uri("https://dagshub.com/muhammedhm/end-to-end-water-potability-prediction.mlflow")

dagshub.init(repo_owner='muhammedhm', repo_name='end-to-end-water-potability-prediction', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)