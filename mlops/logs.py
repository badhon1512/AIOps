import mlflow
import pandas as pd
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MLOps-Experiment")

# Create a sample DataFrame
df = pd.DataFrame({
    "feature1": [1, 2, 3],
    "feature2": [4, 5, 6],
    "target": [0, 1, 0]
})

with mlflow.start_run(run_name="Logging-Example", run_id="2d3a79c48bb8406bbc5f3ea365dccb94") as run:
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.80)
    params = {"epochs": "10", "batch_size": 32, "optimizer": "adam"}
    mlflow.log_params(params)
    mlflow.log_artifact("../test.txt")

    #mlflow.log_figure()

    mlflow.log_table(df, "sample_data.json")