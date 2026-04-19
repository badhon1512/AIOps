import mlflow

with mlflow.start_run(run_name="Nester-Run") as run:
    print("Run ID:", run.info.run_id)

    with mlflow.start_run(run_name="Nested-Run", nested=True) as nested_run:
        print("Nested Run ID:", nested_run.info.run_id)
    with mlflow.start_run(run_name="Nested-Run-2", nested=True) as nested_run_2:
        print("Nested Run 2 ID:", nested_run_2.info.run_id)