import os
import sys
import logging

import mlflow
import mlflow.transformers
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from config import (
    MODEL_DIR,
    MODEL_NAME,
    PROJECT_NAME,
    REGISTRY_EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)


load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(PROJECT_NAME)


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    ex_name = mlflow.set_experiment(REGISTRY_EXPERIMENT_NAME)

    logger.info(
        "registration_started | project=%s | source_model=%s | local_model_dir=%s",
        PROJECT_NAME,
        MODEL_NAME,
        MODEL_DIR,
    )

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Local model directory not found: {MODEL_DIR}. Run download_model.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    input_example = ["This service is fast and reliable."]

    with mlflow.start_run(run_name="register-local-hf-model", experiment_id=ex_name.experiment_id):
        mlflow.log_param("project_name", PROJECT_NAME)
        mlflow.log_param("source_model_name", MODEL_NAME)
        mlflow.log_param("local_model_dir", str(MODEL_DIR))

        model_info = mlflow.transformers.log_model(
            transformers_model=sentiment_pipeline,
            name="hf-model",
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        logger.info(
            "registration_completed | model_uri=%s | registered_model_name=%s",
            model_info.model_uri,
            REGISTERED_MODEL_NAME,
        )


if __name__ == "__main__":
    main()
