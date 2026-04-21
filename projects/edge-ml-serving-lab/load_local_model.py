import os
import logging

import mlflow
import mlflow.transformers
from dotenv import load_dotenv

from config import MODEL_VERSION, PROJECT_NAME, REGISTERED_MODEL_NAME


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(PROJECT_NAME)


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
    logger.info("registry_load_started | project=%s | model_uri=%s", PROJECT_NAME, model_uri)

    sentiment_pipeline = mlflow.transformers.load_model(model_uri)

    text = "This project is simple, fast, and very useful."
    result = sentiment_pipeline(text)

    logger.info("registry_load_completed | input_text=%s | prediction=%s", text, result)


if __name__ == "__main__":
    main()
