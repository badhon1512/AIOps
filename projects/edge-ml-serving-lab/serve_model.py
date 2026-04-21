import os
import time
import logging

import mlflow
import mlflow.transformers
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import (
    APP_HOST,
    APP_PORT,
    APP_RELOAD,
    LOG_DIR,
    LOG_FILE,
    MODEL_VERSION,
    PROJECT_NAME,
    REGISTERED_MODEL_NAME,
    TEMP_DIR,
)


load_dotenv()

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)

logger = logging.getLogger(PROJECT_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
sentiment_pipeline = mlflow.transformers.load_model(model_uri)

app = FastAPI(title=PROJECT_NAME)


class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "project": PROJECT_NAME,
        "model_uri": model_uri,
        "model_version": MODEL_VERSION,
        "status": "running",
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    start_time = time.time()

    try:
        result = sentiment_pipeline(request.text)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "prediction_success | model=%s | version=%s | latency_ms=%s | input=%s | input_length=%s | prediction=%s",
            REGISTERED_MODEL_NAME,
            MODEL_VERSION,
            latency_ms,
            request.text,
            len(request.text),
            result,
        )

        return {
            "input": request.text,
            "prediction": result,
            "model_name": REGISTERED_MODEL_NAME,
            "model_version": MODEL_VERSION,
            "latency_ms": latency_ms,
        }
    except Exception as error:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.exception(
            "prediction_failed | model=%s | version=%s | latency_ms=%s | input_length=%s",
            REGISTERED_MODEL_NAME,
            MODEL_VERSION,
            latency_ms,
            len(request.text),
        )
        raise HTTPException(status_code=500, detail=str(error))



if __name__ == "__main__":
    logger.info(
        "service_starting | project=%s | model_uri=%s | log_file=%s | host=%s | port=%s",
        PROJECT_NAME,
        model_uri,
        LOG_FILE,
        APP_HOST,
        APP_PORT,
    )
    uvicorn.run("serve_model:app", host=APP_HOST, port=APP_PORT, reload=APP_RELOAD)
