import os
import logging

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ARTIFACT_DIR, MODEL_DIR, MODEL_NAME, PROJECT_NAME


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(PROJECT_NAME)


def main():
    logger.info("download_started | project=%s | model=%s | artifact_dir=%s", PROJECT_NAME, MODEL_NAME, ARTIFACT_DIR)

    os.makedirs(MODEL_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

    logger.info("download_completed | model_dir=%s", MODEL_DIR)


if __name__ == "__main__":
    main()
