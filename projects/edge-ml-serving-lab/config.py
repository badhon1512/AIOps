from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
TEMP_DIR = ARTIFACTS_DIR / "tmp"
LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_FILE = LOG_DIR / "inference.log"

PROJECT_NAME = "edge-ml-serving-lab"
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
REGISTRY_EXPERIMENT_NAME = "edge-model-registry"
EVALUATION_EXPERIMENT_NAME = "edge-model-evaluation"
REGISTERED_MODEL_NAME = "edge-sentiment-distilbert-model"
MODEL_VERSION = "1"
APP_HOST = "127.0.0.1"
APP_PORT = 8000
APP_RELOAD = False
