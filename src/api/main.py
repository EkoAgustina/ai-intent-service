import os
import time
import torch
import logging

from datetime import datetime, UTC
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# Logging Configuration
# ============================================================

logger = logging.getLogger("api")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)


# ============================================================
# Model Configuration
# ============================================================

MODEL_DIR = os.getenv("MODEL_DIR")

if not MODEL_DIR:
    raise RuntimeError("MODEL_DIR environment variable is not set")

MAX_LENGTH = 128

print(f"Model directory: {MODEL_DIR}")


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Banking77 Intent Classification API",
    description="Transformer-based intent classification service for Banking77 dataset.",
    version="1.0.0"
)


# ============================================================
# Request / Response Schema
# ============================================================

class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    input_text: str
    predicted_label_id: int
    predicted_label_name: str


# ============================================================
# Load Model
# ============================================================

print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)


# ============================================================
# Device Configuration
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()


# ============================================================
# Model Information
# ============================================================

MODEL_TYPE = model.config.model_type

MODEL_ARCHITECTURE = model.__class__.__name__

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    os.path.basename(os.path.normpath(MODEL_DIR))
)


# ============================================================
# Startup Model Logging
# ============================================================

logger.info(
    "MODEL_LOADED name=%s model_type=%s architecture=%s device=%s",
    MODEL_NAME,
    MODEL_TYPE,
    MODEL_ARCHITECTURE,
    device
)

logger.info(
    "MODEL_CONFIG num_labels=%s",
    model.config.num_labels
)

print(f"Model loaded on device: {device}")
print(f"Model name: {MODEL_NAME}")
print(f"Model type: {MODEL_TYPE}")
print(f"Model architecture: {MODEL_ARCHITECTURE}")
print(f"Number of threads: {torch.get_num_threads()}")
print(f"Number of interop threads: {torch.get_num_interop_threads()}")
print(f"Tokenizer type: {type(tokenizer)}")


# ============================================================
# Request Logging Middleware
# ============================================================

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000

    timestamp = (
        datetime.now(UTC)
        .strftime("%Y/%m/%dT%H:%M:%S.%f")[:-3] + "Z"
    )

    client_host = (
        request.client.host
        if request.client
        else "-"
    )

    client_port = (
        request.client.port
        if request.client
        else "-"
    )

    logger.info(
        '%s - model=%s model_type=%s architecture=%s - '
        '%s:%s - "%s %s %s" %s - %.2fms',
        timestamp,
        MODEL_NAME,
        MODEL_TYPE,
        MODEL_ARCHITECTURE,
        client_host,
        client_port,
        request.method,
        request.url.path,
        request.scope.get("http_version", "1.1"),
        response.status_code,
        duration_ms
    )

    return response


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "model_architecture": MODEL_ARCHITECTURE,
        "device": str(device),
        "num_labels": model.config.num_labels
    }


# ============================================================
# Prediction
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    start_time = time.time()

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    # --------------------------------------------------------
    # Move input tensors to device
    # --------------------------------------------------------

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        predicted_label_id = torch.argmax(
            logits,
            dim=-1
        )

    # --------------------------------------------------------
    # Convert tensor to integer
    # --------------------------------------------------------

    predicted_label_id = predicted_label_id.item()

    # --------------------------------------------------------
    # Get label name
    # --------------------------------------------------------

    predicted_label_name = model.config.id2label[
        predicted_label_id
    ]

    # --------------------------------------------------------
    # Prediction logging
    # --------------------------------------------------------

    duration_ms = (
        time.time() - start_time
    ) * 1000

    logger.info(
        "PREDICTION model=%s model_type=%s "
        "label_id=%s label_name=%s duration=%.2fms",
        MODEL_NAME,
        MODEL_TYPE,
        predicted_label_id,
        predicted_label_name,
        duration_ms
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "input_text": request.text,
        "predicted_label_id": predicted_label_id,
        "predicted_label_name": predicted_label_name
    }
