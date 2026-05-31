import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import time
from datetime import datetime, UTC

logger = logging.getLogger("api")

# konfigurasi logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)


MODEL_DIR = "model/distilbert-banking77"
MAX_LENGTH = 128

app = FastAPI(
    title="Banking77 Intent Classification API",
    description="DistilBERT-based intent classification service for Banking77 dataset.",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    input_text: str
    predicted_label_id: int
    predicted_label_name: str


print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded on device: {device}")

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000

    timestamp = (
        datetime.now(UTC)
        .strftime("%Y/%m/%dT%H:%M:%S.%f")[:-3] + "Z"
    )

    client_host = request.client.host if request.client else "-"
    client_port = request.client.port if request.client else "-"

    logger.info(
        '%s - %s:%s - "%s %s %s" %s - %.2fms',
        timestamp,
        client_host,
        client_port,
        request.method,
        request.url.path,
        request.scope.get("http_version", "1.1"),
        response.status_code,
        duration_ms
    )

    return response

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.time()

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label_id = torch.argmax(logits, dim=-1)

    predicted_label_id = predicted_label_id.item()
    predicted_label_name = model.config.id2label[predicted_label_id]

    return {
        "input_text": request.text,
        "predicted_label_id": predicted_label_id,
        "predicted_label_name": predicted_label_name,
    }