import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
    response_time: float


print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded on device: {device}")


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

    response_time = time.time() - start_time

    return {
        "input_text": request.text,
        "predicted_label_id": predicted_label_id,
        "predicted_label_name": predicted_label_name,
        "response_time": response_time
    }