import os
import json
import numpy as np
import pandas as pd
import torch
import evaluate

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


DATA_DIR = "data/processed"
MODEL_OUTPUT_DIR = "model/distilbert-banking77"

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128


def load_csv_dataset(path):
    df = pd.read_csv(path)

    df = df[[
        "input_request",
        "expected_label_id"
    ]].copy()

    df = df.rename(
        columns={
            "input_request": "text",
            "expected_label_id": "label"
        }
    )

    return Dataset.from_pandas(df)


def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    train_dataset = load_csv_dataset(os.path.join(DATA_DIR, "banking77_train.csv"))
    val_dataset = load_csv_dataset(os.path.join(DATA_DIR, "banking77_validation.csv"))

    with open(os.path.join(DATA_DIR, "id2label.json"), "r") as f:
        id2label = json.load(f)

    with open(os.path.join(DATA_DIR, "label2id.json"), "r") as f:
        label2id = json.load(f)

    # JSON menyimpan key sebagai string, ubah kembali menjadi int
    id2label = {int(k): v for k, v in id2label.items()}

    num_labels = len(id2label)

    print(f"Loading tokenizer and model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy.compute(
            predictions=predictions,
            references=labels
        )

        f1_macro = f1.compute(
            predictions=predictions,
            references=labels,
            average="macro"
        )

        return {
            "accuracy": acc["accuracy"],
            "macro_f1": f1_macro["f1"]
        }

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=False
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
)

    print("Starting fine-tuning...")
    trainer.train()

    print("Evaluating model on validation set...")
    eval_result = trainer.evaluate()

    print("Saving model, tokenizer, and evaluation result...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    with open(os.path.join(MODEL_OUTPUT_DIR, "eval_result.json"), "w") as f:
        json.dump(eval_result, f, indent=4)

    with open(os.path.join(MODEL_OUTPUT_DIR, "id2label.json"), "w") as f:
        json.dump(id2label, f, indent=4)

    with open(os.path.join(MODEL_OUTPUT_DIR, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=4)

    print("\nFine-tuning completed.")
    print(json.dumps(eval_result, indent=4))


if __name__ == "__main__":
    main()