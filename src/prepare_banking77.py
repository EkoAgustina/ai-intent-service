import os
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_split_id(df, prefix):
    """
    Membuat ID unik untuk setiap data pada masing-masing subset.
    Untuk test suite utama, prefix yang digunakan adalah TC.
    """
    df = df.reset_index(drop=True).copy()
    df["test_id"] = [f"{prefix}_{i+1:05d}" for i in range(len(df))]
    return df


def convert_to_research_schema(df):
    """
    Mengubah struktur asli Banking77 menjadi struktur yang sesuai
    dengan rancangan Bab 3.5 tesis.
    """
    return df.rename(
        columns={
            "text": "input_request",
            "label": "expected_label_id",
            "label_name": "expected_label_name"
        }
    )[[
        "test_id",
        "input_request",
        "expected_label_id",
        "expected_label_name"
    ]]


def main():
    print("Loading Banking77 dataset...")

    dataset = load_dataset("PolyAI/banking77")

    train_df_original = pd.DataFrame(dataset["train"])
    test_df_original = pd.DataFrame(dataset["test"])

    full_df = pd.concat(
        [train_df_original, test_df_original],
        ignore_index=True
    )

    full_df = full_df[["text", "label"]].copy()

    label_names = dataset["train"].features["label"].names

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in id2label.items()}

    full_df["label_name"] = full_df["label"].map(id2label)

    print(f"Total data: {len(full_df)}")
    print(f"Total intent: {len(label_names)}")

    # Split 70% training dan 30% temporary
    train_data, temp_data = train_test_split(
        full_df,
        test_size=0.30,
        random_state=42,
        stratify=full_df["label"]
    )

    # Split temporary menjadi 15% validation dan 15% test
    validation_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=42,
        stratify=temp_data["label"]
    )

    # Tambahkan ID unik.
    # TR dan VAL dipakai untuk menjaga konsistensi data eksperimen.
    # TC dipakai untuk test suite utama sesuai Bab 3.5.
    train_data = add_split_id(train_data, "TR")
    validation_data = add_split_id(validation_data, "VAL")
    test_data = add_split_id(test_data, "TC")

    # Ubah semua subset ke skema penelitian.
    train_research = convert_to_research_schema(train_data)
    validation_research = convert_to_research_schema(validation_data)
    test_suite = convert_to_research_schema(test_data)

    # Simpan dataset dengan struktur yang konsisten dengan Bab 3.5.
    train_research.to_csv(
        os.path.join(OUTPUT_DIR, "banking77_train.csv"),
        index=False
    )

    validation_research.to_csv(
        os.path.join(OUTPUT_DIR, "banking77_validation.csv"),
        index=False
    )

    test_suite.to_csv(
        os.path.join(OUTPUT_DIR, "banking77_test_suite.csv"),
        index=False
    )

    # Simpan label mapping untuk training model dan API.
    with open(os.path.join(OUTPUT_DIR, "id2label.json"), "w") as f:
        json.dump(id2label, f, indent=4)

    with open(os.path.join(OUTPUT_DIR, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=4)

    summary = {
        "dataset": "Banking77",
        "total_data": len(full_df),
        "total_intent": len(label_names),
        "split_ratio": {
            "training": "70%",
            "validation": "15%",
            "test": "15%"
        },
        "training_data": len(train_research),
        "validation_data": len(validation_research),
        "test_suite_data": len(test_suite),
        "test_case_schema": [
            "test_id",
            "input_request",
            "expected_label_id",
            "expected_label_name"
        ],
        "test_suite_file": "banking77_test_suite.csv"
    }

    with open(os.path.join(OUTPUT_DIR, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\nDataset preparation completed.")
    print(json.dumps(summary, indent=4))

    print("\nSample test suite:")
    print(test_suite.head())


if __name__ == "__main__":
    main()