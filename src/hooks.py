import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune transformer model"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base Hugging Face model"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model"
    )

    return parser.parse_args()