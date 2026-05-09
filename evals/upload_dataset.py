"""Upload the local eval dataset to LangSmith.

Creates (or updates) a LangSmith dataset called "struct-agent" with examples
from dataset.jsonl. Each example has:
  - inputs: {"question": "..."}
  - outputs: {"expected_tools": [...], "expected_args": {...}, "expected_in_response": [...]}

Usage:
    uv run python evals/upload_dataset.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "struct-agent"
DATASET_FILE = Path(__file__).parent / "dataset.jsonl"


def main():
    client = Client()
    cases = [json.loads(line) for line in DATASET_FILE.read_text().strip().splitlines()]

    # Create or fetch existing dataset
    try:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Eval cases for the struct financial markets agent.",
        )
        print(f"Created dataset: {DATASET_NAME}")
    except Exception:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset already exists: {DATASET_NAME}")

    # Upload examples
    examples = []
    for case in cases:
        examples.append(
            {
                "inputs": {"question": case["input"]},
                "outputs": {
                    "expected_tools": case["expected_tools"],
                    "expected_args": case.get("expected_args", {}),
                    "expected_in_response": case.get("expected_in_response", []),
                },
            }
        )

    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Uploaded {len(examples)} examples to '{DATASET_NAME}'")


if __name__ == "__main__":
    main()
