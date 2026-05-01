"""Utility functions for propaganda detection experiments."""

import base64
import json
import difflib
from pathlib import Path
from typing import Optional

from openai import OpenAI


def get_openai_client(
    api_key: str = "EMPTY",
    base_url: str = "http://localhost:8000/v1",
    timeout: int = 3600
) -> OpenAI:
    """Create and return an OpenAI client for VLLM."""
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def load_jsonl(file_path: str | Path) -> list[dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_data(
    data_dir: str = "../DIPLOMA_DATA_SYMLINK/propaganda_950/datasets/propaganda/defaults/annotations"
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load train, test, and validation data."""
    data_path = Path(data_dir)
    train_data = load_jsonl(data_path / "train.jsonl")
    test_data = load_jsonl(data_path / "test.jsonl")
    val_data = load_jsonl(data_path / "val.jsonl")
    return train_data, test_data, val_data


def get_unique_labels(data: list[dict]) -> set[str]:
    """Extract unique labels from dataset."""
    unique_labels = set()
    for item in data:
        for label in item.get('labels', []):
            unique_labels.add(label)
    return unique_labels


def encode_image_base64(image_path: str | Path) -> str:
    """Load and encode an image to base64."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")


def inference_model(
    client: OpenAI,
    prompt: str,
    image_path: Optional[str | Path] = None,
    model: str = "/home/nazara/Data2/DIPLOMA/qwen3-8b-finetuned-fp16",
    max_tokens: int = 2048,
    **kwargs
) -> str:
    """Run inference on the model with optional image input."""
    content = []
    try:
        if image_path:
            image_b64 = encode_image_base64(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })
        
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        

        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )
    except Exception as e:
        print("Trying unimodal model ")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            **kwargs
        )
    print(prompt)
    print(response.choices[0].message.content)
    return response.choices[0].message.content



def standardize_label(predicted_output: str, valid_labels: set[str], cutoff: float = 0.6) -> list[str]:
    """
    Parse predicted output and map to valid labels using fuzzy matching.
    
    Args:
        predicted_output: Comma-separated string of predicted labels
        valid_labels: Set of valid label names
        cutoff: Minimum similarity score for fuzzy matching
    
    Returns:
        List of standardized labels
    """
    if not predicted_output:
        return []
    
    # Parse comma-separated labels
    raw_labels = [x.strip().lower() for x in predicted_output.split(",") if x.strip()]
    valid_labels_lower = {label.lower(): label for label in valid_labels}
    
    mapped_labels = set()
    for pred_label in raw_labels:
        # Try exact match first
        if pred_label in valid_labels_lower:
            mapped_labels.add(valid_labels_lower[pred_label])
            continue
        
        # Try fuzzy matching
        matches = difflib.get_close_matches(
            pred_label, 
            valid_labels_lower.keys(), 
            n=1, 
            cutoff=cutoff
        )
        if matches:
            mapped_labels.add(valid_labels_lower[matches[0]])
    
    return list(mapped_labels)


def save_predictions(predictions: list[dict], output_path: str | Path) -> None:
    """Save predictions to a JSONL file."""
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def load_predictions(input_path: str | Path) -> list[dict]:
    """Load predictions from a JSONL file."""
    return load_jsonl(input_path)