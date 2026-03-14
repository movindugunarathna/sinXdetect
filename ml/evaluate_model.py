"""
Offline evaluation script for sinXdetect models.

Runs a trained model against a labeled validation/test dataset,
computes standard classification metrics (accuracy, precision, recall, F1,
confusion matrix), and appends the results to ml/evaluations/model_evaluations.json.

Usage:
    python evaluate_model.py \
        --dataset path/to/test.jsonl \
        --model-version v1-sinbert-large \
        --dataset-name sinxdetect-test-v2

The dataset must be JSONL with {"text": "...", "label": "HUMAN"|"AI"} per line,
or CSV with `text` and `label` columns.
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent / "evaluations"
VERSIONS_FILE = EVAL_DIR / "model_versions.json"
EVALUATIONS_FILE = EVAL_DIR / "model_evaluations.json"

LABEL_ORDER = ["HUMAN", "AI"]


def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".jsonl":
        df = pd.read_json(p, lines=True)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .jsonl or .csv")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    df["label"] = df["label"].str.upper().str.strip()
    unknown = set(df["label"].unique()) - set(LABEL_ORDER)
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}. Expected {LABEL_ORDER}")

    return df


def get_model_version(version_id: str) -> dict:
    with open(VERSIONS_FILE) as f:
        versions = json.load(f)
    for v in versions:
        if v["id"] == version_id:
            return v
    raise ValueError(
        f"Model version '{version_id}' not found in {VERSIONS_FILE}. "
        f"Available: {[v['id'] for v in versions]}"
    )


def run_inference(model_path: str, texts: list[str]) -> list[str]:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from classify_text import SinhalaTextClassifier

    classifier = SinhalaTextClassifier(model_path=model_path)
    results = classifier.classify_batch(texts, return_probabilities=False)
    return [r["label"] for r in results]


def compute_metrics(
    y_true: list[str], y_pred: list[str]
) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted")
    rec = recall_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted")
    f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, output_dict=True)

    per_class = {}
    for label in LABEL_ORDER:
        entry = report[label]
        per_class[label] = {
            "precision": round(entry["precision"], 4),
            "recall": round(entry["recall"], 4),
            "f1_score": round(entry["f1-score"], 4),
            "support": int(entry["support"]),
        }

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "labels": LABEL_ORDER,
            "matrix": cm.tolist(),
        },
        "classification_report": per_class,
        "total_samples": len(y_true),
    }


def save_evaluation(model_version_id: str, dataset_name: str, metrics: dict) -> dict:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    evaluation = {
        "id": f"eval-{model_version_id}-{uuid.uuid4().hex[:8]}",
        "model_version_id": model_version_id,
        "dataset_name": dataset_name,
        **metrics,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    if EVALUATIONS_FILE.exists():
        with open(EVALUATIONS_FILE) as f:
            evaluations = json.load(f)
    else:
        evaluations = []

    evaluations.append(evaluation)

    with open(EVALUATIONS_FILE, "w") as f:
        json.dump(evaluations, f, indent=2)

    return evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate a sinXdetect model")
    parser.add_argument("--dataset", required=True, help="Path to test dataset (.jsonl or .csv)")
    parser.add_argument("--model-version", required=True, help="Model version ID from model_versions.json")
    parser.add_argument("--dataset-name", default=None, help="Name for the dataset (defaults to filename)")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    args = parser.parse_args()

    dataset_name = args.dataset_name or Path(args.dataset).stem

    print(f"Loading dataset: {args.dataset}")
    df = load_dataset(args.dataset)
    print(f"  Loaded {len(df)} samples ({df['label'].value_counts().to_dict()})")

    version = get_model_version(args.model_version)
    model_path = str((REPO_ROOT / version["model_path"]).resolve())
    print(f"Model: {version['version_name']}  ({model_path})")

    print("Running inference...")
    texts = df["text"].tolist()
    predictions = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        predictions.extend(run_inference(model_path, batch))
        done = min(i + args.batch_size, len(texts))
        print(f"  {done}/{len(texts)} samples processed")

    print("Computing metrics...")
    metrics = compute_metrics(df["label"].tolist(), predictions)

    print(f"\n{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  Confusion Matrix:")
    cm = metrics["confusion_matrix"]["matrix"]
    print(f"             {'HUMAN':>8} {'AI':>8}")
    print(f"    HUMAN    {cm[0][0]:>8} {cm[0][1]:>8}")
    print(f"    AI       {cm[1][0]:>8} {cm[1][1]:>8}")
    print(f"{'='*50}\n")

    evaluation = save_evaluation(args.model_version, dataset_name, metrics)
    print(f"Saved evaluation: {evaluation['id']}")
    print(f"  -> {EVALUATIONS_FILE}")


if __name__ == "__main__":
    main()
