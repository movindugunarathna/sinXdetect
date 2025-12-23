"""
FastAPI backend to serve the Sinhala Human vs AI text classifier.
Exposes endpoints for single and batch classification while loading the
BERT-based model once at startup.
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from classify_text import SinhalaTextClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "ml" / "models" / "bert_multilingual_model"


def _resolve_model_path(raw_path: str) -> str:
    """Return absolute model path; accept relative paths for convenience."""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


MODEL_PATH = _resolve_model_path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

app = FastAPI(
    title="Sinhala Human vs AI Text Classifier",
    version="1.0.0",
    description="API for classifying Sinhala text as human- or AI-generated",
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins like ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_classifier: Optional[SinhalaTextClassifier] = None


def _log_single(text: str, result: "PredictionResponse") -> None:
    preview = (text[:80] + "...") if len(text) > 80 else text
    print("=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"Input: {preview}")
    print(f"Prediction: {result.label}")
    print(f"Confidence: {result.confidence:.2%}")
    if result.probabilities:
        probs = result.probabilities
        print("\nClass Probabilities:")
        print(f"  HUMAN: {probs.get('HUMAN', 0):.2%}")
        print(f"  AI:    {probs.get('AI', 0):.2%}")
    print("=" * 50)


def get_classifier() -> SinhalaTextClassifier:
    """Lazily create and cache the classifier to avoid repeated model loads."""
    global _classifier
    if _classifier is None:
        _classifier = SinhalaTextClassifier(model_path=MODEL_PATH)
    return _classifier


class TextRequest(BaseModel):
    text: str
    return_probabilities: bool = False


class BatchRequest(BaseModel):
    texts: List[str]
    return_probabilities: bool = False


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Optional[dict] = None


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/classify", response_model=PredictionResponse)
async def classify(request: TextRequest) -> PredictionResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    classifier = get_classifier()
    try:
        result = classifier.classify(
            request.text, return_probabilities=request.return_probabilities
        )
    except Exception as exc:  # pragma: no cover - surface runtime issues to client
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    response = PredictionResponse(**result)
    _log_single(request.text, response)
    return response


@app.post("/classify-batch", response_model=BatchPredictionResponse)
async def classify_batch(request: BatchRequest) -> BatchPredictionResponse:
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty.")

    classifier = get_classifier()
    try:
        results = classifier.classify_batch(
            request.texts, return_probabilities=request.return_probabilities
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    responses = [PredictionResponse(**r) for r in results]
    for text, resp in zip(request.texts, responses):
        _log_single(text, resp)
    return BatchPredictionResponse(results=responses)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
