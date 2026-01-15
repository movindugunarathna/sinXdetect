"""
FastAPI Integration Example for BiLSTM Classifier

This file shows how to integrate the BiLSTM classifier alongside 
the existing BERT classifier in your FastAPI application.

You can either:
1. Add these routes to your existing app.py
2. Run this as a standalone API server
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from classify_text import SinhalaTextClassifier
from classify_text_bilstm import SinhalaBiLSTMClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
BERT_MODEL_PATH = REPO_ROOT / "ml" / "models" / "bert_multilingual_model"
BILSTM_MODEL_PATH = REPO_ROOT / "ml" / "models" / "bilstm_sinhala_model"


# Request/Response Models
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
    model_type: Optional[str] = None  # Added to indicate which model was used


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


class ModelComparisonResponse(BaseModel):
    text: str
    bert_result: PredictionResponse
    bilstm_result: PredictionResponse
    agreement: bool


# Create FastAPI app
app = FastAPI(
    title="Sinhala Text Classifier - Multi-Model API",
    version="2.0.0",
    description="API for classifying Sinhala text using BERT or BiLSTM models",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model instances (lazy loaded)
_bert_classifier: Optional[SinhalaTextClassifier] = None
_bilstm_classifier: Optional[SinhalaBiLSTMClassifier] = None


def get_bert_classifier() -> SinhalaTextClassifier:
    """Lazily load BERT classifier."""
    global _bert_classifier
    if _bert_classifier is None:
        try:
            _bert_classifier = SinhalaTextClassifier(model_path=str(BERT_MODEL_PATH))
        except Exception as e:
            print(f"Warning: Failed to load BERT classifier: {e}")
            print("BERT endpoints will not be available.")
            raise HTTPException(
                status_code=503,
                detail=f"BERT model is not available. Error: {str(e)}. Try installing: pip install tf-keras"
            )
    return _bert_classifier


def get_bilstm_classifier() -> SinhalaBiLSTMClassifier:
    """Lazily load BiLSTM classifier."""
    global _bilstm_classifier
    if _bilstm_classifier is None:
        try:
            _bilstm_classifier = SinhalaBiLSTMClassifier(model_path=str(BILSTM_MODEL_PATH))
        except Exception as e:
            print(f"Warning: Failed to load BiLSTM classifier: {e}")
            print("BiLSTM endpoints will not be available.")
            raise HTTPException(
                status_code=503,
                detail=f"BiLSTM model is not available. Error: {str(e)}"
            )
    return _bilstm_classifier


# Helper function for logging
def _log_result(text: str, result: PredictionResponse, model_type: str) -> None:
    preview = (text[:80] + "...") if len(text) > 80 else text
    print("=" * 60)
    print(f"[{model_type}] CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Input: {preview}")
    print(f"Prediction: {result.label}")
    print(f"Confidence: {result.confidence:.2%}")
    if result.probabilities:
        print("Probabilities:")
        for k, v in result.probabilities.items():
            print(f"  {k}: {v:.2%}")
    print("=" * 60)


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "models": {
            "bert": _bert_classifier is not None,
            "bilstm": _bilstm_classifier is not None
        }
    }


# ============================================================================
# BERT CLASSIFIER ENDPOINTS (Original)
# ============================================================================

@app.post("/classify", response_model=PredictionResponse)
async def classify_bert(request: TextRequest) -> PredictionResponse:
    """
    Classify text using BERT model (default/original endpoint).
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    
    classifier = get_bert_classifier()
    try:
        result = classifier.classify(
            request.text, return_probabilities=request.return_probabilities
        )
        result['model_type'] = 'BERT'
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    response = PredictionResponse(**result)
    _log_result(request.text, response, "BERT")
    return response


@app.post("/classify-batch", response_model=BatchPredictionResponse)
async def classify_batch_bert(request: BatchRequest) -> BatchPredictionResponse:
    """
    Batch classify texts using BERT model.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty.")
    
    classifier = get_bert_classifier()
    try:
        results = classifier.classify_batch(
            request.texts, return_probabilities=request.return_probabilities
        )
        for r in results:
            r['model_type'] = 'BERT'
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    responses = [PredictionResponse(**r) for r in results]
    return BatchPredictionResponse(results=responses)


# ============================================================================
# BiLSTM CLASSIFIER ENDPOINTS (New)
# ============================================================================

@app.post("/classify-bilstm", response_model=PredictionResponse)
async def classify_bilstm(request: TextRequest) -> PredictionResponse:
    """
    Classify text using BiLSTM model (faster, lighter).
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    
    classifier = get_bilstm_classifier()
    try:
        result = classifier.classify(
            request.text, return_probabilities=request.return_probabilities
        )
        result['model_type'] = 'BiLSTM'
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    response = PredictionResponse(**result)
    _log_result(request.text, response, "BiLSTM")
    return response


@app.post("/classify-bilstm-batch", response_model=BatchPredictionResponse)
async def classify_batch_bilstm(request: BatchRequest) -> BatchPredictionResponse:
    """
    Batch classify texts using BiLSTM model.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty.")
    
    classifier = get_bilstm_classifier()
    try:
        results = classifier.classify_batch(
            request.texts, return_probabilities=request.return_probabilities
        )
        for r in results:
            r['model_type'] = 'BiLSTM'
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    responses = [PredictionResponse(**r) for r in results]
    return BatchPredictionResponse(results=responses)


# ============================================================================
# COMPARISON ENDPOINT (New)
# ============================================================================

@app.post("/compare-models", response_model=ModelComparisonResponse)
async def compare_models(request: TextRequest) -> ModelComparisonResponse:
    """
    Compare predictions from both BERT and BiLSTM models on the same text.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    
    try:
        # Get BERT prediction
        try:
            bert_classifier = get_bert_classifier()
            bert_result = bert_classifier.classify(
                request.text, return_probabilities=request.return_probabilities
            )
            bert_result['model_type'] = 'BERT'
        except HTTPException:
            raise HTTPException(
                status_code=503,
                detail="BERT model is not available. Cannot compare models."
            )
        
        # Get BiLSTM prediction
        try:
            bilstm_classifier = get_bilstm_classifier()
            bilstm_result = bilstm_classifier.classify(
                request.text, return_probabilities=request.return_probabilities
            )
            bilstm_result['model_type'] = 'BiLSTM'
        except HTTPException:
            raise HTTPException(
                status_code=503,
                detail="BiLSTM model is not available. Cannot compare models."
            )
        
        # Check agreement
        agreement = bert_result['label'] == bilstm_result['label']
        
        return ModelComparisonResponse(
            text=request.text,
            bert_result=PredictionResponse(**bert_result),
            bilstm_result=PredictionResponse(**bilstm_result),
            agreement=agreement
        )
    
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ============================================================================
# MODEL INFO ENDPOINT
# ============================================================================

@app.get("/models/info")
async def models_info() -> dict:
    """
    Get information about available models.
    """
    return {
        "models": [
            {
                "name": "BERT Multilingual",
                "endpoint": "/classify",
                "type": "Transformer (BERT)",
                "size": "~500MB",
                "speed": "Slower (~500-1000ms)",
                "accuracy": "Higher",
                "best_for": "High accuracy requirements, offline processing"
            },
            {
                "name": "BiLSTM Sinhala",
                "endpoint": "/classify-bilstm",
                "type": "Recurrent Neural Network",
                "size": "~8MB",
                "speed": "Faster (~50-100ms)",
                "accuracy": "Good",
                "best_for": "Real-time applications, production at scale"
            }
        ],
        "comparison_endpoint": "/compare-models"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Multi-Model Sinhala Text Classifier API...")
    print("=" * 60)
    print("Available endpoints:")
    print("  - POST /classify              (BERT model)")
    print("  - POST /classify-bilstm       (BiLSTM model)")
    print("  - POST /classify-batch        (BERT batch)")
    print("  - POST /classify-bilstm-batch (BiLSTM batch)")
    print("  - POST /compare-models        (Compare both)")
    print("  - GET  /models/info           (Model information)")
    print("  - GET  /health                (Health check)")
    print("=" * 60)
    
    uvicorn.run("app_multimodel:app", host="0.0.0.0", port=8000, reload=True)
