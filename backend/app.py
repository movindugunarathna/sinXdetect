import os
from pathlib import Path
from typing import List, Optional
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

try:
    from classify_text import SinhalaTextClassifier
except ImportError:
    from backend.classify_text import SinhalaTextClassifier
from lime.lime_text import LimeTextExplainer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "ml" / "models" / "sinbert_sinhala_classifier"


def _resolve_model_path(raw_path: str) -> str:
    """Return absolute model path; accept relative paths for convenience."""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


MODEL_PATH = _resolve_model_path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

app = FastAPI(
    title="Sinhala Human vs AI Text Classifier with Explainability",
    version="2.0.0",
    description="API for classifying Sinhala text as human- or AI-generated using SinBERT model with LIME explanations",
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sinxdetect.movindu.com",
        "http://sinxdetect.movindu.com",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_classifier: Optional[SinhalaTextClassifier] = None
_executor = ThreadPoolExecutor(max_workers=2)  # Thread pool for LIME computations


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


class ExplainRequest(BaseModel):
    text: str
    num_samples: int = 50  # Reduced from 100 for faster response
    num_features: Optional[int] = None


class ExplanationResponse(BaseModel):
    explanation_data: dict
    highlighted_text: List[dict]
    predicted_class: str
    confidence: float
    error: Optional[str] = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information"""
    return {
        "message": "Sinhala Human vs AI Text Classifier API",
        "version": "2.0.0",
        "model": "SinBERT",
        "endpoints": {
            "/classify": "POST - Classify a single text as human or AI-generated",
            "/classify-batch": "POST - Classify multiple texts in batch",
            "/explain": "POST - Get LIME explanation for text classification with word highlighting",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation (Swagger UI)"
        },
        "features": {
            "classification": "Binary classification (HUMAN vs AI)",
            "batch_processing": "Efficient batch text classification",
            "explainability": "LIME-based word importance highlighting",
            "multilingual": "Optimized for Sinhala text"
        }
    }


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


# ==================== LIME EXPLANATION FUNCTIONALITY ====================

def predict_for_lime(texts: List[str]) -> np.ndarray:
    """
    Predict probability for each text (used by LIME explainer).
    Optimized for batch processing.
    
    Args:
        texts: List of text strings
        
    Returns:
        Array of probabilities for each class [human, ai]
    """
    try:
        if isinstance(texts, str):
            texts = [texts]
        
        classifier = get_classifier()
        
        # Use batch classification for better performance
        if len(texts) > 1:
            results = classifier.classify_batch(texts, return_probabilities=True)
            probabilities_list = []
            for result in results:
                probs = result['probabilities']
                probabilities_list.append([probs['HUMAN'], probs['AI']])
            return np.array(probabilities_list)
        else:
            result = classifier.classify(texts[0], return_probabilities=True)
            probs = result['probabilities']
            return np.array([[probs['HUMAN'], probs['AI']]])
            
    except Exception as e:
        print(f"Error in predict_for_lime: {e}")
        # Return neutral probabilities if prediction fails
        return np.array([[0.5, 0.5]] * len(texts))


def extract_word_importance(explanation, tokens: List[str], class_idx: int = 1) -> dict:
    """
    Extract word importance scores from LIME explanation.
    
    Args:
        explanation: LIME explanation object
        tokens: List of words in the text
        class_idx: Class index (1 for AI-generated, 0 for Human-written)
        
    Returns:
        Dictionary with word importance data
    """
    word_importance = {}
    
    # Get the explanation for the specified class
    if class_idx in explanation.local_exp:
        for word_idx, weight in explanation.local_exp[class_idx]:
            if 0 <= word_idx < len(tokens):
                word = tokens[word_idx]
                # Red for supporting AI-generated, green for supporting human-written
                color = 'red' if weight > 0 else 'green'
                word_importance[word_idx] = {
                    'weight': weight,
                    'color': color,
                    'token': word
                }
    
    return word_importance


def group_into_phrases(word_importance: dict, tokens: List[str], token_positions: List[tuple], max_gap: int = 1) -> List[dict]:
    """
    Group consecutive or nearby important words into phrases.
    
    Args:
        word_importance: Dict mapping word indices to their importance data
        tokens: List of all tokens
        token_positions: List of (start, end) positions for each token
        max_gap: Maximum number of non-important words between important words to still group them
        
    Returns:
        List of phrase dictionaries
    """
    if not word_importance:
        return []
    
    # Sort indices
    sorted_indices = sorted(word_importance.keys())
    phrases = []
    current_phrase = {
        'indices': [sorted_indices[0]],
        'weights': [word_importance[sorted_indices[0]]['weight']],
        'color': word_importance[sorted_indices[0]]['color']
    }
    
    for i in range(1, len(sorted_indices)):
        curr_idx = sorted_indices[i]
        prev_idx = current_phrase['indices'][-1]
        gap = curr_idx - prev_idx - 1
        
        # Check if same color and within gap threshold
        same_color = word_importance[curr_idx]['color'] == current_phrase['color']
        within_gap = gap <= max_gap
        
        if same_color and within_gap:
            # Extend current phrase
            current_phrase['indices'].append(curr_idx)
            current_phrase['weights'].append(word_importance[curr_idx]['weight'])
        else:
            # Finalize current phrase and start new one
            phrases.append(current_phrase)
            current_phrase = {
                'indices': [curr_idx],
                'weights': [word_importance[curr_idx]['weight']],
                'color': word_importance[curr_idx]['color']
            }
    
    # Add the last phrase
    phrases.append(current_phrase)
    
    # Convert phrases to output format
    highlighted_phrases = []
    for phrase_group in phrases:
        indices = phrase_group['indices']
        weights = phrase_group['weights']
        color = phrase_group['color']
        
        # Build the phrase text
        phrase_words = [tokens[idx] for idx in indices]
        phrase_text = ' '.join(phrase_words)
        
        # Calculate average weight for the phrase
        avg_weight = sum(weights) / len(weights)
        
        # Get start and end positions in original text
        start_pos = token_positions[indices[0]][0] if indices[0] < len(token_positions) else 0
        end_pos = token_positions[indices[-1]][1] if indices[-1] < len(token_positions) else 0
        
        # Determine what this phrase indicates
        indicates = 'AI-generated' if color == 'red' else 'Human-written'
        
        highlighted_phrases.append({
            'phrase': phrase_text,
            'color': color,
            'weight': float(avg_weight),
            'start': start_pos,
            'end': end_pos,
            'word_count': len(phrase_words),
            'indicates': indicates
        })
    
    return highlighted_phrases


def _run_lime_explanation(text: str, tokens: List[str], token_positions: List[tuple], 
                          num_features: int, num_samples: int) -> dict:
    """
    Run LIME explanation in a separate function (can be executed in thread pool).
    
    Returns:
        dict with explanation results or error
    """
    try:
        # Create LimeTextExplainer instance
        explainer = LimeTextExplainer(
            class_names=['Human-written', 'AI-generated'],
            split_expression=r'\s+',  # Split on whitespace
            bow=False  # Keep word order
        )
        
        explanation = explainer.explain_instance(
            text,
            predict_for_lime,
            labels=(0, 1),
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get prediction probabilities
        prediction_proba = predict_for_lime([text])[0]
        
        # Extract explanation information
        explanation_data = {
            'class_names': list(map(str, explanation.class_names)),
            'predicted_probability': list(map(float, prediction_proba)),
            'local_exp': {
                str(class_name): {
                    str(idx): float(weight)
                    for idx, weight in exp
                }
                for class_name, exp in explanation.local_exp.items()
            },
            'intercept': list(map(float, explanation.intercept)) if hasattr(explanation, 'intercept') else [0.0, 0.0]
        }
        
        # Extract word importance for AI-generated class (class 1)
        word_importance = extract_word_importance(explanation, tokens, class_idx=1)
        
        # Filter by minimum importance threshold
        word_importance = {
            idx: data for idx, data in word_importance.items()
            if abs(data['weight']) > 0.01
        }
        
        # Group words into phrases
        highlighted_text = group_into_phrases(word_importance, tokens, token_positions, max_gap=1)
        
        # Sort by absolute weight (most important first)
        highlighted_text.sort(key=lambda x: abs(x['weight']), reverse=True)
        
        predicted_class = 'AI-generated' if prediction_proba[1] > 0.5 else 'Human-written'
        confidence = float(max(prediction_proba))
        
        return {
            'success': True,
            'explanation_data': explanation_data,
            'highlighted_text': highlighted_text,
            'predicted_class': predicted_class,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"LIME error: {e}")
        # Return basic prediction if LIME fails
        try:
            prediction_proba = predict_for_lime([text])[0]
            predicted_class = 'AI-generated' if prediction_proba[1] > 0.5 else 'Human-written'
            return {
                'success': False,
                'explanation_data': {
                    'class_names': ['Human-written', 'AI-generated'],
                    'predicted_probability': list(map(float, prediction_proba)),
                    'local_exp': {},
                    'intercept': [0.0, 0.0]
                },
                'highlighted_text': [],
                'predicted_class': predicted_class,
                'confidence': float(max(prediction_proba)),
                'error': str(e)
            }
        except:
            return {
                'success': False,
                'error': str(e)
            }


@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplainRequest) -> ExplanationResponse:
    """
    Explain the prediction for a given text using LIME.
    Highlights words/phrases that contribute to AI vs Human classification.
    
    Args:
        request: ExplainRequest with text field and optional parameters
        
    Returns:
        JSON with explanation data and highlighted text
    """
    try:
        text = request.text
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Tokenize the text (word-level tokenization for LIME)
        word_pattern = re.compile(r'\S+')
        matches = list(word_pattern.finditer(text))
        tokens = [match.group() for match in matches]
        token_positions = [(match.start(), match.end()) for match in matches]
        
        # Check if we have enough tokens for LIME
        if len(tokens) < 2:
            raise HTTPException(status_code=400, detail="Text must contain at least 2 words for explanation")
        
        # Limit token count to prevent timeout (LIME complexity grows with tokens)
        MAX_TOKENS = 200
        if len(tokens) > MAX_TOKENS:
            print(f"Warning: Text has {len(tokens)} tokens, truncating to {MAX_TOKENS} for LIME analysis")
            # Keep first MAX_TOKENS tokens
            tokens = tokens[:MAX_TOKENS]
            token_positions = token_positions[:MAX_TOKENS]
            # Truncate text to match
            text = text[:token_positions[-1][1]]
        
        # Calculate appropriate num_features (reduced for performance)
        num_features = request.num_features
        if num_features is None:
            num_features = max(1, min(len(tokens), 10))  # Reduced from 15 to 10
        else:
            num_features = max(1, min(num_features, len(tokens), 15))  # Cap at 15
        
        print(f"Explaining text with {len(tokens)} tokens, using {num_features} features, {request.num_samples} samples...")
        
        # Run LIME explanation with timeout (120 seconds)
        TIMEOUT_SECONDS = 120
        loop = asyncio.get_event_loop()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _executor,
                    _run_lime_explanation,
                    text, tokens, token_positions, num_features, request.num_samples
                ),
                timeout=TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            print(f"LIME explanation timed out after {TIMEOUT_SECONDS} seconds")
            # Return a basic prediction without explanation
            prediction_proba = predict_for_lime([text])[0]
            predicted_class = 'AI-generated' if prediction_proba[1] > 0.5 else 'Human-written'
            
            return ExplanationResponse(
                explanation_data={
                    'class_names': ['Human-written', 'AI-generated'],
                    'predicted_probability': list(map(float, prediction_proba)),
                    'local_exp': {},
                    'intercept': [0.0, 0.0]
                },
                highlighted_text=[],
                predicted_class=predicted_class,
                confidence=float(max(prediction_proba)),
                error='Explanation timed out. Try with shorter text or fewer samples.'
            )
        
        if result.get('success', False):
            return ExplanationResponse(
                explanation_data=result['explanation_data'],
                highlighted_text=result['highlighted_text'],
                predicted_class=result['predicted_class'],
                confidence=result['confidence']
            )
        else:
            return ExplanationResponse(
                explanation_data=result.get('explanation_data', {
                    'class_names': ['Human-written', 'AI-generated'],
                    'predicted_probability': [0.5, 0.5],
                    'local_exp': {},
                    'intercept': [0.0, 0.0]
                }),
                highlighted_text=result.get('highlighted_text', []),
                predicted_class=result.get('predicted_class', 'Unknown'),
                confidence=result.get('confidence', 0.5),
                error=result.get('error', 'Unable to generate detailed explanation')
            )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in explain endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# ==================== END LIME EXPLANATION FUNCTIONALITY ====================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
