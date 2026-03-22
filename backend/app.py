import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Fix for TensorFlow version attribute issue with transformers
# Some TensorFlow installations have version info at different locations
if not hasattr(tf, 'version'):
    class TFVersion:
        VERSION = tf.__version__
    tf.version = TFVersion()

from lime.lime_text import LimeTextExplainer

try:
    from classify_text import SinhalaTextClassifier
except ImportError:
    from backend.classify_text import SinhalaTextClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "ml" / "models" / "sinbert_sinhala_classifier"
EVAL_DIR = REPO_ROOT / "ml" / "evaluations"
DATA_DIR = Path(os.getenv("FEEDBACK_DATA_DIR", str(REPO_ROOT / "data")))
FEEDBACK_DB = DATA_DIR / "feedback.db"

# ---------------------------------------------------------------------------
# SQLite feedback store
# ---------------------------------------------------------------------------

_FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS classification_feedback (
    id              TEXT PRIMARY KEY,
    analysis_item_id TEXT,
    model_version_id TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    corrected_label TEXT NOT NULL,
    comment         TEXT,
    text_hash       TEXT NOT NULL,
    raw_text_encrypted TEXT,
    user_name       TEXT,
    user_email      TEXT,
    status          TEXT NOT NULL DEFAULT 'NEW',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_status ON classification_feedback(status);
"""

_FEEDBACK_MIGRATIONS = [
    "ALTER TABLE classification_feedback ADD COLUMN user_name TEXT",
    "ALTER TABLE classification_feedback ADD COLUMN user_email TEXT",
]


_db_initialized = False


def _init_db() -> None:
    """Create the database and run migrations once at startup."""
    global _db_initialized
    if _db_initialized:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(FEEDBACK_DB))
    try:
        conn.executescript(_FEEDBACK_SCHEMA)
        for stmt in _FEEDBACK_MIGRATIONS:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()
    finally:
        conn.close()
    _db_initialized = True


def _get_db() -> sqlite3.Connection:
    """Return a connection to the feedback database."""
    _init_db()
    conn = sqlite3.connect(str(FEEDBACK_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _active_model_version_id() -> str:
    """Return the id of the currently active model version."""
    versions = _load_json(EVAL_DIR / "model_versions.json")
    active = next((v for v in versions if v.get("is_active")), None)
    return active["id"] if active else "unknown"


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

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sinxdetect.movindu.com",
        "http://sinxdetect.movindu.com",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all so unhandled errors still get CORS headers via the middleware."""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
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


MAX_BATCH_SIZE = 32


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
    sentence_explanations: List[dict] = []
    evidence_summary: Optional[dict] = None
    predicted_class: str
    confidence: float
    error: Optional[str] = None


# ==================== FEEDBACK MODELS ====================


class FeedbackCreateRequest(BaseModel):
    predicted_label: str
    corrected_label: str
    text_hash: str
    analysis_item_id: Optional[str] = None
    comment: Optional[str] = None
    raw_text: Optional[str] = None
    user_name: Optional[str] = None
    user_email: Optional[str] = None


class FeedbackRecord(BaseModel):
    id: str
    analysis_item_id: Optional[str]
    model_version_id: str
    predicted_label: str
    corrected_label: str
    comment: Optional[str]
    text_hash: str
    user_name: Optional[str]
    user_email: Optional[str]
    status: str
    created_at: str


class FeedbackPatchRequest(BaseModel):
    status: str  # APPROVED or REJECTED


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup so migrations run once."""
    _init_db()


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
            "/feedback": "POST - Submit feedback on a wrong classification",
            "/feedback?status=NEW": "GET - List feedback records (admin/reviewer)",
            "/feedback/{id}": "PATCH - Approve or reject a feedback record",
            "/metrics/current": "GET - Latest evaluation metrics for the active model",
            "/metrics/history": "GET - All evaluation snapshots grouped by model version",
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


def _classify_sync(text: str, return_probabilities: bool) -> dict:
    """Run classification in a thread (model.predict blocks the event loop)."""
    classifier = get_classifier()
    return classifier.classify(text, return_probabilities=return_probabilities)


def _classify_batch_sync(texts: list, return_probabilities: bool) -> list:
    """Run batch classification in a thread."""
    classifier = get_classifier()
    return classifier.classify_batch(texts, return_probabilities=return_probabilities)


@app.post("/classify", response_model=PredictionResponse)
async def classify(request: TextRequest) -> PredictionResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor, _classify_sync, request.text, request.return_probabilities
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
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.texts)} exceeds maximum of {MAX_BATCH_SIZE}.",
        )

    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(
            _executor, _classify_batch_sync, request.texts, request.return_probabilities
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    responses = [PredictionResponse(**r) for r in results]
    for text, resp in zip(request.texts, responses):
        _log_single(text, resp)
    return BatchPredictionResponse(results=responses)


# ==================== MODEL PERFORMANCE METRICS ====================


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


class PerClassMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    support: int


class ConfusionMatrixData(BaseModel):
    labels: List[str]
    matrix: List[List[int]]


class EvaluationSnapshot(BaseModel):
    id: str
    model_version_id: str
    dataset_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: ConfusionMatrixData
    classification_report: dict[str, PerClassMetrics]
    total_samples: int
    evaluated_at: str


class ModelVersionInfo(BaseModel):
    id: str
    version_name: str
    base_model: Optional[str] = None
    is_active: bool
    created_at: str


class CurrentMetricsResponse(BaseModel):
    model: ModelVersionInfo
    evaluation: Optional[EvaluationSnapshot] = None


class HistoryMetricsResponse(BaseModel):
    model: ModelVersionInfo
    evaluations: List[EvaluationSnapshot]


@app.get("/metrics/current", response_model=CurrentMetricsResponse)
async def metrics_current() -> CurrentMetricsResponse:
    """Return the latest evaluation snapshot for the active deployed model."""
    versions = _load_json(EVAL_DIR / "model_versions.json")
    evaluations = _load_json(EVAL_DIR / "model_evaluations.json")

    active = next((v for v in versions if v.get("is_active")), None)
    if active is None:
        raise HTTPException(status_code=404, detail="No active model version found")

    model_info = ModelVersionInfo(
        id=active["id"],
        version_name=active["version_name"],
        base_model=active.get("base_model"),
        is_active=active["is_active"],
        created_at=active["created_at"],
    )

    related = [e for e in evaluations if e["model_version_id"] == active["id"]]
    related.sort(key=lambda e: e.get("evaluated_at", ""), reverse=True)

    latest = EvaluationSnapshot(**related[0]) if related else None
    return CurrentMetricsResponse(model=model_info, evaluation=latest)


@app.get("/metrics/history", response_model=List[HistoryMetricsResponse])
async def metrics_history() -> List[HistoryMetricsResponse]:
    """Return all evaluations grouped by model version (newest first)."""
    versions = _load_json(EVAL_DIR / "model_versions.json")
    evaluations = _load_json(EVAL_DIR / "model_evaluations.json")

    results: List[HistoryMetricsResponse] = []
    for v in versions:
        model_info = ModelVersionInfo(
            id=v["id"],
            version_name=v["version_name"],
            base_model=v.get("base_model"),
            is_active=v.get("is_active", False),
            created_at=v["created_at"],
        )
        related = [e for e in evaluations if e["model_version_id"] == v["id"]]
        related.sort(key=lambda e: e.get("evaluated_at", ""), reverse=True)
        evals = [EvaluationSnapshot(**e) for e in related]
        results.append(HistoryMetricsResponse(model=model_info, evaluations=evals))

    return results


# ==================== FEEDBACK ENDPOINTS ====================


@app.post("/feedback", response_model=FeedbackRecord, status_code=201)
async def submit_feedback(req: FeedbackCreateRequest) -> FeedbackRecord:
    """Accept user feedback on a classification result and queue it for review."""
    if req.corrected_label not in ("HUMAN", "AI"):
        raise HTTPException(status_code=400, detail="corrected_label must be HUMAN or AI")
    if req.predicted_label not in ("HUMAN", "AI"):
        raise HTTPException(status_code=400, detail="predicted_label must be HUMAN or AI")

    feedback_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    model_ver = _active_model_version_id()

    raw_encrypted = None
    if req.raw_text:
        raw_encrypted = req.raw_text

    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO classification_feedback
               (id, analysis_item_id, model_version_id, predicted_label,
                corrected_label, comment, text_hash, raw_text_encrypted,
                user_name, user_email, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'NEW', ?)""",
            (
                feedback_id,
                req.analysis_item_id,
                model_ver,
                req.predicted_label,
                req.corrected_label,
                req.comment,
                req.text_hash,
                raw_encrypted,
                req.user_name,
                req.user_email,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return FeedbackRecord(
        id=feedback_id,
        analysis_item_id=req.analysis_item_id,
        model_version_id=model_ver,
        predicted_label=req.predicted_label,
        corrected_label=req.corrected_label,
        comment=req.comment,
        text_hash=req.text_hash,
        user_name=req.user_name,
        user_email=req.user_email,
        status="NEW",
        created_at=now,
    )


@app.get("/feedback", response_model=List[FeedbackRecord])
async def list_feedback(status: Optional[str] = Query(None)) -> List[FeedbackRecord]:
    """List feedback records, optionally filtered by status (admin/reviewer)."""
    conn = _get_db()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM classification_feedback WHERE status = ? ORDER BY created_at DESC",
                (status.upper(),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM classification_feedback ORDER BY created_at DESC"
            ).fetchall()
    finally:
        conn.close()

    return [
        FeedbackRecord(
            id=r["id"],
            analysis_item_id=r["analysis_item_id"],
            model_version_id=r["model_version_id"],
            predicted_label=r["predicted_label"],
            corrected_label=r["corrected_label"],
            comment=r["comment"],
            text_hash=r["text_hash"],
            user_name=r["user_name"],
            user_email=r["user_email"],
            status=r["status"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


@app.patch("/feedback/{feedback_id}", response_model=FeedbackRecord)
async def update_feedback(feedback_id: str, req: FeedbackPatchRequest) -> FeedbackRecord:
    """Approve or reject a feedback record (admin/reviewer)."""
    if req.status not in ("APPROVED", "REJECTED"):
        raise HTTPException(status_code=400, detail="status must be APPROVED or REJECTED")

    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM classification_feedback WHERE id = ?", (feedback_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Feedback record not found")

        conn.execute(
            "UPDATE classification_feedback SET status = ? WHERE id = ?",
            (req.status, feedback_id),
        )
        conn.commit()

        row = conn.execute(
            "SELECT * FROM classification_feedback WHERE id = ?", (feedback_id,)
        ).fetchone()
    finally:
        conn.close()

    return FeedbackRecord(
        id=row["id"],
        analysis_item_id=row["analysis_item_id"],
        model_version_id=row["model_version_id"],
        predicted_label=row["predicted_label"],
        corrected_label=row["corrected_label"],
        comment=row["comment"],
        text_hash=row["text_hash"],
        user_name=row["user_name"],
        user_email=row["user_email"],
        status=row["status"],
        created_at=row["created_at"],
    )


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


def group_into_phrases(word_importance: dict, tokens: List[str], token_positions: List[tuple],
                       original_text: str, max_gap: int = 2) -> List[dict]:
    """
    Group nearby important words into readable phrases.
    Includes gap words in the phrase text so the output reads naturally.
    Does not group across sentence boundaries.
    """
    if not word_importance:
        return []

    # Detect sentence boundary positions (indices of tokens that end a sentence)
    _SENT_END = re.compile(r'[.!?।\u0dea]$')  # period, !, ?, Sinhala purna virama
    sentence_breaks: set = set()
    for idx, tok in enumerate(tokens):
        if _SENT_END.search(tok):
            sentence_breaks.add(idx)

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

        # Check if a sentence boundary sits between prev and curr
        crosses_sentence = any(b >= prev_idx and b < curr_idx for b in sentence_breaks)

        same_color = word_importance[curr_idx]['color'] == current_phrase['color']
        within_gap = gap <= max_gap

        if same_color and within_gap and not crosses_sentence:
            current_phrase['indices'].append(curr_idx)
            current_phrase['weights'].append(word_importance[curr_idx]['weight'])
        else:
            phrases.append(current_phrase)
            current_phrase = {
                'indices': [curr_idx],
                'weights': [word_importance[curr_idx]['weight']],
                'color': word_importance[curr_idx]['color']
            }

    phrases.append(current_phrase)

    highlighted_phrases = []
    for phrase_group in phrases:
        indices = phrase_group['indices']
        weights = phrase_group['weights']
        color = phrase_group['color']

        # Use the original text span (first token start → last token end)
        # so gap words are included and the phrase reads naturally
        start_pos = token_positions[indices[0]][0] if indices[0] < len(token_positions) else 0
        end_pos = token_positions[indices[-1]][1] if indices[-1] < len(token_positions) else 0
        phrase_text = original_text[start_pos:end_pos]

        avg_weight = sum(weights) / len(weights)
        indicates = 'AI-generated' if color == 'red' else 'Human-written'

        highlighted_phrases.append({
            'phrase': phrase_text,
            'color': color,
            'weight': float(avg_weight),
            'start': start_pos,
            'end': end_pos,
            'word_count': len(indices),
            'indicates': indicates
        })

    return highlighted_phrases


def group_into_sentences(word_importance: dict, tokens: List[str],
                         token_positions: List[tuple], original_text: str) -> List[dict]:
    """
    Aggregate word-level importance into sentence-level explanations.
    Each sentence gets a net score: positive → AI-generated, negative → Human-written.
    """
    if not tokens:
        return []

    # Split tokens into sentences by detecting sentence-ending punctuation
    _SENT_END = re.compile(r'[.!?।\u0dea]$')
    sentences: list = []  # list of (start_tok_idx, end_tok_idx) inclusive
    sent_start = 0
    for idx, tok in enumerate(tokens):
        if _SENT_END.search(tok) or idx == len(tokens) - 1:
            sentences.append((sent_start, idx))
            sent_start = idx + 1

    if not sentences:
        sentences = [(0, len(tokens) - 1)]

    result = []
    for sent_start_idx, sent_end_idx in sentences:
        # Collect importance weights for tokens in this sentence
        sent_weights = []
        important_count = 0
        for tok_idx in range(sent_start_idx, sent_end_idx + 1):
            if tok_idx in word_importance:
                sent_weights.append(word_importance[tok_idx]['weight'])
                important_count += 1
            else:
                sent_weights.append(0.0)

        total_tokens = sent_end_idx - sent_start_idx + 1
        if total_tokens == 0:
            continue

        # Net weight: positive means AI-leaning, negative means Human-leaning
        net_weight = sum(sent_weights)
        abs_total = sum(abs(w) for w in sent_weights)

        # Extract the sentence text from the original
        s_start = token_positions[sent_start_idx][0] if sent_start_idx < len(token_positions) else 0
        s_end = token_positions[sent_end_idx][1] if sent_end_idx < len(token_positions) else len(original_text)
        sentence_text = original_text[s_start:s_end]

        # Classify: negligible signal → neutral (unhighlighted in the text view)
        if abs_total < 0.02:
            color = 'neutral'
            indicates = 'Neutral'
        elif net_weight > 0:
            color = 'red'
            indicates = 'AI-generated'
        else:
            color = 'green'
            indicates = 'Human-written'

        result.append({
            'sentence': sentence_text,
            'color': color,
            'net_weight': float(net_weight),
            'abs_weight': float(abs_total),
            'important_words': important_count,
            'total_words': total_tokens,
            'start': s_start,
            'end': s_end,
            'indicates': indicates,
        })

    # Sort by position for the highlighted-text view (frontend re-sorts if needed)
    result.sort(key=lambda x: x['start'])
    return result


def build_evidence_summary(word_importance: dict) -> dict:
    """
    Aggregate all word-level evidence into an overall summary.
    Returns total weight pointing to AI and to Human, plus a ratio.
    """
    ai_total = 0.0
    human_total = 0.0
    for data in word_importance.values():
        w = data['weight']
        if w > 0:
            ai_total += w
        else:
            human_total += abs(w)

    grand = ai_total + human_total
    return {
        'ai_evidence': float(ai_total),
        'human_evidence': float(human_total),
        'ai_ratio': float(ai_total / grand) if grand > 0 else 0.5,
        'human_ratio': float(human_total / grand) if grand > 0 else 0.5,
        'total_important_words': len(word_importance),
    }


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

        # Group words into readable phrases (includes gap words)
        highlighted_text = group_into_phrases(
            word_importance, tokens, token_positions, text, max_gap=2
        )
        highlighted_text.sort(key=lambda x: abs(x['weight']), reverse=True)

        # Sentence-level aggregation for easier interpretation
        sentence_explanations = group_into_sentences(
            word_importance, tokens, token_positions, text
        )

        # Overall evidence summary
        evidence_summary = build_evidence_summary(word_importance)

        predicted_class = 'AI-generated' if prediction_proba[1] > 0.5 else 'Human-written'
        confidence = float(max(prediction_proba))

        return {
            'success': True,
            'explanation_data': explanation_data,
            'highlighted_text': highlighted_text,
            'sentence_explanations': sentence_explanations,
            'evidence_summary': evidence_summary,
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
        loop = asyncio.get_running_loop()
        
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
                sentence_explanations=result.get('sentence_explanations', []),
                evidence_summary=result.get('evidence_summary'),
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
                sentence_explanations=result.get('sentence_explanations', []),
                evidence_summary=result.get('evidence_summary'),
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
