import json
import os
import re
import sqlite3
import sys
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Must be set before importing TensorFlow/Transformers to avoid Keras 3 API breakages.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf

try:
    import keras
except Exception:
    keras = None


CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
for search_path in (CURRENT_DIR, PARENT_DIR):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)


def _patch_keras_backend() -> None:
    """Provide missing legacy Keras backend helpers used by older transformers stacks."""
    if keras is None or not hasattr(keras, "backend"):
        return
    tf_backend = getattr(tf.keras, "backend", None)

    if not hasattr(keras.backend, "int_shape"):
        def _int_shape(x):
            shape = getattr(x, "shape", None)
            if shape is None:
                return None
            if hasattr(shape, "as_list"):
                return tuple(shape.as_list())
            try:
                return tuple(shape)
            except TypeError:
                return None

        keras.backend.int_shape = _int_shape

    if not hasattr(keras.backend, "batch_set_value"):
        if tf_backend is not None and hasattr(tf_backend, "batch_set_value"):
            keras.backend.batch_set_value = tf_backend.batch_set_value
        else:
            def _batch_set_value(tuples):
                for variable, value in tuples:
                    tf.keras.backend.set_value(variable, value)

            keras.backend.batch_set_value = _batch_set_value

    if not hasattr(keras.backend, "set_value") and tf_backend is not None and hasattr(tf_backend, "set_value"):
        keras.backend.set_value = tf_backend.set_value

    if not hasattr(keras.backend, "get_value") and tf_backend is not None and hasattr(tf_backend, "get_value"):
        keras.backend.get_value = tf_backend.get_value


_patch_keras_backend()

# Fix for TensorFlow version attribute issue with transformers
if not hasattr(tf, 'version'):
    class TFVersion:
        VERSION = tf.__version__
    tf.version = TFVersion()

try:
    from classify_text import SinhalaTextClassifier
except ImportError:
    from backend.classify_text import SinhalaTextClassifier

try:
    from lime_explainer import (
        ExplainRequest, ExplanationResponse, LimeExplainer, tokenize,
    )
except ImportError:
    from backend.lime_explainer import (
        ExplainRequest, ExplanationResponse, LimeExplainer, tokenize,
    )

try:
    from shap_explainer import ShapExplainRequest, ShapExplainer
except ImportError:
    from backend.shap_explainer import ShapExplainRequest, ShapExplainer

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

# Sinhala Unicode block (basic detection for supported input language)
_SINHALA_SCRIPT_RE = re.compile(r"[\u0D80-\u0DFF]")

SINHALA_REQUIRED_DETAIL = (
    "This model is trained for Sinhala text only. Other languages are not supported."
)


def _require_sinhala_script(text: str) -> None:
    """Raise 400 if the text contains no Sinhala script characters."""
    if not _SINHALA_SCRIPT_RE.search(text):
        raise HTTPException(status_code=400, detail=SINHALA_REQUIRED_DETAIL)


app = FastAPI(
    title="Sinhala Human vs AI Text Classifier with Explainability",
    version="2.0.0",
    description="API for classifying Sinhala text as human- or AI-generated using SinBERT model with LIME and SHAP explanations",
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
            "/explain-shap": "POST - Get SHAP explanation for text classification with word highlighting",
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
            "explainability": "LIME and SHAP-based word importance highlighting",
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
    _require_sinhala_script(request.text)

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
    for i, t in enumerate(request.texts):
        s = t.strip()
        if s and not _SINHALA_SCRIPT_RE.search(s):
            raise HTTPException(
                status_code=400,
                detail=f"{SINHALA_REQUIRED_DETAIL} (item {i})",
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


# ==================== LIME EXPLANATION ENDPOINT ====================

_lime_explainer: Optional[LimeExplainer] = None


def _get_lime_explainer() -> LimeExplainer:
    global _lime_explainer
    if _lime_explainer is None:
        _lime_explainer = LimeExplainer(get_classifier)
    return _lime_explainer


@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplainRequest) -> ExplanationResponse:
    """Explain a classification using LIME word / sentence highlights."""
    text = request.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    _require_sinhala_script(text)

    tokens, token_positions, text = tokenize(text)

    if len(tokens) < 2:
        raise HTTPException(
            status_code=400,
            detail="Text must contain at least 2 words for explanation",
        )

    num_features = request.num_features
    if num_features is None:
        num_features = max(1, min(len(tokens), 10))
    else:
        num_features = max(1, min(num_features, len(tokens), 15))

    print(
        f"Explaining text with {len(tokens)} tokens, "
        f"using {num_features} features, {request.num_samples} samples..."
    )

    lime = _get_lime_explainer()
    TIMEOUT_SECONDS = 120
    loop = asyncio.get_running_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                lime.explain,
                text,
                tokens,
                token_positions,
                num_features,
                request.num_samples,
            ),
            timeout=TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print(f"LIME explanation timed out after {TIMEOUT_SECONDS}s")
        fb = lime.fallback_prediction(text)
        return ExplanationResponse(
            explanation_data=fb["explanation_data"],
            highlighted_text=[],
            predicted_class=fb["predicted_class"],
            confidence=fb["confidence"],
            error="Explanation timed out. Try with shorter text or fewer samples.",
        )

    if result.get("success", False):
        return ExplanationResponse(
            explanation_data=result["explanation_data"],
            highlighted_text=result["highlighted_text"],
            sentence_explanations=result.get("sentence_explanations", []),
            evidence_summary=result.get("evidence_summary"),
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
        )

    return ExplanationResponse(
        explanation_data=result.get(
            "explanation_data",
            {
                "class_names": ["Human-written", "AI-generated"],
                "predicted_probability": [0.5, 0.5],
                "local_exp": {},
                "intercept": [0.0, 0.0],
            },
        ),
        highlighted_text=result.get("highlighted_text", []),
        sentence_explanations=result.get("sentence_explanations", []),
        evidence_summary=result.get("evidence_summary"),
        predicted_class=result.get("predicted_class", "Unknown"),
        confidence=result.get("confidence", 0.5),
        error=result.get("error", "Unable to generate detailed explanation"),
    )



# ==================== SHAP EXPLANATION ENDPOINT ====================

_shap_explainer: Optional[ShapExplainer] = None


def _get_shap_explainer() -> ShapExplainer:
    global _shap_explainer
    if _shap_explainer is None:
        _shap_explainer = ShapExplainer(get_classifier)
    return _shap_explainer


@app.post("/explain-shap", response_model=ExplanationResponse)
async def explain_prediction_shap(
    request: ShapExplainRequest,
) -> ExplanationResponse:
    """Explain a classification using SHAP (Shapley values) word / sentence highlights."""
    text = request.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    _require_sinhala_script(text)

    tokens, token_positions, text = tokenize(text)

    if len(tokens) < 2:
        raise HTTPException(
            status_code=400,
            detail="Text must contain at least 2 words for explanation",
        )

    num_features = request.num_features
    if num_features is None:
        num_features = max(1, min(len(tokens), 10))
    else:
        num_features = max(1, min(num_features, len(tokens), 15))

    print(
        f"[SHAP] Explaining text with {len(tokens)} tokens, "
        f"top {num_features} features, {request.num_samples} samples..."
    )

    shap_exp = _get_shap_explainer()
    TIMEOUT_SECONDS = 180  # SHAP can be slower than LIME
    loop = asyncio.get_running_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                shap_exp.explain,
                text,
                tokens,
                token_positions,
                num_features,
                request.num_samples,
            ),
            timeout=TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print(f"[SHAP] Explanation timed out after {TIMEOUT_SECONDS}s")
        fb = shap_exp.fallback_prediction(text)
        return ExplanationResponse(
            explanation_data=fb["explanation_data"],
            highlighted_text=[],
            predicted_class=fb["predicted_class"],
            confidence=fb["confidence"],
            error="SHAP explanation timed out. Try with shorter text or fewer samples.",
        )

    if result.get("success", False):
        return ExplanationResponse(
            explanation_data=result["explanation_data"],
            highlighted_text=result["highlighted_text"],
            sentence_explanations=result.get("sentence_explanations", []),
            evidence_summary=result.get("evidence_summary"),
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
        )

    return ExplanationResponse(
        explanation_data=result.get(
            "explanation_data",
            {
                "method": "shap",
                "class_names": ["Human-written", "AI-generated"],
                "predicted_probability": [0.5, 0.5],
                "base_values": [0.5, 0.5],
                "shap_values": {},
                "tokens": [],
            },
        ),
        highlighted_text=result.get("highlighted_text", []),
        sentence_explanations=result.get("sentence_explanations", []),
        evidence_summary=result.get("evidence_summary"),
        predicted_class=result.get("predicted_class", "Unknown"),
        confidence=result.get("confidence", 0.5),
        error=result.get("error", "Unable to generate SHAP explanation"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
