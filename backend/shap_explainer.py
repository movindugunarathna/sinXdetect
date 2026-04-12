"""
SHAP (SHapley Additive exPlanations) module for Sinhala text
classification explainability.

Uses KernelSHAP to compute Shapley values for each token, producing
word-level, phrase-level, and sentence-level importance scoring that
explains why the classifier labelled text as human- or AI-generated.

The public API mirrors ``lime_explainer.LimeExplainer`` so both
methods can be used interchangeably behind the same endpoint contract.
"""

import logging
from typing import Callable, List, Optional

import numpy as np
import shap
from pydantic import BaseModel

try:
    from lime_explainer import (
        ExplanationResponse,
        group_into_phrases,
        group_into_sentences,
        build_evidence_summary,
    )
except ImportError:
    from backend.lime_explainer import (
        ExplanationResponse,
        group_into_phrases,
        group_into_sentences,
        build_evidence_summary,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request schema (SHAP defaults differ slightly from LIME)
# ---------------------------------------------------------------------------

# KernelSHAP needs a reasonable minimum sample count to produce stable
# Shapley value estimates.  Values below this floor are silently clamped.
_MIN_KERNEL_SAMPLES = 16


class ShapExplainRequest(BaseModel):
    text: str
    num_samples: int = 100  # KernelSHAP converges better with more samples
    num_features: Optional[int] = None


# ---------------------------------------------------------------------------
# ShapExplainer — main service object
# ---------------------------------------------------------------------------


class ShapExplainer:
    """Wraps SHAP text explanation using the project's classifier.

    Uses ``shap.KernelExplainer`` with binary token-presence masks so
    Shapley values are computed per-token in a model-agnostic way.

    Parameters
    ----------
    get_classifier : callable
        Zero-argument function that returns a ready-to-use classifier
        instance (with ``.classify`` and ``.classify_batch`` methods).
    """

    CLASS_NAMES = ["Human-written", "AI-generated"]
    IMPORTANCE_THRESHOLD = 0.01

    def __init__(self, get_classifier: Callable):
        self._get_classifier = get_classifier

    # ----- prediction bridge -----

    def predict(self, texts: List[str]) -> np.ndarray:
        """Return class probabilities ``[human, ai]`` for each text."""
        try:
            if isinstance(texts, str):
                texts = [texts]

            classifier = self._get_classifier()

            if len(texts) > 1:
                results = classifier.classify_batch(
                    texts, return_probabilities=True
                )
                return np.array(
                    [
                        [r["probabilities"]["HUMAN"], r["probabilities"]["AI"]]
                        for r in results
                    ]
                )

            result = classifier.classify(texts[0], return_probabilities=True)
            probs = result["probabilities"]
            return np.array([[probs["HUMAN"], probs["AI"]]])
        except Exception as e:
            logger.warning("predict error: %s", e)
            return np.array([[0.5, 0.5]] * len(texts))

    # ----- core explanation pipeline (runs in worker thread) -----

    def explain(
        self,
        text: str,
        tokens: List[str],
        token_positions: List[tuple],
        num_features: int,
        num_samples: int,
    ) -> dict:
        """Compute SHAP values and post-process into highlights.

        Designed to be called via ``loop.run_in_executor`` so the event
        loop is not blocked.

        The *num_features* argument is used to keep only the top-K most
        important tokens after SHAP values are computed (SHAP always
        evaluates all features, unlike LIME which samples a subset).
        """
        try:
            n_tokens = len(tokens)
            effective_samples = max(num_samples, _MIN_KERNEL_SAMPLES)

            def _predict_masked(masks: np.ndarray) -> np.ndarray:
                """Convert binary token masks to text and batch-predict."""
                batch: list[str] = []
                non_empty_indices: list[int] = []
                for row_idx, mask in enumerate(masks):
                    parts = [
                        tokens[j] for j in range(n_tokens) if mask[j]
                    ]
                    t = " ".join(parts)
                    batch.append(t)
                    if t.strip():
                        non_empty_indices.append(row_idx)

                result = np.full((len(batch), 2), 0.5)
                if non_empty_indices:
                    non_empty_texts = [batch[i] for i in non_empty_indices]
                    preds = self.predict(non_empty_texts)
                    for i, pred in zip(non_empty_indices, preds):
                        result[i] = pred
                return result

            # Baseline: no tokens present -> neutral prediction
            background = np.zeros((1, n_tokens))
            instance = np.ones((1, n_tokens))

            explainer = shap.KernelExplainer(_predict_masked, background)
            raw_shap = explainer.shap_values(
                instance, nsamples=effective_samples, silent=True
            )

            # raw_shap may be a list of arrays (one per class) or a single
            # Explanation/array depending on the shap version.
            ai_shap, human_shap = self._extract_shap_arrays(raw_shap, n_tokens)

            base_values = list(map(float, explainer.expected_value))

            prediction_proba = self.predict([text])[0]

            explanation_data = {
                "method": "shap",
                "class_names": list(self.CLASS_NAMES),
                "predicted_probability": list(map(float, prediction_proba)),
                "base_values": base_values,
                "shap_values": {
                    "Human-written": [float(v) for v in human_shap],
                    "AI-generated": [float(v) for v in ai_shap],
                },
                "tokens": list(tokens),
            }

            # Build word-importance dict (same schema as LIME explainer)
            word_imp: dict[int, dict] = {}
            for idx in range(n_tokens):
                val = float(ai_shap[idx])
                if abs(val) > self.IMPORTANCE_THRESHOLD:
                    word_imp[idx] = {
                        "weight": val,
                        "color": "red" if val > 0 else "green",
                        "token": tokens[idx],
                    }

            # Keep only top-K by magnitude when num_features is set
            if num_features and len(word_imp) > num_features:
                top_keys = sorted(
                    word_imp, key=lambda k: abs(word_imp[k]["weight"]), reverse=True
                )[:num_features]
                word_imp = {k: word_imp[k] for k in top_keys}

            highlighted_text = group_into_phrases(
                word_imp, tokens, token_positions, text, max_gap=2
            )
            highlighted_text.sort(
                key=lambda x: abs(x["weight"]), reverse=True
            )

            sentence_explanations = group_into_sentences(
                word_imp, tokens, token_positions, text
            )

            evidence_summary = build_evidence_summary(word_imp)

            predicted_class = (
                "AI-generated" if prediction_proba[1] > 0.5 else "Human-written"
            )

            return {
                "success": True,
                "explanation_data": explanation_data,
                "highlighted_text": highlighted_text,
                "sentence_explanations": sentence_explanations,
                "evidence_summary": evidence_summary,
                "predicted_class": predicted_class,
                "confidence": float(max(prediction_proba)),
            }

        except Exception as e:
            logger.exception("SHAP error")
            return self._fallback_result(text, error=str(e))

    # ----- helpers -----

    @staticmethod
    def _extract_shap_arrays(
        raw_shap: object, n_tokens: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalise the heterogeneous output of ``shap_values()``.

        Depending on the shap library version the return value can be:
        - A list of 2 arrays each shaped ``(1, n_tokens)`` (legacy).
        - A single ``shap.Explanation`` object with ``.values`` of shape
          ``(1, n_tokens, 2)`` or ``(n_tokens, 2)`` (newer versions).
        - A single numpy array of shape ``(1, n_tokens, 2)``.

        Returns ``(ai_shap, human_shap)`` each of length ``n_tokens``.
        """
        # Legacy list-of-arrays format
        if isinstance(raw_shap, list) and len(raw_shap) == 2:
            human = np.asarray(raw_shap[0]).flatten()[:n_tokens]
            ai = np.asarray(raw_shap[1]).flatten()[:n_tokens]
            return ai, human

        # shap.Explanation object
        values = getattr(raw_shap, "values", raw_shap)
        arr = np.asarray(values)

        # (1, n_tokens, 2) or (n_tokens, 2)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[-1] == 2:
            return arr[:, 1].flatten()[:n_tokens], arr[:, 0].flatten()[:n_tokens]

        # Unexpected shape — treat as single-class AI values, zero out human
        logger.warning("Unexpected shap_values shape %s, falling back", arr.shape)
        flat = arr.flatten()[:n_tokens]
        return flat, np.zeros_like(flat)

    def fallback_prediction(self, text: str) -> dict:
        """Minimal prediction result (no explanation) for timeout / error."""
        return self._fallback_result(text, error=None)

    def _fallback_result(self, text: str, error: Optional[str]) -> dict:
        try:
            proba = self.predict([text])[0]
            predicted_class = (
                "AI-generated" if proba[1] > 0.5 else "Human-written"
            )
            return {
                "success": False,
                "explanation_data": {
                    "method": "shap",
                    "class_names": self.CLASS_NAMES,
                    "predicted_probability": list(map(float, proba)),
                    "base_values": [0.5, 0.5],
                    "shap_values": {},
                    "tokens": [],
                },
                "highlighted_text": [],
                "sentence_explanations": [],
                "evidence_summary": None,
                "predicted_class": predicted_class,
                "confidence": float(max(proba)),
                "error": error,
            }
        except Exception:
            return {
                "success": False,
                "explanation_data": {
                    "method": "shap",
                    "class_names": self.CLASS_NAMES,
                    "predicted_probability": [0.5, 0.5],
                    "base_values": [0.5, 0.5],
                    "shap_values": {},
                    "tokens": [],
                },
                "highlighted_text": [],
                "sentence_explanations": [],
                "evidence_summary": None,
                "predicted_class": "Unknown",
                "confidence": 0.5,
                "error": error or "Prediction failed",
            }
