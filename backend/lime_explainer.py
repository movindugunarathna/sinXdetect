"""
LIME (Local Interpretable Model-agnostic Explanations) module for
Sinhala text classification explainability.

Provides word-level, phrase-level, and sentence-level importance scoring
to explain why the classifier labelled text as human- or AI-generated.
"""

import re
from typing import Callable, List, Optional

import numpy as np
from lime.lime_text import LimeTextExplainer
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ExplainRequest(BaseModel):
    text: str
    num_samples: int = 50
    num_features: Optional[int] = None


class ExplanationResponse(BaseModel):
    explanation_data: dict
    highlighted_text: List[dict]
    sentence_explanations: List[dict] = []
    evidence_summary: Optional[dict] = None
    predicted_class: str
    confidence: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

MAX_TOKENS = 200
_WORD_RE = re.compile(r"\S+")
_SENT_END_RE = re.compile(r"[.!?।\u0dea]$")


def tokenize(text: str) -> tuple[list[str], list[tuple[int, int]], str]:
    """Split *text* on whitespace, returning (tokens, positions, text).

    If the token count exceeds ``MAX_TOKENS`` the text is truncated so
    that LIME analysis stays tractable.
    """
    matches = list(_WORD_RE.finditer(text))
    tokens = [m.group() for m in matches]
    positions = [(m.start(), m.end()) for m in matches]

    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        positions = positions[:MAX_TOKENS]
        text = text[: positions[-1][1]]

    return tokens, positions, text


# ---------------------------------------------------------------------------
# Pure helper functions (no external dependencies beyond their arguments)
# ---------------------------------------------------------------------------


def extract_word_importance(
    explanation, tokens: List[str], class_idx: int = 1
) -> dict[int, dict]:
    """Extract per-word importance scores from a LIME explanation object.

    Red (weight > 0) supports AI-generated; green supports human-written.
    """
    word_importance: dict[int, dict] = {}
    if class_idx not in explanation.local_exp:
        return word_importance

    for word_idx, weight in explanation.local_exp[class_idx]:
        if 0 <= word_idx < len(tokens):
            word_importance[word_idx] = {
                "weight": weight,
                "color": "red" if weight > 0 else "green",
                "token": tokens[word_idx],
            }
    return word_importance


def group_into_phrases(
    word_importance: dict,
    tokens: List[str],
    token_positions: List[tuple],
    original_text: str,
    max_gap: int = 2,
) -> List[dict]:
    """Group nearby important words into readable phrases.

    Gap words are included so the output reads naturally.
    Grouping does not cross sentence boundaries.
    """
    if not word_importance:
        return []

    sentence_breaks: set[int] = set()
    for idx, tok in enumerate(tokens):
        if _SENT_END_RE.search(tok):
            sentence_breaks.add(idx)

    sorted_indices = sorted(word_importance.keys())
    phrases: list[dict] = []
    current: dict = {
        "indices": [sorted_indices[0]],
        "weights": [word_importance[sorted_indices[0]]["weight"]],
        "color": word_importance[sorted_indices[0]]["color"],
    }

    for i in range(1, len(sorted_indices)):
        curr_idx = sorted_indices[i]
        prev_idx = current["indices"][-1]
        gap = curr_idx - prev_idx - 1

        crosses_sentence = any(
            b >= prev_idx and b < curr_idx for b in sentence_breaks
        )
        same_color = word_importance[curr_idx]["color"] == current["color"]

        if same_color and gap <= max_gap and not crosses_sentence:
            current["indices"].append(curr_idx)
            current["weights"].append(word_importance[curr_idx]["weight"])
        else:
            phrases.append(current)
            current = {
                "indices": [curr_idx],
                "weights": [word_importance[curr_idx]["weight"]],
                "color": word_importance[curr_idx]["color"],
            }

    phrases.append(current)

    highlighted: List[dict] = []
    for phrase in phrases:
        indices = phrase["indices"]
        weights = phrase["weights"]
        color = phrase["color"]

        start_pos = (
            token_positions[indices[0]][0]
            if indices[0] < len(token_positions)
            else 0
        )
        end_pos = (
            token_positions[indices[-1]][1]
            if indices[-1] < len(token_positions)
            else 0
        )

        highlighted.append(
            {
                "phrase": original_text[start_pos:end_pos],
                "color": color,
                "weight": float(sum(weights) / len(weights)),
                "start": start_pos,
                "end": end_pos,
                "word_count": len(indices),
                "indicates": "AI-generated" if color == "red" else "Human-written",
            }
        )

    return highlighted


def group_into_sentences(
    word_importance: dict,
    tokens: List[str],
    token_positions: List[tuple],
    original_text: str,
) -> List[dict]:
    """Aggregate word-level importance into sentence-level explanations.

    Positive net score -> AI-generated, negative -> Human-written.
    """
    if not tokens:
        return []

    sentences: list[tuple[int, int]] = []
    sent_start = 0
    for idx, tok in enumerate(tokens):
        if _SENT_END_RE.search(tok) or idx == len(tokens) - 1:
            sentences.append((sent_start, idx))
            sent_start = idx + 1

    if not sentences:
        sentences = [(0, len(tokens) - 1)]

    result: List[dict] = []
    for s_start_idx, s_end_idx in sentences:
        sent_weights: list[float] = []
        important_count = 0
        for tok_idx in range(s_start_idx, s_end_idx + 1):
            if tok_idx in word_importance:
                sent_weights.append(word_importance[tok_idx]["weight"])
                important_count += 1
            else:
                sent_weights.append(0.0)

        total_tokens = s_end_idx - s_start_idx + 1
        if total_tokens == 0:
            continue

        net_weight = sum(sent_weights)
        abs_total = sum(abs(w) for w in sent_weights)

        s_start = (
            token_positions[s_start_idx][0]
            if s_start_idx < len(token_positions)
            else 0
        )
        s_end = (
            token_positions[s_end_idx][1]
            if s_end_idx < len(token_positions)
            else len(original_text)
        )

        if abs_total < 0.02:
            color, indicates = "neutral", "Neutral"
        elif net_weight > 0:
            color, indicates = "red", "AI-generated"
        else:
            color, indicates = "green", "Human-written"

        result.append(
            {
                "sentence": original_text[s_start:s_end],
                "color": color,
                "net_weight": float(net_weight),
                "abs_weight": float(abs_total),
                "important_words": important_count,
                "total_words": total_tokens,
                "start": s_start,
                "end": s_end,
                "indicates": indicates,
            }
        )

    result.sort(key=lambda x: x["start"])
    return result


def build_evidence_summary(word_importance: dict) -> dict:
    """Aggregate word-level evidence into an AI-vs-Human summary."""
    ai_total = 0.0
    human_total = 0.0
    for data in word_importance.values():
        w = data["weight"]
        if w > 0:
            ai_total += w
        else:
            human_total += abs(w)

    grand = ai_total + human_total
    return {
        "ai_evidence": float(ai_total),
        "human_evidence": float(human_total),
        "ai_ratio": float(ai_total / grand) if grand > 0 else 0.5,
        "human_ratio": float(human_total / grand) if grand > 0 else 0.5,
        "total_important_words": len(word_importance),
    }


# ---------------------------------------------------------------------------
# LimeExplainer — main service object
# ---------------------------------------------------------------------------


class LimeExplainer:
    """Wraps LIME text explanation using the project's classifier.

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

    # ----- prediction bridge for LIME -----

    def predict(self, texts: List[str]) -> np.ndarray:
        """Return class probabilities ``[human, ai]`` for each text.

        This is the function LIME calls internally to probe the model.
        """
        try:
            if isinstance(texts, str):
                texts = [texts]

            classifier = self._get_classifier()

            if len(texts) > 1:
                results = classifier.classify_batch(
                    texts, return_probabilities=True
                )
                return np.array(
                    [[r["probabilities"]["HUMAN"], r["probabilities"]["AI"]] for r in results]
                )

            result = classifier.classify(texts[0], return_probabilities=True)
            probs = result["probabilities"]
            return np.array([[probs["HUMAN"], probs["AI"]]])
        except Exception as e:
            print(f"Error in predict: {e}")
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
        """Run LIME and post-process into phrase / sentence highlights.

        Designed to be called via ``loop.run_in_executor`` so the event
        loop is not blocked.
        """
        try:
            explainer = LimeTextExplainer(
                class_names=self.CLASS_NAMES,
                split_expression=r"\s+",
                bow=False,
            )

            explanation = explainer.explain_instance(
                text,
                self.predict,
                labels=(0, 1),
                num_features=num_features,
                num_samples=num_samples,
            )

            prediction_proba = self.predict([text])[0]

            explanation_data = {
                "class_names": list(map(str, explanation.class_names)),
                "predicted_probability": list(map(float, prediction_proba)),
                "local_exp": {
                    str(cn): {str(idx): float(w) for idx, w in exp}
                    for cn, exp in explanation.local_exp.items()
                },
                "intercept": (
                    list(map(float, explanation.intercept))
                    if hasattr(explanation, "intercept")
                    else [0.0, 0.0]
                ),
            }

            word_imp = extract_word_importance(explanation, tokens, class_idx=1)
            word_imp = {
                idx: d
                for idx, d in word_imp.items()
                if abs(d["weight"]) > self.IMPORTANCE_THRESHOLD
            }

            highlighted_text = group_into_phrases(
                word_imp, tokens, token_positions, text, max_gap=2
            )
            highlighted_text.sort(key=lambda x: abs(x["weight"]), reverse=True)

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
            print(f"LIME error: {e}")
            return self._fallback_result(text, error=str(e))

    # ----- helpers -----

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
                    "class_names": self.CLASS_NAMES,
                    "predicted_probability": list(map(float, proba)),
                    "local_exp": {},
                    "intercept": [0.0, 0.0],
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
                    "class_names": self.CLASS_NAMES,
                    "predicted_probability": [0.5, 0.5],
                    "local_exp": {},
                    "intercept": [0.0, 0.0],
                },
                "highlighted_text": [],
                "sentence_explanations": [],
                "evidence_summary": None,
                "predicted_class": "Unknown",
                "confidence": 0.5,
                "error": error or "Prediction failed",
            }
