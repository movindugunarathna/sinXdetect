#!/usr/bin/env python3
"""Compare the project's LIME and SHAP explanation outputs for one Sinhala text."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for search_path in (CURRENT_DIR, ROOT_DIR):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)


from classify_text import SinhalaTextClassifier
from lime_explainer import LimeExplainer, tokenize
from shap_explainer import ShapExplainer

logger = logging.getLogger(__name__)


DEFAULT_TEXT = dedent(
    """
    දේශගුණික විපර්යාසය අද ලෝකය මුහුණ දෙන විශාලතම අභියෝගයන්ගෙන් එකකි. උෂ්ණත්වය ඉහළ යාම, අධික වැසි, වියළි කාල දිගුවීම, මුහුදු මට්ටම ඉහළ යාම සහ ජීව විවිධත්වය අඩුවීම වැනි ගැටලු මේ හේතුවෙන් වැඩිවෙමින් පවතී. එබැවින්, මෙම ගැටලුවට විසඳුම් සෙවීම සෑම රටකම සහ සෑම පුද්ගලයෙකුගේම වගකීමකි.

    දේශගුණික විපර්යාසයට ප්‍රධාන හේතුව වන්නේ ඉන්ධන දහනයෙන් සහ කර්මාන්ත ක්‍රියාකාරකම්වලින් වායුගෝලයට නිකුත් වන හරිතාගාර වායුය. ඒ නිසා පළමු විසඳුම වන්නේ අලුත්කරගත හැකි බලශක්ති ප්‍රභවයන් භාවිතය වැඩි කිරීමයි. සූර්ය බලශක්තිය, සුළං බලශක්තිය සහ ජල විදුලිය වැනි ක්‍රම භාවිතා කිරීමෙන් ගල්අඟුරු, තෙල් සහ ගෑස් වැනි ඉන්ධන මත ඇති ආශ්‍රිතාව අඩු කළ හැක. මෙය පරිසරයට හිතකර පියවරක් වන අතර දිගුකාලීනව ආර්ථික වශයෙන්ද වාසිදායකය.

    දෙවනුව, වනාන්තර ආරක්ෂා කිරීම සහ නැවත වනාන්තරකරණය ඉතා වැදගත්ය. ගස් කාබන් ඩයොක්සයිඩ් අවශෝෂණය කරන බැවින්, ගස් කැපීම අඩු කර නව ගස් රෝපණය කළ යුතුය. ගම්මාන, පාසල් සහ නගර මට්ටමින් වෘක්ෂ රෝපණ වැඩසටහන් ක්‍රියාත්මක කළහොත් පරිසරය සුරකින්නට මහත් දායකත්වයක් ලබා දිය හැක.

    තෙවනුව, අපගේ දෛනික ජීවිතයේ පුරුදු වෙනස් කළ යුතුය. විදුලිය අපතේ නොයවා භාවිතා කිරීම, පොදු ප්‍රවාහනය භාවිතා කිරීම, ප්ලාස්ටික් භාවිතය අඩු කිරීම, අපද්‍රව්‍ය වෙන් කර ප්‍රතිචක්‍රීකරණය කිරීම සහ ජලය සුරකිමින් භාවිතා කිරීම වැනි ක්‍රියා පරමාර්ථවත් වේ. කුඩා පියවරක් සේ පෙනුණත්, බොහෝ දෙනා එකතු වී මේවා අනුගමනය කළහොත් විශාල වෙනසක් ඇති කළ හැක.

    අවසාන වශයෙන්, දේශගුණික විපර්යාසයට විසඳුම් සෙවීම සරල කාර්යයක් නොවූවත්, එකමුතුව සහ වගකීම් සහිත ක්‍රියාමාර්ග මඟින් එය පාලනය කළ හැක. ආණ්ඩු, පෞද්ගලික ආයතන සහ සාමාන්‍ය ජනතාව එක්ව ක්‍රියා කළහොත්, අනාගත පරපුරට සුරක්ෂිත සහ සෞඛ්‍ය සම්පන්න පෘථිවියක් උරුම කර දිය හැක. දේශගුණික විපර්යාසය වැළැක්වීම සඳහා අද ගන්නා තීරණ හෙට දවසේ ජීවිතය තීරණය කරනු ඇත.
    """
).strip()


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalise_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _load_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text.strip()
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return DEFAULT_TEXT


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _top_highlights(items: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ordered = sorted(items, key=lambda item: abs(float(item.get("weight", 0.0))), reverse=True)
    return [dict(item) for item in ordered[:limit]]


def _highlight_phrases(items: Sequence[Dict[str, Any]]) -> List[str]:
    return [_normalise_phrase(str(item.get("phrase", ""))) for item in items if item.get("phrase")]


def _shared_top_phrases(lime_items: Sequence[Dict[str, Any]], shap_items: Sequence[Dict[str, Any]]) -> List[str]:
    lime_set = set(_highlight_phrases(lime_items))
    shap_set = set(_highlight_phrases(shap_items))
    return sorted(lime_set.intersection(shap_set))


def _phrase_weight_map(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in items:
        phrase = _normalise_phrase(str(item.get("phrase", "")))
        if phrase:
            weights[phrase] = float(item.get("weight", 0.0))
    return weights


# ---------------------------------------------------------------------------
# Token-level weight extraction (more reliable than phrase-level)
# ---------------------------------------------------------------------------


def _token_weights_from_result(result: Dict[str, Any], tokens: List[str]) -> Dict[int, float]:
    """Extract per-token weights from an explainer result.

    Works for both LIME and SHAP by inspecting the explanation_data
    structure each method produces.
    """
    data = result.get("explanation_data", {})

    # SHAP: direct token-level Shapley values
    shap_vals = data.get("shap_values", {}).get("AI-generated")
    if shap_vals and len(shap_vals) == len(tokens):
        return {i: float(v) for i, v in enumerate(shap_vals) if abs(v) > 1e-9}

    # LIME: local_exp keyed by class index, values are {token_idx: weight}
    local_exp = data.get("local_exp", {})
    lime_exp = local_exp.get("1") or local_exp.get(1)
    if lime_exp and isinstance(lime_exp, dict):
        return {int(k): float(v) for k, v in lime_exp.items() if abs(v) > 1e-9}

    return {}


# ---------------------------------------------------------------------------
# Statistical comparison metrics
# ---------------------------------------------------------------------------


def _safe_pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2 or len(right) < 2:
        return None
    try:
        value = float(np.corrcoef(np.array(left, dtype=float), np.array(right, dtype=float))[0, 1])
        return None if np.isnan(value) else value
    except Exception:
        return None


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 3 or len(right) < 3:
        return None
    try:
        coeff, _ = stats.spearmanr(left, right)
        return None if np.isnan(coeff) else float(coeff)
    except Exception:
        return None


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _sign_agreement(weights_a: Dict[int, float], weights_b: Dict[int, float]) -> Optional[float]:
    """Fraction of shared tokens where both methods agree on sign (direction)."""
    shared = set(weights_a) & set(weights_b)
    if not shared:
        return None
    agree = sum(
        1 for idx in shared
        if (weights_a[idx] > 0) == (weights_b[idx] > 0)
    )
    return agree / len(shared)


def _rank_agreement_at_k(
    weights_a: Dict[int, float], weights_b: Dict[int, float], k: int
) -> float:
    """Jaccard similarity of the top-k tokens by absolute weight."""
    top_a = set(sorted(weights_a, key=lambda i: abs(weights_a[i]), reverse=True)[:k])
    top_b = set(sorted(weights_b, key=lambda i: abs(weights_b[i]), reverse=True)[:k])
    return _jaccard(top_a, top_b)


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


def _summarise_explanation(result: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    return {
        "predicted_class": result.get("predicted_class"),
        "confidence": float(result.get("confidence", 0.0)),
        "success": result.get("success", False),
        "error": result.get("error"),
        "evidence_summary": result.get("evidence_summary"),
        "highlighted_text": _top_highlights(result.get("highlighted_text", []), top_k),
        "sentence_explanations": result.get("sentence_explanations", []),
        "explanation_data": result.get("explanation_data", {}),
    }


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------


def run_comparison(
    text: str,
    num_samples: int,
    num_features: Optional[int],
    top_k: int,
    classifier: Optional[SinhalaTextClassifier] = None,
) -> Dict[str, Any]:
    """Run LIME and SHAP on the same text and produce a comparison report.

    Parameters
    ----------
    classifier : optional
        Pre-built classifier instance.  When ``None`` a new one is created.
        Passing an existing classifier avoids redundant model loading when
        called from the API layer.
    """
    if classifier is None:
        classifier = SinhalaTextClassifier()

    lime_explainer = LimeExplainer(lambda: classifier)
    shap_explainer = ShapExplainer(lambda: classifier)

    tokens, token_positions, normalised_text = tokenize(text)

    feature_count = num_features if num_features is not None else max(1, min(len(tokens), 10))
    feature_count = max(1, min(feature_count, len(tokens), 15))

    # --- Run LIME with timing ---
    t0 = time.perf_counter()
    try:
        lime_result = lime_explainer.explain(
            normalised_text, tokens, token_positions, feature_count, num_samples,
        )
    except Exception as exc:
        logger.exception("LIME explainer failed")
        lime_result = {
            "success": False, "error": str(exc),
            "predicted_class": None, "confidence": 0.0,
            "highlighted_text": [], "sentence_explanations": [],
            "evidence_summary": None, "explanation_data": {},
        }
    lime_elapsed = time.perf_counter() - t0

    # --- Run SHAP with timing ---
    t0 = time.perf_counter()
    try:
        shap_result = shap_explainer.explain(
            normalised_text, tokens, token_positions, feature_count, num_samples,
        )
    except Exception as exc:
        logger.exception("SHAP explainer failed")
        shap_result = {
            "success": False, "error": str(exc),
            "predicted_class": None, "confidence": 0.0,
            "highlighted_text": [], "sentence_explanations": [],
            "evidence_summary": None, "explanation_data": {},
        }
    shap_elapsed = time.perf_counter() - t0

    lime_summary = _summarise_explanation(lime_result, top_k)
    shap_summary = _summarise_explanation(shap_result, top_k)

    # --- Phrase-level metrics ---
    lime_highlights = lime_summary["highlighted_text"]
    shap_highlights = shap_summary["highlighted_text"]
    shared_phrases = _shared_top_phrases(lime_highlights, shap_highlights)

    lime_phrase_weights = _phrase_weight_map(lime_highlights)
    shap_phrase_weights = _phrase_weight_map(shap_highlights)
    common_phrases = [p for p in lime_phrase_weights if p in shap_phrase_weights]
    phrase_pearson = _safe_pearson(
        [lime_phrase_weights[p] for p in common_phrases],
        [shap_phrase_weights[p] for p in common_phrases],
    )

    # --- Token-level metrics ---
    lime_token_weights = _token_weights_from_result(lime_result, tokens)
    shap_token_weights = _token_weights_from_result(shap_result, tokens)

    shared_token_indices = sorted(set(lime_token_weights) & set(shap_token_weights))
    token_pearson = _safe_pearson(
        [lime_token_weights[i] for i in shared_token_indices],
        [shap_token_weights[i] for i in shared_token_indices],
    )
    token_spearman = _safe_spearman(
        [lime_token_weights[i] for i in shared_token_indices],
        [shap_token_weights[i] for i in shared_token_indices],
    )
    sign_agree = _sign_agreement(lime_token_weights, shap_token_weights)

    # Top-k Jaccard at various k values
    top_k_jaccard: Dict[str, float] = {}
    for k in (5, 10, 15):
        if k <= len(tokens):
            top_k_jaccard[f"top_{k}"] = _rank_agreement_at_k(lime_token_weights, shap_token_weights, k)

    comparison: Dict[str, Any] = {
        "token_count": len(tokens),
        "feature_count": feature_count,
        "prediction_agreement": lime_summary["predicted_class"] == shap_summary["predicted_class"],
        "confidence_delta": abs(lime_summary["confidence"] - shap_summary["confidence"]),
        # Phrase-level
        "shared_top_phrases": shared_phrases,
        "shared_top_phrase_count": len(shared_phrases),
        "phrase_pearson_correlation": phrase_pearson,
        "lime_only_phrases": sorted(set(lime_phrase_weights) - set(shap_phrase_weights)),
        "shap_only_phrases": sorted(set(shap_phrase_weights) - set(lime_phrase_weights)),
        # Token-level
        "lime_important_token_count": len(lime_token_weights),
        "shap_important_token_count": len(shap_token_weights),
        "shared_important_token_count": len(shared_token_indices),
        "token_pearson_correlation": token_pearson,
        "token_spearman_correlation": token_spearman,
        "token_sign_agreement": sign_agree,
        "top_k_jaccard": top_k_jaccard,
        # Timing
        "lime_elapsed_seconds": round(lime_elapsed, 3),
        "shap_elapsed_seconds": round(shap_elapsed, 3),
        "speedup_ratio": round(lime_elapsed / shap_elapsed, 2) if shap_elapsed > 0 else None,
    }

    return {
        "text": text,
        "normalised_text": normalised_text,
        "tokens": tokens,
        "lime": lime_summary,
        "shap": shap_summary,
        "comparison": comparison,
    }


# ---------------------------------------------------------------------------
# CLI report formatting
# ---------------------------------------------------------------------------


def _print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _print_top_items(label: str, items: Sequence[Dict[str, Any]], limit: int) -> None:
    print(f"{label} top highlights:")
    if not items:
        print("  No highlight data returned.")
        return
    for index, item in enumerate(_top_highlights(items, limit), start=1):
        phrase = item.get("phrase", "")
        weight = float(item.get("weight", 0.0))
        direction = item.get("indicates", "Unknown")
        print(f"  {index}. {phrase} | weight={weight:.4f} | {direction}")


def _fmt_optional(value: Optional[float], fmt: str = ".4f") -> str:
    return f"{value:{fmt}}" if value is not None else "N/A (insufficient data)"


def print_report(report: Dict[str, Any], top_k: int) -> None:
    _print_section("Input")
    print(f"Tokens: {len(report['tokens'])}")
    print(f"First 20 tokens: {' | '.join(report['tokens'][:20])}")

    for method in ("lime", "shap"):
        _print_section(method.upper())
        data = report[method]
        print(f"Prediction: {data['predicted_class']}")
        print(f"Confidence: {data['confidence']:.4f}")
        print(f"Success: {data.get('success', 'N/A')}")
        if data.get("error"):
            print(f"Error: {data['error']}")
        if data.get("evidence_summary"):
            print(f"Evidence summary: {json.dumps(data['evidence_summary'], ensure_ascii=False)}")
        _print_top_items(method.upper(), data["highlighted_text"], top_k)

    _print_section("Comparison")
    c = report["comparison"]

    print(f"Prediction agreement: {c['prediction_agreement']}")
    print(f"Confidence delta: {c['confidence_delta']:.4f}")

    print()
    print("-- Phrase-level --")
    print(f"Shared top phrases ({c['shared_top_phrase_count']}): {c['shared_top_phrases']}")
    print(f"Phrase Pearson correlation: {_fmt_optional(c['phrase_pearson_correlation'])}")
    print(f"LIME-only phrases: {c['lime_only_phrases']}")
    print(f"SHAP-only phrases: {c['shap_only_phrases']}")

    print()
    print("-- Token-level --")
    print(f"LIME important tokens: {c['lime_important_token_count']}")
    print(f"SHAP important tokens: {c['shap_important_token_count']}")
    print(f"Shared important tokens: {c['shared_important_token_count']}")
    print(f"Token Pearson correlation: {_fmt_optional(c['token_pearson_correlation'])}")
    print(f"Token Spearman correlation: {_fmt_optional(c['token_spearman_correlation'])}")
    print(f"Token sign agreement: {_fmt_optional(c['token_sign_agreement'], '.2%')}")
    if c["top_k_jaccard"]:
        for label, score in c["top_k_jaccard"].items():
            print(f"Jaccard similarity ({label.replace('_', ' ')}): {score:.4f}")

    print()
    print("-- Timing --")
    print(f"LIME elapsed: {c['lime_elapsed_seconds']:.3f}s")
    print(f"SHAP elapsed: {c['shap_elapsed_seconds']:.3f}s")
    if c["speedup_ratio"] is not None:
        if c["speedup_ratio"] > 1.0:
            faster, ratio = "SHAP", c["speedup_ratio"]
        elif c["speedup_ratio"] > 0:
            faster, ratio = "LIME", 1.0 / c["speedup_ratio"]
        else:
            faster, ratio = "LIME", float("inf")
        print(f"{faster} was {ratio:.1f}x faster")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a comparative analysis between the project's LIME and SHAP explainers."
    )
    parser.add_argument(
        "--text",
        help="Text to analyze. If omitted, the script uses the Sinhala climate-change passage.",
    )
    parser.add_argument(
        "--text-file",
        help="Path to a UTF-8 text file containing the text to analyze.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of perturbation samples passed to the explainers.",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=None,
        help="Number of features to request from the explainers.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highlighted phrases to include in the comparison report.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the full comparison report as JSON.",
    )
    args = parser.parse_args()

    text = _load_text(args)
    report = run_comparison(text, args.num_samples, args.num_features, args.top_k)
    print_report(report, args.top_k)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"Saved JSON report to {output_path}")


if __name__ == "__main__":
    main()
