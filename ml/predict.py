# -*- coding: utf-8 -*-
"""
Model Usage / Inference Script
Load the local multilingual BERT (MobileBERT) sequence classifier and make
predictions on Sinhala text. The model artifacts are expected under
models/bert_multilingual_model (config.json, tokenizer.json, tf_model.h5, etc.).

Dependencies:
- transformers (TF backend)
- tensorflow
"""

from pathlib import Path
import re
from typing import Dict, Tuple, List, Union

import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError(
        "TensorFlow is required for this predictor. Please install it, e.g.\n"
        "  pip install tensorflow\n"
        "If you prefer PyTorch, provide a PyTorch (.bin) model instead of tf_model.h5."
    ) from e

try:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
except ImportError as e:
    raise ImportError(
        "transformers is required. Please install it, e.g.\n"
        "  pip install transformers sentencepiece\n"
    ) from e


# Default id-to-label mapping for binary classification.
# Adjust if your trained model used a different ordering.
ID2LABEL: Dict[int, str] = {
    0: 'HUMAN',
    1: 'AI',
}


def load_model(model_dir: Union[str, Path] = 'models/bert_multilingual_model') -> Tuple[AutoTokenizer, TFAutoModelForSequenceClassification]:
    """
    Load tokenizer and TensorFlow sequence classification model from a local directory.

    Args:
        model_dir: Path to directory containing config.json, tokenizer.json, tf_model.h5, etc.

    Returns:
        (tokenizer, model)
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = TFAutoModelForSequenceClassification.from_pretrained(str(model_dir))

    print(f"Loaded tokenizer and model from {model_dir}")
    print(f"Model config: {model.config}")
    return tokenizer, model


def clean_text(s: str) -> str:
    """
    Clean text: remove URLs and extra whitespace.
    
    Args:
        s (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(s, str):
        return ''
    # Remove URLs
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def predict(
    text: str,
    tokenizer: AutoTokenizer,
    model: TFAutoModelForSequenceClassification,
    return_probabilities: bool = False,
    max_length: int = 256,
) -> Union[str, Dict[str, Union[str, float, Dict[str, float]]]]:
    """
    Predict whether text is AI-generated or Human-written using the BERT classifier.

    Args:
        text: Input Sinhala text.
        tokenizer: Loaded tokenizer.
        model: Loaded TF sequence classification model.
        return_probabilities: If True, return class probabilities.
        max_length: Tokenization max sequence length.

    Returns:
        Predicted label ('AI' or 'HUMAN') or a dict with label, probabilities, confidence.
    """
    cleaned_text = clean_text(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors='tf',
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )

    outputs = model(**inputs)
    logits = outputs.logits  # shape: (1, num_labels)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    pred_id = int(np.argmax(probs))
    label = ID2LABEL.get(pred_id, str(pred_id))

    if return_probabilities:
        prob_dict = {ID2LABEL.get(i, str(i)): float(p) for i, p in enumerate(probs)}
        return {
            'prediction': label,
            'probabilities': prob_dict,
            'confidence': float(probs[pred_id]),
        }

    return label


def batch_predict(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: TFAutoModelForSequenceClassification,
    return_probabilities: bool = False,
    max_length: int = 256,
) -> List[Union[str, Dict[str, Union[str, float, Dict[str, float]]]]]:
    """
    Make predictions on multiple texts.

    Args:
        texts: List of input texts.
        tokenizer: Loaded tokenizer.
        model: Loaded TF sequence classification model.
        return_probabilities: If True, return probabilities for each prediction.
        max_length: Tokenization max sequence length.

    Returns:
        List of predictions or dicts with predictions and probabilities.
    """
    results: List[Union[str, Dict[str, Union[str, float, Dict[str, float]]]]] = []
    for text in texts:
        result = predict(
            text,
            tokenizer,
            model,
            return_probabilities=return_probabilities,
            max_length=max_length,
        )
        results.append(result)
    return results


if __name__ == '__main__':
    # Minimal CLI test
    print("=" * 60)
    print("BERT Model Usage Example")
    print("=" * 60)

    tokenizer, model = load_model('models/bert_multilingual_model')

    # Example 1: Single prediction without probabilities
    print("\n--- Example 1: Single Prediction ---")
    sample_text = (
        "ශ්‍රී ලංකාවේ කෘෂිකර්මය ප්‍රධාන වශයෙන් සහල් නිෂ්පාදනය මත රදා පවතී. එහි ප්‍රධාන අරමුණ වන්නේ වැඩි දියුණු කරන ලද කෘෂිකාර්මික තාක්ෂණය ව්‍යාප්තිය හා සංවර්ධනය තුළින් ස්ථිර හා සාධාරණ කෘෂිකාර්මික දියුණුවක් ලබා ගැනීමයි. මෙම අවසානය ලබා ගැනීම සදහා ශ්‍රී ලංකා රජය සතුව කෘෂිකාර්මික දෙපාර්තමේන්තුවක් පිහිටුවා ඇත. Department of Agriculture - Sri Lanka (DOASL). මෙහි මූලික ක්‍රියාවලි ලෙස පර්යේෂණ,විස්තාරිත, බීජ රෝපණය සහ පැළ සිටවීම, ද්‍රව්‍ය නිෂ්පාදනය, නියාමක සේවා, පැළෑටි නිරෝධායනය, කෘමිනාශක ද්‍රව්‍ය ලියාපදිංචිය සැළකිය හැක. දෙපාර්තමේන්තුවෙහි මාධ්‍ය සැපයුම් ඒකකය වන්නේ ශ්‍රී ලංකා ශ්‍රව්‍ය දෘශ්‍ය කේන්ද්‍රයයි Audio Visual Centre(AVC)-Sri Lanka"
    )
    prediction = predict(sample_text, tokenizer, model)
    print(f"Text: {sample_text}")
    print(f"Prediction: {prediction}")

    # Example 2: With probabilities
    print("\n--- Example 2: With Probabilities ---")
    result = predict(sample_text, tokenizer, model, return_probabilities=True)
    print(result)
    
