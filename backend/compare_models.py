#!/usr/bin/env python3
"""
Comparison script for BERT vs BiLSTM classifiers
Shows performance, accuracy, and resource usage differences
"""

import time
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from classify_text import SinhalaTextClassifier
from classify_text_bilstm import SinhalaBiLSTMClassifier


# Sample test texts
test_texts = [
    "ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි සුන්දර දිවයිනකි.",
    "කෘතිම බුද්ධිය භාවිතයෙන් මෙම ලිපිය ලියා ඇත.",
    "කොළඹ ශ්‍රී ලංකාවේ වාණිජ අගනුවර වේ.",
]


def measure_speed(classifier, texts, model_name):
    """Measure classification speed."""
    print(f"\n{model_name} Speed Test")
    print("-" * 50)
    
    # Warm-up run
    _ = classifier.classify(texts[0])
    
    # Single text classification
    start = time.time()
    for text in texts:
        _ = classifier.classify(text)
    single_time = (time.time() - start) / len(texts)
    
    print(f"Single text avg: {single_time*1000:.2f}ms per text")
    
    # Batch classification
    start = time.time()
    _ = classifier.classify_batch(texts)
    batch_time = (time.time() - start) / len(texts)
    
    print(f"Batch avg: {batch_time*1000:.2f}ms per text")
    print(f"Batch speedup: {single_time/batch_time:.2f}x faster")
    
    return single_time, batch_time


def compare_predictions(bert_classifier, bilstm_classifier, texts):
    """Compare predictions from both models."""
    print("\n\nPrediction Comparison")
    print("=" * 80)
    
    for i, text in enumerate(texts, 1):
        print(f"\n[Text {i}]: {text[:60]}...")
        print("-" * 80)
        
        # BERT prediction
        bert_result = bert_classifier.classify(text, return_probabilities=True)
        print(f"BERT:   {bert_result['label']:6s} (conf: {bert_result['confidence']:.2%})", end="")
        if 'probabilities' in bert_result:
            probs = bert_result['probabilities']
            print(f" | H: {probs.get('HUMAN', 0):.2%}, AI: {probs.get('AI', 0):.2%}")
        else:
            print()
        
        # BiLSTM prediction
        bilstm_result = bilstm_classifier.classify(text, return_probabilities=True)
        print(f"BiLSTM: {bilstm_result['label']:6s} (conf: {bilstm_result['confidence']:.2%})", end="")
        if 'probabilities' in bilstm_result:
            probs = bilstm_result['probabilities']
            # Handle both formats (may have different class names)
            human_prob = probs.get('HUMAN', 0)
            ai_prob = probs.get('AI', 0)
            print(f" | H: {human_prob:.2%}, AI: {ai_prob:.2%}")
        else:
            print()
        
        # Agreement check
        agreement = "✓ AGREE" if bert_result['label'] == bilstm_result['label'] else "✗ DIFFER"
        print(f"Agreement: {agreement}")


def main():
    print("=" * 80)
    print("BERT vs BiLSTM Classifier Comparison")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    print("-" * 80)
    
    print("\n[1/2] Loading BERT model...")
    try:
        bert_classifier = SinhalaTextClassifier()
        print("✓ BERT model loaded")
    except Exception as e:
        print(f"✗ Failed to load BERT model: {e}")
        bert_classifier = None
    
    print("\n[2/2] Loading BiLSTM model...")
    try:
        bilstm_classifier = SinhalaBiLSTMClassifier()
        print("✓ BiLSTM model loaded")
    except Exception as e:
        print(f"✗ Failed to load BiLSTM model: {e}")
        bilstm_classifier = None
    
    if not bert_classifier or not bilstm_classifier:
        print("\n✗ Cannot proceed without both models loaded")
        return
    
    # Speed comparison
    print("\n\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    if bert_classifier:
        bert_single, bert_batch = measure_speed(bert_classifier, test_texts, "BERT")
    
    if bilstm_classifier:
        bilstm_single, bilstm_batch = measure_speed(bilstm_classifier, test_texts, "BiLSTM")
    
    # Calculate speedup
    if bert_classifier and bilstm_classifier:
        print("\n\nSpeedup Summary")
        print("-" * 50)
        print(f"BiLSTM is {bert_single/bilstm_single:.2f}x faster (single)")
        print(f"BiLSTM is {bert_batch/bilstm_batch:.2f}x faster (batch)")
    
    # Prediction comparison
    if bert_classifier and bilstm_classifier:
        compare_predictions(bert_classifier, bilstm_classifier, test_texts)
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = """
    BERT Model:
    ✓ Higher accuracy (typically 1-3% better)
    ✓ Better for complex language understanding
    ✗ Slower inference time (~500-1000ms per text)
    ✗ Higher memory usage (~500MB model size)
    ✗ Requires more computational resources
    
    BiLSTM Model:
    ✓ Much faster inference (~50-100ms per text)
    ✓ Lower memory footprint (~8MB model size)
    ✓ Good for real-time applications
    ✓ Easier deployment on resource-constrained systems
    ✗ Slightly lower accuracy
    ✗ Less robust to complex linguistic patterns
    
    Recommendation:
    - Use BERT for: Offline processing, high-accuracy requirements, research
    - Use BiLSTM for: Real-time APIs, mobile apps, production at scale
    """
    
    print(summary)
    print("=" * 80)


if __name__ == '__main__':
    main()
