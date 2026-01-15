#!/usr/bin/env python3
"""
Example usage of the BiLSTM Sinhala Text Classifier
Demonstrates both single and batch classification
"""

from classify_text_bilstm import SinhalaBiLSTMClassifier
from pathlib import Path

# Sample Sinhala texts for testing
sample_texts = [
    "ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දිවයිනකි.",
    "මෙම පාඨය කෘතිම බුද්ධියෙන් ජනනය කරන ලදී.",
    "අද කාලගුණය ඉතා හොඳයි සහ අහස පැහැදිලියි.",
]


def main():
    print("=" * 70)
    print("BiLSTM Sinhala Text Classifier - Example Usage")
    print("=" * 70)
    
    # Initialize the classifier
    print("\n1. Initializing classifier...")
    classifier = SinhalaBiLSTMClassifier()
    
    # Single text classification
    print("\n2. Single Text Classification")
    print("-" * 70)
    
    text = sample_texts[0]
    print(f"Input text: {text}")
    
    result = classifier.classify(text, return_probabilities=True)
    
    print(f"\nPrediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nProbabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.2%}")
    
    # Batch classification
    print("\n\n3. Batch Classification")
    print("-" * 70)
    
    results = classifier.classify_batch(sample_texts, return_probabilities=True)
    
    for i, (text, result) in enumerate(zip(sample_texts, results), 1):
        print(f"\n[Text {i}]")
        print(f"Input: {text}")
        print(f"Prediction: {result['label']} (confidence: {result['confidence']:.2%})")
        
        if 'probabilities' in result:
            probs_str = " | ".join([f"{k}: {v:.2%}" for k, v in result['probabilities'].items()])
            print(f"Probabilities: {probs_str}")
    
    # Compare with simple classification (without probabilities)
    print("\n\n4. Quick Classification (without probabilities)")
    print("-" * 70)
    
    for i, text in enumerate(sample_texts, 1):
        result = classifier.classify(text, return_probabilities=False)
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"[{i}] {preview}")
        print(f"    -> {result['label']} ({result['confidence']:.2%})")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
