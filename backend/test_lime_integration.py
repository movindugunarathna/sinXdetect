#!/usr/bin/env python3
"""
Quick test script to verify LIME integration without starting the server.
Tests the core LIME functions directly.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from classify_text import SinhalaTextClassifier
import numpy as np
import re

# Test if lime is installed
try:
    from lime.lime_text import LimeTextExplainer
    print("✓ LIME package is installed")
except ImportError:
    print("✗ LIME package is NOT installed")
    print("  Please run: pip install lime")
    sys.exit(1)

# Test data
TEST_TEXTS = [
    "මම අද පාසලට ගියෙමි",  # I went to school today
    "ඔවුන් එකට වැඩ කරති",     # They work together
]

def test_classifier_loading():
    """Test if classifier loads successfully"""
    print("\n" + "="*60)
    print("TEST 1: Loading SinhalaTextClassifier")
    print("="*60)
    
    try:
        classifier = SinhalaTextClassifier()
        print("✓ Classifier loaded successfully")
        return classifier
    except Exception as e:
        print(f"✗ Failed to load classifier: {e}")
        return None


def test_basic_prediction(classifier):
    """Test basic prediction functionality"""
    print("\n" + "="*60)
    print("TEST 2: Basic Prediction")
    print("="*60)
    
    if classifier is None:
        print("✗ Skipping (classifier not loaded)")
        return False
    
    try:
        text = TEST_TEXTS[0]
        result = classifier.classify(text, return_probabilities=True)
        
        print(f"Text: {text}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities: {result['probabilities']}")
        print("✓ Basic prediction works")
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def predict_for_lime_test(classifier, texts):
    """Test version of predict_for_lime function"""
    if isinstance(texts, str):
        texts = [texts]
    
    probabilities_list = []
    for text in texts:
        result = classifier.classify(text, return_probabilities=True)
        probs = result['probabilities']
        probabilities_list.append([probs['HUMAN'], probs['AI']])
    
    return np.array(probabilities_list)


def test_lime_explainer(classifier):
    """Test LIME explanation generation"""
    print("\n" + "="*60)
    print("TEST 3: LIME Explanation")
    print("="*60)
    
    if classifier is None:
        print("✗ Skipping (classifier not loaded)")
        return False
    
    try:
        text = TEST_TEXTS[0]
        
        # Tokenize text
        word_pattern = re.compile(r'\S+')
        matches = list(word_pattern.finditer(text))
        tokens = [match.group() for match in matches]
        
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        
        if len(tokens) < 2:
            print("✗ Text too short for LIME (need at least 2 words)")
            return False
        
        # Create LIME explainer
        explainer = LimeTextExplainer(
            class_names=['Human-written', 'AI-generated'],
            split_expression=r'\s+',
            bow=False
        )
        
        print("\n✓ LIME explainer created")
        
        # Generate explanation
        num_features = max(1, min(len(tokens), 15))
        print(f"Generating explanation with {num_features} features...")
        
        explanation = explainer.explain_instance(
            text,
            lambda texts: predict_for_lime_test(classifier, texts),
            labels=(0, 1),
            num_features=num_features,
            num_samples=50  # Reduced for faster testing
        )
        
        print("✓ Explanation generated successfully")
        
        # Get prediction
        prediction_proba = predict_for_lime_test(classifier, [text])[0]
        print(f"\nPrediction probabilities:")
        print(f"  HUMAN: {prediction_proba[0]:.2%}")
        print(f"  AI:    {prediction_proba[1]:.2%}")
        
        # Show top features
        print(f"\nTop features for AI-generated class:")
        if 1 in explanation.local_exp:
            for idx, weight in explanation.local_exp[1][:5]:
                if 0 <= idx < len(tokens):
                    indicator = "🔴" if weight > 0 else "🟢"
                    print(f"  {indicator} {tokens[idx]}: {weight:.3f}")
        
        print("\n✓ LIME integration test PASSED")
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ LIME explanation failed: {e}")
        print(traceback.format_exc())
        return False


def test_batch_prediction(classifier):
    """Test batch prediction for LIME"""
    print("\n" + "="*60)
    print("TEST 4: Batch Prediction (for LIME)")
    print("="*60)
    
    if classifier is None:
        print("✗ Skipping (classifier not loaded)")
        return False
    
    try:
        predictions = predict_for_lime_test(classifier, TEST_TEXTS)
        
        print(f"Tested {len(TEST_TEXTS)} texts")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Expected shape: ({len(TEST_TEXTS)}, 2)")
        
        for i, (text, pred) in enumerate(zip(TEST_TEXTS, predictions)):
            print(f"\nText {i+1}: {text}")
            print(f"  HUMAN: {pred[0]:.2%}, AI: {pred[1]:.2%}")
        
        assert predictions.shape == (len(TEST_TEXTS), 2), "Incorrect output shape"
        print("\n✓ Batch prediction works")
        return True
        
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("LIME INTEGRATION TEST SUITE")
    print("="*60)
    
    # Run tests
    classifier = test_classifier_loading()
    test_basic_prediction(classifier)
    test_batch_prediction(classifier)
    test_lime_explainer(classifier)
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)
    print("\nIf all tests passed, you can now:")
    print("1. Start the server: python app.py")
    print("2. Test the API: python example_explanation.py")
    print("3. View API docs: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
