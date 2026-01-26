#!/usr/bin/env python3
"""
Sinhala Text Classification Script using SinBERT Model
Classifies text as HUMAN-generated or AI-generated
"""

import os
import sys
import warnings
import argparse
from pathlib import Path
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set environment variable for TensorFlow/Keras compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "ml" / "models" / "sinbert_sinhala_classifier"


def _resolve_model_path(raw_path: str) -> str:
    """Return absolute model path; accept relative paths for convenience."""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


class SinhalaTextClassifier:
    """
    A classifier for Sinhala text to detect if it's human or AI generated.
    Uses SinBERT (Sinhala BERT) model for classification.
    """
    def __init__(self, model_path=None):
        raw_model_path = model_path or str(DEFAULT_MODEL_DIR)
        self.model_path = _resolve_model_path(raw_model_path)
        self.label_mapping = {0: 'HUMAN', 1: 'AI'}
        
        print("Loading model and tokenizer...")
        print(f"Model path: {self.model_path}")
        
        try:
            # Verify model directory exists
            model_dir = Path(self.model_path)
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            # Load tokenizer from local path
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=False
            )
            
            # Load model from local path
            self.model = TFAutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=False
            )
            print("✓ Model loaded successfully!\n")
            print(self.model)
        except Exception as e:
            error_msg = f"Error loading model: {e}"
            print(f"✗ {error_msg}")
            
            # Check if it's a Keras compatibility issue
            if "tf-keras" in str(e) or "tf_keras" in str(e):
                print("\n" + "=" * 60)
                print("KERAS COMPATIBILITY ISSUE DETECTED")
                print("=" * 60)
                print("Solution: Install tf-keras package")
                print("Run: pip install tf-keras")
                print("=" * 60 + "\n")
            
            # Raise exception instead of sys.exit to allow FastAPI to handle it
            raise RuntimeError(error_msg) from e
    
    def preprocess_text(self, text):
        """
        Preprocess text before classification.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            dict: Tokenized inputs with input_ids and attention_mask
        """
        # Tokenize text
        encodings = self.tokenizer(
            text,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='tf'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
    
    def classify(self, text, return_probabilities=False):
        """
        Classify a given text as HUMAN or AI generated.
        
        Args:
            text (str): Text to classify
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Classification results containing label, confidence, and optionally probabilities
        """
        # Preprocess the text
        inputs = self.preprocess_text(text)
        
        # Make prediction
        predictions = self.model.predict(inputs, verbose=0)
        
        # Get logits and convert to probabilities
        logits = predictions.logits[0]
        probabilities = tf.nn.softmax(logits).numpy()
        
        # Get predicted class and confidence
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        predicted_label = self.label_mapping[predicted_class]
        
        result = {
            'label': predicted_label,
            'confidence': float(confidence)
        }
        
        if return_probabilities:
            result['probabilities'] = {
                'HUMAN': float(probabilities[0]),
                'AI': float(probabilities[1])
            }
        
        return result
    
    def classify_batch(self, texts, return_probabilities=False):
        """
        Classify multiple texts at once.
        
        Args:
            texts (list): List of texts to classify
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            list: List of classification results for each text
        """
        results = []
        for text in texts:
            result = self.classify(text, return_probabilities)
            results.append(result)
        return results


def main():
    """
    Main function to run the classifier from command line.
    """
    parser = argparse.ArgumentParser(
        description='Classify Sinhala text as HUMAN or AI generated'
    )
    parser.add_argument(
        'text',
        nargs='?',
        help='Text to classify (optional if using --file)'
    )
    parser.add_argument(
        '--file',
        '-f',
        help='Path to file containing text to classify'
    )    
    parser.add_argument(
        '--model',
        '-m',
        default='ml/models/sinbert_sinhala_classifier',
        help='Path to the SinBERT model directory (default: ml/models/sinbert_sinhala_classifier)'
    )
    parser.add_argument(
        '--probabilities',
        '-p',
        action='store_true',
        help='Show probabilities for both classes'
    )
    
    args = parser.parse_args()
    
    # Get text to classify
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        print("Error: Please provide text to classify or use --file option")
        parser.print_help()
        sys.exit(1)
    
    # Initialize classifier
    classifier = SinhalaTextClassifier(model_path=args.model)
    
    # Classify text
    print("Classifying text...")
    print(f"Input: {text[:100]}{'...' if len(text) > 100 else ''}\n")
    
    result = classifier.classify(text, return_probabilities=args.probabilities)
    
    # Display results
    print("=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    if args.probabilities and 'probabilities' in result:
        print("\nClass Probabilities:")
        print(f"  HUMAN: {result['probabilities']['HUMAN']:.2%}")
        print(f"  AI:    {result['probabilities']['AI']:.2%}")
    
    print("=" * 50)


if __name__ == '__main__':
    main()
