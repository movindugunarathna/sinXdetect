#!/usr/bin/env python3
"""
Sinhala Text Classification Script using BiLSTM Model
Classifies text as HUMAN-generated or AI-generated
"""

import os
import sys
import warnings
import argparse
from pathlib import Path
import numpy as np
import json

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import joblib


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "ml" / "models" / "bilstm_sinhala_model"


def _resolve_model_path(raw_path: str) -> str:
    """Return absolute model path; accept relative paths for convenience."""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


class SinhalaBiLSTMClassifier:
    """
    A BiLSTM-based classifier for Sinhala text to detect if it's human or AI generated.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the classifier with the trained BiLSTM model.
        
        Args:
            model_path (str): Path to the saved BiLSTM model directory
        """
        raw_model_path = model_path or str(DEFAULT_MODEL_DIR)
        self.model_path = Path(_resolve_model_path(raw_model_path))
        
        print("Loading BiLSTM model and preprocessing components...")
        try:
            # Load the Keras model
            model_file = self.model_path / 'saved_model.keras'
            if not model_file.exists():
                # Try alternative name
                model_file = self.model_path / 'checkpoint.keras'
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found in {self.model_path}")
            
            self.model = tf.keras.models.load_model(model_file)
            print(f"✓ Model loaded from {model_file.name}")
            
            # Load label encoder
            label_encoder_file = self.model_path / 'label_encoder.joblib'
            if not label_encoder_file.exists():
                raise FileNotFoundError(f"Label encoder not found at {label_encoder_file}")
            
            self.label_encoder = joblib.load(label_encoder_file)
            print(f"✓ Label encoder loaded: {self.label_encoder.classes_}")
            
            # Load text vectorizer configuration and weights
            vectorizer_config_file = self.model_path / 'vectorizer_config.json'
            vectorizer_weights_file = self.model_path / 'vectorizer_weights.npz'
            
            if not vectorizer_config_file.exists() or not vectorizer_weights_file.exists():
                raise FileNotFoundError("Vectorizer config or weights not found")
            
            # Reconstruct the TextVectorization layer
            with open(vectorizer_config_file, 'r', encoding='utf-8') as f:
                vectorizer_config = json.load(f)
            
            self.text_vectorizer = tf.keras.layers.TextVectorization.from_config(vectorizer_config)
            
            # Load and set the weights (vocabulary)
            vectorizer_weights = np.load(vectorizer_weights_file, allow_pickle=True)
            weights_list = [vectorizer_weights[f'arr_{i}'] for i in range(len(vectorizer_weights.files))]
            
            # Adapt with dummy data to initialize, then set weights
            self.text_vectorizer.adapt(['dummy'])
            self.text_vectorizer.set_weights(weights_list)
            print("✓ Text vectorizer loaded and configured")
            
            print("✓ All components loaded successfully!\n")
            
        except Exception as e:
            print(f"✗ Error loading model components: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def preprocess_text(self, text):
        """
        Preprocess text using the text vectorizer.
        
        Args:
            text (str or list): Input text(s) to preprocess
            
        Returns:
            tf.Tensor: Vectorized text ready for model input
        """
        if isinstance(text, str):
            text = [text]
        
        # Convert to tensor and vectorize
        text_tensor = tf.constant(text)
        vectorized = self.text_vectorizer(text_tensor)
        
        return vectorized
    
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
        
        # Get probabilities
        probabilities = predictions[0]
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        predicted_label = self.label_encoder.classes_[predicted_class_idx]
        
        result = {
            'label': predicted_label,
            'confidence': float(confidence)
        }
        
        if return_probabilities:
            result['probabilities'] = {
                class_name: float(probabilities[i])
                for i, class_name in enumerate(self.label_encoder.classes_)
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
        # Preprocess all texts at once
        inputs = self.preprocess_text(texts)
        
        # Make predictions
        predictions = self.model.predict(inputs, verbose=0)
        
        # Process results
        results = []
        for i, probabilities in enumerate(predictions):
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            predicted_label = self.label_encoder.classes_[predicted_class_idx]
            
            result = {
                'label': predicted_label,
                'confidence': float(confidence)
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    class_name: float(probabilities[j])
                    for j, class_name in enumerate(self.label_encoder.classes_)
                }
            
            results.append(result)
        
        return results


def main():
    """
    Main function to run the classifier from command line.
    """
    parser = argparse.ArgumentParser(
        description='Classify Sinhala text as HUMAN or AI generated using BiLSTM model'
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
        default=str(DEFAULT_MODEL_DIR),
        help=f'Path to the BiLSTM model directory (default: {DEFAULT_MODEL_DIR})'
    )
    parser.add_argument(
        '--probabilities',
        '-p',
        action='store_true',
        help='Show probabilities for both classes'
    )
    parser.add_argument(
        '--batch',
        '-b',
        help='Path to file containing multiple texts (one per line) for batch classification'
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = SinhalaBiLSTMClassifier(model_path=args.model)
    
    # Handle batch classification
    if args.batch:
        try:
            with open(args.batch, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Classifying {len(texts)} texts in batch mode...")
            results = classifier.classify_batch(texts, return_probabilities=args.probabilities)
            
            print("=" * 70)
            print("BATCH CLASSIFICATION RESULTS")
            print("=" * 70)
            
            for i, (text, result) in enumerate(zip(texts, results), 1):
                preview = (text[:60] + "...") if len(text) > 60 else text
                print(f"\n[{i}] {preview}")
                print(f"    Prediction: {result['label']} | Confidence: {result['confidence']:.2%}")
                
                if args.probabilities and 'probabilities' in result:
                    probs = result['probabilities']
                    print(f"    Probabilities: ", end="")
                    print(" | ".join([f"{k}: {v:.2%}" for k, v in probs.items()]))
            
            print("=" * 70)
            return
        
        except Exception as e:
            print(f"Error processing batch file: {e}")
            sys.exit(1)
    
    # Handle single text classification
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
        print("Error: Please provide text to classify or use --file/--batch option")
        parser.print_help()
        sys.exit(1)
    
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
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")
    
    print("=" * 50)


if __name__ == '__main__':
    main()
