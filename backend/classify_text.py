#!/usr/bin/env python3

import os
import sys
import warnings
import argparse
import json
import joblib
from pathlib import Path
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "ml" / "models" / "bilstm_sinhala_model"


def _resolve_model_path(raw_path: str) -> str:
    """Return absolute model path; accept relative paths for convenience."""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


class SinhalaTextClassifier:
    """
    A classifier for Sinhala text to detect if it's human or AI generated.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the classifier with the trained BiLSTM model.
        
        Args:
            model_path (str): Path to the saved BiLSTM model directory
        """
        raw_model_path = model_path or str(DEFAULT_MODEL_DIR)
        self.model_path = Path(_resolve_model_path(raw_model_path))
        
        print("Loading BiLSTM model and preprocessing assets...")
        try:
            # Load the Keras model
            model_file = self.model_path / 'saved_model.keras'
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            self.model = tf.keras.models.load_model(model_file)
            print("✓ Model loaded successfully!")
            
            # Load label encoder
            label_encoder_file = self.model_path / 'label_encoder.joblib'
            if not label_encoder_file.exists():
                raise FileNotFoundError(f"Label encoder not found: {label_encoder_file}")
            
            self.label_encoder = joblib.load(label_encoder_file)
            print("✓ Label encoder loaded!")
            
            # Load text vectorizer configuration and weights
            vectorizer_config_file = self.model_path / 'vectorizer_config.json'
            vectorizer_weights_file = self.model_path / 'vectorizer_weights.npz'
            
            if not vectorizer_config_file.exists():
                raise FileNotFoundError(f"Vectorizer config not found: {vectorizer_config_file}")
            if not vectorizer_weights_file.exists():
                raise FileNotFoundError(f"Vectorizer weights not found: {vectorizer_weights_file}")
            
            with open(vectorizer_config_file, 'r', encoding='utf-8') as f:
                vectorizer_config = json.load(f)
            
            # Create the text vectorizer with the same configuration
            self.text_vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=vectorizer_config['max_tokens'],
                output_mode=vectorizer_config['output_mode'],
                output_sequence_length=vectorizer_config['output_sequence_length'],
                standardize=vectorizer_config.get('standardize', 'lower_and_strip_punctuation')
            )
            
            # Load weights which should contain the vocabulary
            weights_data = np.load(vectorizer_weights_file, allow_pickle=True)
            
            if len(weights_data.files) == 0:
                # Weights file is empty - need to re-adapt the vectorizer
                print("⚠ Vectorizer weights file is empty. The model needs to be retrained with proper vectorizer saving.")
                print("⚠ Using BERT model as fallback...")
                # Try to use BERT model instead
                bert_model_path = self.model_path.parent / 'bert_multilingual_model'
                if bert_model_path.exists():
                    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
                    self.tokenizer = AutoTokenizer.from_pretrained(str(bert_model_path))
                    self.model = TFAutoModelForSequenceClassification.from_pretrained(str(bert_model_path))
                    self.use_bert = True
                    print("✓ BERT model loaded as fallback!")
                    return
                else:
                    raise ValueError("BiLSTM vectorizer weights are missing and BERT model not found. Please retrain the BiLSTM model.")
            
            weights = [weights_data[key] for key in weights_data.files]
            
            # Extract vocabulary from weights
            # The first weight array contains the vocabulary
            vocab = weights[0]
            
            # Adapt using the vocabulary
            self.text_vectorizer.set_vocabulary(vocab)
            self.use_bert = False
            print("✓ Text vectorizer loaded!")
            
            print("✓ All components loaded successfully!\n")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def preprocess_text(self, text):
        """
        Preprocess text before classification.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            Vectorized text ready for model input
        """
        if hasattr(self, 'use_bert') and self.use_bert:
            # BERT preprocessing
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
        else:
            # BiLSTM preprocessing
            vectorized = self.text_vectorizer([text])
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
        if hasattr(self, 'use_bert') and self.use_bert:
            predictions = self.model.predict(inputs, verbose=0)
            logits = predictions.logits[0]
            probabilities = tf.nn.softmax(logits).numpy()
        else:
            probabilities = self.model.predict(inputs, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        result = {
            'label': predicted_label,
            'confidence': float(confidence)
        }
        
        if return_probabilities:
            probs_dict = {}
            for idx, class_name in enumerate(self.label_encoder.classes_):
                probs_dict[class_name] = float(probabilities[idx])
            result['probabilities'] = probs_dict
        
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
        default='models/bilstm_sinhala_model',
        help='Path to the BILSTM Sinhala model directory (default: models/bilstm_sinhala_model)'
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
