# -*- coding: utf-8 -*-
"""
Model Usage / Inference Script
Load the trained TF-IDF + LogisticRegression model and make predictions on new Sinhala text.
"""

import joblib
from pathlib import Path
import re


def load_model(model_path='models/tfidf_logreg_sinhala.joblib'):
    """
    Load the trained model and vectorizer from a joblib file.
    
    Args:
        model_path (str): Path to the saved model bundle
        
    Returns:
        tuple: (vectorizer, model)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    bundle = joblib.load(model_path)
    vectorizer = bundle['vectorizer']
    model = bundle['model']
    
    print(f"Model loaded from {model_path}")
    return vectorizer, model


def clean_text(s):
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


def predict(text, vectorizer, model, return_probabilities=False):
    """
    Predict whether text is AI-generated or Human-written.
    
    Args:
        text (str): Input Sinhala text
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained LogisticRegression model
        return_probabilities (bool): If True, return class probabilities
        
    Returns:
        str or dict: Predicted label ('AI' or 'HUMAN') or dict with label and probabilities
    """
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    
    if return_probabilities:
        probabilities = model.predict_proba(text_tfidf)[0]
        class_labels = model.classes_
        prob_dict = {label: prob for label, prob in zip(class_labels, probabilities)}
        return {
            'prediction': prediction,
            'probabilities': prob_dict,
            'confidence': max(probabilities)
        }
    
    return prediction


def batch_predict(texts, vectorizer, model, return_probabilities=False):
    """
    Make predictions on multiple texts.
    
    Args:
        texts (list): List of input texts
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained LogisticRegression model
        return_probabilities (bool): If True, return probabilities for each prediction
        
    Returns:
        list: List of predictions or dicts with predictions and probabilities
    """
    results = []
    for text in texts:
        result = predict(text, vectorizer, model, return_probabilities)
        results.append(result)
    return results


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("Model Usage Example")
    print("=" * 60)
    
    # Load model
    vectorizer, model = load_model('models/tfidf_logreg_sinhala.joblib')
    
    # Example 1: Single prediction without probabilities
    print("\n--- Example 1: Single Prediction ---")
    sample_text = "අනාගතයේ නගරය අතිශය නවීන තාක්ෂණික විප්ලවයක මධ්‍යස්ථානයකි. ගගනචුම්බී ගොඩනැගිලි සූර්ය ශක්තියෙන් ක්‍රියා කරමින්, වායු දූෂණය අවම කරයි. මෝටර් රථ නොව, ගුවන් මාර්ගවල ගමන් කරන නවීන ගුවන් රථ භාවිතා වේ. නගරයේ සියලුම පද්ධති – විදුලි, ජලය, ආහාර බෙදාහැරීම – කෘත්‍රිම බුද්ධියෙන් පාලනය වේ. පුරවැසියන්ට නිදහස් වෙලාව වැඩි වන අතර, පරිසරය සහ තාක්ෂණය එකට එකතු වී ශාන්ත සහ සුරක්ෂිත ජීවිතයකට පත් කරයි. මේ නගරය යනු මානව සහ යන්ත්‍ර අතර සම්පූර්ණ සමගියක ලෝකයකට මඟ පෙන්වන අනාගතයේ දර්ශනයකි."
    prediction = predict(sample_text, vectorizer, model)
    print(f"Text: {sample_text}")
    print(f"Prediction: {prediction}")
    
