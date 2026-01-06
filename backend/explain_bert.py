# Importing required libraries
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer
import numpy as np
import os
import json
import h5py

# Creating a FastAPI application instance
app = FastAPI(title="BERT Text Classifier Explainer")

# Enabling Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the BERT model and tokenizer
MODEL_PATH = "../ml/models/bert_multilingual_model"

# Check if model path exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Maximum sequence length for BERT
MAX_LENGTH = 128

# Load config to understand model architecture
config_path = os.path.join(MODEL_PATH, "config.json")
with open(config_path, 'r') as f:
    model_config = json.load(f)

print(f"Model type: {model_config.get('model_type', 'unknown')}")
print(f"Architecture: {model_config.get('architectures', ['unknown'])[0]}")

# Load the BERT model
try:
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    from transformers import TFAutoModelForSequenceClassification
    
    # Load model with explicit settings to avoid segfaults
    bert_model = TFAutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        from_pt=False
    )
    print("BERT model loaded successfully using transformers")
    
    # Test the model to ensure it works
    print("Testing model...")
    test_text = ["This is a test"]
    test_encoded = tokenizer(
        test_text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    test_output = bert_model(test_encoded, training=False)
    print(f"Model test successful! Output shape: {test_output.logits.shape}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define request model
class ExplainRequest(BaseModel):
    text: str

########Using LIME to explain the model decisions########

# Function to preprocess text for BERT
def preprocess_text_bert(texts):
    """
    Preprocess text for BERT model
    Args:
        texts: List of text strings
    Returns:
        Dictionary with input_ids and attention_mask
    """
    if isinstance(texts, str):
        texts = [texts]
    
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    
    return encoded

# Function to predict using BERT model
def predict(texts):
    """
    Predict probability for each text
    Args:
        texts: List of text strings
    Returns:
        Array of probabilities for each class [human, ai]
    """
    try:
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess the texts
        encoded = preprocess_text_bert(texts)
        
        # Get predictions from BERT model - use numpy to avoid graph issues
        with tf.device('/CPU:0'):  # Force CPU to avoid GPU memory issues
            outputs = bert_model(encoded, training=False)
        
        # Handle different output formats from transformers
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        
        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        
        # Format predictions as [human_prob, ai_prob]
        returnable = []
        for prob in probabilities:
            # Ensure we have a 2-class probability distribution
            if len(prob) >= 2:
                returnable.append(np.array([float(prob[0]), float(prob[1])]))
            else:
                # If single output, assume it's probability of class 1
                prob_val = float(prob[0]) if len(prob) > 0 else float(prob)
                returnable.append(np.array([1 - prob_val, prob_val]))
        
        return np.array(returnable)
    except Exception as e:
        print(f"Error in predict function: {e}")
        # Return neutral probabilities if prediction fails
        return np.array([[0.5, 0.5]] * len(texts))

# Function to get word-level importance scores
def get_word_importance(explanation, tokens):
    """
    Extract word importance scores from LIME explanation
    Args:
        explanation: LIME explanation object
        tokens: List of words in the text
    Returns:
        Dictionary mapping words to their importance scores
    """
    word_scores = {}
    
    # Get the explanation for AI-generated class (class 1)
    for word_idx, weight in explanation.local_exp[1]:
        if 0 <= word_idx < len(tokens):
            word = tokens[word_idx]
            word_scores[word] = weight
    
    return word_scores

# Handling POST request to the /explain endpoint
@app.post('/explain')
async def explain_prediction(request: ExplainRequest):
    """
    Explain the prediction for a given text using LIME
    Args:
        request: ExplainRequest with text field
    Returns:
        JSON with explanation data and highlighted text
    """
    try:
        text = request.text
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Tokenize the text (simple word-level tokenization for LIME)
        tokens = text.lower().split()
        
        # Create LimeTextExplainer instance
        explainer = LimeTextExplainer(class_names=['Human-written', 'AI-generated'])
        
        # Explain prediction with controlled settings to avoid memory issues
        explanation = explainer.explain_instance(
            text, 
            predict, 
            labels=(0, 1), 
            num_features=min(len(tokens), 15),  # Limit features
            num_samples=100  # Reduce samples for stability
        )
        
        # Get prediction probabilities
        prediction_proba = predict([text])[0]
        
        # Extract explanation information
        explanation_data = {
            'class_names': list(map(str, explanation.class_names)),
            'predicted_probability': list(map(float, prediction_proba)),
            'local_exp': {
                str(class_name): {
                    str(idx): float(weight) 
                    for idx, weight in exp
                } 
                for class_name, exp in explanation.local_exp.items()
            },
            'intercept': list(map(float, explanation.intercept)) if hasattr(explanation, 'intercept') else [0.0, 0.0]
        }
        
        # Extract local_exp from explanation_data and align with tokens
        local_exp = explanation_data.get('local_exp', {})
        aligned_local_exp = {}
        for class_name, class_exp in local_exp.items():
            aligned_exp = {}
            for word_idx, word in enumerate(tokens):
                word_key = str(word_idx)
                if word_key in class_exp:
                    aligned_exp[str(word_idx)] = class_exp[word_key]
            aligned_local_exp[class_name] = aligned_exp
        
        # Creating a dictionary to store highlighted tokens with their colors
        highlighted_text_dict = {}
        
        # Highlight words based on their importance
        for class_index, class_exp in explanation_data["local_exp"].items():
            for word_idx, weight in class_exp.items():
                idx = int(word_idx)
                if idx < len(tokens):
                    word = tokens[idx]
                    if abs(weight) > 0.01:  # Only highlight words with significant weight
                        # Determine the color based on the class index
                        # Green for human-written (class 0), Red for AI-generated (class 1)
                        if class_index == '1':  # AI-generated class
                            color = 'red' if weight > 0 else 'green'
                        else:  # Human-written class
                            color = 'green' if weight > 0 else 'red'
                        
                        # Store with absolute weight for sorting
                        if word not in highlighted_text_dict or abs(weight) > abs(highlighted_text_dict[word]['weight']):
                            highlighted_text_dict[word] = {'color': color, 'weight': weight}
        
        # Convert the dictionary to a list of dictionaries for JSON serialization
        highlighted_text = [
            {'word': word, 'color': data['color'], 'weight': float(data['weight'])} 
            for word, data in highlighted_text_dict.items()
        ]
        
        # Sort by absolute weight (most important first)
        highlighted_text.sort(key=lambda x: abs(x['weight']), reverse=True)
        
        return {
            'explanation_data': explanation_data,
            'highlighted_text': highlighted_text,
            'predicted_class': 'AI-generated' if prediction_proba[1] > 0.5 else 'Human-written',
            'confidence': float(max(prediction_proba))
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Health check endpoint
@app.get('/health')
async def health_check():
    """Check if the service is running"""
    return {"status": "healthy", "model": "BERT Multilingual"}

# Root endpoint
@app.get('/')
async def root():
    """Root endpoint with API information"""
    return {
        "message": "BERT Text Classifier Explainer API",
        "endpoints": {
            "/explain": "POST - Explain text classification",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
