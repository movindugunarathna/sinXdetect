# BiLSTM Text Classifier for Sinhala Text

This script provides a classifier for Sinhala text using a Bidirectional LSTM (BiLSTM) neural network model. It can classify text as either HUMAN-generated or AI-generated.

## Features

- **BiLSTM Architecture**: Uses a bidirectional LSTM model with embedding layers
- **Batch Processing**: Can classify multiple texts at once for efficiency
- **Probability Scores**: Optional probability output for both classes
- **Command-line Interface**: Easy to use from the terminal
- **API-compatible**: Can be imported and used in other Python applications

## Requirements

```bash
pip install tensorflow numpy joblib
```

## Model Files

The script expects the following files in the model directory:

- `saved_model.keras` or `checkpoint.keras` - The trained Keras model
- `label_encoder.joblib` - The label encoder for class names
- `vectorizer_config.json` - TextVectorization layer configuration
- `vectorizer_weights.npz` - TextVectorization layer weights (vocabulary)

Default model path: `ml/models/bilstm_sinhala_model/`

## Usage

### Command Line

#### Single Text Classification

```bash
# Classify text directly
python classify_text_bilstm.py "Your Sinhala text here"

# Classify text from a file
python classify_text_bilstm.py --file input.txt

# Show probability scores
python classify_text_bilstm.py "Your text" --probabilities
```

#### Batch Classification

```bash
# Classify multiple texts from a file (one per line)
python classify_text_bilstm.py --batch texts.txt --probabilities
```

#### Custom Model Path

```bash
# Use a different model directory
python classify_text_bilstm.py "Your text" --model path/to/model
```

### Python API

```python
from classify_text_bilstm import SinhalaBiLSTMClassifier

# Initialize classifier
classifier = SinhalaBiLSTMClassifier(model_path="ml/models/bilstm_sinhala_model")

# Classify single text
result = classifier.classify("Your Sinhala text here", return_probabilities=True)
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

# Classify multiple texts at once
texts = ["Text 1", "Text 2", "Text 3"]
results = classifier.classify_batch(texts, return_probabilities=True)
for text, result in zip(texts, results):
    print(f"{text[:50]}... -> {result['label']} ({result['confidence']:.2%})")
```

## Integration with FastAPI

To use this classifier with your existing FastAPI app (`app.py`), you can create a separate endpoint:

```python
from classify_text_bilstm import SinhalaBiLSTMClassifier

# Add to app.py
_bilstm_classifier: Optional[SinhalaBiLSTMClassifier] = None

def get_bilstm_classifier() -> SinhalaBiLSTMClassifier:
    """Lazily create and cache the BiLSTM classifier."""
    global _bilstm_classifier
    if _bilstm_classifier is None:
        bilstm_model_path = REPO_ROOT / "ml" / "models" / "bilstm_sinhala_model"
        _bilstm_classifier = SinhalaBiLSTMClassifier(model_path=str(bilstm_model_path))
    return _bilstm_classifier

@app.post("/classify-bilstm", response_model=PredictionResponse)
async def classify_bilstm(request: TextRequest) -> PredictionResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    classifier = get_bilstm_classifier()
    try:
        result = classifier.classify(
            request.text, return_probabilities=request.return_probabilities
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = PredictionResponse(**result)
    _log_single(request.text, response)
    return response
```

## Model Architecture

The BiLSTM model uses the following architecture:

- **Embedding Layer**: Character/word embeddings (64 dimensions)
- **Bidirectional LSTM Layers**: Two stacked BiLSTM layers (64 units each)
- **Dropout Layers**: Regularization to prevent overfitting
- **Dense Output Layer**: Softmax activation for binary classification

## Output Format

### Single Classification

```json
{
  "label": "HUMAN",
  "confidence": 0.9234,
  "probabilities": {
    "HUMAN": 0.9234,
    "AI": 0.0766
  }
}
```

### Batch Classification

```json
[
  {
    "label": "HUMAN",
    "confidence": 0.9234,
    "probabilities": { "HUMAN": 0.9234, "AI": 0.0766 }
  },
  {
    "label": "AI",
    "confidence": 0.8891,
    "probabilities": { "HUMAN": 0.1109, "AI": 0.8891 }
  }
]
```

## Differences from BERT Classifier

| Feature      | BERT Classifier     | BiLSTM Classifier                 |
| ------------ | ------------------- | --------------------------------- |
| Model Type   | Transformer (BERT)  | Recurrent Neural Network          |
| Model Size   | ~500MB              | ~8MB                              |
| Speed        | Slower              | Faster                            |
| Accuracy     | Higher              | Good                              |
| Memory Usage | Higher              | Lower                             |
| Best For     | High accuracy needs | Resource-constrained environments |

## Troubleshooting

### Model Not Found

Ensure the model directory exists and contains all required files:

```bash
ls ml/models/bilstm_sinhala_model/
# Should show: saved_model.keras, label_encoder.joblib, vectorizer_config.json, vectorizer_weights.npz
```

### Import Errors

Install required dependencies:

```bash
pip install tensorflow numpy joblib
```

### Memory Issues

The BiLSTM model is much lighter than BERT. If you still face memory issues, try:

- Processing texts in smaller batches
- Using CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=-1`

## Performance

- **Single text classification**: ~50-100ms
- **Batch classification (100 texts)**: ~2-3 seconds
- **Memory usage**: ~200-300MB
- **Model size**: ~8MB

## License

Same as the parent project.

## Contributing

For improvements or bug reports, please submit issues or pull requests to the main repository.
