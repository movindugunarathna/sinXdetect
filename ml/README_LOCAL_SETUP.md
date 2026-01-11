# Local Setup Instructions for BiLSTM Text Classifier

## Prerequisites

- Python 3.7+ installed
- `pip` package manager

## Step-by-Step Setup

### 1. Navigate to the ML Directory
```bash
cd d:\MTG\sinXdetect\sinXdetect\ml
```

### 2. Install Required Dependencies
```bash
pip install pandas scikit-learn tensorflow matplotlib seaborn joblib
```

For CPU-only (faster installation):
```bash
pip install pandas scikit-learn tensorflow-cpu matplotlib seaborn joblib
```

### 3. Prepare Your Data

Place your data files in the `dataset/` folder. You have two options:

#### Option A: Use Pre-Split Data (Recommended)
Create three JSONL files with exact names:
- `dataset/train.jsonl`
- `dataset/val.jsonl`
- `dataset/test.jsonl`

Each file should contain one JSON object per line:
```json
{"text": "your text here", "label": "category"}
{"text": "another text", "label": "category"}
```

#### Option B: Auto-Split from Original Data
Place your original data in:
- `dataset/train_original.jsonl`

The notebook will automatically:
- Remove duplicates
- Create stratified train/val/test splits (70/15/15)
- Ensure zero data leakage

### 4. Run the Notebook

**Option A: Jupyter Notebook**
```bash
jupyter notebook bilstm_text_classifier.ipynb
```

**Option B: VS Code**
- Open this folder in VS Code
- Click "Run All" or run cells sequentially

**Option C: Programmatically**
```bash
jupyter nbconvert --to notebook --execute bilstm_text_classifier.ipynb
```

### 5. Verify Setup

After running the first few cells, you should see:
```
✓ Found pre-split data (train.jsonl, val.jsonl, test.jsonl)
Data directory: /path/to/dataset
Model directory: /path/to/models/bilstm_sinhala

Dataset sizes:
  Train :  XX,XXX samples
  Val   :   X,XXX samples
  Test  :   X,XXX samples
```

## Troubleshooting

### Error: "No data found in dataset/"
- **Cause:** Data files are missing
- **Fix:** Place either `train_original.jsonl` or the three split files in `dataset/`

### Error: "Missing train.jsonl at..."
- **Cause:** Pre-split files don't exist
- **Fix:** Either:
  1. Place `train_original.jsonl` and let the notebook create splits, OR
  2. Manually create the three split files

### Error: "ModuleNotFoundError: No module named 'tensorflow'"
- **Cause:** Dependencies not installed
- **Fix:** Run `pip install tensorflow matplotlib seaborn`

### Error: Out of memory during training
- **Fix:** Reduce `BATCH_SIZE` in the Configuration cell (try 32 or 16)

### Very slow training
- **Cause:** Running on CPU
- **Fix:** Install GPU support: `pip install tensorflow[and-cuda]` (requires NVIDIA GPU + CUDA toolkit)

## Expected Output

The notebook will produce:

1. **Leakage Detection Report** - Shows any data contamination between splits
2. **Data Quality Diagnostics** - Identifies trivial patterns that might inflate metrics
3. **Training Curves** - Shows loss/accuracy over epochs
4. **Confusion Matrix** - Shows per-class prediction accuracy
5. **ROC Curve** - Shows classifier discrimination ability
6. **Saved Model** - Stored in `models/bilstm_sinhala/`

## Key Evaluation Metrics

- **Test Accuracy:** Percentage of correct predictions
- **ROC AUC:** Area under ROC curve (1.0 = perfect, 0.5 = random)
- **Classification Report:** Precision, Recall, F1-score per class

⚠️ **If AUC = 1.0:** Your data likely has:
- Trivial separability (e.g., class-specific keywords)
- Leakage between splits
- Duplicate samples

Run the "Data Quality Diagnostics" cell to identify the issue.

## Output Files

After successful training:
```
models/bilstm_sinhala/
├── checkpoint.keras          # Best model weights
├── saved_model/              # Complete model
├── label_encoder.joblib      # Label encoding
├── vectorizer_config.json    # Text preprocessing config
└── vectorizer_weights.npz    # Vocabulary weights
```

## Next Steps

1. **Evaluate Model:** Run the ROC curve and confusion matrix cells
2. **Test Inference:** Use the "Inference and Testing" cell with custom text
3. **Cross-Validation:** Uncomment the k-fold cell for robust evaluation
4. **Production:** Use the saved model with inference code

## Notes

- The notebook uses stratified k-fold to ensure balanced class distribution
- `TextVectorization` is fitted **only on training data** to prevent leakage
- All metrics use the held-out test set (not training data)
- Models are regularized with dropout and L2 to prevent overfitting
