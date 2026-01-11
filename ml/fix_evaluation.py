"""
Fix ROC Curve Evaluation

This script diagnoses and fixes ROC curve calculation issues:
1. Verify correct positive class identification
2. Check that probabilities are being used (not hard predictions)
3. Ensure we're evaluating on the test set (not train)
4. Add comprehensive diagnostics
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report

print("="*70)
print("ROC CURVE EVALUATION DIAGNOSTICS")
print("="*70)

# Load the model and data
MODEL_DIR = Path('models/bilstm_sinhala')
DATA_DIR = Path('dataset/final')

print("\n1. Loading model and data...")

# Load label encoder
label_encoder = joblib.load(MODEL_DIR / 'label_encoder.joblib')
print(f"   Classes: {label_encoder.classes_}")
print(f"   Class encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Identify AI class index
ai_class_idx = label_encoder.transform(['AI'])[0]
human_class_idx = label_encoder.transform(['HUMAN'])[0]
print(f"\n   AI class index: {ai_class_idx}")
print(f"   HUMAN class index: {human_class_idx}")

# Load test data
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

test_df = load_jsonl(DATA_DIR / 'test.jsonl')
print(f"\n   Test set size: {len(test_df):,}")
print(f"   Test label distribution: {dict(test_df['label'].value_counts())}")

# Encode test labels
y_test = label_encoder.transform(test_df['label'])
print(f"   Encoded labels - AI: {(y_test == ai_class_idx).sum()}, HUMAN: {(y_test == human_class_idx).sum()}")

# Load model
print("\n2. Loading trained model...")
try:
    model = tf.keras.models.load_model(MODEL_DIR / 'saved_model')
    print("   Model loaded successfully")
except Exception as e:
    print(f"   ERROR: Could not load model: {e}")
    print("\n   Please ensure you have trained the model first!")
    exit(1)

# Load vectorizer
print("\n3. Loading text vectorizer...")
with open(MODEL_DIR / 'vectorizer_config.json', 'r', encoding='utf-8') as f:
    vectorizer_config = json.load(f)

text_vectorizer = tf.keras.layers.TextVectorization.from_config(vectorizer_config)
vectorizer_weights = np.load(MODEL_DIR / 'vectorizer_weights.npz', allow_pickle=True)
text_vectorizer.set_weights([vectorizer_weights[f'arr_{i}'] for i in range(len(vectorizer_weights.files))])

print("   Vectorizer loaded successfully")

# Create test dataset
print("\n4. Creating test dataset...")
BATCH_SIZE = 64

def make_test_dataset(texts, labels):
    ds = tf.data.Dataset.from_tensor_slices((texts.values, labels))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds.map(lambda x, y: (text_vectorizer(x), y))

test_ds = make_test_dataset(test_df['text'], y_test)
print(f"   Test dataset created with {len(test_df)} samples")

# Get predictions
print("\n5. Getting model predictions...")
test_probs = model.predict(test_ds, verbose=0)
test_pred = np.argmax(test_probs, axis=1)

print(f"   Predictions shape: {test_probs.shape}")
print(f"   Predictions are probabilities: {np.all((test_probs >= 0) & (test_probs <= 1))}")
print(f"   Probabilities sum to 1: {np.allclose(test_probs.sum(axis=1), 1.0)}")

# Diagnostic information
print("\n6. Analyzing predictions...")
print(f"   Predicted labels: {np.unique(test_pred, return_counts=True)}")
print(f"   Accuracy: {(test_pred == y_test).mean():.4f}")

# Check for perfect predictions (suspicious)
correct = (test_pred == y_test).sum()
total = len(y_test)
print(f"   Correct: {correct}/{total} ({correct/total*100:.2f}%)")

if correct == total:
    print("   ⚠️  WARNING: 100% accuracy detected!")
    print("   This suggests:")
    print("     - You may be using the wrong dataset (train instead of test)")
    print("     - Or the model is still learning shortcuts")

# Probability analysis
print("\n7. Analyzing probability distributions...")

ai_probs = test_probs[:, ai_class_idx]
human_probs = test_probs[:, human_class_idx]

print(f"\n   AI class probabilities:")
print(f"     Min: {ai_probs.min():.6f}")
print(f"     Max: {ai_probs.max():.6f}")
print(f"     Mean: {ai_probs.mean():.6f}")
print(f"     Unique values: {len(np.unique(ai_probs))}")

# Check if probabilities are too extreme (0 or 1)
extreme_probs = ((ai_probs < 0.01) | (ai_probs > 0.99)).sum()
print(f"     Extreme probabilities (<0.01 or >0.99): {extreme_probs}/{len(ai_probs)} ({extreme_probs/len(ai_probs)*100:.1f}%)")

if extreme_probs / len(ai_probs) > 0.9:
    print("   ⚠️  WARNING: Most probabilities are extreme (near 0 or 1)!")
    print("   This suggests the model is overconfident")

# Probability distribution by true label
print(f"\n   AI probability by true label:")
print(f"     When true=AI:    mean={ai_probs[y_test == ai_class_idx].mean():.6f}")
print(f"     When true=HUMAN: mean={ai_probs[y_test == human_class_idx].mean():.6f}")

# Calculate ROC curve with CORRECT positive class (AI)
print("\n8. Calculating ROC curve (with AI as positive class)...")

# Use AI class probabilities
fpr, tpr, thresholds = roc_curve(
    y_true=y_test,
    y_score=ai_probs,
    pos_label=ai_class_idx  # CRITICAL: Specify AI as positive class
)

roc_auc = auc(fpr, tpr)

print(f"   AUC (using AI as positive): {roc_auc:.6f}")

# Also calculate with sklearn's roc_auc_score for verification
roc_auc_sklearn = roc_auc_score(
    y_true=(y_test == ai_class_idx).astype(int),  # Binary: 1 if AI, 0 if HUMAN
    y_score=ai_probs
)
print(f"   AUC (sklearn verification): {roc_auc_sklearn:.6f}")

# Print ROC curve points
print(f"\n   ROC curve points:")
print(f"     Total thresholds: {len(thresholds)}")
print(f"     FPR range: [{fpr.min():.6f}, {fpr.max():.6f}]")
print(f"     TPR range: [{tpr.min():.6f}, {tpr.max():.6f}]")

# Check for perfect ROC (TPR=1.0 at FPR=0.0)
if tpr[fpr < 0.01].max() > 0.99:
    print("   ⚠️  WARNING: Near-perfect ROC curve detected!")
    print("   TPR is very high even at very low FPR")

# Classification report
print("\n9. Classification Report:")
report = classification_report(
    y_test,
    test_pred,
    target_names=label_encoder.classes_,
    digits=4
)
print(report)

# Check for data contamination
print("\n10. Checking for data contamination...")

# Load train data to check for overlap
train_df = load_jsonl(DATA_DIR / 'train.jsonl')
train_texts = set(train_df['text'].values)
test_texts = set(test_df['text'].values)

overlap = train_texts & test_texts
print(f"   Train-Test overlap: {len(overlap)} texts")

if len(overlap) > 0:
    print("   ⚠️  CRITICAL: Found overlapping texts between train and test!")
    print("   Examples:")
    for text in list(overlap)[:3]:
        print(f"     - {text[:80]}...")
else:
    print("   ✓ No overlap detected")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n✓ Correct positive class identified: AI (index {ai_class_idx})")
print(f"✓ Using probabilities: {test_probs.shape}")
print(f"✓ Evaluating on test set: {len(test_df)} samples")
print(f"\nFinal AUC (AI as positive): {roc_auc:.6f}")

if roc_auc > 0.98:
    print("\n⚠️  AUC is still very high (>0.98)!")
    print("Possible causes:")
    print("  1. Model is learning remaining shortcuts")
    print("  2. Task is genuinely very easy")
    print("  3. Data contamination still exists")
    print("\nRecommendations:")
    print("  - Manually inspect misclassified examples")
    print("  - Check for class-specific keywords")
    print("  - Verify preprocessing was applied correctly")
elif roc_auc > 0.95:
    print("\n✓ AUC is high (0.95-0.98) - Model performs very well")
    print("  This is acceptable if no obvious shortcuts remain")
elif roc_auc > 0.85:
    print("\n✓ AUC is good (0.85-0.95) - Healthy semantic learning")
    print("  Model is learning content differences")
elif roc_auc > 0.75:
    print("\n⚠️  AUC is moderate (0.75-0.85) - Room for improvement")
    print("  Consider: architecture changes, hyperparameter tuning")
else:
    print("\n⚠️  AUC is low (<0.75) - Model needs improvement")
    print("  Check: model architecture, training process, data quality")

print("="*70)
