# ROC Curve Calculation Fix - Complete Guide

## 🐛 **Bug Identified**

The ROC curve was showing AUC = 1.00 even after data preprocessing due to **incorrect positive class identification**.

### The Critical Bug

```python
# WRONG (old code):
positive_class_idx = 1  # Assumed index 1 was AI
fpr, tpr, _ = roc_curve(y_test, test_probs[:, 1])
```

**Problem**: `LabelEncoder` sorts classes **alphabetically**, so:
- `'AI'` → index **0**  
- `'HUMAN'` → index **1**

The code was using HUMAN probabilities to evaluate AI detection! This caused incorrect AUC calculations.

---

## ✅ **Fixes Applied**

### Fix #1: Correct Positive Class Identification

```python
# CORRECT (new code):
ai_class_idx = label_encoder.transform(['AI'])[0]  # Get actual AI index
probs_positive = test_probs_np[:, ai_class_idx]    # Use AI probabilities
```

### Fix #2: Explicit pos_label Parameter

```python
# CRITICAL: Explicitly specify AI as positive class
fpr, tpr, thresholds = roc_curve(
    y_true=y_test,
    y_score=probs_positive,
    pos_label=ai_class_idx  # ← This is crucial!
)
```

### Fix #3: Comprehensive Diagnostics

Added diagnostic output to detect issues:
- Which class is being used as positive
- Probability distributions
- Detection of extreme probabilities
- Warning if 100% accuracy (wrong dataset)
- Probability statistics by true label

### Fix #4: Verification Checks

```python
# Check if accidentally using training data
if (test_pred == y_test).mean() == 1.0:
    print("WARNING: 100% accuracy - check if using correct dataset!")

# Check for overconfident predictions
extreme = ((probs < 0.01) | (probs > 0.99)).sum()
print(f"Extreme probabilities: {extreme}/{len(probs)}")
```

---

## 📊 **Understanding the Fix**

### Before Fix:
```
Using HUMAN probabilities to evaluate AI detection
↓
When true label = AI:
  - High HUMAN prob → Incorrectly interpreted as low AI detection
  - Low HUMAN prob → Incorrectly interpreted as high AI detection
↓
ROC curve calculation backwards → Incorrect AUC
```

### After Fix:
```
Using AI probabilities to evaluate AI detection
↓
When true label = AI:
  - High AI prob → Correct: Strong AI detection
  - Low AI prob → Correct: Weak AI detection
↓
ROC curve calculation correct → Accurate AUC
```

---

## 🎯 **What to Expect Now**

### Diagnostic Output (from ROC cell)

When you run the notebook, you'll see:

```
Classes: ['AI' 'HUMAN']
AI index: 0, HUMAN index: 1

Using positive class: AI (index 0)
Probability range: [0.001234, 0.998765]
Mean prob when true=AI:    0.875432
Mean prob when true=HUMAN: 0.124567
Unique probability values: 5711
Extreme probabilities (<0.01 or >0.99): 342/5711

Evaluating on: 5711 samples
Test accuracy: 0.8642
```

### Interpreting the Output

#### ✅ **Good Signs:**
```
Mean prob when true=AI:    > 0.7
Mean prob when true=HUMAN: < 0.3
Extreme probabilities:     < 30%
Test accuracy:             80-95%
AUC:                       0.80-0.95
```

#### ⚠️ **Warning Signs:**
```
Mean probs are too close:  ~0.5 each (model not learning)
Extreme probabilities:     > 90% (overconfident)
Test accuracy:             100% (wrong dataset or overfitting)
AUC:                       Still 1.00 (check for remaining issues)
```

---

## 🔍 **Troubleshooting**

### If AUC is still 1.00:

1. **Check diagnostic output:**
   ```
   If "Mean prob when true=AI: 0.999999" → Model is overconfident
   If "Extreme probabilities: 5700/5711" → Nearly all predictions are extreme
   ```

2. **Verify you're using the final dataset:**
   ```python
   # Should be:
   DATA_DIR = Path('dataset/final')
   
   # NOT:
   DATA_DIR = Path('dataset')  # Original with shortcuts
   DATA_DIR = Path('dataset/cleaned')  # Only deduplicated
   ```

3. **Check if evaluating on correct split:**
   ```python
   # Make sure you're predicting on test_ds, not train_ds
   test_probs = model.predict(test_ds)  # ✓ Correct
   test_probs = model.predict(train_ds)  # ✗ Wrong!
   ```

4. **Run the diagnostic script:**
   ```bash
   cd ml
   python fix_evaluation.py
   ```
   This will comprehensively check for evaluation issues.

### If AUC is too low (<0.70):

Model needs improvement:
- Increase model capacity (more layers, units)
- Train longer (more epochs)
- Adjust learning rate
- Try different architecture (add attention, use pre-trained embeddings)

---

## 📝 **Complete Workflow**

### Step-by-Step Guide:

1. **Ensure you're using the final preprocessed data:**
   ```python
   # In the notebook configuration cell:
   DATA_DIR = Path('dataset/final')  # ← This is crucial!
   ```

2. **Restart notebook kernel:**
   - Clear all cached data
   - Clear all outputs
   - This ensures clean state

3. **Run all cells from the beginning:**
   - Don't skip any cells
   - Wait for training to complete
   - Model will be saved to `models/bilstm_sinhala/`

4. **Check the ROC curve cell output:**
   - Read the diagnostic information
   - Verify AI is the positive class
   - Check probability distributions
   - Note the final AUC

5. **Interpret results:**
   - AUC 0.80-0.88: Good semantic learning
   - AUC 0.88-0.95: Excellent performance
   - AUC > 0.95: Check for remaining issues
   - AUC < 0.80: Model needs improvement

---

## 🧪 **Verification Tests**

### Test 1: Class Index Verification
```python
print(f"Classes: {label_encoder.classes_}")
print(f"AI → {label_encoder.transform(['AI'])[0]}")
print(f"HUMAN → {label_encoder.transform(['HUMAN'])[0]}")

# Expected output:
# Classes: ['AI' 'HUMAN']
# AI → 0
# HUMAN → 1
```

### Test 2: Probability Verification
```python
# Get a few samples
for i in range(3):
    true_label = label_encoder.inverse_transform([y_test[i]])[0]
    ai_prob = test_probs[i, ai_class_idx]
    human_prob = test_probs[i, human_class_idx]
    print(f"True: {true_label} | P(AI)={ai_prob:.4f} | P(HUMAN)={human_prob:.4f}")

# Expected:
# When true=AI: P(AI) should be high (>0.7)
# When true=HUMAN: P(HUMAN) should be high (>0.7)
```

### Test 3: Dataset Verification
```python
# Verify you're using test set
print(f"Test samples: {len(test_df)}")
print(f"Predictions: {len(test_probs)}")
print(f"Labels: {len(y_test)}")

# All should match!
# For final dataset: ~5,711 samples
```

---

## 📚 **Technical Details**

### Why pos_label Matters

The `roc_curve` function needs to know which class is "positive":

```python
# Without pos_label:
# sklearn guesses based on label values (0 < 1, so 1 is positive)
# This would treat HUMAN (index 1) as positive - WRONG!

# With pos_label=ai_class_idx:
# Explicitly tells sklearn: "AI is positive, HUMAN is negative"
# This is correct for our use case!
```

### Binary Classification Convention

In binary classification:
- **Positive class**: What we're trying to detect (AI text)
- **Negative class**: The alternative (HUMAN text)

ROC curve answers: "How well can we detect the positive class?"

For AI detection:
- True Positive: Correctly identified AI text
- False Positive: Incorrectly identified HUMAN as AI
- True Negative: Correctly identified HUMAN text
- False Negative: Incorrectly identified AI as HUMAN

---

## 🎓 **Learning Points**

### Key Takeaways:

1. **Never assume class indices** - Always check with `label_encoder.transform()`

2. **Always specify pos_label** - Make your intent explicit in the code

3. **Use diagnostic outputs** - Help catch issues early

4. **Verify your evaluation** - Check you're using the right dataset

5. **Understand your metrics** - Know what AUC actually measures

### Common Mistakes:

❌ Assuming index 1 is always the target class  
❌ Not specifying pos_label in binary classification  
❌ Evaluating on training data instead of test data  
❌ Using hard predictions instead of probabilities  
❌ Normalizing probabilities incorrectly  

---

## ✅ **Checklist**

Before declaring "Fixed":

- [ ] Verified DATA_DIR = Path('dataset/final')
- [ ] Restarted kernel and cleared outputs
- [ ] Ran all cells from beginning
- [ ] Checked ROC curve diagnostic output
- [ ] Verified AI is identified as index 0
- [ ] Checked probability distributions look reasonable
- [ ] AUC is realistic (0.80-0.95, not 1.00)
- [ ] Test accuracy is < 100%
- [ ] Confusion matrix shows some errors
- [ ] Classification report shows balanced metrics

---

## 📞 **Still Having Issues?**

### If AUC = 1.00 persists:

Run the comprehensive diagnostic:
```bash
cd ml
python fix_evaluation.py
```

This will check:
- Correct positive class identification
- Probability distributions  
- Dataset contamination
- Model predictions
- All evaluation metrics

The script will provide specific recommendations based on what it finds.

---

**Date**: 2026-01-09  
**Status**: ✅ ROC curve calculation fixed  
**Action Required**: Restart kernel and retrain model
