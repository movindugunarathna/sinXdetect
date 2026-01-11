"""
Adversarial Preprocessing Script

This script applies adversarial preprocessing to prevent the model from learning
spurious patterns and shortcuts:

1. Normalize length distribution across classes
2. Strip formatting cues
3. Remove metadata before modeling
4. Adversarial preprocessing (punctuation normalization, etc.)
"""

import json
import re
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
INPUT_DIR = Path('dataset/cleaned')
OUTPUT_DIR = Path('dataset/final')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ADVERSARIAL PREPROCESSING")
print("="*70)

# ============================================================================
# 1. LOAD CLEAN DATA
# ============================================================================
print("\n1. Loading clean data...")

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

train_df = load_jsonl(INPUT_DIR / 'train.jsonl')
val_df = load_jsonl(INPUT_DIR / 'val.jsonl')
test_df = load_jsonl(INPUT_DIR / 'test.jsonl')

train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"   Loaded {len(all_data):,} samples")
print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# ============================================================================
# 2. ANALYZE LENGTH DISTRIBUTION
# ============================================================================
print("\n2. Analyzing length distribution by class...")

all_data['text_length'] = all_data['text'].str.len()
all_data['word_count'] = all_data['text'].str.split().str.len()

print("\n   Character Length Statistics:")
for label in ['HUMAN', 'AI']:
    subset = all_data[all_data['label'] == label]
    print(f"   {label}:")
    print(f"     Mean:   {subset['text_length'].mean():.1f}")
    print(f"     Median: {subset['text_length'].median():.1f}")
    print(f"     Std:    {subset['text_length'].std():.1f}")
    print(f"     Min:    {subset['text_length'].min()}")
    print(f"     Max:    {subset['text_length'].max()}")

print("\n   Word Count Statistics:")
for label in ['HUMAN', 'AI']:
    subset = all_data[all_data['label'] == label]
    print(f"   {label}:")
    print(f"     Mean:   {subset['word_count'].mean():.1f}")
    print(f"     Median: {subset['word_count'].median():.1f}")

# Check if lengths are significantly different
human_lengths = all_data[all_data['label'] == 'HUMAN']['text_length']
ai_lengths = all_data[all_data['label'] == 'AI']['text_length']

mean_diff = abs(human_lengths.mean() - ai_lengths.mean())
print(f"\n   Mean length difference: {mean_diff:.1f} characters")

if mean_diff > 100:
    print(f"   WARNING: Significant length difference detected!")
    print(f"   This could be a shortcut for the model.")
else:
    print(f"   OK: Length distributions are similar")

# ============================================================================
# 3. STRIP FORMATTING CUES
# ============================================================================
print("\n3. Stripping formatting cues...")

def strip_formatting_cues(text):
    """Remove formatting artifacts that might distinguish AI from human text"""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove multiple consecutive punctuation (e.g., "!!!" -> "!")
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # Remove excessive newlines (keep max 1)
    text = re.sub(r'\n+', ' ', text)
    
    # Remove tabs
    text = re.sub(r'\t+', ' ', text)
    
    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove markdown-style formatting if present
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold** -> bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic* -> italic
    text = re.sub(r'__(.*?)__', r'\1', text)      # __bold__ -> bold
    text = re.sub(r'_(.*?)_', r'\1', text)        # _italic_ -> italic
    
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', '', text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply formatting cleanup
print("   Applying formatting cleanup...")
all_data['text_cleaned'] = all_data['text'].apply(strip_formatting_cues)

# Check how many texts changed
changed = (all_data['text'] != all_data['text_cleaned']).sum()
print(f"   Modified {changed:,} texts ({changed/len(all_data)*100:.1f}%)")

# Update text column
all_data['text'] = all_data['text_cleaned']
all_data.drop(columns=['text_cleaned'], inplace=True)

# ============================================================================
# 4. ADVERSARIAL PUNCTUATION NORMALIZATION
# ============================================================================
print("\n4. Applying adversarial preprocessing...")

def normalize_punctuation(text):
    """Normalize punctuation patterns to prevent model shortcuts"""
    
    # Normalize spaces around punctuation
    # Add space after punctuation if missing
    text = re.sub(r'([.!?,:;])([A-Za-z])', r'\1 \2', text)
    
    # Remove space before punctuation
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    
    # Normalize ellipsis
    text = re.sub(r'\.{3,}', '...', text)
    
    # Normalize dashes
    text = text.replace('—', '-').replace('–', '-')
    
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def analyze_punctuation_patterns(df):
    """Analyze punctuation usage patterns by class"""
    patterns = {
        'exclamation_marks': r'!',
        'question_marks': r'\?',
        'commas': r',',
        'periods': r'\.',
        'semicolons': r';',
        'colons': r':',
        'quotes': r'["\']',
        'parentheses': r'[()]',
        'dashes': r'-',
    }
    
    results = {}
    for label in ['HUMAN', 'AI']:
        subset = df[df['label'] == label]['text']
        results[label] = {}
        
        for name, pattern in patterns.items():
            # Count average occurrences per text
            counts = subset.apply(lambda x: len(re.findall(pattern, x)))
            results[label][name] = counts.mean()
    
    return results

# Analyze before normalization
print("   Analyzing punctuation patterns...")
punct_before = analyze_punctuation_patterns(all_data)

print("\n   Punctuation usage (average per text):")
print("   Pattern          | HUMAN  | AI     | Diff")
print("   " + "-"*50)
for pattern in punct_before['HUMAN'].keys():
    human_val = punct_before['HUMAN'][pattern]
    ai_val = punct_before['AI'][pattern]
    diff = abs(human_val - ai_val)
    print(f"   {pattern:15} | {human_val:6.2f} | {ai_val:6.2f} | {diff:5.2f}")

# Apply normalization
print("\n   Normalizing punctuation...")
all_data['text'] = all_data['text'].apply(normalize_punctuation)

# ============================================================================
# 5. NORMALIZE LENGTH DISTRIBUTION
# ============================================================================
print("\n5. Normalizing length distribution across classes...")

# Define acceptable length ranges (in characters)
MIN_LENGTH = 50
MAX_LENGTH = 2000

# Filter by length
all_data = all_data[
    (all_data['text_length'] >= MIN_LENGTH) & 
    (all_data['text_length'] <= MAX_LENGTH)
].copy()

print(f"   Filtered to texts between {MIN_LENGTH}-{MAX_LENGTH} characters")
print(f"   Remaining: {len(all_data):,} samples")

# Recalculate lengths after preprocessing
all_data['text_length'] = all_data['text'].str.len()
all_data['word_count'] = all_data['text'].str.split().str.len()

# Balance length distribution by sampling
print("\n   Balancing length distributions...")

# Create length bins
all_data['length_bin'] = pd.cut(
    all_data['text_length'], 
    bins=10, 
    labels=False
)

# Sample to balance length distribution across classes
balanced_samples = []

for bin_id in all_data['length_bin'].unique():
    bin_data = all_data[all_data['length_bin'] == bin_id]
    
    # Get counts per class in this bin
    human_count = len(bin_data[bin_data['label'] == 'HUMAN'])
    ai_count = len(bin_data[bin_data['label'] == 'AI'])
    
    # Sample to match the smaller class
    target_count = min(human_count, ai_count)
    
    if target_count > 0:
        human_sample = bin_data[bin_data['label'] == 'HUMAN'].sample(
            n=target_count, random_state=42
        )
        ai_sample = bin_data[bin_data['label'] == 'AI'].sample(
            n=target_count, random_state=42
        )
        
        balanced_samples.append(human_sample)
        balanced_samples.append(ai_sample)

all_data_balanced = pd.concat(balanced_samples, ignore_index=True)

print(f"   After balancing: {len(all_data_balanced):,} samples")
print(f"   Removed: {len(all_data) - len(all_data_balanced):,} samples")

print("\n   New length statistics:")
for label in ['HUMAN', 'AI']:
    subset = all_data_balanced[all_data_balanced['label'] == label]
    print(f"   {label}:")
    print(f"     Count:  {len(subset):,}")
    print(f"     Mean:   {subset['text_length'].mean():.1f}")
    print(f"     Median: {subset['text_length'].median():.1f}")
    print(f"     Std:    {subset['text_length'].std():.1f}")

# ============================================================================
# 6. REMOVE METADATA AND KEEP ONLY TEXT + LABEL
# ============================================================================
print("\n6. Removing metadata...")

# Keep only essential columns
all_data_final = all_data_balanced[['text', 'label', 'split']].copy()

print(f"   Kept only 'text' and 'label' columns")
print(f"   All metadata removed")

# ============================================================================
# 7. RECREATE SPLITS
# ============================================================================
print("\n7. Recreating train/val/test splits...")

# Split by the original split assignment (but shuffled)
train_final = all_data_final[all_data_final['split'] == 'train'].drop(columns=['split'])
val_final = all_data_final[all_data_final['split'] == 'val'].drop(columns=['split'])
test_final = all_data_final[all_data_final['split'] == 'test'].drop(columns=['split'])

# If splits are too imbalanced after filtering, recreate them
min_split_size = min(len(train_final), len(val_final), len(test_final))

if len(val_final) < 1000 or len(test_final) < 1000:
    print("   Recreating splits due to size imbalance...")
    all_data_final = all_data_final.drop(columns=['split'])
    
    # 70/15/15 split
    train_final, temp = train_test_split(
        all_data_final, test_size=0.3, random_state=42, stratify=all_data_final['label']
    )
    val_final, test_final = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp['label']
    )

print(f"\n   Final split sizes:")
print(f"   Train: {len(train_final):,} ({len(train_final)/len(all_data_final)*100:.1f}%)")
print(f"   Val:   {len(val_final):,} ({len(val_final)/len(all_data_final)*100:.1f}%)")
print(f"   Test:  {len(test_final):,} ({len(test_final)/len(all_data_final)*100:.1f}%)")

# Verify class balance
print("\n   Class distribution per split:")
for split_name, split_data in [('Train', train_final), ('Val', val_final), ('Test', test_final)]:
    dist = split_data['label'].value_counts()
    print(f"   {split_name}:")
    for label, count in dist.items():
        print(f"     {label}: {count:,} ({count/len(split_data)*100:.1f}%)")

# ============================================================================
# 8. FINAL QUALITY CHECKS
# ============================================================================
print("\n8. Running final quality checks...")

# Check for empty texts
empty_count = sum([
    (train_final['text'].str.strip() == '').sum(),
    (val_final['text'].str.strip() == '').sum(),
    (test_final['text'].str.strip() == '').sum()
])

if empty_count > 0:
    print(f"   WARNING: Found {empty_count} empty texts!")
else:
    print(f"   OK: No empty texts")

# Check for duplicates across splits
train_texts = set(train_final['text'].values)
val_texts = set(val_final['text'].values)
test_texts = set(test_final['text'].values)

overlap = len(train_texts & val_texts) + len(train_texts & test_texts) + len(val_texts & test_texts)

if overlap > 0:
    print(f"   WARNING: Found {overlap} duplicate texts across splits!")
else:
    print(f"   OK: No duplicates across splits")

# Check length distribution similarity
human_train_len = train_final[train_final['label'] == 'HUMAN']['text'].str.len().mean()
ai_train_len = train_final[train_final['label'] == 'AI']['text'].str.len().mean()
len_diff = abs(human_train_len - ai_train_len)

print(f"\n   Length distribution check:")
print(f"   HUMAN mean: {human_train_len:.1f}")
print(f"   AI mean:    {ai_train_len:.1f}")
print(f"   Difference: {len_diff:.1f}")

if len_diff < 50:
    print(f"   OK: Length distributions are well balanced")
else:
    print(f"   NOTE: Some length difference remains ({len_diff:.1f} chars)")

# ============================================================================
# 9. SAVE FINAL PREPROCESSED DATA
# ============================================================================
print("\n9. Saving final preprocessed data...")

def save_jsonl(df, path):
    with open(path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            record = {'text': row['text'], 'label': row['label']}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

save_jsonl(train_final, OUTPUT_DIR / 'train.jsonl')
save_jsonl(val_final, OUTPUT_DIR / 'val.jsonl')
save_jsonl(test_final, OUTPUT_DIR / 'test.jsonl')

print(f"   SUCCESS: Final preprocessed data saved to: {OUTPUT_DIR.absolute()}")
print(f"   Files:")
print(f"     - train.jsonl ({len(train_final):,} samples)")
print(f"     - val.jsonl ({len(val_final):,} samples)")
print(f"     - test.jsonl ({len(test_final):,} samples)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nPreprocessing complete!")
print(f"\nTransformations applied:")
print(f"  1. Stripped formatting cues (markdown, HTML, excessive whitespace)")
print(f"  2. Normalized punctuation patterns")
print(f"  3. Balanced length distribution across classes")
print(f"  4. Removed all metadata (only text + label remain)")
print(f"  5. Filtered to {MIN_LENGTH}-{MAX_LENGTH} character range")

print(f"\nDataset statistics:")
print(f"  Input:  {len(all_data):,} samples")
print(f"  Output: {len(all_data_final):,} samples")
print(f"  Removed: {len(all_data) - len(all_data_final):,} ({(len(all_data) - len(all_data_final))/len(all_data)*100:.1f}%)")

print(f"\nFinal dataset location: {OUTPUT_DIR.absolute()}")

print(f"\nNEXT STEPS:")
print(f"  1. Update your training notebook:")
print(f"     DATA_DIR = Path('dataset/final')")
print(f"  2. Retrain your model")
print(f"  3. Expected AUC: 0.80-0.95 (model must learn semantic features)")
print(f"  4. Lower than before is GOOD - means no spurious shortcuts!")

print("="*70)
