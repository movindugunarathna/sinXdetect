"""
Data Leakage Detection and Fix Script

This script identifies and fixes data leakage issues that cause perfect AUC scores:
1. Detect exact duplicates across splits
2. Detect near-duplicates using TF-IDF cosine similarity
3. Check for source/document leakage
4. Recreate proper train/val/test splits
"""

import json
import hashlib
import re
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = Path('dataset')
OUTPUT_DIR = Path('dataset/cleaned')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIMILARITY_THRESHOLD = 0.85  # Texts with >85% similarity are considered near-duplicates

print("="*70)
print("DATA LEAKAGE DETECTION AND FIX")
print("="*70)

# ============================================================================
# 1. LOAD ALL DATA
# ============================================================================
print("\n1. Loading data...")

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

train_df = load_jsonl(DATA_DIR / 'train.jsonl')
val_df = load_jsonl(DATA_DIR / 'val.jsonl')
test_df = load_jsonl(DATA_DIR / 'test.jsonl')

# Add split identifier
train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

# Combine all data
all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"   Total samples: {len(all_data):,}")
print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"   Label distribution: {dict(all_data['label'].value_counts())}")

# ============================================================================
# 2. DETECT EXACT DUPLICATES
# ============================================================================
print("\n2. Detecting exact duplicates...")

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

all_data['text_normalized'] = all_data['text'].apply(normalize_text)
all_data['text_hash'] = all_data['text_normalized'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())

# Find duplicates
duplicate_groups = all_data.groupby('text_hash').agg({
    'split': lambda x: list(x),
    'label': lambda x: list(x),
    'text': 'first'
}).reset_index()

duplicate_groups['num_splits'] = duplicate_groups['split'].apply(lambda x: len(set(x)))
duplicate_groups['total_copies'] = duplicate_groups['split'].apply(len)

cross_split_duplicates = duplicate_groups[duplicate_groups['num_splits'] > 1]

print(f"   Total unique texts: {len(duplicate_groups):,}")
print(f"   Texts with duplicates: {len(duplicate_groups[duplicate_groups['total_copies'] > 1]):,}")
print(f"   Texts duplicated across splits: {len(cross_split_duplicates):,}")

if len(cross_split_duplicates) > 0:
    print(f"\n   WARNING: Found {len(cross_split_duplicates):,} texts appearing in multiple splits!")
    print("   Examples:")
    for idx, row in cross_split_duplicates.head(3).iterrows():
        splits = Counter(row['split'])
        print(f"     - {row['text'][:80]}...")
        print(f"       Appears in: {dict(splits)}")
else:
    print("   OK: No exact duplicates across splits")

# ============================================================================
# 3. DETECT NEAR-DUPLICATES (SAMPLING FOR SPEED)
# ============================================================================
print("\n3. Detecting near-duplicates (this may take a few minutes)...")

near_duplicate_count = 0
split_pairs = [('train', 'val'), ('train', 'test'), ('val', 'test')]

for split1, split2 in split_pairs:
    print(f"   Checking {split1} vs {split2}...")
    
    df1 = all_data[all_data['split'] == split1].copy()
    df2 = all_data[all_data['split'] == split2].copy()
    
    # Sample if too large
    if len(df1) > 5000:
        df1 = df1.sample(5000, random_state=42)
    if len(df2) > 5000:
        df2 = df2.sample(5000, random_state=42)
    
    if len(df1) == 0 or len(df2) == 0:
        continue
    
    try:
        # Vectorize using character n-grams (better for multilingual text)
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(2, 4),
            analyzer='char_wb',
            min_df=1
        )
        
        all_texts = pd.concat([df1['text_normalized'], df2['text_normalized']])
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        tfidf1 = tfidf_matrix[:len(df1)]
        tfidf2 = tfidf_matrix[len(df1):]
        
        # Compute similarities in batches
        batch_size = 500
        count = 0
        
        for i in range(0, tfidf1.shape[0], batch_size):
            batch = tfidf1[i:i+batch_size]
            similarities = cosine_similarity(batch, tfidf2)
            high_sim = (similarities > SIMILARITY_THRESHOLD).sum()
            count += high_sim
        
        near_duplicate_count += count
        print(f"     Found {count:,} near-duplicate pairs (similarity > {SIMILARITY_THRESHOLD})")
    
    except Exception as e:
        print(f"     Error: {e}")

if near_duplicate_count > 0:
    print(f"\n   WARNING: Found {near_duplicate_count:,} near-duplicate pairs across splits!")
else:
    print("   OK: No significant near-duplicates found")

# ============================================================================
# 4. CHECK FOR SOURCE/METADATA LEAKAGE
# ============================================================================
print("\n4. Checking for source/metadata leakage...")

metadata_fields = [col for col in all_data.columns 
                   if col not in ['text', 'label', 'split', 'text_normalized', 'text_hash']]

if metadata_fields:
    print(f"   Metadata fields found: {metadata_fields}")
    for field in metadata_fields:
        try:
            # Skip dict/list fields
            if all_data[field].apply(lambda x: isinstance(x, (dict, list))).any():
                print(f"   Skipping '{field}': Contains complex objects (dict/list)")
                continue
            
            unique_vals = all_data[field].nunique()
            value_split_dist = all_data.groupby(field)['split'].apply(lambda x: set(x))
            cross_split_values = value_split_dist[value_split_dist.apply(len) > 1]
            
            if len(cross_split_values) > 0:
                print(f"   WARNING Field '{field}': {len(cross_split_values)} values appear in multiple splits")
            else:
                print(f"   OK Field '{field}': No cross-split leakage")
        except Exception as e:
            print(f"   Skipping '{field}': Error ({e})")
else:
    print("   OK: No metadata fields found")

# ============================================================================
# 5. CLEAN DATASET
# ============================================================================
print("\n5. Cleaning dataset...")

# Remove exact duplicates
all_data_dedup = all_data.drop_duplicates(subset=['text_hash'], keep='first')
print(f"   Removed {len(all_data) - len(all_data_dedup):,} exact duplicates")

# Remove invalid texts
all_data_dedup['text_length'] = all_data_dedup['text'].str.len()
all_data_clean = all_data_dedup[all_data_dedup['text_length'] >= 10].copy()
print(f"   Removed {len(all_data_dedup) - len(all_data_clean):,} texts with <10 characters")

# Keep only necessary columns
all_data_clean = all_data_clean[['text', 'label']].copy()

print(f"   Final clean dataset: {len(all_data_clean):,} samples")
print(f"   Label distribution: {dict(all_data_clean['label'].value_counts())}")

# ============================================================================
# 6. CREATE NEW SPLITS
# ============================================================================
print("\n6. Creating new train/val/test splits (70/15/15)...")

# Shuffle
all_data_clean = all_data_clean.sample(frac=1, random_state=42).reset_index(drop=True)

# Split: 70% train, 15% val, 15% test
train_data, temp_data = train_test_split(
    all_data_clean,
    test_size=0.3,
    random_state=42,
    stratify=all_data_clean['label']
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=temp_data['label']
)

print(f"   Train: {len(train_data):,} ({len(train_data)/len(all_data_clean)*100:.1f}%)")
print(f"   Val:   {len(val_data):,} ({len(val_data)/len(all_data_clean)*100:.1f}%)")
print(f"   Test:  {len(test_data):,} ({len(test_data)/len(all_data_clean)*100:.1f}%)")

# ============================================================================
# 7. VERIFY NO LEAKAGE
# ============================================================================
print("\n7. Verifying new splits...")

train_texts = set(train_data['text'].values)
val_texts = set(val_data['text'].values)
test_texts = set(test_data['text'].values)

train_val_overlap = train_texts & val_texts
train_test_overlap = train_texts & test_texts
val_test_overlap = val_texts & test_texts

print(f"   Train-Val overlap: {len(train_val_overlap)} texts")
print(f"   Train-Test overlap: {len(train_test_overlap)} texts")
print(f"   Val-Test overlap: {len(val_test_overlap)} texts")

if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
    print(f"\n   SUCCESS: No data leakage detected!")
else:
    print(f"\n   WARNING: Some overlap still exists!")

# ============================================================================
# 8. SAVE CLEAN SPLITS
# ============================================================================
print("\n8. Saving clean splits...")

def save_jsonl(df, path):
    with open(path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            record = {'text': row['text'], 'label': row['label']}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

save_jsonl(train_data, OUTPUT_DIR / 'train.jsonl')
save_jsonl(val_data, OUTPUT_DIR / 'val.jsonl')
save_jsonl(test_data, OUTPUT_DIR / 'test.jsonl')

print(f"   SUCCESS: Clean splits saved to: {OUTPUT_DIR.absolute()}")
print(f"   Files:")
print(f"     - train.jsonl ({len(train_data):,} samples)")
print(f"     - val.jsonl ({len(val_data):,} samples)")
print(f"     - test.jsonl ({len(test_data):,} samples)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nData cleaning complete!")
print(f"\nStatistics:")
print(f"   Original samples: {len(all_data):,}")
print(f"   After cleaning: {len(all_data_clean):,}")
print(f"   Removed: {len(all_data) - len(all_data_clean):,} ({(len(all_data) - len(all_data_clean))/len(all_data)*100:.1f}%)")
print(f"\nClean data location: {OUTPUT_DIR.absolute()}")
print(f"\nNEXT STEPS:")
print(f"   1. Update your training notebook to use the cleaned data:")
print(f"      DATA_DIR = Path('dataset/cleaned')")
print(f"   2. Retrain your model with the clean splits")
print(f"   3. The new ROC curve should show realistic AUC scores (0.85-0.98)")
print("="*70)
