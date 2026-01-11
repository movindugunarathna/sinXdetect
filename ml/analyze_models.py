import json
from pathlib import Path
from collections import Counter

def analyze_models_in_dataset(file_path):
    """Extract model information from a JSONL dataset file."""
    models = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if 'meta' in data and 'gen_model' in data['meta']:
                        model = data['meta']['gen_model']
                        if model:  # Only add non-null/non-empty models
                            models.append(model)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Error parsing line {line_num}: {e}")
    except FileNotFoundError:
        print(f"  File not found: {file_path}")
        return []
    except Exception as e:
        print(f"  Error reading file: {e}")
        return []
    
    return models

def main():
    # Define dataset files
    dataset_files = [
        'ml/dataset/dataset.jsonl',
        'ml/dataset/test.jsonl',
        'ml/dataset/train_original.jsonl',
        'ml/dataset/train.jsonl',
        'ml/dataset/val.jsonl'
    ]
    
    print("=" * 80)
    print("Model Analysis Report")
    print("=" * 80)
    
    all_models = []
    
    for dataset_file in dataset_files:
        print(f"\n📁 {dataset_file}")
        models = analyze_models_in_dataset(dataset_file)
        
        if models:
            model_counts = Counter(models)
            print(f"  Total samples with model info: {len(models)}")
            print(f"  Unique models: {len(model_counts)}")
            print(f"  Models found:")
            for model, count in model_counts.most_common():
                print(f"    - {model}: {count} samples")
            all_models.extend(models)
        else:
            print(f"  No model information found or file is empty")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("Overall Summary")
    print("=" * 80)
    
    if all_models:
        all_model_counts = Counter(all_models)
        print(f"Total samples with model info across all datasets: {len(all_models)}")
        print(f"Total unique models: {len(all_model_counts)}")
        print(f"\nAll models used:")
        for model, count in all_model_counts.most_common():
            print(f"  - {model}: {count} samples")
    else:
        print("No model information found in any dataset")

if __name__ == "__main__":
    main()
