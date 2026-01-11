#!/usr/bin/env python3
"""Script to extract and print models used in each dataset file."""

import json
import os
from pathlib import Path

# Dataset files to process (relative to this script's directory)
dataset_files = [
    "dataset/dataset.jsonl",
    "dataset/test.jsonl",
    "dataset/train_drive.jsonl",
    "dataset/train_original.jsonl",
    "dataset/train.jsonl",
    "dataset/val.jsonl",
]

def extract_models_from_file(filepath):
    """Extract unique models from a JSONL file."""
    models = set()
    
    # Make path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(filepath):
        filepath = os.path.join(script_dir, filepath)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        # Search for "models/" pattern in the line
                        import re
                        model_matches = re.findall(r'"models/[^"]*"', line)
                        for match in model_matches:
                            # Remove quotes and add to set
                            models.add(match.strip('"'))
                        
                        # Also parse as JSON and check all field values
                        data = json.loads(line)
                        for key, value in data.items():
                            if isinstance(value, str) and 'models/' in value:
                                models.add(value)
                            elif isinstance(value, dict):
                                for nested_key, nested_value in value.items():
                                    if isinstance(nested_value, str) and 'models/' in nested_value:
                                        models.add(nested_value)
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Failed to parse line {line_num}: {e}")
    except FileNotFoundError:
        print(f"  Error: File not found - {filepath}")
    except Exception as e:
        print(f"  Error: {e}")
    
    return models


def main():
    """Main function to process all dataset files."""
    print("=" * 70)
    print("MODELS USED IN EACH DATASET")
    print("=" * 70)
    
    for dataset_file in dataset_files:
        print(f"\nFile: {dataset_file}")
        models = extract_models_from_file(dataset_file)
        
        if models:
            for model in sorted(models):
                print(f"  - {model}")
        else:
            print("  No 'model' field found in this dataset")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
