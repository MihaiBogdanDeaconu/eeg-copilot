# In file: scripts/preprocess_and_save.py

import os
import sys
import yaml
import json
import numpy as np
from tqdm import tqdm
import argparse

# Add project root to path to allow importing from 'src'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_pipeline import EEGPreprocessor, EEGDataset

def get_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_preprocessing(config_path: str, output_dir: str):
    """
    Scans the TUH dataset, runs the full SOTA preprocessing pipeline,
    and saves the clean epochs and labels to a new directory.
    """
    config = get_config(config_path)
    
    # Use the same file split we created earlier
    split_file = config['paths']['dataset_split_file']
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"dataset_split.json not found. Please run scripts/prepare_dataset.py first.")
    
    with open(split_file, 'r') as f:
        file_splits = json.load(f)
    
    all_files = file_splits['train'] + file_splits['validation'] + file_splits['test']

    # --- We use EEGDataset here as a convenient way to access the preprocessor and label generation ---
    # This is a bit of a "hack" but avoids duplicating code.
    dataset = EEGDataset(all_files, config)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting preprocessing for {len(all_files)} files. Output will be saved to: {output_dir}")

    for i in tqdm(range(len(dataset)), desc="Preprocessing TUH Dataset"):
        try:
            # We call the dataset's __getitem__ which contains all our preprocessing logic
            processed_item = dataset[i]
            
            if processed_item['data'].nelement() > 0:
                # Create a unique name for the output file
                original_file_path = dataset.file_paths[i]
                file_basename = os.path.basename(original_file_path).replace('.edf', '.npz')
                output_path = os.path.join(output_dir, file_basename)

                # Save the processed data and labels dictionary to a compressed .npz file
                np.savez_compressed(
                    output_path,
                    data=processed_item['data'].numpy(),
                    # Convert labels from tensors to numpy arrays for saving
                    **{k: v.numpy() for k, v in processed_item['labels'].items()}
                )
        except Exception as e:
            original_file_path = dataset.file_paths[i]
            print(f"\nCould not process file {original_file_path}. Error: {e}")
            continue
            
    print(f"\n[✅] Preprocessing complete. Clean data saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess the entire TUH dataset and save results.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config.yaml file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the preprocessed .npz files.")
    args = parser.parse_args()
    run_preprocessing(args.config_path, args.output_dir)