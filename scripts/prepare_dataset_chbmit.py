import os
import json
import glob
from collections import defaultdict
import yaml
import argparse
from sklearn.model_selection import train_test_split

def create_split(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    processed_dir = config['paths']['chb_mit_processed_dir']
    all_files = glob.glob(os.path.join(processed_dir, '*.npz'))
    
    patient_files = defaultdict(list)
    for f in all_files:
        patient_id = os.path.basename(f).split('_')[0]
        patient_files[patient_id].append(f)
        
    all_patients = sorted(list(patient_files.keys()))
    
    train_val_patients, test_patients = train_test_split(all_patients, test_size=0.15, random_state=42)
    train_patients, val_patients = train_test_split(train_val_patients, test_size=0.18, random_state=42)

    file_splits = {
        'train': [f for p in train_patients for f in patient_files[p]],
        'validation': [f for p in val_patients for f in patient_files[p]],
        'test': [f for p in test_patients for f in patient_files[p]]
    }

    output_path = config['paths']['dataset_split_file']
    with open(output_path, 'w') as f:
        json.dump(file_splits, f, indent=4)
        
    print(f"\n✅ Successfully created CHB-MIT dataset split file at: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create train/val/test splits for preprocessed CHB-MIT.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config_chb_mit.yaml file.")
    args = parser.parse_args()
    create_split(args.config_path)