# In file: scripts/prepare_dataset.py

import os
import json
import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
import yaml

def create_dataset_split(config_path: str):
    """
    Scans the CHB-MIT data directory, performs a patient-level split,
    and saves the file lists to a JSON file.
    """
    print("--- Starting Dataset Preparation ---")
    
    # Load base data directory from main config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    data_dir = config['paths']['chb_mit_dir']
    
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found at: {data_dir}. Please check your config.yaml.")

    # 1. Discover all patient directories
    all_patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('chb')])
    print(f"Found {len(all_patients)} patients: {all_patients}")

    # if len(all_patients) < 3:   #UNCOMMENT THIS!
    #     raise ValueError("Cannot create train/val/test splits with fewer than 3 patients. Please download more data.")

    # 2. Perform patient-level split (e.g., 70% train, 15% val, 15% test)
    train_val_patients, test_patients = train_test_split(all_patients, test_size=0.15, random_state=42)
    train_patients, val_patients = train_test_split(train_val_patients, test_size=0.18, random_state=42) # 0.18 * 0.85 ~= 0.15

    print(f"Training patients: {train_patients}")
    print(f"Validation patients: {val_patients}")
    print(f"Testing patients: {test_patients}")

    sets = {
        'train': train_patients,
        'validation': val_patients,
        'test': test_patients
    }
    
    # 3. Collect all .edf files for each set
    file_splits = defaultdict(list)
    for set_name, patient_list in sets.items():
        for patient_id in patient_list:
            patient_dir = os.path.join(data_dir, patient_id)
            edf_files = glob.glob(os.path.join(patient_dir, '*.edf'))
            file_splits[set_name].extend(sorted(edf_files))

    # 4. Save the split to a JSON file
    output_path = os.path.join(os.path.dirname(config_path), 'dataset_split.json')
    with open(output_path, 'w') as f:
        json.dump(file_splits, f, indent=4)

    print(f"\n✅ Successfully created dataset split file at: {output_path}")
    print(f" -> {len(file_splits['train'])} training files")
    print(f" -> {len(file_splits['validation'])} validation files")
    print(f" -> {len(file_splits['test'])} testing files")
    print("\nUpdate your config.yaml to use this file.")


if __name__ == '__main__':
    # We assume the main config is in the parent directory of 'scripts'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_config_path = os.path.join(project_root, 'config', 'config.yaml')
    create_dataset_split(main_config_path)