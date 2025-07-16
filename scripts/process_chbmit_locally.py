# In file: scripts/process_chbmit_sota.py

import os
import sys
import yaml
import numpy as np
import argparse
import mne
import re
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

# ===================================================================
# ANNOTATION LOGIC (Self-Contained)
# ===================================================================
def load_annotations_for_file(edf_file_path: str) -> List[Dict[str, int]]:
    """
    Master annotation loader. It looks for a corresponding summary file
    in the patient's directory (.txt or .tse).
    """
    base_name_lower = os.path.basename(edf_file_path).lower()
    patient_dir = os.path.dirname(edf_file_path)
    try:
        for f_name in os.listdir(patient_dir):
            if f_name.lower().endswith(('.txt', '.tse')):
                summary_file_path = os.path.join(patient_dir, f_name)
                with open(summary_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                seizure_info = {}
                file_blocks = re.finditer(r"File Name: ([\w\d_\-\.]+\.edf)[\s\S]*?(?=File Name:|$)", content, re.IGNORECASE)
                for block in file_blocks:
                    file_name = block.group(1).lower()
                    seizures = []
                    seizure_times = re.finditer(r"Seizure(?: \w+)? Start Time: ([\d\.]+) seconds\nSeizure(?: \w+)? End Time: ([\d\.]+) seconds", block.group(0), re.IGNORECASE)
                    for seizure in seizure_times:
                        seizures.append({'start': int(float(seizure.group(1))), 'end': int(float(seizure.group(2)))})
                    if seizures:
                        seizure_info[file_name] = seizures
                
                if base_name_lower in seizure_info:
                    return seizure_info[base_name_lower]
    except Exception:
        pass
    return []

# ===================================================================
# PREPROCESSING BLUEPRINT (Self-Contained & Corrected)
# ===================================================================
class SOTA_Bipolar_Preprocessor:
    """
    Definitive SOTA Preprocessing Pipeline for Bipolar Data (CHB-MIT).
    This version correctly enforces a consistent channel set.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config['preprocessing']
        self.sfreq = self.config['sfreq']
        self.l_freq = self.config['l_freq']
        self.h_freq = self.config['h_freq']
        self.notch_freq = self.config['notch_freq']
        self.epoch_duration_s = self.config['epoch_duration_s']
        self.epoch_overlap_s = self.config['epoch_overlap_s']
        self.initial_reject_v = 1000 * 1e-6
        # A robust, non-duplicate, 18-channel standard bipolar set.
        self.TARGET_CHANNELS = [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
            'FZ-CZ', 'CZ-PZ'
        ]

    def _universal_reader(self, file_path: str) -> mne.io.Raw:
        try:
            return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception:
            return None

    def preprocess(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = self._universal_reader(file_path)
        if raw is None: return None, None, None

        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        
        # --- CORRECTED LOGIC: ENFORCE CONSISTENCY ---
        # 1. Check if the file contains ALL required channels (case-insensitive)
        ch_names_upper = [ch.upper() for ch in raw.ch_names]
        if not set(ch.upper() for ch in self.TARGET_CHANNELS).issubset(ch_names_upper):
            # This file is non-compliant. Skip it entirely.
            print(f"INFO: Skipping {os.path.basename(file_path)} - missing required channels.")
            return None, None, None

        # 2. Pick the exact set of channels in the correct order.
        # This ensures every single output file has the same channels in the same order.
        raw.pick(self.TARGET_CHANNELS)
        
        # --- Continue with SOTA pipeline on the now-uniform data ---
        raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose=False)
        raw.notch_filter(self.notch_freq, fir_design='firwin', verbose=False)
        raw.resample(self.sfreq, npad='auto', verbose=False)

        epochs = mne.make_fixed_length_epochs(raw, duration=self.epoch_duration_s, overlap=self.epoch_overlap_s, preload=True, reject_by_annotation=True, verbose=False)
        if len(epochs) == 0: return None, None, None

        # Two-Stage Hybrid Artifact Rejection
        epochs.drop_bad(reject=dict(eeg=self.initial_reject_v))
        if len(epochs) == 0: return None, None, None
        
        epochs_data = epochs.get_data()
        features = np.hstack([epochs_data.var(axis=2), np.mean(epochs_data**2, axis=2)])
        iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=1)
        is_inlier = iso_forest.fit_predict(features)
        
        # Intelligent Labeling
        seizure_intervals = load_annotations_for_file(file_path)
        pathology_labels, artifact_labels = [], []
        
        for i in range(len(epochs)):
            epoch_start_s = epochs.events[i, 0] / epochs.info['sfreq']
            is_seizure = 0
            for seizure in seizure_intervals:
                if seizure['start'] <= epoch_start_s < seizure['end']:
                    is_seizure = 1; break
            
            if is_inlier[i] == -1:
                artifact_labels.append(1)
                pathology_labels.append(-100)
            else:
                artifact_labels.append(0)
                pathology_labels.append(is_seizure)
        
        return epochs_data, np.array(pathology_labels), np.array(artifact_labels)

# ===================================================================
# ORCHESTRATION LOGIC (The part you run)
# ===================================================================
def process_and_save_file(preprocessor, file_path, output_dir):
    """A wrapper function for our parallel processing job."""
    try:
        result = preprocessor.preprocess(file_path)
        if result is not None:
            data, pathology, artifact = result
            if data is not None and data.shape[0] > 0:
                output_basename = os.path.basename(file_path).replace('.edf', '.npz')
                output_path = os.path.join(output_dir, output_basename)
                np.savez_compressed(
                    output_path,
                    data=data,
                    seizure_binary=pathology,
                    artifact=artifact
                )
    except Exception as e:
        print(f"CRITICAL ERROR processing {file_path}: {e}", file=sys.stderr)

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['paths']['chb_mit_raw_dir']
    output_dir = config['paths']['chb_mit_processed_dir']

    preprocessor = SOTA_Bipolar_Preprocessor(config)
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = [os.path.join(root, name) for root, _, files in os.walk(data_dir) for name in files if name.endswith(".edf")]
    print(f"Found {len(all_files)} .edf files in '{data_dir}' to process.")
    
    Parallel(n_jobs=-1)(
        delayed(process_and_save_file)(preprocessor, file_path, output_dir)
        for file_path in tqdm(all_files, desc="Processing CHB-MIT")
    )
    print(f"\n[✅] Finished processing all files. Clean data saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processes the local CHB-MIT dataset with a SOTA bipolar-aware pipeline.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config_chb_mit.yaml file.")
    args = parser.parse_args()
    main(args.config_path)