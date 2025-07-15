import os
import sys
import yaml
import numpy as np
import argparse
import mne
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_pipeline.annotations import load_annotations_for_file

class SOTA_Preprocessor:
    def __init__(self, config: dict):
        self.config = config['preprocessing']
        self.sfreq = self.config['sfreq']
        self.l_freq = self.config['l_freq']
        self.h_freq = self.config['h_freq']
        self.notch_freq = self.config['notch_freq']
        self.epoch_duration_s = self.config['epoch_duration_s']
        self.epoch_overlap_s = self.config['epoch_overlap_s']
        self.montage = mne.channels.make_standard_montage(self.config['montage'])
        self.initial_reject_v = 1000 * 1e-6

    def _rename_channels(self, raw: mne.io.Raw):
        mapping = {
            'FP1-F7': 'Fp1', 'F7-T7': 'F7', 'T7-P7': 'T7', 'P7-O1': 'O1',
            'FP1-F3': 'F3', 'F3-C3': 'C3', 'C3-P3': 'P3', 'P3-O1': 'P3',
            'FP2-F8': 'Fp2', 'F8-T8': 'F8', 'T8-P8': 'T8', 'P8-O2': 'P8',
            'FP2-F4': 'F4', 'F4-C4': 'C4', 'C4-P4': 'P4', 'P4-O2': 'O2',
            'FZ-CZ': 'Fz', 'CZ-PZ': 'Cz', 'P7-T7': 'P7', 'T8-P8-1': 'P8',
        }
        current_ch_names_lower = {ch.lower(): ch for ch in raw.ch_names}
        map_keys_lower = {key.lower(): key for key in mapping.keys()}
        current_mapping = {current_ch_names_lower[ch_lower]: mapping[map_keys_lower[ch_lower]]
                           for ch_lower in current_ch_names_lower if ch_lower in map_keys_lower}
        raw.rename_channels(current_mapping, allow_duplicates=True)

    def _universal_reader(self, file_path: str) -> mne.io.Raw:
        try:
            return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception:
            return None

    def process_single_file(self, file_path: str, output_dir: str):
        try:
            raw = self._universal_reader(file_path)
            if raw is None: return

            raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
            self._rename_channels(raw)
            raw.set_montage(self.montage, on_missing='ignore')
            
            ch_with_locs = [ch for ch in raw.ch_names if np.all(np.isfinite(raw._get_channel_positions([ch])[0]))]
            raw.pick(ch_with_locs)
            if len(raw.ch_names) == 0: return

            raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose=False)
            raw.notch_filter(self.notch_freq, fir_design='firwin', verbose=False)
            raw.resample(self.sfreq, npad='auto', verbose=False)
            raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)

            epochs = mne.make_fixed_length_epochs(raw, duration=self.epoch_duration_s, overlap=self.epoch_overlap_s, preload=True, reject_by_annotation=True, verbose=False)
            if len(epochs) == 0: return
            
            epochs.drop_bad(reject=dict(eeg=self.initial_reject_v))
            if len(epochs) == 0: return
            
            epochs_data = epochs.get_data()
            features = np.hstack([epochs_data.var(axis=2), np.mean(epochs_data**2, axis=2)])
            iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=1)
            is_inlier = iso_forest.fit_predict(features)

            seizure_intervals = load_annotations_for_file(file_path)
            pathology_labels, artifact_labels = [], []
            final_indices_to_keep = np.arange(len(epochs))

            for i, inlier_status in enumerate(is_inlier):
                is_seizure = 0
                epoch_start_s = i * (self.epoch_duration_s - self.epoch_overlap_s)
                epoch_end_s = epoch_start_s + self.epoch_duration_s
                for seizure in seizure_intervals:
                    if max(epoch_start_s, seizure['start']) < min(epoch_end_s, seizure['end']):
                        is_seizure = 1
                        break
                
                if inlier_status == -1:
                    artifact_labels.append(1)
                    pathology_labels.append(-100)
                else:
                    artifact_labels.append(0)
                    pathology_labels.append(is_seizure)
            
            final_data = epochs_data
            
            output_basename = os.path.basename(file_path).replace('.edf', '.npz')
            output_path = os.path.join(output_dir, output_basename)
            np.savez_compressed(
                output_path,
                data=final_data,
                seizure_binary=np.array(pathology_labels),
                artifact=np.array(artifact_labels)
            )
        except Exception as e:
            print(f"CRITICAL ERROR processing {file_path}: {e}")

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['paths']['chb_mit_raw_dir']
    output_dir = config['paths']['chb_mit_processed_dir']

    preprocessor = SOTA_Preprocessor(config)
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = [os.path.join(root, name) for root, _, files in os.walk(data_dir) for name in files if name.endswith(".edf")]
    print(f"Found {len(all_files)} .edf files in '{data_dir}' to process.")
    
    Parallel(n_jobs=-1)(
        delayed(preprocessor.process_single_file)(file_path, output_dir)
        for file_path in tqdm(all_files, desc="Processing CHB-MIT")
    )
    print(f"\n[✅] Finished processing all files. Clean data saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processes the local CHB-MIT dataset.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config_chb_mit.yaml file.")
    args = parser.parse_args()
    main(args.config_path)