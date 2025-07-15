import os
import sys
import yaml
import numpy as np
import argparse
import mne
import requests
import shutil
from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed

# --- Add project root for local imports ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_pipeline.annotations import load_annotations_for_file

# --- THE SOTA PREPROCESSOR (Integrated into this script) ---
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

    def process_single_file(self, file_path: str) -> (np.ndarray, np.ndarray, np.ndarray):
        try:
            raw = self._universal_reader(file_path)
            if raw is None: return None
            
            raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
            self._rename_channels(raw)
            raw.set_montage(self.montage, on_missing='ignore')
            ch_with_locs = [ch for ch in raw.ch_names if np.all(np.isfinite(raw._get_channel_positions([ch])[0]))]
            raw.pick(ch_with_locs)
            if len(raw.ch_names) == 0: return None

            raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose=False)
            raw.notch_filter(self.notch_freq, fir_design='firwin', verbose=False)
            raw.resample(self.sfreq, npad='auto', verbose=False)
            raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)

            epochs = mne.make_fixed_length_epochs(raw, duration=self.epoch_duration_s, overlap=self.epoch_overlap_s, preload=True, reject_by_annotation=True, verbose=False)
            if len(epochs) == 0: return None
            
            epochs.drop_bad(reject=dict(eeg=self.initial_reject_v))
            if len(epochs) == 0: return None
            
            epochs_data = epochs.get_data()
            features = np.hstack([epochs_data.var(axis=2), np.mean(epochs_data**2, axis=2)])
            iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=1)
            is_inlier = iso_forest.fit_predict(features)

            seizure_intervals = load_annotations_for_file(file_path)
            pathology_labels, artifact_labels = [], []
            final_indices = []

            for i, inlier_status in enumerate(is_inlier):
                is_seizure = 0
                epoch_start_s = i * (self.epoch_duration_s - self.epoch_overlap_s)
                epoch_end_s = epoch_start_s + self.epoch_duration_s
                for seizure in seizure_intervals:
                    if max(epoch_start_s, seizure['start']) < min(epoch_end_s, seizure['end']):
                        is_seizure = 1
                        break
                
                if inlier_status == -1:
                    pathology_labels.append(-100)
                    artifact_labels.append(1)
                else:
                    pathology_labels.append(is_seizure)
                    artifact_labels.append(0)
                final_indices.append(i)
            
            final_data = epochs_data[final_indices]
            return final_data, np.array(pathology_labels), np.array(artifact_labels)
        except Exception:
            return None


def get_tuh_file_urls(base_url):
    print("Scraping TUH server for file list... This may take a minute.")
    all_urls = []
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href.startswith('v') and href.endswith('/'):
            version_url = base_url + href
            v_response = requests.get(version_url)
            v_soup = BeautifulSoup(v_response.text, 'html.parser')
            for sub_link in v_soup.find_all('a'):
                sub_href = sub_link.get('href')
                if sub_href.startswith('edf/'):
                    edf_url = version_url + sub_href
                    e_response = requests.get(edf_url)
                    e_soup = BeautifulSoup(e_response.text, 'html.parser')
                    for patient_link in e_soup.find_all('a'):
                        patient_href = patient_link.get('href')
                        if patient_href.startswith('000'):
                            patient_url = edf_url + patient_href
                            p_response = requests.get(patient_url)
                            p_soup = BeautifulSoup(p_response.text, 'html.parser')
                            for file_link in p_soup.find_all('a'):
                                file_href = file_link.get('href')
                                if file_href.endswith('.edf'):
                                    all_urls.append(patient_url + file_href)
    print(f"Found {len(all_urls)} .edf files.")
    return all_urls

def download_file(url, target_folder):
    local_filename = os.path.join(target_folder, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return local_filename


def main(config_path, output_dir, temp_dir):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocessor = SOTA_Preprocessor(config)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # This base URL points to the TUH EEG Corpus
    base_url = "https://isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/"
    all_file_urls = get_tuh_file_urls(base_url)

    for i in tqdm(range(len(all_file_urls)), desc="Overall Progress"):
        url = all_file_urls[i]
        try:
            # 1. Download
            local_file_path = download_file(url, temp_dir)
            
            # 2. Process
            result = preprocessor.process_single_file(local_file_path)
            
            if result is not None:
                data, pathology, artifact = result
                if data.shape[0] > 0:
                    # 3. Save
                    output_basename = os.path.basename(local_file_path).replace('.edf', '.npz')
                    output_path = os.path.join(output_dir, output_basename)
                    np.savez_compressed(output_path, data=data, pathology=pathology, artifact=artifact)

        except Exception as e:
            print(f"Failed processing URL {url}. Error: {e}")
        finally:
            # 4. Delete
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

    print("\n[✅] Entire dataset processed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fully automated orchestrator for downloading and preprocessing the TUH dataset.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config.yaml file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Final directory to save the permanent preprocessed .npz files.")
    parser.add_argument('--temp_dir', type=str, required=True, help="Temporary directory for downloading raw .edf files. This will be cleared automatically.")
    args = parser.parse_args()
    main(args.config_path, args.output_dir, args.temp_dir)