import mne
import numpy as np
import os
from typing import Dict, Any, List
from sklearn.ensemble import IsolationForest

class EEGPreprocessor:
    """
    Final SOTA Preprocessing Pipeline.
    This version enforces a fixed, standard 19-channel set for all recordings
    and uses a robust Two-Stage Hybrid Artifact Rejection method.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config['preprocessing']
        self.sfreq = self.config['sfreq']
        self.l_freq = self.config['l_freq']
        self.h_freq = self.config['h_freq']
        self.notch_freq = self.config['notch_freq']
        self.epoch_duration_s = self.config['epoch_duration_s']
        self.epoch_overlap_s = self.config['epoch_overlap_s']
        self.montage = mne.channels.make_standard_montage(self.config['montage'])
        self.initial_reject_v = 1000 * 1e-6
        # The definitive 19-channel set based on the 10-20 system
        self.standard_channels = [
            'Fp1', 'F7', 'T7', 'P7', 'O1', 'F3', 'C3', 'P3',
            'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4',
            'F8', 'T8', 'P8', 'O2'
        ]

    def _rename_channels(self, raw: mne.io.Raw):
        ch_map = {
            'FP1-F7': 'Fp1', 'F7-T7': 'F7', 'T7-P7': 'T7', 'P7-O1': 'O1',
            'FP1-F3': 'F3', 'F3-C3': 'C3', 'C3-P3': 'P3', 'P3-O1': 'P3',
            'FP2-F8': 'Fp2', 'F8-T8': 'F8', 'T8-P8': 'T8', 'P8-O2': 'P8',
            'FP2-F4': 'F4', 'F4-C4': 'C4', 'C4-P4': 'P4', 'P4-O2': 'O2',
            'FZ-CZ': 'Fz', 'CZ-PZ': 'Cz', 'P7-T7': 'P7', 'T8-P8-1': 'P8',
        }
        current_map = {ch: ch_map[ch] for ch in ch_map if ch in raw.ch_names}
        raw.rename_channels(current_map, allow_duplicates=False)

    def _universal_reader(self, file_path: str) -> mne.io.Raw:
        try:
            return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception:
            return None

    def preprocess(self, file_path: str) -> tuple:
        raw = self._universal_reader(file_path)
        if raw is None: return None, None, None

        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        self._rename_channels(raw)
        
        # Enforce the standard channel set and interpolate missing
        present_channels = [ch for ch in self.standard_channels if ch in raw.ch_names]
        info = mne.create_info(ch_names=self.standard_channels, sfreq=raw.info['sfreq'], ch_types='eeg')
        standard_raw = mne.io.RawArray(np.zeros((len(self.standard_channels), raw.n_times)), info, verbose=False)
        standard_raw.set_montage(self.montage)
        
        # Copy data for channels that exist before interpolation
        map_from = [raw.ch_names.index(ch) for ch in present_channels]
        map_to = [standard_raw.ch_names.index(ch) for ch in present_channels]
        standard_raw._data[map_to, :] = raw._data[map_from, :]

        missing_channels = [ch for ch in self.standard_channels if ch not in present_channels]
        if missing_channels:
            standard_raw.info['bads'] = missing_channels
            standard_raw.interpolate_bads(reset_bads=True, mode='accurate', verbose=False)
        
        raw = standard_raw
        raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose=False)
        raw.notch_filter(self.notch_freq, fir_design='firwin', verbose=False)
        raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)

        epochs = mne.make_fixed_length_epochs(raw, duration=self.epoch_duration_s, overlap=self.epoch_overlap_s, preload=True, reject_by_annotation=True, verbose=False)
        if len(epochs) == 0: return None, None, None

        # --- Stage 1: Global "Bad Span" Rejection ---
        epochs.drop_bad(reject=dict(eeg=self.initial_reject_v))
        if len(epochs) == 0: return None, None, None
        
        # --- Stage 2: Local Outlier Detection with Isolation Forest ---
        epochs_data_for_features = epochs.get_data()
        features = np.hstack([epochs_data_for_features.var(axis=2), np.mean(epochs_data_for_features**2, axis=2)])
        iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=1)
        is_inlier = iso_forest.fit_predict(features)
        
        # --- THE FIX: Create a mask of indices to keep ---
        inlier_indices = np.where(is_inlier == 1)[0]
        artifact_indices = np.where(is_inlier == -1)[0]

        seizure_intervals = load_annotations_for_file(file_path)
        pathology_labels = []
        
        # Label pathology ONLY for the clean (inlier) epochs
        clean_epochs = epochs[inlier_indices]
        for i in range(len(clean_epochs)):
            epoch_start_s = clean_epochs.events[i, 0] / clean_epochs.info['sfreq']
            is_seizure = 0
            for seizure in seizure_intervals:
                if seizure['start'] <= epoch_start_s < seizure['end']:
                    is_seizure = 1
                    break
            pathology_labels.append(is_seizure)

        # Combine clean and artifact epochs for the final data tensor
        final_epochs_to_keep = epochs[np.hstack([inlier_indices, artifact_indices])]
        final_data = final_epochs_to_keep.get_data()

        # Create final artifact labels (0 for clean, 1 for artifact)
        final_artifact_labels = np.concatenate([np.zeros(len(inlier_indices)), np.ones(len(artifact_indices))])
        
        # Create final pathology labels (-100 for artifacts, 0/1 for clean)
        final_pathology_labels = np.concatenate([np.array(pathology_labels), np.full(len(artifact_indices), -100)])
        
        return final_data, final_pathology_labels, final_artifact_labels