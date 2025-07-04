# In file: src/data_pipeline/preprocessing.py

import mne
import numpy as np
import os
from typing import Dict, Any, List
from autoreject import AutoReject

class EEGPreprocessor:
    """
    The "Data Refinery" pipeline.
    Transforms raw EEG files into clean, epoched, and normalized data ready for the model.
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
        self.reject_threshold_v = self.config['reject_threshold_uv']

    def _rename_chb_mit_channels(self, raw: mne.io.Raw):
        chb_to_standard_map = {
            'FP1-F7': 'Fp1', 'F7-T7': 'F7', 'T7-P7': 'T7', 'P7-O1': 'O1',
            'FP1-F3': 'F3', 'F3-C3': 'C3', 'C3-P3': 'P3', 'P3-O1': 'P3',
            'FP2-F8': 'Fp2', 'F8-T8': 'F8', 'T8-P8': 'T8', 'P8-O2': 'P8',
            'FP2-F4': 'F4', 'F4-C4': 'C4', 'C4-P4': 'P4', 'P4-O2': 'O2',
            'FZ-CZ': 'Fz', 'CZ-PZ': 'Cz',
            'P7-T7': 'P7', 'T8-P8-1': 'P8',
        }
        current_ch_names_lower = {ch.lower(): ch for ch in raw.ch_names}
        map_keys_lower = {key.lower(): key for key in chb_to_standard_map.keys()}
        current_mapping = {}
        for ch_lower in current_ch_names_lower:
            if ch_lower in map_keys_lower:
                original_case_key = map_keys_lower[ch_lower]
                target_name = chb_to_standard_map[original_case_key]
                if target_name not in current_mapping.values():
                    current_mapping[original_case_key] = target_name
        raw.rename_channels(current_mapping, allow_duplicates=False)

    def _universal_reader(self, file_path: str) -> mne.io.Raw:
        readers = {'.edf': mne.io.read_raw_edf, '.bdf': mne.io.read_raw_bdf, '.gdf': mne.io.read_raw_gdf}
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in readers:
            try:
                return readers[file_extension](file_path, preload=True, verbose=False)
            except Exception as e:
                raise IOError(f"Failed to read {file_path} with MNE: {e}")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    # In src/data_pipeline/preprocessing.py
    def preprocess(self, file_path: str) -> np.ndarray:
        """
        Executes a FAST preprocessing pipeline using a fixed rejection threshold for testing.
        """
        raw = self._universal_reader(file_path)
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        self._rename_chb_mit_channels(raw)
        raw.set_montage(self.montage, on_missing='ignore')
        ch_with_locs = [ch for ch in raw.ch_names if np.all(np.isfinite(raw._get_channel_positions([ch])[0]))]
        raw.pick(ch_with_locs)
        if len(raw.ch_names) == 0: return np.array([])

        raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose=False)
        raw.notch_filter(self.notch_freq, fir_design='firwin', verbose=False)
        raw.resample(self.sfreq, npad='auto', verbose=False)
        raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)
        epochs = mne.make_fixed_length_epochs(raw, duration=self.epoch_duration_s, overlap=self.epoch_overlap_s, preload=True, reject_by_annotation=True, verbose=False)
        if len(epochs) == 0: return np.array([])

        # Use the fast, fixed-threshold rejection
        reject_criteria = dict(eeg=self.reject_threshold_v)
        epochs.drop_bad(reject=reject_criteria)

        epochs_data = epochs.get_data(copy=False)
        mean = np.mean(epochs_data, axis=2, keepdims=True)
        std = np.std(epochs_data, axis=2, keepdims=True)
        std[std == 0] = 1 
        normalized_epochs = (epochs_data - mean) / std
        return normalized_epochs