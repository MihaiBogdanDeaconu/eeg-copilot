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
        """
        Initializes the preprocessor with settings from the config file.
        """
        self.config = config['preprocessing']
        self.sfreq = self.config['sfreq']
        self.l_freq = self.config['l_freq']
        self.h_freq = self.config['h_freq']
        self.notch_freq = self.config['notch_freq']
        self.epoch_duration_s = self.config['epoch_duration_s']
        self.epoch_overlap_s = self.config['epoch_overlap_s']
        # We no longer load a standard montage here.

    def _create_dummy_montage(self, ch_names: List[str]) -> mne.channels.DigMontage:
        """
        Creates a dummy montage with arbitrary 3D positions to satisfy autoreject.
        The exact positions are not critical; their existence and uniqueness are.
        """
        n_channels = len(ch_names)
        # Create positions arranged in a circle on the XY plane
        radius = 0.1  # 10 cm
        angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
        positions = np.array([
            [radius * np.cos(a), radius * np.sin(a), 0] for a in angles
        ])
        
        ch_pos_dict = dict(zip(ch_names, positions))
        return mne.channels.make_dig_montage(ch_pos=ch_pos_dict, coord_frame='head')

    def _universal_reader(self, file_path: str) -> mne.io.Raw:
        """
        Flexible MNE reader that attempts to load various raw file formats.
        """
        readers = {
            '.edf': mne.io.read_raw_edf,
            '.bdf': mne.io.read_raw_bdf,
            '.gdf': mne.io.read_raw_gdf,
        }
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in readers:
            try:
                return readers[file_extension](file_path, preload=True, verbose=False)
            except Exception as e:
                raise IOError(f"Failed to read {file_path} with MNE: {e}")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def preprocess(self, file_path: str) -> np.ndarray:
        """
        Executes the full preprocessing pipeline using a custom-generated montage.
        """
        # 1. Data Loading
        raw = self._universal_reader(file_path)
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})

        # 2. **THE NEW FIX**: Create and set a dummy montage
        dummy_montage = self._create_dummy_montage(raw.ch_names)
        raw.set_montage(dummy_montage)
        
        # 3. Filtering
        raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose=False)
        raw.notch_filter(self.notch_freq, fir_design='firwin', verbose=False)

        # 4. Resampling
        raw.resample(self.sfreq, npad='auto', verbose=False)

        # 5. Re-referencing (optional, but good practice)
        raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)

        # 6. Epoching
        epochs = mne.make_fixed_length_epochs(
            raw, 
            duration=self.epoch_duration_s, 
            overlap=self.epoch_overlap_s,
            preload=True,
            reject_by_annotation=True,
            verbose=False
        )
        
        if len(epochs) == 0:
            return np.array([])

        # 7. Adaptive Artifact Rejection with AutoReject
        # This will now succeed as the epochs object has channel locations.
        ar = AutoReject(picks='eeg', n_interpolate=[1, 2, 4], n_jobs=-1, random_state=42, verbose=False)
        epochs_clean = ar.fit_transform(epochs)

        # 8. Normalization
        epochs_data = epochs_clean.get_data(copy=False)
        
        mean = np.mean(epochs_data, axis=2, keepdims=True)
        std = np.std(epochs_data, axis=2, keepdims=True)
        std[std == 0] = 1 
        
        normalized_epochs = (epochs_data - mean) / std
        
        return normalized_epochs