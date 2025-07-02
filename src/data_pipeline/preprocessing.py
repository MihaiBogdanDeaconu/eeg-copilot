import mne
import numpy as np
from typing import Dict, Any

class EEGPreprocessor:
    """
    The "Data Refinery" pipeline.
    Transforms raw EEG files into clean, epoched, and normalized data ready for the model.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the preprocessor with settings from the config file.
        
        Args:
            config (Dict[str, Any]): A dictionary containing preprocessing parameters.
        """
        self.config = config['preprocessing']
        self.sfreq = self.config['sfreq']
        self.l_freq = self.config['l_freq']
        self.h_freq = self.config['h_freq']
        self.notch_freq = self.config['notch_freq']
        self.epoch_duration_s = self.config['epoch_duration_s']
        self.epoch_overlap_s = self.config['epoch_overlap_s']
        self.reject_threshold = self.config['reject_threshold_uv'] * 1e-6  # Convert uV to V
        self.montage = mne.channels.make_standard_montage(self.config['montage'])

    def _universal_reader(self, file_path: str) -> mne.io.Raw:
        """
        Flexible MNE reader that attempts to load various raw file formats.
        As per blueprint, this handles the data extraction problem.
        """
        # Add other readers as needed based on identified file formats
        readers = {
            '.edf': mne.io.read_raw_edf,
            '.bdf': mne.io.read_raw_bdf,
            '.gdf': mne.io.read_raw_gdf,
            # '.vhdr': mne.io.read_raw_brainvision, # Example for BrainVision
            # '.dat': mne.io.read_raw_nicolet,     # Example for Nicolet
        }
        
        file_extension = file_path[file_path.rfind('.'):].lower()
        
        if file_extension in readers:
            try:
                raw = readers[file_extension](file_path, preload=True, verbose=False)
                return raw
            except Exception as e:
                raise IOError(f"Failed to read {file_path} with MNE: {e}")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Please add a reader for it.")

    def preprocess(self, file_path: str) -> np.ndarray:
        """
        Executes the full preprocessing pipeline on a single EEG file.
        
        Args:
            file_path (str): The path to the raw EEG file.
            
        Returns:
            np.ndarray: A numpy array of shape (n_epochs, n_channels, n_times)
                        containing clean, normalized data.
        """
        # 1. Data Loading
        raw = self._universal_reader(file_path)

        # 2. Montage Selection & Referencing
        # Select only channels present in the standard montage
        raw.pick_channels([ch for ch in raw.ch_names if ch in self.montage.ch_names])
        raw.set_montage(self.montage, on_missing='ignore')
        
        # 3. Filtering
        raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose=False)
        raw.notch_filter(self.notch_freq, fir_design='firwin', verbose=False)

        # 4. Resampling
        raw.resample(self.sfreq, npad='auto', verbose=False)

        # 5. Re-referencing
        raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)

        # 6. Epoching
        epoch_len_samples = int(self.epoch_duration_s * self.sfreq)
        epoch_overlap_samples = int(self.epoch_overlap_s * self.sfreq)
        
        epochs = mne.make_fixed_length_epochs(
            raw, 
            duration=self.epoch_duration_s, 
            overlap=self.epoch_overlap_s,
            preload=True,
            reject_by_annotation=True,
            verbose=False
        )
        
        # 7. Preliminary Artifact Rejection
        reject_criteria = dict(eeg=self.reject_threshold)
        epochs.drop_bad(reject=reject_criteria, verbose=False)
        
        if len(epochs) == 0:
            return np.array([]) # Return empty array if no good epochs found

        # 8. Normalization (Epoch-wise Z-score)
        epochs_data = epochs.get_data() # (n_epochs, n_channels, n_times)
        
        mean = np.mean(epochs_data, axis=2, keepdims=True)
        std = np.std(epochs_data, axis=2, keepdims=True)
        # Avoid division by zero for flat channels
        std[std == 0] = 1 
        
        normalized_epochs = (epochs_data - mean) / std
        
        return normalized_epochs