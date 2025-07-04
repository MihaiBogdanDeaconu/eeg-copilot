# In file: src/data_pipeline/dataset.py

import torch
import os
from torch.utils.data import Dataset
from typing import List, Dict, Any
from .preprocessing import EEGPreprocessor
from .annotations import load_annotations_for_file

class EEGDataset(Dataset):
    """
    Production-ready PyTorch Dataset for EEG analysis.
    """
    def __init__(self, file_paths: List[str], config: Dict[str, Any]):
        self.file_paths = file_paths
        self.config = config
        self.preprocessor = EEGPreprocessor(config)
        self.epoch_duration_s = config['preprocessing']['epoch_duration_s']
        self.epoch_overlap_s = config['preprocessing']['epoch_overlap_s']
        
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.file_paths[idx]
        epochs_data = self.preprocessor.preprocess(file_path)
        
        if epochs_data.shape[0] == 0:
            return {'data': torch.empty(0), 'labels': {}}

        num_epochs = epochs_data.shape[0]
        seizure_intervals = load_annotations_for_file(file_path)
        tasks = self.config['model']['tasks']
        
        # --- Real Label Generation ---
        pathology_labels = torch.zeros(num_epochs, dtype=torch.long) # 0 = Non-Seizure
        for i in range(num_epochs):
            epoch_start_s = i * (self.epoch_duration_s - self.epoch_overlap_s)
            epoch_end_s = epoch_start_s + self.epoch_duration_s
            for seizure in seizure_intervals:
                if max(epoch_start_s, seizure['start']) < min(epoch_end_s, seizure['end']):
                    pathology_labels[i] = 1 # Class 1 = Seizure
                    break
        
        localization_labels = torch.zeros(num_epochs, dtype=torch.long)
        localization_labels[pathology_labels == 1] = 2 # 2 = Temporal
        
        # --- Handle Missing Labels Correctly ---
        # For tasks where CHB-MIT has no labels, we create tensors with an ignore_index.
        # This tells the loss function to skip these tasks for this batch of data.
        ignore_index = -100 # PyTorch's default ignore_index for CrossEntropyLoss
        
        rhythm_labels = torch.full((num_epochs,), ignore_index, dtype=torch.long)
        
        # For multi-label BCE loss, we can't use an ignore index in the same way.
        # We will create a tensor of zeros, but also pass a weight mask.
        # (This will be handled in the training loop for simplicity here).
        artifact_labels = torch.zeros(num_epochs, tasks['artifact']['num_classes'], dtype=torch.float)

        labels = {
            'rhythm': rhythm_labels,
            'artifact': artifact_labels,
            'pathology': pathology_labels,
            'localization': localization_labels,
        }

        return {
            'data': torch.from_numpy(epochs_data).float(),
            'labels': labels
        }