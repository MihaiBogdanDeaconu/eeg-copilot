import torch
import os
from torch.utils.data import Dataset
from typing import List, Dict, Any
import numpy as np

class EEGDataset(Dataset):
    """
    SOTA PyTorch Dataset that loads PREPROCESSED data from .npz files.
    This version is for running on the training server. It is extremely fast
    as all heavy preprocessing has been done offline.
    """
    def __init__(self, file_paths: List[str], config: Dict[str, Any]):
        """
        Args:
            file_paths (List[str]): List of paths to the preprocessed .npz files.
            config (Dict[str, Any]): The project configuration dictionary.
        """
        self.file_paths = file_paths
        self.config = config
        self.tasks = self.config['model']['tasks']
        self.ignore_index = -100

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads a single preprocessed .npz file containing multiple epochs.
        """
        npz_file_path = self.file_paths[idx]
        
        try:
            with np.load(npz_file_path) as npz_file:
                data = torch.from_numpy(npz_file['data']).float()
                
                # --- Assemble all labels for this file ---
                num_epochs = data.shape[0]
                
                # Get the real labels we saved
                pathology_labels = torch.from_numpy(npz_file['pathology']).long()
                artifact_binary = torch.from_numpy(npz_file['artifact']).long()
                
                # For multi-label artifact head, convert binary to one-hot encoding
                artifact_labels = torch.nn.functional.one_hot(artifact_binary, num_classes=self.tasks['artifact']['num_classes']).float()
                
                # Create dummy/ignore labels for tasks we don't have ground truth for
                epileptiform_labels = torch.full((num_epochs,), self.ignore_index, dtype=torch.long)
                
                labels = {
                    'seizure_binary': pathology_labels,
                    'epileptiform_event': epileptiform_labels,
                    'artifact': artifact_labels,
                }
                
                return {'data': data, 'labels': labels}

        except Exception as e:
            print(f"Warning: Could not load or process file {npz_file_path}. Error: {e}. Skipping.")
            return {'data': torch.empty(0), 'labels': {}}