import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Any
from .preprocessing import EEGPreprocessor

class EEGDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing EEG data for multi-task learning.
    """
    def __init__(self, file_paths: List[str], config: Dict[str, Any]):
        """
        Args:
            file_paths (List[str]): List of paths to EEG files.
            config (Dict[str, Any]): Project configuration dictionary.
        """
        self.file_paths = file_paths
        self.config = config
        self.preprocessor = EEGPreprocessor(config)
        
        # In a real project, labels would come from annotations or metadata files.
        # Here, we generate mock labels for demonstration purposes.
        self.mock_labels = self._generate_mock_labels(file_paths)

    def _generate_mock_labels(self, file_paths: List[str]) -> pd.DataFrame:
        """Generates a DataFrame of mock labels for each file."""
        print("INFO: Generating mock labels for demonstration. In a real project, load these from annotations.")
        num_files = len(file_paths)
        tasks = self.config['model']['tasks']
        
        data = {
            'file_path': file_paths,
            'rhythm': np.random.randint(0, tasks['rhythm']['num_classes'], num_files),
            # Multi-label for artifacts, so each row is a list of 0s and 1s
            'artifact': [list(np.random.randint(0, 2, tasks['artifact']['num_classes'])) for _ in range(num_files)],
            'pathology': np.random.randint(0, tasks['pathology']['num_classes'], num_files),
            'localization': np.random.randint(0, tasks['localization']['num_classes'], num_files)
        }
        return pd.DataFrame(data).set_index('file_path')

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Processes one EEG file and returns its epochs and corresponding labels.
        
        NOTE: This implementation returns ALL epochs from a file as a single batch.
              A more common approach for large datasets is to create a mapping from
              (file_idx, epoch_idx) to a single epoch. This simplified version
              is for clarity.
        """
        file_path = self.file_paths[idx]
        
        # Preprocess the entire file to get all its valid epochs
        epochs_data = self.preprocessor.preprocess(file_path)
        
        if epochs_data.shape[0] == 0:
            # Handle cases where a file has no valid epochs
            # Return a dummy batch or skip. Here we'll try the next file.
            print(f"Warning: No valid epochs found in {file_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # Get the mock labels for this file
        file_labels = self.mock_labels.loc[file_path]
        
        # For each epoch from this file, assign the same file-level label
        num_epochs = epochs_data.shape[0]
        
        labels = {
            'rhythm': torch.tensor([file_labels['rhythm']] * num_epochs, dtype=torch.long),
            'artifact': torch.tensor([file_labels['artifact']] * num_epochs, dtype=torch.float),
            'pathology': torch.tensor([file_labels['pathology']] * num_epochs, dtype=torch.long),
            'localization': torch.tensor([file_labels['localization']] * num_epochs, dtype=torch.long),
        }
        
        return {
            'data': torch.from_numpy(epochs_data).float(),
            'labels': labels
        }