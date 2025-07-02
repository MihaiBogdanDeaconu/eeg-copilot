import torch
import torch.nn as nn
from typing import Dict, Any

from .cnn_transformer import CNNTransformerBackbone

class MTLModel(nn.Module):
    """
    Multi-Task Learning model with a shared backbone and task-specific heads.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the MTL model.
        
        Args:
            config (Dict[str, Any]): The project configuration dictionary.
        """
        super().__init__()
        self.config = config
        model_config = config['model']
        tasks = model_config['tasks']
        
        # 1. Shared Backbone
        self.backbone = CNNTransformerBackbone(config)
        
        # 2. Task-Specific Heads
        self.heads = nn.ModuleDict()
        backbone_out_dim = model_config['cnn_out_channels']
        
        for task_name, task_config in tasks.items():
            self.heads[task_name] = nn.Linear(backbone_out_dim, task_config['num_classes'])
            
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        """
        Performs a forward pass through the backbone and all task heads.
        
        Args:
            x (torch.Tensor): Input EEG data of shape (Batch, Channels, Timepoints).
            return_attention (bool): Whether to return attention maps.
            
        Returns:
            Tuple containing:
            - A dictionary of task outputs (logits). Keys are task names.
            - Attention weights tensor if requested.
        """
        # Get the final feature representation from the backbone
        features, attention_weights = self.backbone(x, return_attention)
        
        # Pass the features through each task-specific head
        outputs = {
            task_name: head(features) for task_name, head in self.heads.items()
        }
        
        return outputs, attention_weights