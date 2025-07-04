# In file: scripts/count_parameters.py

import torch
import yaml
import argparse
import os
import sys

# Add the project root to the Python path to allow importing from 'src'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.model_architecture import MTLModel

def count_parameters(config_path: str):
    """
    Loads the model based on a config file and prints the number of trainable parameters.
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Instantiate the model (no need to move to GPU for parameter counting)
    model = MTLModel(config)
    
    print("\n" + "="*50)
    print("Model Parameter Count Breakdown")
    print("="*50)

    # --- Calculate parameter counts for different components ---
    
    # Backbone parameters
    cnn_params = sum(p.numel() for p in model.backbone.temporal_conv.parameters() if p.requires_grad)
    # The spatial_conv is created dynamically, so we can't count it this way.
    # We'll count the whole backbone and subtract the easily separable parts.
    
    transformer_params = sum(p.numel() for p in model.backbone.transformer_encoder.parameters() if p.requires_grad)
    pos_encoder_params = sum(p.numel() for p in model.backbone.pos_encoder.parameters() if p.requires_grad)
    cls_token_params = model.backbone.cls_token.numel()

    # Heads parameters
    heads_params = sum(p.numel() for p in model.heads.parameters() if p.requires_grad)

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate the remaining CNN params
    other_backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad) - (transformer_params + pos_encoder_params + cls_token_params)

    print(f"  - CNN Feature Extractor (Backbone):\t {other_backbone_params:,}")
    print(f"  - Transformer Encoder (Backbone):\t {transformer_params:,}")
    print(f"  - Other Backbone (Positional Enc, etc):\t {pos_encoder_params + cls_token_params:,}")
    print(f"  - Multi-Task Heads:\t\t\t {heads_params:,}")
    print("-"*50)
    print(f"  Total Trainable Parameters:\t\t {total_params:,}")
    print("="*50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count the parameters of the EEG AI Co-Pilot model.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config.yaml file.")
    args = parser.parse_args()
    count_parameters(args.config_path)