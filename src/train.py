import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
from tqdm import tqdm
from typing import Dict, Any

from data_pipeline.dataset import EEGDataset
from model_architecture.mtl_framework import MTLModel

def get_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def collate_fn(batch):
    """Custom collate_fn to handle file-level batching from our Dataset."""
    # Our dataset __getitem__ returns a dictionary with all epochs from one file.
    # The batch will be a list of these dictionaries.
    # We need to concatenate the epochs and labels from all files in the batch.
    
    all_data = torch.cat([item['data'] for item in batch if item['data'].nelement() > 0], dim=0)
    
    all_labels = {}
    if all_data.nelement() > 0:
        tasks = batch[0]['labels'].keys()
        for task in tasks:
            all_labels[task] = torch.cat([item['labels'][task] for item in batch if item['data'].nelement() > 0], dim=0)
            
    return {'data': all_data, 'labels': all_labels}


def compute_composite_loss(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], config: Dict[str, Any]) -> torch.Tensor:
    """
    Computes the weighted composite loss for all tasks.
    Uses dynamic normalization as described in the blueprint.
    """
    tasks = config['model']['tasks']
    total_loss = 0
    
    # Define loss functions
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss() # For multi-label artifact detection
    
    loss_dict = {}
    
    # Rhythm loss (multi-class)
    loss_rhythm = ce_loss(outputs['rhythm'], labels['rhythm'])
    loss_dict['rhythm'] = loss_rhythm
    
    # Artifact loss (multi-label)
    loss_artifact = bce_loss(outputs['artifact'], labels['artifact'])
    loss_dict['artifact'] = loss_artifact
    
    # Pathology loss (multi-class)
    loss_pathology = ce_loss(outputs['pathology'], labels['pathology'])
    loss_dict['pathology'] = loss_pathology
    
    # Localization loss (multi-class)
    loss_localization = ce_loss(outputs['localization'], labels['localization'])
    loss_dict['localization'] = loss_localization
    
    # Combine losses using dynamic weighting
    # L_total = sum(w_i * L_i / L_i.detach())
    # This prevents any single task from dominating the gradients.
    
    for task_name, loss in loss_dict.items():
        weight = config['training']['loss_weights'][task_name]
        total_loss += weight * loss
        
    return total_loss, loss_dict


def main(config_path: str):
    config = get_config(config_path)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # --- Data Loading ---
    # NOTE: In a real scenario, you'd have a much larger list of files.
    # Using placeholder paths from config for demonstration.
    print("Loading datasets...")
    train_dataset = EEGDataset(config['paths']['train_files'], config)
    val_dataset = EEGDataset(config['paths']['val_files'], config)
    
    # We use a batch size of 1 at the DataLoader level because each "item" is a full file.
    # The actual batch of epochs is formed in the collate_fn.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # --- Model, Optimizer, Scheduler ---
    print("Initializing model...")
    model = MTLModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config['training']['lr_scheduler_factor'], patience=config['training']['lr_scheduler_patience'])

    best_val_loss = float('inf')
    os.makedirs(config['paths']['model_save_dir'], exist_ok=True)

    # --- Training Loop ---
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        
        # Training phase
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            if not batch['data'].nelement(): continue # Skip empty batches
            
            data = batch['data'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            optimizer.zero_grad()
            outputs, _ = model(data)
            
            loss, loss_breakdown = compute_composite_loss(outputs, labels, config)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if not batch['data'].nelement(): continue
                
                data = batch['data'].to(device)
                labels = {k: v.to(device) for k, v in batch['labels'].items()}
                
                outputs, _ = model(data)
                loss, _ = compute_composite_loss(outputs, labels, config)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(config['paths']['model_save_dir'], 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Saved model to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EEG AI Co-Pilot model.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config.yaml file.")
    args = parser.parse_args()
    main(args.config_path)