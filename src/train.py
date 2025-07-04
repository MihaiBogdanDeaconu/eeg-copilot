# In file: src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import os
import json
from tqdm import tqdm
from typing import Dict, Any

from data_pipeline.dataset import EEGDataset
from model_architecture.mtl_framework import MTLModel

def get_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def collate_fn(batch):
    valid_items = [item for item in batch if item['data'].nelement() > 0]
    if not valid_items:
        return {'data': torch.empty(0), 'labels': {}}
    all_data = torch.cat([item['data'] for item in valid_items], dim=0)
    all_labels = {}
    tasks = valid_items[0]['labels'].keys()
    for task in tasks:
        all_labels[task] = torch.cat([item['labels'][task] for item in valid_items], dim=0)
    return {'data': all_data, 'labels': all_labels}

def compute_composite_loss(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], config: Dict[str, Any]) -> torch.Tensor:
    """
    Computes the weighted composite loss, correctly ignoring tasks without valid labels.
    """
    total_loss = 0
    loss_breakdown = {}
    
    # --- Define Loss Functions (with ignore_index for classification) ---
    ignore_index = -100
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    bce_loss = nn.BCEWithLogitsLoss()

    # --- Calculate loss for each task only if valid labels exist ---
    if (labels['pathology'] != ignore_index).any():
        loss_pathology = ce_loss(outputs['pathology'], labels['pathology'])
        loss_breakdown['pathology'] = loss_pathology.item()
        total_loss += config['training']['loss_weights']['pathology'] * loss_pathology
        
    if (labels['localization'] != ignore_index).any():
        loss_localization = ce_loss(outputs['localization'], labels['localization'])
        loss_breakdown['localization'] = loss_localization.item()
        total_loss += config['training']['loss_weights']['localization'] * loss_localization

    # For CHB-MIT, rhythm and artifact will have all ignore_index/zero labels,
    # so their losses won't be calculated, which is the correct behavior.
    if (labels['rhythm'] != ignore_index).any():
        loss_rhythm = ce_loss(outputs['rhythm'], labels['rhythm'])
        loss_breakdown['rhythm'] = loss_rhythm.item()
        total_loss += config['training']['loss_weights']['rhythm'] * loss_rhythm
    
    # We won't calculate artifact loss as we have no labels for it.
    
    return total_loss, loss_breakdown

def main(config_path: str):
    config = get_config(config_path)
    if torch.backends.mps.is_available(): device = torch.device('mps')
    elif torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Data Loading with Multiple Workers ---
    split_file = config['paths']['dataset_split_file']
    with open(split_file, 'r') as f:
        file_splits = json.load(f)
    
    train_dataset = EEGDataset(file_splits['train'], config)
    val_dataset = EEGDataset(file_splits['validation'], config)
    
    num_workers = config['training']['data_loader']['num_workers']
    print(f"Using {num_workers} workers for data loading.")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    # --- Model, Optimizer, Scheduler ---
    model = MTLModel(config)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        # 2. Wrap the model in DataParallel
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config['training']['lr_scheduler_factor'], patience=config['training']['lr_scheduler_patience'])
    os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        model.train()
        total_train_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            if not batch['data'].nelement(): continue
            
            data = batch['data'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            optimizer.zero_grad()
            outputs, _ = model(data)
            
            loss, loss_breakdown = compute_composite_loss(outputs, labels, config)
            
            if torch.is_tensor(loss) and loss != 0:
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                pbar.set_postfix(loss_breakdown)
            
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if not batch['data'].nelement(): continue
                data = batch['data'].to(device)
                labels = {k: v.to(device) for k, v in batch['labels'].items()}
                outputs, _ = model(data)
                loss, _ = compute_composite_loss(outputs, labels, config)
                if torch.is_tensor(loss):
                    total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)

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