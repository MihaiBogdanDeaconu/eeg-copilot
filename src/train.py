import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix

from data_pipeline.dataset import EEGDataset
from model_architecture.mtl_framework import MTLModel

def get_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_file_path: str):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write("epoch,avg_train_loss,avg_val_loss,val_auroc,val_auprc,val_f1,val_sensitivity,val_specificity,val_precision\n")

def log_metrics(log_file_path: str, epoch: int, metrics: dict):
    with open(log_file_path, 'a') as f:
        f.write(f"{epoch},{metrics['avg_train_loss']:.4f},{metrics['avg_val_loss']:.4f},"
                f"{metrics.get('val_auroc', 0):.4f},{metrics.get('val_auprc', 0):.4f},{metrics.get('val_f1', 0):.4f},"
                f"{metrics.get('val_sensitivity', 0):.4f},{metrics.get('val_specificity', 0):.4f},{metrics.get('val_precision', 0):.4f}\n")

def collate_fn(batch):
    valid_items = [item for item in batch if item['data'].nelement() > 0]
    if not valid_items: return {'data': torch.empty(0), 'labels': {}}
    all_data = torch.cat([item['data'] for item in valid_items], dim=0)
    all_labels = {}
    tasks = valid_items[0]['labels'].keys()
    for task in tasks:
        all_labels[task] = torch.cat([item['labels'][task] for item in valid_items], dim=0)
    return {'data': all_data, 'labels': all_labels}

def compute_composite_loss(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], config: Dict[str, Any]) -> tuple:
    total_loss = 0
    loss_breakdown = {}
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    # --- CORRECTED: Use the 'seizure_binary' key ---
    if 'seizure_binary' in outputs and (labels['seizure_binary'] != -100).any():
        loss = ce_loss(outputs['seizure_binary'], labels['seizure_binary'])
        loss_breakdown['seizure_loss'] = loss.item()
        total_loss += config['training']['loss_weights']['seizure_binary'] * loss

    if 'artifact' in outputs and (labels['artifact'] != -100).any():
        loss = ce_loss(outputs['artifact'], labels['artifact'])
        loss_breakdown['artifact_loss'] = loss.item()
        total_loss += config['training']['loss_weights']['artifact'] * loss
        
    return total_loss, loss_breakdown

def main(config_path: str):
    config = get_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    log_file = config['paths']['log_file']
    setup_logging(log_file)

    split_file = config['paths']['dataset_split_file']
    with open(split_file, 'r') as f: file_splits = json.load(f)
    train_dataset = EEGDataset(file_splits['train'], config)
    val_dataset = EEGDataset(file_splits['validation'], config)
    
    num_workers = config['training']['data_loader']['num_workers']
    print(f"Using {num_workers} workers for data loading.")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    model = MTLModel(config)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config['training']['lr_scheduler_factor'], patience=config['training']['lr_scheduler_patience'])
    best_val_loss = float('inf')

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

        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if not batch['data'].nelement(): continue
                data = batch['data'].to(device)
                labels = {k: v.to(device) for k, v in batch['labels'].items()}
                outputs, _ = model(data)
                loss, _ = compute_composite_loss(outputs, labels, config)
                if torch.is_tensor(loss): total_val_loss += loss.item()
                
                probs = F.softmax(outputs['seizure_binary'], dim=1)[:, 1]
                true_labels = labels['seizure_binary']
                valid_indices = true_labels != -100
                all_preds.extend(probs[valid_indices].cpu().numpy())
                all_labels.extend(true_labels[valid_indices].cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        val_metrics = {'avg_train_loss': avg_train_loss, 'avg_val_loss': avg_val_loss}
        if len(all_labels) > 0:
            binary_preds = (np.array(all_preds) >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds, labels=[0, 1]).ravel()
            val_metrics['val_auroc'] = roc_auc_score(all_labels, all_preds)
            val_metrics['val_auprc'] = average_precision_score(all_labels, all_preds)
            val_metrics['val_f1'] = f1_score(all_labels, binary_preds, zero_division=0)
            val_metrics['val_sensitivity'] = recall_score(all_labels, binary_preds, zero_division=0)
            val_metrics['val_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            val_metrics['val_precision'] = precision_score(all_labels, binary_preds, zero_division=0)

        print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} | Val AUROC: {val_metrics.get('val_auroc', 0):.4f}")
        log_metrics(log_file, epoch + 1, val_metrics)
        
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(config['paths']['model_save_dir'], 'best_model.pth')
            # Save the underlying model state when using DataParallel
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Saved model to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EEG AI Co-Pilot model.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    main(args.config_path)