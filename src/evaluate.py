# In file: src/evaluate.py

import torch
import torch.nn.functional as F
import yaml
import argparse
import os
import json
from tqdm import tqdm
from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix

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

def evaluate(config_path: str, model_weights_path: str):
    config = get_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Test Data ---
    split_file = config['paths']['dataset_split_file']
    with open(split_file, 'r') as f:
        file_splits = json.load(f)
    test_files = file_splits.get('test')
    if not test_files:
        raise ValueError("No 'test' key found in dataset_split.json. Please create a test set.")
    
    test_dataset = EEGDataset(test_files, config)
    # Use num_workers=0 for evaluation to ensure ordered processing if needed
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # --- Load Model ---
    model = MTLModel(config)
    # Handle DataParallel wrapper if the model was saved that way
    state_dict = torch.load(model_weights_path, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Run Inference and Collect Results ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            if not batch['data'].nelement():
                continue
            
            data = batch['data'].to(device)
            labels = batch['labels']['seizure_binary'] # Focus on the primary binary task
            
            outputs, _ = model(data)
            # Get probabilities for the positive class (seizure)
            probs = F.softmax(outputs['seizure_binary'], dim=1)[:, 1]

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Calculate and Print Metrics ---
    # Convert probabilities to binary predictions for threshold-based metrics
    binary_preds = (all_preds >= 0.5).astype(int)

    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, binary_preds)
    sensitivity = recall_score(all_labels, binary_preds) # Recall is sensitivity
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
    specificity = tn / (tn + fp)
    precision = precision_score(all_labels, binary_preds)

    print("\n" + "="*50)
    print("           Performance on Test Set")
    print("="*50)
    print(f"AUROC (Area Under ROC Curve):       {auroc:.4f}")
    print(f"AUPRC (Area Under Precision-Recall): {auprc:.4f}")
    print(f"F1-Score:                           {f1:.4f}")
    print(f"Sensitivity (Recall):               {sensitivity:.4f}")
    print(f"Specificity:                        {specificity:.4f}")
    print(f"Precision:                          {precision:.4f}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the trained EEG AI Co-Pilot model.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config.yaml file.")
    parser.add_argument('--model_weights_path', type=str, required=True, help="Path to the saved model .pth file from training.")
    args = parser.parse_args()
    evaluate(args.config_path, args.model_weights_path)