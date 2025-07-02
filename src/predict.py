import torch
import yaml
import argparse
import os
import numpy as np
from typing import Dict, Any

from data_pipeline.preprocessing import EEGPreprocessor
from model_architecture.mtl_framework import MTLModel
from xai.attention_maps import visualize_attention_map
from xai.shap_explainer import explain_prediction_with_shap

def get_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def print_predictions(predictions: Dict[str, torch.Tensor], config: Dict[str, Any]):
    """Prints formatted predictions from the model."""
    print("\n--- Model Predictions ---")
    tasks = config['model']['tasks']
    
    for task_name, logits in predictions.items():
        print(f"\nTask: {task_name.capitalize()}")
        class_names = tasks[task_name]['class_names']
        
        if task_name == 'artifact': # Multi-label
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            for i, prob in enumerate(probs):
                print(f"  - {class_names[i]}: {prob:.2%} probability")
        else: # Multi-class
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            pred_class_idx = np.argmax(probs)
            pred_class_name = class_names[pred_class_idx]
            print(f"  Prediction: '{pred_class_name}' (Confidence: {probs[pred_class_idx]:.2%})")
            print("  Full Distribution:")
            for i, prob in enumerate(probs):
                print(f"    - {class_names[i]}: {prob:.2%}")


def main(args):
    config = get_config(args.config_path)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

print(f"Using device: {device}")
    print(f"Using device: {device}")

    # --- Load Model ---
    print("Loading trained model...")
    model = MTLModel(config)
    model.load_state_dict(torch.load(args.model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Preprocess Input EEG File ---
    print(f"Preprocessing EEG file: {args.eeg_file_path}")
    preprocessor = EEGPreprocessor(config)
    epochs_data = preprocessor.preprocess(args.eeg_file_path) # (n_epochs, C, T)

    if epochs_data.shape[0] == 0:
        print("Error: No valid epochs could be extracted from the file.")
        return

    epochs_tensor = torch.from_numpy(epochs_data).float().to(device)

    # --- Inference ---
    # We will explain the prediction for the first epoch
    epoch_to_explain = epochs_tensor[0].unsqueeze(0) # (1, C, T)

    with torch.no_grad():
        predictions, _ = model(epoch_to_explain)
    
    print_predictions(predictions, config)
    
    # --- Explainability (XAI) ---
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    # 1. Attention Map Visualization
    attn_map_path = os.path.join(config['paths']['output_dir'], 'attention_map.png')
    visualize_attention_map(model, epoch_to_explain, config, attn_map_path)
    
    # 2. SHAP Explanation
    # We need a background dataset for SHAP. We'll use a few epochs from the input file itself.
    # In a real application, you'd use a representative subset of the training set.
    background_data = epochs_tensor[:10] # Use up to 10 epochs as background
    
    # Let's explain the 'pathology' prediction
    task_to_explain = 'pathology'
    predicted_class_idx = torch.argmax(predictions[task_to_explain]).item()
    
    shap_plot_path = os.path.join(config['paths']['output_dir'], 'shap_explanation.png')
    explain_prediction_with_shap(
        model=model,
        background_data=background_data,
        instance_to_explain=epoch_to_explain,
        task=task_to_explain,
        class_index=predicted_class_idx,
        config=config,
        save_path=shap_plot_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference and XAI on the EEG AI Co-Pilot model.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config.yaml file.")
    parser.add_argument('--model_weights_path', type=str, required=True, help="Path to the saved model .pth file.")
    parser.add_argument('--eeg_file_path', type=str, required=True, help="Path to the input EEG file (.edf, .bdf, etc.).")
    args = parser.parse_args()
    main(args)