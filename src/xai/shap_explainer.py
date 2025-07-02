import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# This is a placeholder for the model. In a real script, you'd import it.
from ..model_architecture.mtl_framework import MTLModel

def explain_prediction_with_shap(
    model: MTLModel,
    background_data: torch.Tensor,
    instance_to_explain: torch.Tensor,
    task: str,
    class_index: int,
    config: Dict[str, Any],
    save_path: str
):
    """
    Uses SHAP (DeepExplainer) to explain a model's prediction for a specific task.

    Args:
        model (MTLModel): The trained multi-task model.
        background_data (torch.Tensor): A tensor of background samples for SHAP.
        instance_to_explain (torch.Tensor): The specific instance to explain.
        task (str): The name of the task head to explain (e.g., 'pathology').
        class_index (int): The index of the class to explain within the task.
        config (Dict[str, Any]): The project configuration.
        save_path (str): Path to save the SHAP force plot.
    """
    print(f"Generating SHAP explanation for task '{task}', class {class_index}...")
    model.eval()

    # 1. Create a SHAP DeepExplainer
    # SHAP needs a background distribution to compute expected values.
    # A subset of the training data is typically used.
    explainer = shap.DeepExplainer(model, background_data)

    # 2. Compute SHAP values for the instance
    # The output of the model is a dict, but SHAP needs a single tensor output.
    # We need to wrap the model to select the desired output.
    def model_wrapper(x):
        outputs, _ = model(x)
        return outputs[task]

    explainer = shap.DeepExplainer((model_wrapper, model.backbone.cnn_feature_extractor), background_data)
    shap_values = explainer.shap_values(instance_to_explain)

    # `shap_values` is a list (one for each class). We select our target class.
    # Shape: (Batch, Channels, Timepoints) -> (1, C, T)
    shap_values_for_class = shap_values[class_index][0]

    # 3. Generate and save a SHAP plot
    # To make a force plot, we need to flatten the input features.
    # This is less intuitive for image-like data. A summary plot is often better.
    
    # Let's create a summary plot of feature importance per channel
    plt.figure(figsize=(10, 8))
    
    # We sum the absolute SHAP values over the time dimension to get channel importance
    channel_importance = np.abs(shap_values_for_class).sum(axis=1)
    
    plt.barh(np.arange(len(channel_importance)), channel_importance)
    plt.yticks(
        np.arange(len(channel_importance)), 
        [f'Ch {i+1}' for i in range(len(channel_importance))]
    )
    plt.xlabel('Sum of absolute SHAP values (Feature Importance)')
    plt.ylabel('EEG Channel')
    
    task_class_name = config['model']['tasks'][task]['class_names'][class_index]
    plt.title(f'SHAP Feature Importance for Predicting "{task_class_name}"')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"SHAP plot saved to {save_path}")