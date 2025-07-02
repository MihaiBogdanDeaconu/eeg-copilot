import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# This is a placeholder for the model. In a real script, you'd import it.
from ..model_architecture.mtl_framework import MTLModel 

def visualize_attention_map(
    model: MTLModel,
    epoch_data: torch.Tensor,
    config: Dict[str, Any],
    save_path: str
):
    """
    Generates and visualizes the attention map from the Transformer's last layer.
    
    Args:
        model (MTLModel): The trained multi-task model.
        epoch_data (torch.Tensor): A single epoch of data (1, Channels, Timepoints).
        config (Dict[str, Any]): Project configuration.
        save_path (str): Path to save the visualization.
    """
    print("Generating attention map...")
    model.eval()
    
    # --- This is a workaround to get attention weights from a standard nn.TransformerEncoder ---
    # We register a forward hook on the final multi-head attention layer
    attention_weights = None
    
    def hook_fn(module, input, output):
        nonlocal attention_weights
        # output is a tuple (attn_output, attn_output_weights)
        attention_weights = output[1].detach()

    # Register the hook on the last MHA layer of the last encoder block
    handle = model.backbone.transformer_encoder.layers[-1].self_attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(epoch_data)
        
    handle.remove() # Don't forget to remove the hook!
    
    if attention_weights is None:
        print("Could not extract attention weights. Check model architecture and hook.")
        return

    # Process the attention weights
    # Shape: (Batch, Num_Heads, Seq_Len, Seq_Len) -> (1, H, S, S)
    # We care about the attention from the [CLS] token to the sequence tokens
    cls_attention = attention_weights[0, :, 0, 1:] # (Heads, Seq_Len_out)
    
    # Average attention across all heads
    avg_attention = cls_attention.mean(dim=0).cpu().numpy() # (Seq_Len_out,)
    
    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot EEG data
    eeg_data_np = epoch_data.squeeze().cpu().numpy()
    num_channels, num_times = eeg_data_np.shape
    time = np.linspace(0, config['preprocessing']['epoch_duration_s'], num_times)
    
    # Offset for plotting
    offsets = np.arange(num_channels) * eeg_data_np.std() * 3
    ax1.plot(time, (eeg_data_np + offsets[:, np.newaxis]).T, color='black', lw=0.5)
    
    ax1.set_yticks(offsets)
    ax1.set_yticklabels([f'Ch {i+1}' for i in range(num_channels)])
    ax1.set_title('EEG Traces with Model Attention')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Channels')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Upsample attention to match the original time dimension
    # The CNN downsamples the sequence. We need to map attention back.
    # This is a simplification; a precise mapping requires tracking pooling indices.
    # Here, we just repeat the values to fill the space.
    cnn_downsample_factor = config['model']['sequence_length'] // len(avg_attention)
    upsampled_attention = np.repeat(avg_attention, cnn_downsample_factor)
    # Pad if necessary
    if len(upsampled_attention) < num_times:
        upsampled_attention = np.pad(upsampled_attention, (0, num_times - len(upsampled_attention)), 'edge')
    
    # Plot attention as a heatmap below
    im = ax2.imshow(
        upsampled_attention[np.newaxis, :],
        aspect='auto',
        cmap='inferno',
        extent=[time[0], time[-1], 0, 1]
    )
    ax2.set_title('Transformer Attention (from CLS token)')
    ax2.set_xlabel('Time (s)')
    ax2.set_yticks([])
    
    cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.2)
    cbar.set_label('Attention Score')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Attention map saved to {save_path}")