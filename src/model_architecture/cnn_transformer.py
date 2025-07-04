# In file: src/model_architecture/cnn_transformer.py

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional # <-- 1. IMPORT THIS


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CNNTransformerBackbone(nn.Module):
    """
    Hybrid CNN-Transformer Backbone Architecture.
    This version is made FLEXIBLE to handle varying numbers of input channels.
    """
    def __init__(self, config: dict):
        super().__init__()
        model_config = config['model']
        self.d_model = model_config['cnn_out_channels']
        
        # --- CNN Front-End Layers (defined individually for flexibility) ---
        
        # Part 1: Temporal Convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding='same', bias=False),
            nn.BatchNorm2d(16)
        )
        
        # Part 2: Spatial Convolution is now defined in the forward pass
        # This is the layer that needed to be dynamic.
        
        # Part 3: Rest of the CNN pipeline
        self.feature_aggregator = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(model_config['dropout']),
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding='same', groups=32, bias=False),
            nn.Conv2d(32, self.d_model, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(model_config['dropout'])
        )

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(self.d_model, model_config['dropout'])

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=model_config['transformer_heads'],
            dim_feedforward=model_config['transformer_ff_dim'],
            dropout=model_config['dropout'],
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config['transformer_layers'],
            norm=nn.LayerNorm(self.d_model)
        )

        # --- Classification Token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input EEG data of shape (Batch, Channels, Timepoints)
        """
        # Get the actual number of channels from the input tensor
        num_channels = x.size(1)
        
        # Add a channel dimension for Conv2D
        x = x.unsqueeze(1) # (B, 1, C, T)

        # 1. Temporal Convolution
        x = self.temporal_conv(x)
        
        # 2. DYNAMIC Spatial Depthwise Convolution
        # We define and apply it on the fly using the input's channel count
        spatial_conv = nn.Conv2d(16, 32, kernel_size=(num_channels, 1), groups=16, bias=False).to(x.device)
        x = spatial_conv(x) # Output will have height=1, e.g., (B, 32, 1, T_out)
        
        # 3. Rest of the CNN feature extraction
        x = self.feature_aggregator(x) # (B, d_model, 1, T_out)
        
        # 4. Reshape for Transformer
        # This squeeze will now work correctly because the dimension at index 2 is 1
        x = x.squeeze(2)      # (B, d_model, T_out)
        x = x.permute(0, 2, 1) # (B, T_out, d_model) <-- This will now succeed

        # 5. Prepend CLS token and apply Transformer
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        
        x = self.transformer_encoder(x)
        
        cls_representation = x[:, 0, :]
        
        # Note: attention extraction logic remains separate in the XAI script
        return cls_representation, None