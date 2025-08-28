#!/usr/bin/env python3
"""
Prithvi 2.0 Teacher Model Loader
Simple wrapper for loading and using Prithvi as a teacher model.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Tuple, Optional
import numpy as np

class PrithviLoader:
    """
    Simple Prithvi 2.0 teacher model loader for knowledge distillation.
    """
    
    def __init__(self, 
                 model_name: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
                 device: str = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"Loading Prithvi 2.0 from {model_name}...")
        
        # Load model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Model info
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        self.input_channels = getattr(self.config, 'num_channels', 12)
        self.patch_size = getattr(self.config, 'patch_size', 16)
        self.image_size = getattr(self.config, 'image_size', 224)
        
        print(f"✅ Prithvi loaded: {self.num_parameters:,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Prithvi model.
        
        Args:
            x: Input tensor [B, C, H, W] where C=12 for multispectral
            
        Returns:
            Output features from the model
        """
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.model(x)
            return outputs.last_hidden_state
    
    def get_features(self, x: torch.Tensor, layer_names: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for knowledge distillation.
        
        Args:
            x: Input tensor [B, C, H, W]
            layer_names: Specific layers to extract (if None, extracts key layers)
            
        Returns:
            Dictionary of layer_name -> features
        """
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output.detach()
            return hook
        
        # Default important layers for distillation
        if layer_names is None:
            layer_names = [
                'encoder.layer.6',   # Mid-level features
                'encoder.layer.11',  # High-level features
                'pooler'             # Final pooled features
            ]
        
        # Register hooks
        hooks = []
        for name in layer_names:
            try:
                layer = self._get_layer_by_name(name)
                hook = layer.register_forward_hook(hook_fn(name))
                hooks.append(hook)
            except AttributeError:
                print(f"Warning: Layer {name} not found in Prithvi model")
        
        # Forward pass
        with torch.no_grad():
            x = x.to(self.device)
            _ = self.model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return features
    
    def _get_layer_by_name(self, name: str):
        """Get layer by dot-separated name."""
        layer = self.model
        for attr in name.split('.'):
            layer = getattr(layer, attr)
        return layer
    
    def get_logits(self, x: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """
        Get classification logits (adds a classification head if needed).
        
        Args:
            x: Input tensor [B, C, H, W]
            num_classes: Number of output classes
            
        Returns:
            Classification logits [B, num_classes]
        """
        features = self.forward(x)  # [B, seq_len, hidden_dim]
        
        # Global average pooling
        pooled = features.mean(dim=1)  # [B, hidden_dim]
        
        # Simple classification head
        if not hasattr(self, 'classifier'):
            hidden_dim = features.shape[-1]
            self.classifier = nn.Linear(hidden_dim, num_classes).to(self.device)
        
        logits = self.classifier(pooled)
        return logits
    
    def preprocess_input(self, x: np.ndarray) -> torch.Tensor:
        """
        Preprocess input for Prithvi (normalization, etc.).
        
        Args:
            x: Input array [B, C, H, W] or [C, H, W]
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Normalize to [0, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0
        
        # Standard normalization (you may want to use actual Prithvi stats)
        mean = torch.tensor([0.485, 0.456, 0.406] * 4).view(1, -1, 1, 1)  # Approximate
        std = torch.tensor([0.229, 0.224, 0.225] * 4).view(1, -1, 1, 1)   # Approximate
        
        x = (x - mean) / std
        return x
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'name': 'Prithvi-2.0',
            'parameters': self.num_parameters,
            'input_channels': self.input_channels,
            'patch_size': self.patch_size,
            'image_size': self.image_size,
            'device': self.device
        }