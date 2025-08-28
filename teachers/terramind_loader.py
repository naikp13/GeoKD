#!/usr/bin/env python3
"""
TerraMind Teacher Model Loader
Simple wrapper for loading and using TerraMind as a teacher model.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Union
import numpy as np

class TerraMindLoader:
    """
    Simple TerraMind teacher model loader for knowledge distillation.
    """
    
    def __init__(self, 
                 model_name: str = "ibm-esa-geospatial/TerraMind-1.0-large",
                 modalities: List[str] = None,
                 device: str = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.modalities = modalities or ['S2L2A']  # Default to Sentinel-2
        
        print(f"Loading TerraMind from {model_name}...")
        
        # Load model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Model info
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        
        # Modality configurations
        self.modality_config = {
            'S2L2A': {'channels': 12, 'bands': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']},
            'S1GRD': {'channels': 2, 'polarizations': ['VV', 'VH']},
            'DEM': {'channels': 1}
        }
        
        print(f"✅ TerraMind loaded: {self.num_parameters:,} parameters")
        print(f"📡 Supported modalities: {self.modalities}")
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through TerraMind model.
        
        Args:
            x: Input tensor [B, C, H, W] or dict of modality -> tensor
            
        Returns:
            Output features from the model
        """
        with torch.no_grad():
            if isinstance(x, dict):
                # Multi-modal input
                x = {k: v.to(self.device) for k, v in x.items()}
            else:
                # Single tensor input
                x = x.to(self.device)
            
            outputs = self.model(x)
            return outputs.last_hidden_state
    
    def get_features(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                    layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for knowledge distillation.
        
        Args:
            x: Input tensor or dict of modality -> tensor
            layer_names: Specific layers to extract
            
        Returns:
            Dictionary of layer_name -> features
        """
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output.detach()
            return hook
        
        # Default important layers
        if layer_names is None:
            layer_names = [
                'encoder.layer.8',   # Mid-level features
                'encoder.layer.15',  # High-level features
                'decoder.layer.3'    # Decoder features
            ]
        
        # Register hooks
        hooks = []
        for name in layer_names:
            try:
                layer = self._get_layer_by_name(name)
                hook = layer.register_forward_hook(hook_fn(name))
                hooks.append(hook)
            except AttributeError:
                print(f"Warning: Layer {name} not found in TerraMind model")
        
        # Forward pass
        with torch.no_grad():
            if isinstance(x, dict):
                x = {k: v.to(self.device) for k, v in x.items()}
            else:
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
    
    def generate(self, input_modality: str, target_modality: str, 
                x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generate target modality from input modality (Any-to-Any).
        
        Args:
            input_modality: Source modality (e.g., 'S2L2A')
            target_modality: Target modality (e.g., 'S1GRD')
            x: Input tensor [B, C, H, W]
            
        Returns:
            Generated target modality tensor
        """
        with torch.no_grad():
            x = x.to(self.device)
            
            # Prepare input with modality information
            inputs = {
                input_modality: x,
                'target_modality': target_modality
            }
            
            # Generate (simplified - actual implementation depends on TerraMind API)
            outputs = self.model.generate(inputs, **kwargs)
            return outputs
    
    def get_logits(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                  num_classes: int = 10) -> torch.Tensor:
        """
        Get classification logits.
        
        Args:
            x: Input tensor or dict of modality -> tensor
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
    
    def preprocess_input(self, x: np.ndarray, modality: str = 'S2L2A') -> torch.Tensor:
        """
        Preprocess input for specific modality.
        
        Args:
            x: Input array [B, C, H, W] or [C, H, W]
            modality: Input modality type
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Modality-specific preprocessing
        if modality == 'S2L2A':
            # Sentinel-2 preprocessing
            if x.max() > 1.0:
                x = x / 10000.0  # Typical Sentinel-2 scaling
            x = torch.clamp(x, 0, 1)
        
        elif modality == 'S1GRD':
            # Sentinel-1 preprocessing (dB values)
            x = torch.clamp(x, -30, 0)  # Typical SAR range
            x = (x + 30) / 30  # Normalize to [0, 1]
        
        return x
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'name': 'TerraMind-1.0',
            'parameters': self.num_parameters,
            'modalities': self.modalities,
            'modality_config': self.modality_config,
            'device': self.device
        }