import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional
from .base_model import BaseGeospatialModel

class PrithviModel(BaseGeospatialModel):
    """Prithvi 2.0 model wrapper for geospatial tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = 'Prithvi-2.0'
        self.hf_model_name = 'ibm-nasa-geospatial/Prithvi-EO-2.0-600M'
        
        # Load the pretrained model
        self.backbone = AutoModel.from_pretrained(self.hf_model_name)
        self.feature_dim = self.backbone.config.hidden_size
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through Prithvi model."""
        # Assuming x is preprocessed satellite imagery
        outputs = self.backbone(x, **kwargs)
        return outputs.last_hidden_state
    
    def get_features(self, x: torch.Tensor, layer_names: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """Extract intermediate features for knowledge distillation."""
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks for specified layers
        hooks = []
        if layer_names:
            for name, module in self.backbone.named_modules():
                if name in layer_names:
                    hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        output = self.forward(x)
        features['output'] = output
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return features
    
    @classmethod
    def from_pretrained(cls, model_path: str = None, **kwargs):
        """Load pretrained Prithvi model."""
        config = kwargs.get('config', {})
        config['model_name'] = 'Prithvi-2.0'
        return cls(config)