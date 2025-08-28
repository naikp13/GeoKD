from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class BaseGeospatialModel(ABC, nn.Module):
    """Base class for geospatial foundation models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = config.get('model_name', 'unknown')
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def get_features(self, x: torch.Tensor, layer_names: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """Extract intermediate features for knowledge distillation."""
        pass
    
    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load pretrained model."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': self.model_name,
            'parameters': sum(p.numel() for p in self.parameters()),
            'config': self.config
        }