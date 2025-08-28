import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from torchvision.models import resnet18, resnet34, resnet50
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.base_model import BaseGeospatialModel

class ResNetStudent(BaseGeospatialModel):
    """ResNet-based student network for geospatial knowledge distillation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.num_classes = config.get('num_classes', 1000)
        self.in_channels = config.get('in_channels', 3)
        self.architecture = config.get('architecture', 'resnet18')  # resnet18, resnet34, resnet50
        self.pretrained = config.get('pretrained', True)
        
        # Build backbone
        self._build_backbone()
        
        # Feature extraction hooks
        self.feature_hooks = {}
        self.feature_maps = {}
        self._register_hooks()
        
    def _build_backbone(self):
        """Build ResNet backbone."""
        if self.architecture == 'resnet18':
            backbone = resnet18(pretrained=self.pretrained)
        elif self.architecture == 'resnet34':
            backbone = resnet34(pretrained=self.pretrained)
        elif self.architecture == 'resnet50':
            backbone = resnet50(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        # Modify first conv layer for different input channels
        if self.in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Modify classifier for different number of classes
        backbone.fc = nn.Linear(backbone.fc.in_features, self.num_classes)
        
        self.backbone = backbone
        
    def _register_hooks(self):
        """Register forward hooks for feature extraction."""
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output
            return hook
        
        # Register hooks for different layers
        self.feature_hooks['conv1'] = self.backbone.conv1.register_forward_hook(
            get_activation('conv1')
        )
        self.feature_hooks['layer1'] = self.backbone.layer1.register_forward_hook(
            get_activation('layer1')
        )
        self.feature_hooks['layer2'] = self.backbone.layer2.register_forward_hook(
            get_activation('layer2')
        )
        self.feature_hooks['layer3'] = self.backbone.layer3.register_forward_hook(
            get_activation('layer3')
        )
        self.feature_hooks['layer4'] = self.backbone.layer4.register_forward_hook(
            get_activation('layer4')
        )
        self.feature_hooks['avgpool'] = self.backbone.avgpool.register_forward_hook(
            get_activation('avgpool')
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor, layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Extract intermediate features."""
        # Clear previous feature maps
        self.feature_maps.clear()
        
        # Forward pass to populate feature maps
        _ = self.forward(x)
        
        # Return requested features or all features
        if layer_names is None:
            return self.feature_maps.copy()
        else:
            return {name: self.feature_maps[name] for name in layer_names if name in self.feature_maps}
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load pretrained ResNet student model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint.get('config', {})
        config.update(kwargs)
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get classification logits."""
        return self.forward(x)