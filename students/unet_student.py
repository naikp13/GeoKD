import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.base_model import BaseGeospatialModel

class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetStudent(BaseGeospatialModel):
    """U-Net based student network for geospatial segmentation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.in_channels = config.get('in_channels', 3)
        self.num_classes = config.get('num_classes', 1)
        self.bilinear = config.get('bilinear', True)
        self.base_channels = config.get('base_channels', 64)
        
        # Build U-Net architecture
        self._build_unet()
        
        # Feature extraction
        self.feature_maps = {}
        self._register_hooks()
    
    def _build_unet(self):
        """Build U-Net architecture."""
        self.inc = DoubleConv(self.in_channels, self.base_channels)
        self.down1 = Down(self.base_channels, self.base_channels * 2)
        self.down2 = Down(self.base_channels * 2, self.base_channels * 4)
        self.down3 = Down(self.base_channels * 4, self.base_channels * 8)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(self.base_channels * 8, self.base_channels * 16 // factor)
        
        self.up1 = Up(self.base_channels * 16, self.base_channels * 8 // factor, self.bilinear)
        self.up2 = Up(self.base_channels * 8, self.base_channels * 4 // factor, self.bilinear)
        self.up3 = Up(self.base_channels * 4, self.base_channels * 2 // factor, self.bilinear)
        self.up4 = Up(self.base_channels * 2, self.base_channels, self.bilinear)
        self.outc = nn.Conv2d(self.base_channels, self.num_classes, kernel_size=1)
        
        # Global average pooling for classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.base_channels, self.num_classes)
    
    def _register_hooks(self):
        """Register forward hooks for feature extraction."""
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output
            return hook
        
        self.inc.register_forward_hook(get_activation('inc'))
        self.down1.register_forward_hook(get_activation('down1'))
        self.down2.register_forward_hook(get_activation('down2'))
        self.down3.register_forward_hook(get_activation('down3'))
        self.down4.register_forward_hook(get_activation('down4'))
        self.up1.register_forward_hook(get_activation('up1'))
        self.up2.register_forward_hook(get_activation('up2'))
        self.up3.register_forward_hook(get_activation('up3'))
        self.up4.register_forward_hook(get_activation('up4'))
    
    def forward(self, x: torch.Tensor, task='segmentation', **kwargs) -> torch.Tensor:
        """Forward pass."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        if task == 'segmentation':
            return self.outc(x)
        elif task == 'classification':
            # Use features before final upsampling for classification
            features = self.global_pool(x)
            features = features.view(features.size(0), -1)
            return self.classifier(features)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def get_features(self, x: torch.Tensor, layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Extract intermediate features."""
        self.feature_maps.clear()
        _ = self.forward(x)
        
        if layer_names is None:
            return self.feature_maps.copy()
        else:
            return {name: self.feature_maps[name] for name in layer_names if name in self.feature_maps}
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load pretrained U-Net student model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint.get('config', {})
        config.update(kwargs)
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_logits(self, x: torch.Tensor, task='classification') -> torch.Tensor:
        """Get logits for classification or segmentation."""
        return self.forward(x, task=task)