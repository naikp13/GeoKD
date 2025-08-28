import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.base_model import BaseGeospatialModel

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        
        return x

class BasicLayer(nn.Module):
    """A basic Swin Transformer layer."""
    
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop)
            for i in range(depth)])
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class SwinStudent(BaseGeospatialModel):
    """Swin Transformer based student network."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.img_size = config.get('img_size', 224)
        self.patch_size = config.get('patch_size', 4)
        self.in_chans = config.get('in_channels', 3)
        self.num_classes = config.get('num_classes', 1000)
        self.embed_dim = config.get('embed_dim', 96)
        self.depths = config.get('depths', [2, 2, 6, 2])
        self.num_heads = config.get('num_heads', [3, 6, 12, 24])
        self.window_size = config.get('window_size', 7)
        self.mlp_ratio = config.get('mlp_ratio', 4.)
        self.qkv_bias = config.get('qkv_bias', True)
        self.drop_rate = config.get('drop_rate', 0.)
        self.attn_drop_rate = config.get('attn_drop_rate', 0.)
        
        # Build architecture
        self._build_swin()
        
        # Feature extraction
        self.feature_maps = {}
        self._register_hooks()
    
    def _build_swin(self):
        """Build Swin Transformer architecture."""
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, 
            in_chans=self.in_chans, embed_dim=self.embed_dim)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(len(self.depths)):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate
            )
            self.layers.append(layer)
        
        # Final norm and classifier
        self.norm = nn.LayerNorm(int(self.embed_dim * 2 ** (len(self.depths) - 1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(int(self.embed_dim * 2 ** (len(self.depths) - 1)), self.num_classes)
    
    def _register_hooks(self):
        """Register forward hooks for feature extraction."""
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output
            return hook
        
        self.patch_embed.register_forward_hook(get_activation('patch_embed'))
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(get_activation(f'layer_{i}'))
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
        # Patch embedding
        x = self.patch_embed(x)
        
        # Through layers
        for layer in self.layers:
            x = layer(x)
        
        # Final processing
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x
    
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
        """Load pretrained Swin student model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint.get('config', {})
        config.update(kwargs)
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get classification logits."""
        return self.forward(x)