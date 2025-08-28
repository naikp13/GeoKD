#!/usr/bin/env python3
"""Test script for student networks."""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from students import ResNetStudent, UNetStudent, SwinStudent

def test_resnet_student():
    """Test ResNet student network."""
    print("Testing ResNet Student...")
    
    config = {
        'model_name': 'resnet_student',
        'num_classes': 10,
        'in_channels': 12,  # Multispectral
        'architecture': 'resnet18',
        'pretrained': False
    }
    
    model = ResNetStudent(config)
    
    # Test forward pass
    x = torch.randn(2, 12, 224, 224)
    logits = model(x)
    print(f"ResNet input shape: {x.shape}")
    print(f"ResNet output shape: {logits.shape}")
    
    # Test feature extraction
    features = model.get_features(x, ['conv1', 'layer1', 'layer4'])
    print(f"ResNet features: {list(features.keys())}")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    print(f"ResNet parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("ResNet Student test passed!\n")

def test_unet_student():
    """Test U-Net student network."""
    print("Testing U-Net Student...")
    
    config = {
        'model_name': 'unet_student',
        'num_classes': 5,  # Segmentation classes
        'in_channels': 12,
        'base_channels': 64,
        'bilinear': True
    }
    
    model = UNetStudent(config)
    
    # Test segmentation
    x = torch.randn(2, 12, 256, 256)
    seg_output = model(x, task='segmentation')
    print(f"U-Net input shape: {x.shape}")
    print(f"U-Net segmentation output shape: {seg_output.shape}")
    
    # Test classification
    cls_output = model(x, task='classification')
    print(f"U-Net classification output shape: {cls_output.shape}")
    
    # Test feature extraction
    features = model.get_features(x, ['inc', 'down1', 'down4'])
    print(f"U-Net features: {list(features.keys())}")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("U-Net Student test passed!\n")

def test_swin_student():
    """Test Swin Transformer student network."""
    print("Testing Swin Student...")
    
    config = {
        'model_name': 'swin_student',
        'img_size': 224,
        'patch_size': 4,
        'in_channels': 12,
        'num_classes': 1000,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7
    }
    
    model = SwinStudent(config)
    
    # Test forward pass
    x = torch.randn(2, 12, 224, 224)
    logits = model(x)
    print(f"Swin input shape: {x.shape}")
    print(f"Swin output shape: {logits.shape}")
    
    # Test feature extraction
    features = model.get_features(x, ['patch_embed', 'layer_0', 'layer_3'])
    print(f"Swin features: {list(features.keys())}")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    print(f"Swin parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Swin Student test passed!\n")

if __name__ == "__main__":
    print("Testing Student Networks...\n")
    
    test_resnet_student()
    test_unet_student()
    test_swin_student()
    
    print("All student network tests completed successfully!")