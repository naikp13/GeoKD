#!/usr/bin/env python3
"""
Quick test script for teacher model loaders.
"""

import torch
from teachers import PrithviLoader, TerraMindLoader

def test_teachers():
    print("🧪 Testing Teacher Model Loaders\n")
    
    # Test Prithvi
    print("1️⃣ Testing Prithvi Loader...")
    prithvi = PrithviLoader()
    
    # Create dummy satellite data (12 channels, 224x224)
    dummy_data = torch.randn(2, 12, 224, 224)
    
    # Test forward pass
    features = prithvi.forward(dummy_data)
    print(f"   Features shape: {features.shape}")
    
    # Test feature extraction
    layer_features = prithvi.get_features(dummy_data)
    print(f"   Extracted {len(layer_features)} layer features")
    
    # Test logits
    logits = prithvi.get_logits(dummy_data, num_classes=10)
    print(f"   Logits shape: {logits.shape}")
    
    print(f"   Model info: {prithvi.get_model_info()}\n")
    
    # Test TerraMind
    print("2️⃣ Testing TerraMind Loader...")
    terramind = TerraMindLoader()
    
    # Test forward pass
    features = terramind.forward(dummy_data)
    print(f"   Features shape: {features.shape}")
    
    # Test feature extraction
    layer_features = terramind.get_features(dummy_data)
    print(f"   Extracted {len(layer_features)} layer features")
    
    # Test logits
    logits = terramind.get_logits(dummy_data, num_classes=10)
    print(f"   Logits shape: {logits.shape}")
    
    print(f"   Model info: {terramind.get_model_info()}")
    
    print("\n✅ All teacher loaders working correctly!")

if __name__ == "__main__":
    test_teachers()