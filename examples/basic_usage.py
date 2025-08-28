#!/usr/bin/env python3
"""
Basic usage examples for GeoKD framework.

This script demonstrates how to:
1. Load Prithvi 2.0 and TerraMind models
2. Process satellite imagery
3. Extract features
4. Perform basic inference
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import PrithviModel, TerraMindModel

def main():
    print("=== GeoKD Basic Usage Examples ===")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Example 1: Load Prithvi 2.0
    print("\n1. Loading Prithvi 2.0 Model")
    prithvi = PrithviModel.from_pretrained()
    prithvi = prithvi.to(device)
    
    info = prithvi.get_model_info()
    print(f"Model: {info['name']}")
    print(f"Parameters: {info['parameters']:,}")
    
    # Example 2: Load TerraMind
    print("\n2. Loading TerraMind Model")
    terramind = TerraMindModel.from_pretrained(
        config={
            'modalities': ['S2L2A', 'S1GRD'],
            'merge_method': 'mean'
        }
    )
    terramind = terramind.to(device)
    
    info = terramind.get_model_info()
    print(f"Model: {info['name']}")
    print(f"Parameters: {info['parameters']:,}")
    
    # Example 3: Process single modality (Prithvi)
    print("\n3. Single Modality Processing (Prithvi)")
    
    # Simulate Landsat/Sentinel-2 imagery (batch_size=2, channels=12, height=224, width=224)
    satellite_data = torch.randn(2, 12, 224, 224).to(device)
    
    with torch.no_grad():
        prithvi_output = prithvi(satellite_data)
        print(f"Prithvi output shape: {prithvi_output.shape}")
        
        # Extract features
        prithvi_features = prithvi.get_features(satellite_data)
        print(f"Prithvi features: {list(prithvi_features.keys())}")
    
    # Example 4: Multi-modal processing (TerraMind)
    print("\n4. Multi-modal Processing (TerraMind)")
    
    # Multi-modal input
    multi_modal_data = {
        'S2L2A': torch.randn(2, 12, 224, 224).to(device),  # Sentinel-2
        'S1GRD': torch.randn(2, 2, 224, 224).to(device),   # Sentinel-1
    }
    
    with torch.no_grad():
        terramind_output = terramind(multi_modal_data)
        print(f"TerraMind output shape: {terramind_output.shape}")
        
        # Extract features
        terramind_features = terramind.get_features(multi_modal_data)
        print(f"TerraMind features: {list(terramind_features.keys())}")
    
    # Example 5: Generation with TerraMind
    print("\n5. Any-to-Any Generation (TerraMind)")
    
    input_modalities = {
        'S2L2A': torch.randn(1, 12, 224, 224).to(device)
    }
    
    with torch.no_grad():
        generated = terramind.generate(
            input_modalities=input_modalities,
            output_modalities=['S1GRD', 'LULC'],
            timesteps=5  # Reduced for demo
        )
        
        print("Generated modalities:")
        for modality, data in generated.items():
            print(f"  {modality}: {data.shape}")
    
    # Example 6: Feature comparison
    print("\n6. Feature Dimension Comparison")
    
    # Use same input for both models (single modality for fair comparison)
    test_input = torch.randn(1, 12, 224, 224).to(device)
    
    with torch.no_grad():
        prithvi_feat = prithvi.get_features(test_input)
        terramind_feat = terramind.get_features(test_input)
        
        print("Feature dimensions:")
        print(f"  Prithvi output: {prithvi_feat['output'].shape}")
        print(f"  TerraMind output: {terramind_feat['patch_embeddings'].shape}")
    
    print("\n=== Basic Usage Examples Complete ===")

if __name__ == "__main__":
    main()