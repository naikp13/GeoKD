#!/usr/bin/env python3
"""
Knowledge Distillation Example for GeoKD framework.

This script demonstrates:
1. Setting up teacher-student pairs
2. Configuring distillation parameters
3. Running knowledge distillation
4. Evaluating distilled models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import PrithviModel, TerraMindModel
from distillation import GeospatialDistiller

def create_synthetic_dataset(num_samples=1000, batch_size=16):
    """Create synthetic satellite imagery dataset for demonstration."""
    
    # Generate synthetic satellite imagery
    # Simulating multi-spectral data (12 bands) at 224x224 resolution
    images = torch.randn(num_samples, 12, 224, 224)
    
    # Generate synthetic labels (e.g., land use classes)
    labels = torch.randint(0, 10, (num_samples,))  # 10 land use classes
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def create_distillation_config():
    """Create distillation configuration."""
    
    config = {
        'loss': {
            'temperature': 4.0,
            'alpha': 0.7,
            'feature_loss_weight': 0.3
        },
        'optimizer': {
            'name': 'AdamW',
            'lr': 1e-4,
            'weight_decay': 1e-5
        },
        'training': {
            'epochs': 5,  # Reduced for demo
            'log_interval': 10
        }
    }
    
    return config

def example_1_terramind_to_prithvi():
    """Example 1: TerraMind (teacher) -> Prithvi (student)"""
    
    print("\n=== Example 1: TerraMind -> Prithvi Distillation ===")
    
    # Load models
    teacher = TerraMindModel.from_pretrained(
        config={'modalities': ['S2L2A'], 'merge_method': 'mean'}
    )
    student = PrithviModel.from_pretrained()
    
    print(f"Teacher (TerraMind): {teacher.get_model_info()['parameters']:,} parameters")
    print(f"Student (Prithvi): {student.get_model_info()['parameters']:,} parameters")
    
    # Create synthetic dataset
    train_loader = create_synthetic_dataset(num_samples=200, batch_size=8)
    val_loader = create_synthetic_dataset(num_samples=50, batch_size=8)
    
    # Initialize distiller
    distiller = GeospatialDistiller(
        teacher_model=teacher,
        student_model=student
    )
    
    # Prepare data format for distillation
    def format_batch(batch):
        images, labels = batch
        return {'input': images, 'target': labels}
    
    # Convert dataloaders
    train_data = [format_batch(batch) for batch in train_loader]
    val_data = [format_batch(batch) for batch in val_loader]
    
    # Run distillation
    print("Starting distillation...")
    history = distiller.distill(
        train_dataloader=train_data,
        val_dataloader=val_data,
        epochs=3
    )
    
    print(f"Training complete. Final train loss: {history['train_losses'][-1]:.4f}")
    if history['val_losses']:
        print(f"Final validation loss: {history['val_losses'][-1]:.4f}")
    
    # Save distilled model
    distiller.save_student('models/prithvi_distilled_from_terramind.pth')
    
    return distiller

def example_2_prithvi_to_terramind():
    """Example 2: Prithvi (teacher) -> TerraMind (student)"""
    
    print("\n=== Example 2: Prithvi -> TerraMind Distillation ===")
    
    # Load models
    teacher = PrithviModel.from_pretrained()
    student = TerraMindModel.from_pretrained(
        config={'modalities': ['S2L2A'], 'merge_method': 'mean'}
    )
    
    print(f"Teacher (Prithvi): {teacher.get_model_info()['parameters']:,} parameters")
    print(f"Student (TerraMind): {student.get_model_info()['parameters']:,} parameters")
    
    # Create dataset
    train_loader = create_synthetic_dataset(num_samples=200, batch_size=8)
    
    # Custom configuration for this scenario
    config = create_distillation_config()
    config['loss']['alpha'] = 0.6  # Different alpha for this direction
    
    # Initialize distiller with custom config
    distiller = GeospatialDistiller(
        teacher_model=teacher,
        student_model=student
    )
    distiller.config = config
    
    # Format data
    train_data = [{'input': batch[0], 'target': batch[1]} for batch in train_loader]
    
    # Run distillation
    print("Starting distillation...")
    history = distiller.distill(
        train_dataloader=train_data,
        epochs=3
    )
    
    print(f"Training complete. Final train loss: {history['train_losses'][-1]:.4f}")
    
    # Save distilled model
    distiller.save_student('models/terramind_distilled_from_prithvi.pth')
    
    return distiller

def evaluate_distilled_models():
    """Evaluate and compare original vs distilled models."""
    
    print("\n=== Model Evaluation ===")
    
    # Create test dataset
    test_loader = create_synthetic_dataset(num_samples=100, batch_size=16)
    
    # Load original models
    prithvi_original = PrithviModel.from_pretrained()
    terramind_original = TerraMindModel.from_pretrained(
        config={'modalities': ['S2L2A'], 'merge_method': 'mean'}
    )
    
    # Set to eval mode
    prithvi_original.eval()
    terramind_original.eval()
    
    # Evaluate on test data
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= 3:  # Just test a few batches
                break
                
            # Original model outputs
            prithvi_out = prithvi_original(images)
            terramind_out = terramind_original(images)
            
            print(f"Batch {i+1}:")
            print(f"  Prithvi output shape: {prithvi_out.shape}")
            print(f"  TerraMind output shape: {terramind_out.shape}")
            print(f"  Prithvi output mean: {prithvi_out.mean().item():.4f}")
            print(f"  TerraMind output mean: {terramind_out.mean().item():.4f}")

def main():
    """Main function to run all distillation examples."""
    
    print("=== GeoKD Knowledge Distillation Examples ===")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    try:
        # Example 1: TerraMind -> Prithvi
        distiller1 = example_1_terramind_to_prithvi()
        
        # Example 2: Prithvi -> TerraMind
        distiller2 = example_2_prithvi_to_terramind()
        
        # Evaluation
        evaluate_distilled_models()
        
        print("\n=== All Examples Complete ===")
        print("Distilled models saved in 'models/' directory")
        
    except Exception as e:
        print(f"Error during distillation: {e}")
        print("This is expected in demo mode without actual model weights")

if __name__ == "__main__":
    main()