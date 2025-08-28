#!/usr/bin/env python3
"""Training script for knowledge distillation on SEN12MS dataset."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import argparse
import yaml
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.sen12ms_dataset import create_sen12ms_dataloaders
from src.training.trainer import KnowledgeDistillationTrainer
from src.distillation.losses import DistillationLoss
from teachers import PrithviLoader, TerraMindLoader
from students import ResNetStudent, UNetStudent, SwinStudent

def parse_args():
    parser = argparse.ArgumentParser(description='Train knowledge distillation on SEN12MS')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to SEN12MS dataset')
    parser.add_argument('--teacher', type=str, choices=['prithvi', 'terramind'], default='prithvi')
    parser.add_argument('--student', type=str, choices=['resnet', 'unet', 'swin'], default='resnet')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_teacher_model(teacher_type: str, config: dict, device: str):
    """Create teacher model."""
    if teacher_type == 'prithvi':
        teacher = PrithviLoader(config['teacher'])
    elif teacher_type == 'terramind':
        teacher = TerraMindLoader(config['teacher'])
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")
    
    return teacher.to(device)

def create_student_model(student_type: str, config: dict, device: str):
    """Create student model."""
    student_config = config['student']
    
    if student_type == 'resnet':
        student = ResNetStudent(student_config)
    elif student_type == 'unet':
        student = UNetStudent(student_config)
    elif student_type == 'swin':
        student = SwinStudent(student_config)
    else:
        raise ValueError(f"Unknown student type: {student_type}")
    
    return student.to(device)

def create_optimizer(model: nn.Module, config: dict):
    """Create optimizer."""
    optimizer_config = config['optimizer']
    optimizer_type = optimizer_config.get('type', 'adamw')
    
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_config.get('lr', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 1e-4),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_config.get('lr', 1e-3),
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer

def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler."""
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type.lower() == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 100),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_type.lower() == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    else:
        scheduler = None
    
    return scheduler

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if 'training' not in config:
        config['training'] = {}
    
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['device'] = args.device
    
    if 'optimizer' not in config:
        config['optimizer'] = {}
    config['optimizer']['lr'] = args.lr
    
    print(f"Training configuration:")
    print(f"  Teacher: {args.teacher}")
    print(f"  Student: {args.student}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    
    # Create data loaders
    print("Creating data loaders...")
    dataloaders = create_sen12ms_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        target_size=(256, 256),
        modalities=['s1', 's2', 'lc']
    )
    
    print(f"Dataset sizes:")
    for split, loader in dataloaders.items():
        print(f"  {split}: {len(loader.dataset)} samples")
    
    # Create models
    print("Creating models...")
    teacher_model = create_teacher_model(args.teacher, config, args.device)
    student_model = create_student_model(args.student, config, args.device)
    
    print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # Create loss function
    distillation_loss = DistillationLoss(config.get('loss', {}))
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(student_model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create trainer
    trainer = KnowledgeDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        distillation_loss=distillation_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        save_dir=args.save_dir,
        log_wandb=args.wandb,
        project_name=f'geokd-{args.teacher}-to-{args.student}'
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=args.epochs,
        save_every=10,
        validate_every=1
    )

if __name__ == '__main__':
    main()