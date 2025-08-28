import torch
from src.distillation.distiller import GeospatialDistiller
from teachers.prithvi_loader import PrithviTeacher
from students.resnet_student import ResNetStudent

# Load models
teacher = PrithviTeacher('configs/prithvi_config.yaml')
student = ResNetStudent(num_classes=10)

# Example 1: Pure Logit Distillation
logit_distiller = GeospatialDistiller(
    teacher_model=teacher,
    student_model=student,
    distillation_mode='logit'  # Only uses final outputs
)

# Example 2: Pure Feature Distillation  
feature_distiller = GeospatialDistiller(
    teacher_model=teacher,
    student_model=student,
    distillation_mode='feature'  # Only uses intermediate features
)

# Example 3: Hybrid Distillation (Recommended)
hybrid_distiller = GeospatialDistiller(
    teacher_model=teacher,
    student_model=student,
    distillation_mode='hybrid'  # Combines both logit and feature distillation
)

# Training example
for batch in dataloader:
    if distillation_mode == 'logit':
        losses = logit_distiller.distill_step(batch)
        print(f"Logit Loss: {losses['distillation_loss']:.4f}, Task Loss: {losses['task_loss']:.4f}")
    
    elif distillation_mode == 'feature':
        losses = feature_distiller.distill_step(batch)
        print(f"Feature Loss: {losses['feature_loss']:.4f}")
    
    elif distillation_mode == 'hybrid':
        losses = hybrid_distiller.distill_step(batch)
        print(f"Logit: {losses['logit_distillation']:.4f}, "
              f"Feature: {losses['feature_distillation']:.4f}, "
              f"Total: {losses['total_loss']:.4f}")