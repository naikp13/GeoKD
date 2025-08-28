import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import yaml
from .losses import DistillationLoss
from ..models.base_model import BaseGeospatialModel

class GeospatialDistiller:
    """Main knowledge distillation class for geospatial models."""
    
    def __init__(self, 
                 teacher_model: BaseGeospatialModel,
                 student_model: BaseGeospatialModel,
                 distillation_mode: str = 'hybrid',  # 'logit', 'feature', 'hybrid'
                 distillation_config: str = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.device = device
        self.distillation_mode = distillation_mode
        
        # Load configuration
        if distillation_config:
            with open(distillation_config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
            
        # Initialize appropriate distillation loss
        self._init_distillation_loss()
        
        # Initialize optimizer
        self.optimizer = self._init_optimizer()
        
        # Set teacher to eval mode
        self.teacher.eval()
    
    def _init_distillation_loss(self):
        """Initialize distillation loss based on mode."""
        if self.distillation_mode == 'logit':
            from .losses import LogitDistillation
            self.distillation_loss = LogitDistillation(
                temperature=self.config['loss']['temperature'],
                alpha=self.config['loss']['alpha']
            )
        elif self.distillation_mode == 'feature':
            from .losses import FeatureDistillation
            self.distillation_loss = FeatureDistillation(
                feature_loss_weight=self.config['loss'].get('feature_loss_weight', 1.0)
            )
        elif self.distillation_mode == 'hybrid':
            from .losses import HybridDistillation
            self.distillation_loss = HybridDistillation(
                temperature=self.config['loss']['temperature'],
                alpha=self.config['loss']['alpha'],
                beta=self.config['loss'].get('beta', 0.3)
            )
        else:
            # Fallback to original comprehensive loss
            from .losses import DistillationLoss
            self.distillation_loss = DistillationLoss(self.config['loss'])
    
    def distill_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced distillation step supporting different modes."""
        self.student.train()
        self.optimizer.zero_grad()
        
        inputs = batch['input'].to(self.device)
        targets = batch.get('target', None)
        if targets is not None:
            targets = targets.to(self.device)
        
        # Forward pass through both models
        with torch.no_grad():
            if self.distillation_mode in ['feature', 'hybrid']:
                teacher_outputs, teacher_features = self.teacher(inputs, return_features=True)
            else:
                teacher_outputs = self.teacher(inputs)
                teacher_features = {}
        
        if self.distillation_mode in ['feature', 'hybrid']:
            student_outputs, student_features = self.student(inputs, return_features=True)
        else:
            student_outputs = self.student(inputs)
            student_features = {}
        
        # Calculate loss based on distillation mode
        if self.distillation_mode == 'logit':
            losses = self.distillation_loss(student_outputs, teacher_outputs, targets)
        elif self.distillation_mode == 'feature':
            losses = self.distillation_loss(student_features, teacher_features, targets)
        elif self.distillation_mode == 'hybrid':
            losses = self.distillation_loss(student_outputs, teacher_outputs, 
                                          student_features, teacher_features, targets)
        else:
            # Original comprehensive loss
            losses = self.distillation_loss(student_outputs, teacher_outputs,
                                          student_features, teacher_features, targets)
        
        # Backward pass
        losses['total_loss'].backward()
        self.optimizer.step()
        
        # Convert to float for logging
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def _default_config(self) -> Dict[str, Any]:
        """Default distillation configuration."""
        return {
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
                'epochs': 100,
                'log_interval': 10
            }
        }
    
    def _init_optimizer(self) -> optim.Optimizer:
        """Initialize optimizer for student model."""
        opt_config = self.config['optimizer']
        
        if opt_config['name'] == 'AdamW':
            return optim.AdamW(
                self.student.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['name'] == 'SGD':
            return optim.SGD(
                self.student.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['name']}")
    
    def distill(self, 
                train_dataloader: DataLoader,
                val_dataloader: Optional[DataLoader] = None,
                epochs: Optional[int] = None) -> Dict[str, list]:
        """Main distillation training loop."""
        
        epochs = epochs or self.config['training']['epochs']
        log_interval = self.config['training']['log_interval']
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.student.train()
            epoch_train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_dataloader):
                loss_dict = self.distill_step(batch)
                epoch_train_loss += loss_dict['total_loss']
                
                if batch_idx % log_interval == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss_dict["total_loss"]:.4f}')
            
            avg_train_loss = epoch_train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                val_losses.append(val_loss)
                print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}')
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the student model."""
        self.student.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                targets = batch.get('target', None)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # Teacher and student forward passes
                teacher_outputs = self.teacher(inputs)
                teacher_features = self.teacher.get_features(inputs)
                student_outputs = self.student(inputs)
                student_features = self.student.get_features(inputs)
                
                # Calculate loss
                loss_dict = self.distillation_loss(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    student_features=student_features,
                    teacher_features=teacher_features,
                    targets=targets
                )
                
                total_loss += loss_dict['total_loss'].item()
        
        return total_loss / len(dataloader)
    
    def save_student(self, path: str):
        """Save the distilled student model."""
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'config': self.student.config,
            'distillation_config': self.config
        }, path)
        print(f"Student model saved to {path}")