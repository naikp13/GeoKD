import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import wandb
from tqdm import tqdm
import os
from pathlib import Path
import json
import time
from collections import defaultdict

class KnowledgeDistillationTrainer:
    """Trainer for knowledge distillation on geospatial data."""
    
    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 distillation_loss: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 save_dir: str = './checkpoints',
                 log_wandb: bool = True,
                 project_name: str = 'geokd'):
        
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.distillation_loss = distillation_loss.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_wandb = log_wandb
        if log_wandb:
            wandb.init(project=project_name)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.student_model.train()
        
        epoch_losses = defaultdict(list)
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            loss_dict = self._forward_step(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'kd': f"{loss_dict['kd_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to wandb
            if self.log_wandb and batch_idx % 100 == 0:
                log_dict = {f'train/{k}': v.item() for k, v in loss_dict.items()}
                log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                log_dict['epoch'] = epoch
                log_dict['global_step'] = self.global_step
                wandb.log(log_dict)
            
            self.global_step += 1
        
        # Calculate epoch averages
        epoch_avg = {key: sum(values) / len(values) for key, values in epoch_losses.items()}
        
        return epoch_avg
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.student_model.eval()
        
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                batch = self._move_to_device(batch)
                loss_dict = self._forward_step(batch)
                
                for key, value in loss_dict.items():
                    val_losses[key].append(value.item())
        
        # Calculate averages
        val_avg = {key: sum(values) / len(values) for key, values in val_losses.items()}
        
        return val_avg
    
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward step for both teacher and student."""
        # Prepare inputs (combine modalities if needed)
        inputs = self._prepare_inputs(batch)
        targets = batch.get('lc', None)  # Land cover labels
        
        # Teacher forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
            teacher_features = self.teacher_model.get_features(inputs) if hasattr(self.teacher_model, 'get_features') else {}
        
        # Student forward pass
        student_outputs = self.student_model(inputs)
        student_features = self.student_model.get_features(inputs) if hasattr(self.student_model, 'get_features') else {}
        
        # Calculate distillation loss
        loss_dict = self.distillation_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            student_features=student_features,
            teacher_features=teacher_features,
            targets=targets
        )
        
        return loss_dict
    
    def _prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare inputs by combining modalities."""
        inputs = []
        
        # Add modalities in order
        if 's1' in batch:
            inputs.append(batch['s1'])
        if 's2' in batch:
            inputs.append(batch['s2'])
        
        # Concatenate along channel dimension
        if inputs:
            return torch.cat(inputs, dim=1)
        else:
            raise ValueError("No input modalities found in batch")
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value 
                for key, value in batch.items()}
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'student_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int,
              save_every: int = 5,
              validate_every: int = 1):
        """Main training loop."""
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(self.epoch, num_epochs):
            start_time = time.time()
            
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            if epoch % validate_every == 0:
                val_losses = self.validate(val_loader)
                
                # Check if best model
                val_loss = val_losses['total_loss']
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                # Log validation results
                if self.log_wandb:
                    log_dict = {f'val/{k}': v for k, v in val_losses.items()}
                    log_dict.update({f'train_avg/{k}': v for k, v in train_losses.items()})
                    log_dict['epoch'] = epoch
                    wandb.log(log_dict)
                
                print(f"Epoch {epoch}: Train Loss: {train_losses['total_loss']:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f}s")
                
                # Save checkpoint
                if epoch % save_every == 0 or is_best:
                    self.save_checkpoint(epoch, val_loss, is_best)
            
            self.epoch = epoch + 1
        
        print("Training completed!")
        
        if self.log_wandb:
            wandb.finish()