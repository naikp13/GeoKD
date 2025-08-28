import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class DistillationLoss(nn.Module):
    """Knowledge distillation loss for geospatial models."""
    
    def __init__(self, loss_config: Dict[str, Any]):
        super().__init__()
        self.temperature = loss_config.get('temperature', 4.0)
        self.alpha = loss_config.get('alpha', 0.7)
        self.feature_loss_weight = loss_config.get('feature_loss_weight', 0.3)
        
        # Loss functions
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, 
                student_outputs: torch.Tensor,
                teacher_outputs: torch.Tensor,
                student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor],
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Calculate distillation loss."""
        
        losses = {}
        
        # 1. Knowledge Distillation Loss (KL Divergence)
        student_logits = student_outputs / self.temperature
        teacher_logits = teacher_outputs / self.temperature
        
        kd_loss = self.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1)
        ) * (self.temperature ** 2)
        
        losses['kd_loss'] = kd_loss
        
        # 2. Feature Matching Loss
        feature_loss = self._calculate_feature_loss(student_features, teacher_features)
        losses['feature_loss'] = feature_loss
        
        # 3. Task-specific Loss (if targets available)
        task_loss = torch.tensor(0.0, device=student_outputs.device)
        if targets is not None:
            if targets.dtype == torch.long:  # Classification
                task_loss = self.ce_loss(student_outputs, targets)
            else:  # Regression
                task_loss = self.mse_loss(student_outputs, targets)
        
        losses['task_loss'] = task_loss
        
        # 4. Total Loss
        total_loss = (
            self.alpha * kd_loss + 
            (1 - self.alpha) * task_loss + 
            self.feature_loss_weight * feature_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _calculate_feature_loss(self, 
                               student_features: Dict[str, torch.Tensor],
                               teacher_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate feature matching loss between student and teacher."""
        
        feature_loss = torch.tensor(0.0)
        count = 0
        
        # Match features by name
        for name in student_features:
            if name in teacher_features:
                s_feat = student_features[name]
                t_feat = teacher_features[name]
                
                # Ensure same dimensions (add projection if needed)
                if s_feat.shape != t_feat.shape:
                    # Simple adaptation: average pooling or linear projection
                    if len(s_feat.shape) == 3:  # Sequence features
                        if s_feat.shape[-1] != t_feat.shape[-1]:
                            # Different feature dimensions - skip or project
                            continue
                        if s_feat.shape[1] != t_feat.shape[1]:
                            # Different sequence lengths - interpolate
                            s_feat = F.interpolate(
                                s_feat.transpose(1, 2), 
                                size=t_feat.shape[1], 
                                mode='linear'
                            ).transpose(1, 2)
                
                # Calculate MSE loss between features
                feat_loss = self.mse_loss(s_feat, t_feat.detach())
                feature_loss += feat_loss
                count += 1
        
        return feature_loss / max(count, 1)

class AttentionTransferLoss(nn.Module):
    """Attention transfer loss for transformer-based models."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, student_attention: torch.Tensor, teacher_attention: torch.Tensor) -> torch.Tensor:
        """Calculate attention transfer loss."""
        # Normalize attention maps
        student_att = F.normalize(student_attention.view(student_attention.size(0), -1), p=2, dim=1)
        teacher_att = F.normalize(teacher_attention.view(teacher_attention.size(0), -1), p=2, dim=1)
        
        # Calculate MSE loss
        return F.mse_loss(student_att, teacher_att)