import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import math

class DistillationLoss(nn.Module):
    """Enhanced knowledge distillation loss for geospatial models."""
    
    def __init__(self, loss_config: Dict[str, Any]):
        super().__init__()
        self.temperature = loss_config.get('temperature', 4.0)
        self.alpha = loss_config.get('alpha', 0.7)
        self.feature_loss_weight = loss_config.get('feature_loss_weight', 0.3)
        self.attention_loss_weight = loss_config.get('attention_loss_weight', 0.1)
        self.relation_loss_weight = loss_config.get('relation_loss_weight', 0.1)
        self.spatial_loss_weight = loss_config.get('spatial_loss_weight', 0.1)
        
        # Loss functions
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Advanced loss components
        self.attention_transfer = AttentionTransferLoss()
        self.relation_loss = RelationDistillationLoss()
        self.spatial_loss = SpatialAttentionLoss()
        
    def forward(self, 
                student_outputs: torch.Tensor,
                teacher_outputs: torch.Tensor,
                student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor],
                targets: Optional[torch.Tensor] = None,
                student_attention: Optional[Dict[str, torch.Tensor]] = None,
                teacher_attention: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Calculate comprehensive distillation loss."""
        
        losses = {}
        device = student_outputs.device
        
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
        task_loss = torch.tensor(0.0, device=device)
        if targets is not None:
            if targets.dtype == torch.long:  # Classification
                task_loss = self.ce_loss(student_outputs, targets)
            else:  # Regression
                task_loss = self.mse_loss(student_outputs, targets)
        
        losses['task_loss'] = task_loss
        
        # 4. Attention Transfer Loss
        attention_loss = torch.tensor(0.0, device=device)
        if student_attention is not None and teacher_attention is not None:
            attention_loss = self._calculate_attention_loss(student_attention, teacher_attention)
        losses['attention_loss'] = attention_loss
        
        # 5. Relation Distillation Loss
        relation_loss = self.relation_loss(student_outputs, teacher_outputs)
        losses['relation_loss'] = relation_loss
        
        # 6. Spatial Attention Loss
        spatial_loss = torch.tensor(0.0, device=device)
        if student_features and teacher_features:
            spatial_loss = self._calculate_spatial_loss(student_features, teacher_features)
        losses['spatial_loss'] = spatial_loss
        
        # 7. Total Loss
        total_loss = (
            self.alpha * kd_loss + 
            (1 - self.alpha) * task_loss + 
            self.feature_loss_weight * feature_loss +
            self.attention_loss_weight * attention_loss +
            self.relation_loss_weight * relation_loss +
            self.spatial_loss_weight * spatial_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _calculate_feature_loss(self, 
                               student_features: Dict[str, torch.Tensor],
                               teacher_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced feature matching loss with multi-scale adaptation."""
        
        feature_loss = torch.tensor(0.0)
        count = 0
        
        # Match features by name
        for name in student_features:
            if name in teacher_features:
                s_feat = student_features[name]
                t_feat = teacher_features[name]
                
                # Multi-scale feature matching
                if s_feat.shape != t_feat.shape:
                    s_feat, t_feat = self._adapt_features(s_feat, t_feat)
                    if s_feat is None or t_feat is None:
                        continue
                
                # Calculate MSE loss between features
                feat_loss = self.mse_loss(s_feat, t_feat.detach())
                
                # Add L1 regularization for sparsity
                l1_loss = F.l1_loss(s_feat, t_feat.detach())
                feat_loss = feat_loss + 0.1 * l1_loss
                
                feature_loss += feat_loss
                count += 1
        
        return feature_loss / max(count, 1)
    
    def _adapt_features(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor):
        """Adapt features to match dimensions."""
        if len(student_feat.shape) == 4:  # Conv features [B, C, H, W]
            # Spatial adaptation
            if student_feat.shape[2:] != teacher_feat.shape[2:]:
                student_feat = F.interpolate(
                    student_feat, size=teacher_feat.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            # Channel adaptation
            if student_feat.shape[1] != teacher_feat.shape[1]:
                # Use 1x1 conv for channel adaptation
                adapter = nn.Conv2d(
                    student_feat.shape[1], teacher_feat.shape[1], 1
                ).to(student_feat.device)
                student_feat = adapter(student_feat)
                
        elif len(student_feat.shape) == 3:  # Sequence features [B, L, C]
            # Sequence length adaptation
            if student_feat.shape[1] != teacher_feat.shape[1]:
                student_feat = F.interpolate(
                    student_feat.transpose(1, 2), 
                    size=teacher_feat.shape[1], 
                    mode='linear'
                ).transpose(1, 2)
            
            # Feature dimension adaptation
            if student_feat.shape[2] != teacher_feat.shape[2]:
                adapter = nn.Linear(
                    student_feat.shape[2], teacher_feat.shape[2]
                ).to(student_feat.device)
                student_feat = adapter(student_feat)
        
        return student_feat, teacher_feat
    
    def _calculate_attention_loss(self, 
                                 student_attention: Dict[str, torch.Tensor],
                                 teacher_attention: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate attention transfer loss."""
        attention_loss = torch.tensor(0.0)
        count = 0
        
        for name in student_attention:
            if name in teacher_attention:
                s_att = student_attention[name]
                t_att = teacher_attention[name]
                
                att_loss = self.attention_transfer(s_att, t_att)
                attention_loss += att_loss
                count += 1
        
        return attention_loss / max(count, 1)
    
    def _calculate_spatial_loss(self, 
                               student_features: Dict[str, torch.Tensor],
                               teacher_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate spatial attention loss."""
        spatial_loss = torch.tensor(0.0)
        count = 0
        
        for name in student_features:
            if name in teacher_features:
                s_feat = student_features[name]
                t_feat = teacher_features[name]
                
                if len(s_feat.shape) == 4:  # Conv features
                    s_spatial = self.spatial_loss(s_feat)
                    t_spatial = self.spatial_loss(t_feat)
                    
                    if s_spatial.shape != t_spatial.shape:
                        s_spatial = F.interpolate(
                            s_spatial, size=t_spatial.shape[2:], 
                            mode='bilinear', align_corners=False
                        )
                    
                    loss = F.mse_loss(s_spatial, t_spatial.detach())
                    spatial_loss += loss
                    count += 1
        
        return spatial_loss / max(count, 1)

class AttentionTransferLoss(nn.Module):
    """Enhanced attention transfer loss for transformer-based models."""
    
    def __init__(self, beta: float = 1000.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, student_attention: torch.Tensor, teacher_attention: torch.Tensor) -> torch.Tensor:
        """Calculate attention transfer loss with normalization."""
        # Ensure same dimensions
        if student_attention.shape != teacher_attention.shape:
            if len(student_attention.shape) == 4:  # [B, H, L, L]
                teacher_attention = F.interpolate(
                    teacher_attention, size=student_attention.shape[2:], 
                    mode='bilinear', align_corners=False
                )
        
        # Normalize attention maps
        student_att = self._normalize_attention(student_attention)
        teacher_att = self._normalize_attention(teacher_attention)
        
        # Calculate attention transfer loss
        return self.beta * F.mse_loss(student_att, teacher_att.detach())
    
    def _normalize_attention(self, attention: torch.Tensor) -> torch.Tensor:
        """Normalize attention maps."""
        # Sum over last dimension (attention weights should sum to 1)
        attention = F.softmax(attention, dim=-1)
        
        # Flatten spatial dimensions
        batch_size = attention.size(0)
        attention = attention.view(batch_size, -1)
        
        # L2 normalize
        return F.normalize(attention, p=2, dim=1)

class RelationDistillationLoss(nn.Module):
    """Relation-based knowledge distillation loss."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor) -> torch.Tensor:
        """Calculate relation distillation loss."""
        # Calculate pairwise distances
        student_relations = self._compute_relations(student_outputs)
        teacher_relations = self._compute_relations(teacher_outputs)
        
        return F.mse_loss(student_relations, teacher_relations.detach())
    
    def _compute_relations(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute pairwise relations between samples."""
        # Normalize outputs
        outputs_norm = F.normalize(outputs, p=2, dim=1)
        
        # Compute pairwise cosine similarities
        relations = torch.mm(outputs_norm, outputs_norm.t())
        
        return relations

class SpatialAttentionLoss(nn.Module):
    """Spatial attention transfer loss for convolutional features."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        """Generate spatial attention maps from feature maps."""
        # Sum across channel dimension to get spatial attention
        spatial_attention = torch.sum(feature_maps.abs(), dim=1, keepdim=True)
        
        # Normalize
        batch_size, _, height, width = spatial_attention.shape
        spatial_attention = spatial_attention.view(batch_size, -1)
        spatial_attention = F.softmax(spatial_attention, dim=1)
        spatial_attention = spatial_attention.view(batch_size, 1, height, width)
        
        return spatial_attention

class FocalDistillationLoss(nn.Module):
    """Focal loss for knowledge distillation to handle hard examples."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor) -> torch.Tensor:
        """Calculate focal distillation loss."""
        # Soften predictions
        student_probs = F.softmax(student_outputs / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        
        # Calculate KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            teacher_probs,
            reduction='none'
        ).sum(dim=1)
        
        # Apply focal weighting
        pt = torch.exp(-kl_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * kl_loss
        
        return focal_loss.mean() * (self.temperature ** 2)

class MultiScaleDistillationLoss(nn.Module):
    """Multi-scale knowledge distillation for different resolution features."""
    
    def __init__(self, scales: List[int] = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        """Calculate multi-scale distillation loss."""
        total_loss = torch.tensor(0.0, device=student_features.device)
        
        for scale in self.scales:
            # Downsample features
            if scale > 1:
                s_feat = F.avg_pool2d(student_features, kernel_size=scale, stride=scale)
                t_feat = F.avg_pool2d(teacher_features, kernel_size=scale, stride=scale)
            else:
                s_feat = student_features
                t_feat = teacher_features
            
            # Ensure same dimensions
            if s_feat.shape != t_feat.shape:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
            
            # Calculate loss at this scale
            scale_loss = self.mse_loss(s_feat, t_feat.detach())
            total_loss += scale_loss
        
        return total_loss / len(self.scales)


class LogitDistillation(nn.Module):
    """Pure Logit Distillation (Soft Label Distillation).
    
    Uses only the teacher's final output probabilities to guide the student.
    This is the classic knowledge distillation approach from Hinton et al.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation vs task loss
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Calculate logit distillation loss.
        
        Args:
            student_logits: Raw outputs from student model
            teacher_logits: Raw outputs from teacher model  
            targets: Ground truth labels (optional)
            
        Returns:
            Dictionary containing loss components
        """
        losses = {}
        device = student_logits.device
        
        # Soft target distillation loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        losses['distillation_loss'] = distillation_loss
        
        # Task-specific loss (hard targets)
        task_loss = torch.tensor(0.0, device=device)
        if targets is not None:
            if targets.dtype == torch.long:  # Classification
                task_loss = self.ce_loss(student_logits, targets)
            else:  # Regression
                task_loss = self.mse_loss(student_logits, targets)
        losses['task_loss'] = task_loss
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
        losses['total_loss'] = total_loss
        
        return losses


class FeatureDistillation(nn.Module):
    """Feature Distillation using intermediate representations.
    
    Student mimics the teacher's intermediate feature maps rather than
    just the final outputs. Useful for transferring spatial and semantic
    knowledge from deeper layers.
    """
    
    def __init__(self, 
                 feature_loss_weight: float = 1.0,
                 adaptation_layers: Optional[Dict[str, nn.Module]] = None):
        super().__init__()
        self.feature_loss_weight = feature_loss_weight
        self.mse_loss = nn.MSELoss()
        self.adaptation_layers = adaptation_layers or {}
        
    def forward(self,
                student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor],
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Calculate feature distillation loss.
        
        Args:
            student_features: Dictionary of intermediate features from student
            teacher_features: Dictionary of intermediate features from teacher
            targets: Ground truth (optional, for task loss)
            
        Returns:
            Dictionary containing loss components
        """
        losses = {}
        device = next(iter(student_features.values())).device
        
        # Feature matching loss
        feature_loss = torch.tensor(0.0, device=device)
        matched_layers = 0
        
        for layer_name in student_features.keys():
            if layer_name in teacher_features:
                student_feat = student_features[layer_name]
                teacher_feat = teacher_features[layer_name]
                
                # Adapt feature dimensions if needed
                if layer_name in self.adaptation_layers:
                    student_feat = self.adaptation_layers[layer_name](student_feat)
                elif student_feat.shape != teacher_feat.shape:
                    student_feat = self._adapt_feature_dimensions(student_feat, teacher_feat)
                
                # Calculate MSE loss between features
                layer_loss = self.mse_loss(student_feat, teacher_feat.detach())
                feature_loss += layer_loss
                matched_layers += 1
        
        if matched_layers > 0:
            feature_loss = feature_loss / matched_layers
        
        losses['feature_loss'] = feature_loss
        losses['total_loss'] = self.feature_loss_weight * feature_loss
        
        return losses
    
    def _adapt_feature_dimensions(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """Adapt student features to match teacher dimensions."""
        if student_feat.shape == teacher_feat.shape:
            return student_feat
            
        # Handle channel dimension mismatch
        if student_feat.shape[1] != teacher_feat.shape[1]:
            # Use 1x1 conv to match channels
            if not hasattr(self, f'_adapt_conv_{student_feat.shape[1]}_{teacher_feat.shape[1]}'):
                conv = nn.Conv2d(student_feat.shape[1], teacher_feat.shape[1], 1, bias=False)
                conv = conv.to(student_feat.device)
                setattr(self, f'_adapt_conv_{student_feat.shape[1]}_{teacher_feat.shape[1]}', conv)
            
            conv = getattr(self, f'_adapt_conv_{student_feat.shape[1]}_{teacher_feat.shape[1]}')
            student_feat = conv(student_feat)
        
        # Handle spatial dimension mismatch
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            student_feat = F.interpolate(student_feat, size=teacher_feat.shape[2:], 
                                       mode='bilinear', align_corners=False)
        
        return student_feat


class HybridDistillation(nn.Module):
    """Hybrid Distillation combining logit and feature distillation.
    
    This is the most comprehensive approach, combining both final output
    distillation and intermediate feature matching. Most effective in practice.
    """
    
    def __init__(self,
                 temperature: float = 4.0,
                 alpha: float = 0.7,  # Weight for distillation vs task loss
                 beta: float = 0.3,   # Weight for feature vs logit distillation
                 adaptation_layers: Optional[Dict[str, nn.Module]] = None):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # Initialize component distillation modules
        self.logit_distiller = LogitDistillation(temperature, alpha=1.0)  # Pure logit loss
        self.feature_distiller = FeatureDistillation(feature_loss_weight=1.0, 
                                                    adaptation_layers=adaptation_layers)
        
        # Task loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor],
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Calculate hybrid distillation loss.
        
        Args:
            student_logits: Final outputs from student
            teacher_logits: Final outputs from teacher
            student_features: Intermediate features from student
            teacher_features: Intermediate features from teacher
            targets: Ground truth labels (optional)
            
        Returns:
            Dictionary containing all loss components
        """
        losses = {}
        device = student_logits.device
        
        # 1. Logit distillation loss
        logit_losses = self.logit_distiller(student_logits, teacher_logits, targets=None)
        logit_distill_loss = logit_losses['distillation_loss']
        losses['logit_distillation'] = logit_distill_loss
        
        # 2. Feature distillation loss
        feature_losses = self.feature_distiller(student_features, teacher_features)
        feature_distill_loss = feature_losses['feature_loss']
        losses['feature_distillation'] = feature_distill_loss
        
        # 3. Task-specific loss
        task_loss = torch.tensor(0.0, device=device)
        if targets is not None:
            if targets.dtype == torch.long:  # Classification
                task_loss = self.ce_loss(student_logits, targets)
            else:  # Regression
                task_loss = self.mse_loss(student_logits, targets)
        losses['task_loss'] = task_loss
        
        # 4. Combined hybrid loss
        # Combine logit and feature distillation
        distillation_loss = (1 - self.beta) * logit_distill_loss + self.beta * feature_distill_loss
        
        # Combine distillation with task loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
        
        losses['distillation_loss'] = distillation_loss
        losses['total_loss'] = total_loss
        
        return losses