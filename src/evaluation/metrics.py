import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class GeospatialMetrics:
    """Comprehensive metrics for geospatial tasks."""
    
    def __init__(self, task_type: str = 'classification', num_classes: int = None):
        """
        Args:
            task_type: Type of task ('classification', 'segmentation', 'regression')
            num_classes: Number of classes for classification/segmentation
        """
        self.task_type = task_type
        self.num_classes = num_classes
        
        if task_type == 'classification':
            self.metrics_calculator = ClassificationMetrics(num_classes)
        elif task_type == 'segmentation':
            self.metrics_calculator = SegmentationMetrics(num_classes)
        elif task_type == 'regression':
            self.metrics_calculator = RegressionMetrics()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate all relevant metrics for the task."""
        return self.metrics_calculator.calculate(predictions, targets)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new batch."""
        self.metrics_calculator.update(predictions, targets)
    
    def compute(self) -> Dict[str, float]:
        """Compute accumulated metrics."""
        return self.metrics_calculator.compute()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.metrics_calculator.reset()

class ClassificationMetrics:
    """Metrics for classification tasks."""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with new batch."""
        # Convert to numpy and flatten
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)
        
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        self.all_predictions.extend(pred_np)
        self.all_targets.extend(target_np)
    
    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics for single batch."""
        # Convert predictions to class indices if needed
        if predictions.dim() > 1 and predictions.shape[1] > 1:
            predictions = torch.argmax(predictions, dim=1)
        
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        # Calculate metrics
        accuracy = calculate_accuracy(pred_np, target_np)
        f1_macro = calculate_f1_score(pred_np, target_np, average='macro')
        f1_weighted = calculate_f1_score(pred_np, target_np, average='weighted')
        
        # Per-class metrics
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            target_np, pred_np, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': np.mean(precision),
            'recall_macro': np.mean(recall)
        }
        
        # Add per-class metrics if num_classes is reasonable
        if self.num_classes and self.num_classes <= 20:
            for i in range(min(len(f1_per_class), self.num_classes)):
                metrics[f'f1_class_{i}'] = f1_per_class[i]
        
        return metrics
    
    def compute(self) -> Dict[str, float]:
        """Compute accumulated metrics."""
        if not self.all_predictions:
            return {}
        
        pred_array = np.array(self.all_predictions)
        target_array = np.array(self.all_targets)
        
        return self.calculate(torch.from_numpy(pred_array), torch.from_numpy(target_array))

class SegmentationMetrics:
    """Metrics for segmentation tasks."""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.total_pixels = 0
        self.correct_pixels = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with new batch."""
        # Convert predictions to class indices if needed
        if predictions.dim() > 1 and predictions.shape[1] > 1:
            predictions = torch.argmax(predictions, dim=1)
        
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        # Update pixel accuracy
        self.correct_pixels += np.sum(pred_np == target_np)
        self.total_pixels += len(pred_np)
        
        # Update IoU components
        for cls in range(self.num_classes):
            pred_mask = (pred_np == cls)
            target_mask = (target_np == cls)
            
            intersection = np.sum(pred_mask & target_mask)
            union = np.sum(pred_mask | target_mask)
            
            self.intersection[cls] += intersection
            self.union[cls] += union
    
    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics for single batch."""
        # Convert predictions to class indices if needed
        if predictions.dim() > 1 and predictions.shape[1] > 1:
            predictions = torch.argmax(predictions, dim=1)
        
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        # Pixel accuracy
        pixel_accuracy = np.mean(pred_np == target_np)
        
        # IoU per class
        iou_per_class = []
        for cls in range(self.num_classes):
            pred_mask = (pred_np == cls)
            target_mask = (target_np == cls)
            
            intersection = np.sum(pred_mask & target_mask)
            union = np.sum(pred_mask | target_mask)
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0
            iou_per_class.append(iou)
        
        # Mean IoU
        mean_iou = np.mean(iou_per_class)
        
        metrics = {
            'pixel_accuracy': pixel_accuracy,
            'mean_iou': mean_iou,
            'iou_per_class': iou_per_class
        }
        
        # Add per-class IoU if reasonable number of classes
        if self.num_classes <= 20:
            for i, iou in enumerate(iou_per_class):
                metrics[f'iou_class_{i}'] = iou
        
        return metrics
    
    def compute(self) -> Dict[str, float]:
        """Compute accumulated metrics."""
        if self.total_pixels == 0:
            return {}
        
        # Pixel accuracy
        pixel_accuracy = self.correct_pixels / self.total_pixels
        
        # IoU per class
        iou_per_class = []
        for cls in range(self.num_classes):
            if self.union[cls] > 0:
                iou = self.intersection[cls] / self.union[cls]
            else:
                iou = 0.0
            iou_per_class.append(iou)
        
        # Mean IoU
        mean_iou = np.mean(iou_per_class)
        
        metrics = {
            'pixel_accuracy': pixel_accuracy,
            'mean_iou': mean_iou
        }
        
        # Add per-class IoU if reasonable number of classes
        if self.num_classes <= 20:
            for i, iou in enumerate(iou_per_class):
                metrics[f'iou_class_{i}'] = iou
        
        return metrics

class RegressionMetrics:
    """Metrics for regression tasks."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with new batch."""
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        self.all_predictions.extend(pred_np)
        self.all_targets.extend(target_np)
    
    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics for single batch."""
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # MSE and RMSE
        mse = np.mean((pred_np - target_np) ** 2)
        rmse = np.sqrt(mse)
        
        # MAE
        mae = np.mean(np.abs(pred_np - target_np))
        
        # PSNR (assuming pixel values in [0, 1] range)
        psnr = calculate_psnr(pred_np, target_np)
        
        # SSIM (for image-like data)
        ssim = calculate_ssim(pred_np, target_np)
        
        # R² score
        ss_res = np.sum((target_np - pred_np) ** 2)
        ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'psnr': psnr,
            'ssim': ssim,
            'r2': r2
        }
    
    def compute(self) -> Dict[str, float]:
        """Compute accumulated metrics."""
        if not self.all_predictions:
            return {}
        
        pred_array = np.array(self.all_predictions)
        target_array = np.array(self.all_targets)
        
        return self.calculate(torch.from_numpy(pred_array), torch.from_numpy(target_array))

# Utility functions
def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate accuracy."""
    return accuracy_score(targets, predictions)

def calculate_f1_score(predictions: np.ndarray, targets: np.ndarray, average: str = 'macro') -> float:
    """Calculate F1 score."""
    return f1_score(targets, predictions, average=average, zero_division=0)

def calculate_iou(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Calculate IoU for segmentation."""
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        
        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask | target_mask)
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        iou_per_class.append(iou)
    
    return {
        'mean_iou': np.mean(iou_per_class),
        'iou_per_class': iou_per_class
    }

def calculate_psnr(predictions: np.ndarray, targets: np.ndarray, max_val: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((predictions - targets) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def calculate_ssim(predictions: np.ndarray, targets: np.ndarray, 
                  window_size: int = 11, sigma: float = 1.5) -> float:
    """Calculate Structural Similarity Index."""
    try:
        from skimage.metrics import structural_similarity as ssim
        
        # Handle different array shapes
        if predictions.ndim == 1:
            # Assume square image
            size = int(np.sqrt(len(predictions)))
            if size * size == len(predictions):
                predictions = predictions.reshape(size, size)
                targets = targets.reshape(size, size)
            else:
                return 0.0
        
        # Calculate SSIM
        if predictions.ndim == 2:
            return ssim(targets, predictions, data_range=targets.max() - targets.min())
        elif predictions.ndim == 3:
            # Multi-channel image
            return ssim(targets, predictions, data_range=targets.max() - targets.min(), multichannel=True)
        else:
            return 0.0
            
    except ImportError:
        # Fallback simple correlation-based similarity
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Normalize
        pred_norm = (pred_flat - np.mean(pred_flat)) / (np.std(pred_flat) + 1e-8)
        target_norm = (target_flat - np.mean(target_flat)) / (np.std(target_flat) + 1e-8)
        
        # Correlation coefficient as similarity measure
        correlation = np.corrcoef(pred_norm, target_norm)[0, 1]
        return max(0.0, correlation)  # Ensure non-negative

def create_metrics_calculator(task_type: str, num_classes: Optional[int] = None) -> GeospatialMetrics:
    """Factory function to create metrics calculator."""
    return GeospatialMetrics(task_type=task_type, num_classes=num_classes)