#!/usr/bin/env python3
"""
Knowledge Distillation Utilities for CT Reconstruction.

Provides distillation losses and utilities for transferring knowledge
from pre-trained CNN (teacher) to 3D Gaussian model (student).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class VolumeDistillationLoss(nn.Module):
    """
    Volume-level distillation loss between teacher CNN and student Gaussian model.
    
    Measures the difference between:
    1. Teacher's predicted volume (from CNN)
    2. Student's rendered volume (from 3D Gaussians)
    """
    
    def __init__(self, temperature=2.0, alpha=0.7, use_kl=True, use_l1=True, use_ssim=False):
        """
        Args:
            temperature: Temperature for KL divergence softening
            alpha: Weighting factor between distillation and original loss
            use_kl: Use KL divergence loss
            use_l1: Use L1 loss
            use_ssim: Use SSIM loss (computationally expensive for 3D)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.use_kl = use_kl
        self.use_l1 = use_l1
        self.use_ssim = use_ssim
        
        if use_ssim:
            # 3D SSIM is expensive, we'll use 2D slices approximation
            self.ssim_loss = SSIM3DApproximation()
    
    def forward(self, teacher_volume: torch.Tensor, student_volume: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss between teacher and student volumes.
        
        Args:
            teacher_volume: Volume from teacher CNN [batch, 1, D, H, W] or [D, H, W]
            student_volume: Volume from student Gaussians [batch, 1, D, H, W] or [D, H, W]
        
        Returns:
            Dictionary of loss components and total loss
        """
        # Ensure same shape
        if teacher_volume.dim() == 3:
            teacher_volume = teacher_volume.unsqueeze(0).unsqueeze(0)
        if student_volume.dim() == 3:
            student_volume = student_volume.unsqueeze(0).unsqueeze(0)
        
        losses = {}
        
        # KL divergence loss (soft labels)
        if self.use_kl:
            kl_loss = self._kl_divergence_loss(teacher_volume, student_volume)
            losses['kl'] = kl_loss
        
        # L1 loss
        if self.use_l1:
            l1_loss = F.l1_loss(student_volume, teacher_volume)
            losses['l1'] = l1_loss
        
        # SSIM loss (approximation)
        if self.use_ssim:
            ssim_loss = self.ssim_loss(student_volume, teacher_volume)
            losses['ssim'] = ssim_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _kl_divergence_loss(self, teacher_vol: torch.Tensor, student_vol: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between teacher and student volume distributions."""
        # Soften distributions with temperature
        teacher_soft = F.softmax(teacher_vol.view(teacher_vol.size(0), -1) / self.temperature, dim=-1)
        student_soft = F.log_softmax(student_vol.view(student_vol.size(0), -1) / self.temperature, dim=-1)
        
        # KL divergence
        kl_div = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Scale by temperature squared (as in Hinton et al.)
        kl_div = kl_div * (self.temperature ** 2)
        
        return kl_div


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation loss.
    
    For CNN models with intermediate features, we can match feature statistics
    between teacher and student. This requires access to CNN intermediate layers.
    """
    
    def __init__(self, feature_matching_layers=None, weights=None):
        """
        Args:
            feature_matching_layers: List of layer indices/names to match features
            weights: Weight for each feature layer
        """
        super().__init__()
        self.feature_matching_layers = feature_matching_layers or [0, 1, 2, 3]
        self.weights = weights or [1.0, 0.8, 0.6, 0.4]
    
    def forward(self, teacher_features: Dict[str, torch.Tensor], 
                student_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            teacher_features: Dictionary of teacher feature maps {layer_name: feature}
            student_features: Dictionary of student feature maps {layer_name: feature}
        
        Returns:
            Feature matching loss
        """
        total_loss = 0.0
        
        for i, layer_key in enumerate(self.feature_matching_layers):
            if layer_key in teacher_features and layer_key in student_features:
                t_feat = teacher_features[layer_key]
                s_feat = student_features[layer_key]
                
                # Match feature statistics (mean and variance)
                t_mean = t_feat.mean(dim=[2, 3, 4])
                t_var = t_feat.var(dim=[2, 3, 4])
                s_mean = s_feat.mean(dim=[2, 3, 4])
                s_var = s_feat.var(dim=[2, 3, 4])
                
                mean_loss = F.mse_loss(s_mean, t_mean)
                var_loss = F.mse_loss(s_var, t_var)
                
                layer_loss = mean_loss + var_loss
                total_loss += layer_loss * self.weights[i]
        
        return total_loss


class ProjectionConsistencyLoss(nn.Module):
    """
    Projection consistency loss.
    
    Ensures that student's rendered projections match teacher's input projections
    (which should be similar to ground truth).
    """
    
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, student_projections: torch.Tensor, 
                target_projections: torch.Tensor) -> torch.Tensor:
        """
        Compute projection consistency loss.
        
        Args:
            student_projections: Projections rendered from student Gaussians
            target_projections: Target projections (from teacher or ground truth)
        
        Returns:
            Projection consistency loss
        """
        # Simple L1 loss on projections
        loss = F.l1_loss(student_projections, target_projections)
        return loss * self.weight


class SSIM3DApproximation(nn.Module):
    """
    Approximate 3D SSIM loss by computing SSIM on 2D slices and averaging.
    
    Note: True 3D SSIM would be computationally expensive.
    """
    
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self._create_gaussian_window()
    
    def _create_gaussian_window(self):
        """Create 2D Gaussian window for SSIM computation."""
        from scipy.signal import convolve2d
        from scipy.ndimage import gaussian_filter
        
        # Create 1D Gaussian
        gauss_1d = gaussian_filter(np.ones(self.window_size), self.sigma)
        # Create 2D Gaussian by outer product
        gauss_2d = np.outer(gauss_1d, gauss_1d)
        # Normalize
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        self.register_buffer('gaussian_window', torch.from_numpy(gauss_2d).float())
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute approximate 3D SSIM by averaging 2D SSIM over slices.
        
        Args:
            x: First volume [batch, 1, D, H, W] or [D, H, W]
            y: Second volume [batch, 1, D, H, W] or [D, H, W]
        
        Returns:
            Approximate SSIM loss (1 - SSIM)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)
            y = y.unsqueeze(0).unsqueeze(0)
        
        batch_size, channels, depth, height, width = x.shape
        device = x.device
        
        # Move window to device
        window = self.gaussian_window.to(device).unsqueeze(0).unsqueeze(0)
        
        ssim_values = []
        for d in range(depth):
            # Extract slice
            x_slice = x[:, :, d, :, :]  # [batch, 1, H, W]
            y_slice = y[:, :, d, :, :]  # [batch, 1, H, W]
            
            # Compute SSIM for this slice
            mu_x = F.conv2d(x_slice, window, padding=self.window_size//2)
            mu_y = F.conv2d(y_slice, window, padding=self.window_size//2)
            
            mu_x_sq = mu_x ** 2
            mu_y_sq = mu_y ** 2
            mu_x_mu_y = mu_x * mu_y
            
            sigma_x_sq = F.conv2d(x_slice ** 2, window, padding=self.window_size//2) - mu_x_sq
            sigma_y_sq = F.conv2d(y_slice ** 2, window, padding=self.window_size//2) - mu_y_sq
            sigma_xy = F.conv2d(x_slice * y_slice, window, padding=self.window_size//2) - mu_x_mu_y
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim_map = ((2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)) / \
                       ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
            
            ssim_values.append(ssim_map.mean())
        
        # Average over depth
        avg_ssim = torch.stack(ssim_values).mean()
        
        # Return 1 - SSIM as loss (to minimize)
        return 1.0 - avg_ssim


class ProgressiveDistillation:
    """
    Progressive distillation scheduler.
    
    Gradually increases distillation weight during training to avoid
    overwhelming the student with teacher's knowledge early on.
    """
    
    def __init__(self, total_iterations: int, warmup_ratio: float = 0.3,
                 max_distill_weight: float = 0.5, schedule: str = 'linear'):
        """
        Args:
            total_iterations: Total training iterations
            warmup_ratio: Ratio of iterations for warmup (no distillation)
            max_distill_weight: Maximum distillation weight
            schedule: 'linear', 'cosine', or 'step'
        """
        self.total_iterations = total_iterations
        self.warmup_iterations = int(total_iterations * warmup_ratio)
        self.max_weight = max_distill_weight
        self.schedule = schedule
        
    def get_weight(self, iteration: int) -> float:
        """
        Get distillation weight for current iteration.
        
        Args:
            iteration: Current training iteration
        
        Returns:
            Distillation weight (0 to max_weight)
        """
        if iteration < self.warmup_iterations:
            return 0.0
        
        progress = (iteration - self.warmup_iterations) / (self.total_iterations - self.warmup_iterations)
        progress = max(0.0, min(1.0, progress))
        
        if self.schedule == 'linear':
            return self.max_weight * progress
        elif self.schedule == 'cosine':
            return self.max_weight * (1 - np.cos(np.pi * progress)) / 2
        elif self.schedule == 'step':
            # Step increase at 50% and 75%
            if progress < 0.5:
                return self.max_weight * 0.3
            elif progress < 0.75:
                return self.max_weight * 0.6
            else:
                return self.max_weight
        else:
            return self.max_weight * progress


def create_distillation_loss(config: Dict) -> nn.Module:
    """
    Create distillation loss based on configuration.
    
    Args:
        config: Dictionary with distillation configuration
    
    Returns:
        Distillation loss module
    """
    loss_type = config.get('type', 'volume')
    
    if loss_type == 'volume':
        return VolumeDistillationLoss(
            temperature=config.get('temperature', 2.0),
            alpha=config.get('alpha', 0.7),
            use_kl=config.get('use_kl', True),
            use_l1=config.get('use_l1', True),
            use_ssim=config.get('use_ssim', False)
        )
    elif loss_type == 'feature':
        return FeatureDistillationLoss(
            feature_matching_layers=config.get('feature_layers', [0, 1, 2, 3]),
            weights=config.get('feature_weights', [1.0, 0.8, 0.6, 0.4])
        )
    else:
        raise ValueError(f"Unknown distillation loss type: {loss_type}")


# Example usage:
if __name__ == "__main__":
    # Test the distillation loss
    batch_size = 1
    depth, height, width = 64, 64, 64
    
    # Create dummy volumes
    teacher_vol = torch.randn(batch_size, 1, depth, height, width)
    student_vol = torch.randn(batch_size, 1, depth, height, width)
    
    # Create loss
    dist_loss = VolumeDistillationLoss(
        temperature=2.0,
        alpha=0.7,
        use_kl=True,
        use_l1=True,
        use_ssim=False
    )
    
    # Compute loss
    losses = dist_loss(teacher_vol, student_vol)
    
    print("Distillation losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    # Test progressive distillation
    prog_distill = ProgressiveDistillation(
        total_iterations=10000,
        warmup_ratio=0.3,
        max_distill_weight=0.5,
        schedule='linear'
    )
    
    print("\nProgressive distillation weights:")
    for iter in [0, 1000, 3000, 5000, 7000, 10000]:
        weight = prog_distill.get_weight(iter)
        print(f"  Iter {iter:5d}: weight = {weight:.3f}")