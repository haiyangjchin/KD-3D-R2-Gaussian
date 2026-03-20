#!/usr/bin/env python3
"""
Analyze CNN predictions for CT reconstruction.
Loads trained CNN model and evaluates on test data.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import yaml

# Add project root to path
sys.path.append("./")

from cnn_pretrain import CTReconstructionDataset, create_model


def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR between prediction and target."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def compute_ssim_3d(pred, target, window_size=7, sigma=1.5, max_val=1.0):
    """Compute 3D SSIM approximation."""
    # Simple implementation: compute SSIM on each slice and average
    # This is an approximation since true 3D SSIM would be more complex
    
    # Apply Gaussian smoothing to both volumes
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # For 3D SSIM approximation, we'll compute 2D SSIM on each slice
    # and average across slices
    from scipy.signal import convolve2d
    from scipy.ndimage import gaussian_filter
    
    def create_window(window_size, sigma):
        gaussian = np.outer(
            gaussian_filter(np.ones(window_size), sigma),
            gaussian_filter(np.ones(window_size), sigma)
        )
        return gaussian / gaussian.sum()
    
    window = create_window(window_size, sigma)
    
    ssim_values = []
    for depth in range(pred_np.shape[0]):
        img1 = pred_np[depth]
        img2 = target_np[depth]
        
        mu1 = convolve2d(img1, window, mode='valid')
        mu2 = convolve2d(img2, window, mode='valid')
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = convolve2d(img1 ** 2, window, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(img2 ** 2, window, mode='valid') - mu2_sq
        sigma12 = convolve2d(img1 * img2, window, mode='valid') - mu1_mu2
        
        C1 = (0.01 * max_val) ** 2
        C2 = (0.03 * max_val) ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_values.append(ssim_map.mean())
    
    return np.mean(ssim_values)


def compute_mae(pred, target):
    """Compute Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()


def visualize_volume_slices(vol_pred, vol_gt, save_path=None):
    """Visualize slices from predicted and ground truth volumes."""
    # Handle different tensor dimensions
    if vol_gt.dim() == 4:  # [batch, depth, height, width]
        vol_gt = vol_gt.unsqueeze(1)  # Add channel dimension
    if vol_pred.dim() == 4:
        vol_pred = vol_pred.unsqueeze(1)
    
    # Select middle slices
    depth_slice = vol_pred.shape[2] // 2
    height_slice = vol_pred.shape[3] // 2
    width_slice = vol_pred.shape[4] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # XY plane (constant depth)
    axes[0, 0].imshow(vol_gt[0, 0, depth_slice, :, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('GT: XY Plane')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(vol_pred[0, 0, depth_slice, :, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Pred: XY Plane')
    axes[1, 0].axis('off')
    
    # XZ plane (constant height)
    axes[0, 1].imshow(vol_gt[0, 0, :, height_slice, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('GT: XZ Plane')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(vol_pred[0, 0, :, height_slice, :].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Pred: XZ Plane')
    axes[1, 1].axis('off')
    
    # YZ plane (constant width)
    axes[0, 2].imshow(vol_gt[0, 0, :, :, width_slice].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('GT: YZ Plane')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(vol_pred[0, 0, :, :, width_slice].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Pred: YZ Plane')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def visualize_error_map(vol_pred, vol_gt, save_path=None):
    """Visualize error maps between prediction and ground truth."""
    # Handle different tensor dimensions
    if vol_gt.dim() == 4:  # [batch, depth, height, width]
        vol_gt = vol_gt.unsqueeze(1)  # Add channel dimension
    if vol_pred.dim() == 4:
        vol_pred = vol_pred.unsqueeze(1)
    
    error = torch.abs(vol_pred - vol_gt)
    
    # Select middle slices
    depth_slice = vol_pred.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Error on XY plane
    im = axes[0].imshow(error[0, 0, depth_slice, :, :].cpu().numpy(), cmap='hot', vmin=0, vmax=0.5)
    axes[0].set_title('Absolute Error: XY Plane')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Histogram of errors
    error_flat = error.cpu().numpy().flatten()
    axes[1].hist(error_flat, bins=50, log=True)
    axes[1].set_title('Error Distribution (log scale)')
    axes[1].set_xlabel('Absolute Error')
    axes[1].set_ylabel('Frequency')
    
    # Cumulative distribution
    sorted_errors = np.sort(error_flat)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[2].plot(sorted_errors, cdf)
    axes[2].set_title('Error CDF')
    axes[2].set_xlabel('Absolute Error')
    axes[2].set_ylabel('CDF')
    axes[2].grid(True, alpha=0.3)
    
    # Add quantile markers
    for q in [0.5, 0.75, 0.9, 0.95]:
        idx = int(q * len(sorted_errors))
        axes[2].axvline(sorted_errors[idx], color='r', linestyle='--', alpha=0.5)
        axes[2].text(sorted_errors[idx], 0.5, f'{q*100:.0f}%', 
                    rotation=90, verticalalignment='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error visualization to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def analyze_predictions(model_path, config_path, output_dir='./cnn_analysis'):
    """Analyze predictions from trained CNN model."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create dataset
    print("\nCreating test dataset...")
    test_dataset = CTReconstructionDataset(
        data_path=config['data_path'],
        split='test',
        target_proj_size=tuple(config['proj_size']),
        target_vol_size=tuple(config['vol_size']),
        max_projections=config['max_projections'],
        dataset_type=config['dataset_type']
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(type('Args', (), {
        'model_size': config['model_size'],
        'use_dropout': config['use_dropout'],
        'vol_size': config['vol_size']
    })())
    
    # Load model weights
    print(f"\nLoading model from {model_path}")
    if model_path.endswith('.pth'):
        # Check if it's a checkpoint or just model weights
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            # Full checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown'):.6f}")
        else:
            # Just model weights
            model.load_state_dict(checkpoint)
            print("Loaded model weights")
    else:
        raise ValueError(f"Unknown model file format: {model_path}")
    
    model = model.to(device)
    model.eval()
    
    # Get test data
    projections, volume_gt = test_dataset[0]
    projections = projections.to(device)
    volume_gt = volume_gt.to(device)
    
    print(f"\nInput projections shape: {projections.shape}")
    print(f"Ground truth volume shape: {volume_gt.shape}")
    
    # Make prediction
    print("\nMaking prediction...")
    # Add channel dimension: [batch, depth, H, W] -> [batch, channels, depth, H, W]
    # Model expects input shape: [batch, in_channels, depth, height, width]
    projections_with_channel = projections.unsqueeze(1)  # Add channel dimension
    print(f"Projections with channel shape: {projections_with_channel.shape}")
    
    with torch.no_grad():
        volume_pred = model(projections_with_channel)
    
    print(f"Predicted volume shape: {volume_pred.shape}")
    
    # Compute metrics
    print("\nComputing metrics...")
    psnr = compute_psnr(volume_pred, volume_gt)
    mae = compute_mae(volume_pred, volume_gt)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"MAE: {mae:.6f}")
    
    # Try to compute SSIM (might be slow for 3D)
    try:
        ssim = compute_ssim_3d(volume_pred, volume_gt)
        print(f"SSIM (approx): {ssim:.4f}")
    except Exception as e:
        print(f"SSIM computation failed: {e}")
        ssim = None
    
    # Save metrics
    metrics = {
        'psnr': float(psnr),
        'mae': float(mae),
        'ssim': float(ssim) if ssim is not None else None
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Volume slices
    slices_path = os.path.join(output_dir, 'volume_slices.png')
    visualize_volume_slices(volume_pred, volume_gt, slices_path)
    
    # Error analysis
    error_path = os.path.join(output_dir, 'error_analysis.png')
    visualize_error_map(volume_pred, volume_gt, error_path)
    
    # Save predictions for further analysis
    print("\nSaving predictions for further analysis...")
    pred_np = volume_pred.cpu().numpy()
    gt_np = volume_gt.cpu().numpy()
    
    np.save(os.path.join(output_dir, 'predicted_volume.npy'), pred_np)
    np.save(os.path.join(output_dir, 'ground_truth_volume.npy'), gt_np)
    
    # Create a summary report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("CNN Model Prediction Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("Input/Output Shapes:\n")
        f.write(f"  Projections: {tuple(projections.shape)}\n")
        f.write(f"  Ground Truth: {tuple(volume_gt.shape)}\n")
        f.write(f"  Prediction: {tuple(volume_pred.shape)}\n\n")
        
        f.write("Evaluation Metrics:\n")
        f.write(f"  PSNR: {psnr:.2f} dB\n")
        f.write(f"  MAE: {mae:.6f}\n")
        if ssim is not None:
            f.write(f"  SSIM: {ssim:.4f}\n\n")
        
        f.write("\nInterpretation:\n")
        f.write("- PSNR > 30 dB: Excellent reconstruction\n")
        f.write("- PSNR 20-30 dB: Good reconstruction\n")
        f.write("- PSNR < 20 dB: Poor reconstruction\n")
        
        if psnr > 30:
            f.write("\nThe model shows excellent reconstruction quality.\n")
        elif psnr > 20:
            f.write("\nThe model shows good reconstruction quality.\n")
        else:
            f.write("\nThe reconstruction quality needs improvement.\n")
    
    print(f"Saved analysis report to {report_path}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze CNN predictions")
    parser.add_argument('--model_path', type=str, default='./cnn_output_final/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default='./cnn_output_final/config.yaml',
                       help='Path to training config file')
    parser.add_argument('--output_dir', type=str, default='./cnn_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    print("CNN Prediction Analysis")
    print("=" * 50)
    
    metrics = analyze_predictions(args.model_path, args.config_path, args.output_dir)
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")