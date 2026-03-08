#!/usr/bin/env python3
"""
Evaluate CNN model and compute PSNR, MAE, SSIM metrics.
"""

import os
import sys
import torch
import numpy as np
import yaml
import argparse
from pathlib import Path

# Add project root to path
sys.path.append("./")

# Import model and dataset
from cnn_pretrain import CTReconstructionDataset, create_model
from analyze_cnn_predictions import compute_psnr, compute_mae, compute_ssim_3d


def evaluate_model(model_path, config_path, device='cuda'):
    """Evaluate CNN model on test dataset."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(argparse.Namespace(**config)).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Create test dataset
    test_dataset = CTReconstructionDataset(
        data_path=config['data_path'],
        split='test',
        target_proj_size=tuple(config['proj_size']),
        target_vol_size=tuple(config['vol_size']),
        max_projections=config['max_projections'],
        dataset_type=config['dataset_type']
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Evaluate on test set
    psnr_values = []
    mae_values = []
    ssim_values = []
    
    print(f"Evaluating model on {len(test_loader)} test samples...")
    
    with torch.no_grad():
        for batch_idx, (projections, volumes) in enumerate(test_loader):
            projections = projections.to(device)  # [1, num_angles, H, W]
            volumes = volumes.to(device)         # [1, D, H, W]
            
            # Forward pass
            pred_volumes = model(projections)  # [1, 1, D, H, W]
            
            # Remove channel dimension if present
            if pred_volumes.dim() == 5:
                pred_volumes = pred_volumes.squeeze(1)  # [1, D, H, W]
            
            # Compute metrics
            psnr = compute_psnr(pred_volumes, volumes)
            mae = compute_mae(pred_volumes, volumes)
            ssim = compute_ssim_3d(pred_volumes, volumes)
            
            psnr_values.append(psnr)
            mae_values.append(mae)
            ssim_values.append(ssim)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} samples")
    
    # Compute average metrics
    avg_psnr = np.mean(psnr_values)
    avg_mae = np.mean(mae_values)
    avg_ssim = np.mean(ssim_values)
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Test samples: {len(test_loader)}")
    print(f"PSNR: {avg_psnr:.4f} dB")
    print(f"MAE: {avg_mae:.6f}")
    print(f"SSIM: {avg_ssim:.6f}")
    print("="*60)
    
    # Save results to YAML file
    output_dir = Path(model_path).parent
    metrics_path = output_dir / "metrics.yaml"
    
    metrics = {
        'psnr': float(avg_psnr),
        'mae': float(avg_mae),
        'ssim': float(avg_ssim)
    }
    
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    print(f"Metrics saved to: {metrics_path}")
    
    # Also save detailed report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("CNN Model Evaluation Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Test samples: {len(test_loader)}\n\n")
        f.write("Evaluation Metrics:\n")
        f.write(f"  PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"  MAE: {avg_mae:.6f}\n")
        f.write(f"  SSIM: {avg_ssim:.6f}\n\n")
        f.write("Interpretation:\n")
        f.write("- PSNR > 30 dB: Excellent reconstruction\n")
        f.write("- PSNR 20-30 dB: Good reconstruction\n")
        f.write("- PSNR < 20 dB: Poor reconstruction\n\n")
        
        if avg_psnr > 30:
            f.write("The model shows excellent reconstruction quality.\n")
        elif avg_psnr >= 20:
            f.write("The model shows good reconstruction quality.\n")
        else:
            f.write("The reconstruction quality needs improvement.\n")
    
    print(f"Detailed report saved to: {report_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained CNN model (.pth file)")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to model configuration (.yaml file)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    evaluate_model(args.model_path, args.config_path, args.device)


if __name__ == "__main__":
    main()