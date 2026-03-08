#!/usr/bin/env python3
"""Test CNN model on training set to check in-distribution performance."""

import os
import sys
import numpy as np
import torch
import yaml

sys.path.append("./")

from cnn_pretrain import CTReconstructionDataset, create_model


def test_train_set(model_path, config_path):
    """Test model on training data."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Testing on training set...")
    print(f"Data path: {config['data_path']}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create training dataset
    train_dataset = CTReconstructionDataset(
        data_path=config['data_path'],
        split='train',
        target_proj_size=tuple(config['proj_size']),
        target_vol_size=tuple(config['vol_size']),
        max_projections=config['max_projections'],
        dataset_type=config['dataset_type']
    )
    
    # Create model
    model = create_model(type('Args', (), {
        'model_size': config['model_size'],
        'use_dropout': config['use_dropout'],
        'vol_size': config['vol_size']
    })())
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Get training data
    projections, volume_gt = train_dataset[0]
    projections = projections.to(device)
    volume_gt = volume_gt.to(device)
    
    print(f"\nTraining set shapes:")
    print(f"  Projections: {projections.shape}")
    print(f"  Ground truth: {volume_gt.shape}")
    
    # Add channel dimension
    projections_with_channel = projections.unsqueeze(1)
    print(f"  Projections with channel: {projections_with_channel.shape}")
    
    # Make prediction
    with torch.no_grad():
        volume_pred = model(projections_with_channel)
    
    print(f"  Prediction: {volume_pred.shape}")
    
    # Compute metrics
    mse = torch.mean((volume_pred - volume_gt.unsqueeze(1)) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    mae = torch.mean(torch.abs(volume_pred - volume_gt.unsqueeze(1))).item()
    
    print(f"\nMetrics on training set:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MAE: {mae:.6f}")
    print(f"  MSE: {mse.item():.6f}")
    
    # Also test on test set for comparison
    print("\n" + "="*50)
    print("Testing on test set for comparison...")
    
    test_dataset = CTReconstructionDataset(
        data_path=config['data_path'],
        split='test',
        target_proj_size=tuple(config['proj_size']),
        target_vol_size=tuple(config['vol_size']),
        max_projections=config['max_projections'],
        dataset_type=config['dataset_type']
    )
    
    projections_test, volume_gt_test = test_dataset[0]
    projections_test = projections_test.to(device)
    volume_gt_test = volume_gt_test.to(device)
    
    print(f"\nTest set shapes:")
    print(f"  Projections: {projections_test.shape}")
    print(f"  Ground truth: {volume_gt_test.shape}")
    
    # Add channel dimension
    projections_test_with_channel = projections_test.unsqueeze(1)
    
    with torch.no_grad():
        volume_pred_test = model(projections_test_with_channel)
    
    mse_test = torch.mean((volume_pred_test - volume_gt_test.unsqueeze(1)) ** 2)
    psnr_test = 20 * torch.log10(1.0 / torch.sqrt(mse_test)).item()
    mae_test = torch.mean(torch.abs(volume_pred_test - volume_gt_test.unsqueeze(1))).item()
    
    print(f"\nMetrics on test set:")
    print(f"  PSNR: {psnr_test:.2f} dB")
    print(f"  MAE: {mae_test:.6f}")
    print(f"  MSE: {mse_test.item():.6f}")
    
    print(f"\nPerformance drop from train to test:")
    print(f"  PSNR drop: {psnr - psnr_test:.2f} dB")
    print(f"  MAE increase: {mae_test - mae:.6f}")
    print(f"  Relative MAE increase: {(mae_test - mae)/mae*100:.1f}%")
    
    return {
        'train': {'psnr': psnr, 'mae': mae, 'mse': mse.item()},
        'test': {'psnr': psnr_test, 'mae': mae_test, 'mse': mse_test.item()}
    }


if __name__ == "__main__":
    model_path = "./cnn_output_final/best_model.pth"
    config_path = "./cnn_output_final/config.yaml"
    
    results = test_train_set(model_path, config_path)
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"Training PSNR: {results['train']['psnr']:.2f} dB")
    print(f"Test PSNR: {results['test']['psnr']:.2f} dB")
    
    if results['train']['psnr'] > 20 and results['test']['psnr'] < 15:
        print("\nDiagnosis: Model overfits to training data.")
    elif results['train']['psnr'] < 15 and results['test']['psnr'] < 15:
        print("\nDiagnosis: Model underfits - needs more training or capacity.")
    else:
        print("\nDiagnosis: Model generalizes reasonably.")