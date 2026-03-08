#!/usr/bin/env python3
"""
CNN-based initialization for 3D Gaussians.
Alternative to FDK initialization.
"""

import os
import os.path as osp
import sys
import argparse
import numpy as np
import torch
import json

sys.path.append("./")

from r2_gaussian.dataset import Scene
from r2_gaussian.utils.general_utils import t2a
from models.ct_unet3d import CTUNet3D


def load_cnn_model(model_path, target_depth=64, model_size='small'):
    """Load trained CNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine feature sizes
    if model_size == 'small':
        features = [16, 32, 64, 128]
    elif model_size == 'medium':
        features = [32, 64, 128, 256]
    elif model_size == 'large':
        features = [64, 128, 256, 512]
    else:
        features = [32, 64, 128, 256]
    
    # Create model
    model = CTUNet3D(
        in_channels=1,
        out_channels=1,
        features=features,
        use_dropout=False,
        target_depth=target_depth
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device


def reconstruct_volume_with_cnn(model, device, projections, target_proj_size=(64, 64), target_vol_size=(64, 64, 64)):
    """
    Reconstruct 3D volume from projections using CNN.
    
    Args:
        model: Trained CNN model
        device: Torch device
        projections: Projection images [num_angles, height, width]
        target_proj_size: Target projection size (height, width)
        target_vol_size: Target volume size (depth, height, width)
    
    Returns:
        Reconstructed volume [depth, height, width]
    """
    original_height, original_width = projections.shape[1], projections.shape[2]
    target_height, target_width = target_proj_size
    
    # Resize projections if needed
    projs_tensor = torch.from_numpy(projections).float().unsqueeze(1)  # [N, 1, H, W]
    if (original_height, original_width) != (target_height, target_width):
        projs_resized = torch.nn.functional.interpolate(
            projs_tensor,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )
    else:
        projs_resized = projs_tensor
    
    # Prepare input: [batch, channels, depth, height, width]
    # projs_resized shape: [N, 1, H, W] where N = num_angles
    # We need: [1, 1, N, H, W]
    projs_input = projs_resized.unsqueeze(0)  # [1, N, 1, H, W]
    projs_input = projs_input.permute(0, 2, 1, 3, 4)  # [1, 1, N, H, W]
    projs_input = projs_input.to(device)
    
    # Forward pass
    with torch.no_grad():
        vol_pred = model(projs_input)
    
    # Remove batch and channel dimensions
    vol_pred = vol_pred.squeeze(0).squeeze(0)  # [D, H, W]
    
    return vol_pred.cpu().numpy()


def sample_points_from_volume(volume, scanner_cfg, n_points=50000, density_thresh=0.05, density_rescale=0.15):
    """
    Sample points from reconstructed volume.
    
    Args:
        volume: 3D volume [depth, height, width]
        scanner_cfg: Scanner configuration dictionary
        n_points: Number of points to sample
        density_thresh: Density threshold for valid voxels
        density_rescale: Rescale factor for densities
    
    Returns:
        positions: Sampled point positions [n_points, 3]
        densities: Sampled densities [n_points]
    """
    # Apply threshold
    density_mask = volume > density_thresh
    valid_indices = np.argwhere(density_mask)
    
    if len(valid_indices) < n_points:
        raise ValueError(f"Only {len(valid_indices)} valid voxels found, need at least {n_points}. "
                        f"Try lowering density_thresh (currently {density_thresh}).")
    
    # Random sample
    sampled_indices = valid_indices[
        np.random.choice(len(valid_indices), n_points, replace=False)
    ]
    
    # Convert indices to world coordinates
    offOrigin = np.array(scanner_cfg["offOrigin"])
    dVoxel = np.array(scanner_cfg["dVoxel"])
    sVoxel = np.array(scanner_cfg["sVoxel"])
    
    # Note: Volume indices are (depth, height, width) = (Z, Y, X)
    # Scanner coordinates assume (X, Y, Z) order
    # We need to reorder indices
    sampled_indices_xyz = sampled_indices[:, [2, 1, 0]]  # Convert ZYX to XYZ
    
    sampled_positions = sampled_indices_xyz * dVoxel - sVoxel / 2 + offOrigin
    
    # Extract densities
    sampled_densities = volume[
        sampled_indices[:, 0],  # Z
        sampled_indices[:, 1],  # Y
        sampled_indices[:, 2]   # X
    ]
    
    # Rescale densities
    sampled_densities = sampled_densities * density_rescale
    
    return sampled_positions, sampled_densities


def main():
    parser = argparse.ArgumentParser(description="CNN-based initialization for 3D Gaussians")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output .npy file for initialized points')
    parser.add_argument('--cnn_model', type=str, default='./cnn_output_final/best_model.pth',
                       help='Path to trained CNN model')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large'],
                       help='Size of CNN model')
    parser.add_argument('--n_points', type=int, default=50000,
                       help='Number of points to sample')
    parser.add_argument('--density_thresh', type=float, default=0.05,
                       help='Density threshold for valid voxels')
    parser.add_argument('--density_rescale', type=float, default=0.15,
                       help='Rescale factor for densities')
    parser.add_argument('--target_proj_size', type=int, nargs=2, default=[64, 64],
                       help='Target projection size (height, width)')
    parser.add_argument('--target_vol_size', type=int, nargs=3, default=[64, 64, 64],
                       help='Target volume size (depth, height, width)')
    parser.add_argument('--use_test_projections', action='store_true',
                       help='Use test projections instead of train projections')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CNN-based Initialization for 3D Gaussians")
    print("=" * 60)
    
    # Check inputs
    if not osp.exists(args.data):
        raise FileNotFoundError(f"Data path not found: {args.data}")
    
    if not osp.exists(args.cnn_model):
        raise FileNotFoundError(f"CNN model not found: {args.cnn_model}")
    
    # Load scene using CTReconstructionDataset from cnn_pretrain
    print(f"\nLoading scene from {args.data}")
    try:
        from cnn_pretrain import CTReconstructionDataset
        
        split = 'test' if args.use_test_projections else 'train'
        dataset = CTReconstructionDataset(
            data_path=args.data,
            split=split,
            target_proj_size=tuple(args.target_proj_size),
            target_vol_size=tuple(args.target_vol_size),
            max_projections=None,
            dataset_type='Blender'
        )
        
        # Get projections and ground truth volume
        projections_tensor, volume_gt_tensor = dataset[0]
        projections = projections_tensor.squeeze(0).numpy()  # [num_angles, H, W]
        
        # Get scanner config from dataset metadata
        scanner_cfg = dataset.scanner_cfg
        
        print(f"Loaded {projections.shape[0]} projections for {split} split")
        print(f"Projection shape: {projections.shape}")
        print(f"Scanner config: {scanner_cfg['nVoxel']} voxels")
        
    except Exception as e:
        print(f"Error loading with CTReconstructionDataset: {e}")
        print("Trying alternative loading...")
        
        # Fallback: try to load from saved files
        # This is dataset-specific and may need adjustment
        meta_path = osp.join(args.data, "meta_data.json")
        if osp.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            scanner_cfg = meta.get("scanner_cfg", {})
            
            # Load projections from numpy files if available
            # This is a simplified assumption
            proj_files = [f for f in os.listdir(args.data) if f.endswith('.npy') and 'proj' in f]
            if proj_files:
                projections = np.load(osp.join(args.data, proj_files[0]))
                print(f"Loaded projections from {proj_files[0]}, shape: {projections.shape}")
            else:
                raise RuntimeError("Could not load projections. Need full Scene infrastructure.")
        else:
            raise RuntimeError("Could not load scene data.")
    
    # Load CNN model
    print(f"\nLoading CNN model from {args.cnn_model}")
    target_depth = args.target_vol_size[0]
    model, device = load_cnn_model(args.cnn_model, target_depth, args.model_size)
    
    # Reconstruct volume
    print("\nReconstructing volume with CNN...")
    print(f"  Input projections: {projections.shape}")
    print(f"  Target projection size: {args.target_proj_size}")
    print(f"  Target volume size: {args.target_vol_size}")
    
    volume = reconstruct_volume_with_cnn(
        model, device, projections,
        target_proj_size=tuple(args.target_proj_size),
        target_vol_size=tuple(args.target_vol_size)
    )
    
    print(f"  Reconstructed volume shape: {volume.shape}")
    print(f"  Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
    
    # Resize volume to original scanner dimensions if needed
    original_size = tuple(scanner_cfg['nVoxel'][::-1])  # Convert XYZ to ZYX
    if volume.shape != original_size:
        print(f"\nResizing volume from {volume.shape} to {original_size}")
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        volume_resized = torch.nn.functional.interpolate(
            volume_tensor,
            size=original_size,
            mode='trilinear',
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()
        volume = volume_resized
        print(f"  Resized volume range: [{volume.min():.3f}, {volume.max():.3f}]")
    
    # Sample points
    print(f"\nSampling {args.n_points} points from volume...")
    print(f"  Density threshold: {args.density_thresh}")
    print(f"  Density rescale: {args.density_rescale}")
    
    positions, densities = sample_points_from_volume(
        volume, scanner_cfg,
        n_points=args.n_points,
        density_thresh=args.density_thresh,
        density_rescale=args.density_rescale
    )
    
    print(f"  Sampled {len(positions)} points")
    print(f"  Position range: X [{positions[:,0].min():.3f}, {positions[:,0].max():.3f}]")
    print(f"                 Y [{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]")
    print(f"                 Z [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}]")
    print(f"  Density range: [{densities.min():.3f}, {densities.max():.3f}]")
    
    # Save to file
    output_data = np.concatenate([positions, densities[:, None]], axis=-1)
    np.save(args.output, output_data)
    
    print(f"\nSaved initialization to {args.output}")
    print(f"File shape: {output_data.shape}")
    
    # Optional: evaluate against ground truth if available
    # Ground truth volume is available in dataset.vol_gt
    # For future evaluation, can compare with ground truth
    pass
    
    print("\n" + "=" * 60)
    print("CNN initialization completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()