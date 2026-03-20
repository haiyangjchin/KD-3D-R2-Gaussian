#!/usr/bin/env python3
"""
Evaluate initialization point cloud quality.
"""
import os
import os.path as osp
import sys
import argparse
import numpy as np
import torch

sys.path.append("./")
sys.path.append("./r2_gaussian/submodules/simple-knn")
sys.path.append("./r2_gaussian/submodules/xray-gaussian-rasterization-voxelization")

from r2_gaussian.dataset import Scene
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, query, initialize_gaussian
from r2_gaussian.utils.image_utils import metric_vol

def evaluate_initial_quality(init_points, scene, scanner_cfg, pipe_args):
    """
    Evaluate initial point cloud quality by rendering volume and comparing to GT.
    
    Args:
        init_points: Initial point cloud [N, 4] (xyz + density)
        scene: Scene object with ground truth volume
        scanner_cfg: Scanner configuration
        pipe_args: Pipeline parameters
    
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Create temporary Gaussian model
        volume_to_world = max(scanner_cfg["sVoxel"])
        scale_bound = [volume_to_world * 0.001, volume_to_world * 0.1]
        gaussians = GaussianModel(scale_bound)
        
        # Create temporary model args
        class TempModelArgs:
            def __init__(self):
                self.ply_path = None
                self.source_path = None
                self.scale_min = None
                self.scale_max = None
        
        model_args = TempModelArgs()
        model_args.ply_path = "temp_init.npy"
        
        # Save points to temporary file
        np.save("temp_init.npy", init_points)
        
        # Initialize Gaussians
        initialize_gaussian(gaussians, model_args, None)
        
        # Ensure scanner_cfg values are torch tensors on correct device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        offOrigin = torch.tensor(scanner_cfg["offOrigin"], device=device)
        nVoxel = torch.tensor(scanner_cfg["nVoxel"], device=device)
        sVoxel = torch.tensor(scanner_cfg["sVoxel"], device=device)
        
        # Query volume
        with torch.no_grad():
            vol_pred = query(
                gaussians,
                offOrigin,
                nVoxel,
                sVoxel,
                pipe_args,
            )["vol"]
            
            vol_gt = scene.vol_gt.to(device)
            
            # Debug: print volume statistics
            print(f"  Volume stats: GT shape {vol_gt.shape}, min={vol_gt.min().item():.3f}, max={vol_gt.max().item():.3f}, mean={vol_gt.mean().item():.3f}")
            print(f"  Volume stats: Pred shape {vol_pred.shape}, min={vol_pred.min().item():.3f}, max={vol_pred.max().item():.3f}, mean={vol_pred.mean().item():.3f}")
            
            # Compute metrics
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            mse_3d, _ = metric_vol(vol_gt, vol_pred, "mse")
            l1_3d = torch.mean(torch.abs(vol_gt - vol_pred)).item()
        
        # Clean up
        if osp.exists("temp_init.npy"):
            os.remove("temp_init.npy")
        
        return {
            'psnr': psnr_3d,
            'mse': mse_3d,
            'l1': l1_3d,
            'num_points': init_points.shape[0]
        }
    except Exception as e:
        print(f"Error in evaluate_initial_quality: {e}")
        import traceback
        traceback.print_exc()
        # Return placeholder metrics
        return {
            'psnr': 0.0,
            'mse': 1.0,
            'l1': 1.0,
            'num_points': init_points.shape[0],
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate initialization point cloud")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--init', type=str, required=True,
                       help='Path to initialization .npy file')
    args = parser.parse_args()
    
    # Setup scene
    parser = argparse.ArgumentParser()
    model_args = ModelParams(parser)
    model_args.source_path = args.data
    model_args.model_path = None
    model_args.eval = False
    pipe_args = PipelineParams(parser)
    
    scene = Scene(model_args)
    scanner_cfg = scene.scanner_cfg
    
    print(f"Scene loaded: {len(scene.getTrainCameras())} training views")
    print(f"Volume size: {scanner_cfg['nVoxel']}")
    print()
    
    # Load point cloud
    if not osp.exists(args.init):
        raise FileNotFoundError(f"Initialization file not found: {args.init}")
    
    init_points = np.load(args.init)
    print(f"Loaded {init_points.shape[0]} points from {args.init}")
    print(f"  Position range: [{init_points[:,:3].min(axis=0)}, {init_points[:,:3].max(axis=0)}]")
    print(f"  Density range: [{init_points[:,3].min():.3f}, {init_points[:,3].max():.3f}]")
    
    # Evaluate
    metrics = evaluate_initial_quality(init_points, scene, scanner_cfg, pipe_args)
    
    print("\nEvaluation metrics:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  L1: {metrics['l1']:.4f}")
    print(f"  Number of points: {metrics['num_points']}")
    
    if 'error' in metrics:
        print(f"  Error: {metrics['error']}")

if __name__ == "__main__":
    main()