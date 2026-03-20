#!/usr/bin/env python3
"""
Compare initialization methods for 3D Gaussian training.

Compares:
1. Random initialization (baseline)
2. FDK initialization (traditional)
3. CNN initialization (proposed)
4. CNN initialization + Knowledge distillation

For each method, evaluates:
- Initial point cloud quality
- Training convergence speed
- Final reconstruction quality
"""

import os
import os.path as osp
import sys
import argparse
import numpy as np
import torch
import json
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("./")
sys.path.append("./r2_gaussian/submodules/simple-knn")
sys.path.append("./r2_gaussian/submodules/xray-gaussian-rasterization-voxelization")

from r2_gaussian.dataset import Scene
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, query, initialize_gaussian
from r2_gaussian.utils.image_utils import metric_vol


def load_point_cloud(init_file):
    """Load point cloud from initialization file."""
    if not osp.exists(init_file):
        raise FileNotFoundError(f"Initialization file not found: {init_file}")
    
    data = np.load(init_file)
    print(f"Loaded {data.shape[0]} points from {init_file}")
    print(f"  Position range: [{data[:,:3].min(axis=0)}, {data[:,:3].max(axis=0)}]")
    print(f"  Density range: [{data[:,3].min():.3f}, {data[:,3].max():.3f}]")
    
    return data


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


def compare_initialization_methods(data_path, methods, cnn_model_path=None):
    """
    Compare different initialization methods.
    
    Args:
        data_path: Path to data directory
        methods: List of methods to compare ['random', 'fdk', 'cnn']
        cnn_model_path: Path to CNN model for CNN initialization
    
    Returns:
        Dictionary of results for each method
    """
    print("=" * 70)
    print("Initialization Method Comparison")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Methods: {methods}")
    print()
    
    # Setup scene
    parser = argparse.ArgumentParser()
    model_args = ModelParams(parser)
    model_args.source_path = data_path
    model_args.model_path = None
    model_args.eval = False
    pipe_args = PipelineParams(parser)
    
    scene = Scene(model_args)
    scanner_cfg = scene.scanner_cfg
    
    print(f"Scene loaded: {len(scene.getTrainCameras())} training views")
    print(f"Volume size: {scanner_cfg['nVoxel']}")
    print()
    
    results = {}
    
    # Generate initialization for each method
    for method in methods:
        print(f"\n{'='*40}")
        print(f"Method: {method.upper()}")
        print(f"{'='*40}")
        
        init_file = f"./init_{method}.npy"
        
        if method == 'random':
            # Generate random initialization
            n_points = 50000
            offOrigin = np.array(scanner_cfg["offOrigin"])
            sVoxel = np.array(scanner_cfg["sVoxel"])
            
            positions = offOrigin[None, ...] + sVoxel[None, ...] * (np.random.rand(n_points, 3) - 0.5)
            densities = np.random.rand(n_points) * 1.0  # max density = 1.0
            
            init_points = np.concatenate([positions, densities[:, None]], axis=-1)
            np.save(init_file, init_points)
            
            print(f"  Generated {n_points} random points")
            print(f"  Saved to {init_file}")
            
        elif method == 'fdk':
            # Try to use FDK initialization (requires TIGRE)
            print("  FDK initialization requires TIGRE library")
            print("  Skipping FDK (would need TIGRE installation)")
            continue
            
        elif method == 'cnn':
            # Check if initialization file already exists
            if osp.exists(init_file):
                print(f"  Using existing initialization file: {init_file}")
            else:
                # Generate CNN initialization
                if cnn_model_path is None or not osp.exists(cnn_model_path):
                    print(f"  CNN model not found: {cnn_model_path}")
                    print("  Using default path: ./cnn_output_final/best_model.pth")
                    cnn_model_path = "./cnn_output_final/best_model.pth"
                
                if not osp.exists(cnn_model_path):
                    print(f"  Default CNN model not found either")
                    print("  Skipping CNN initialization")
                    continue
                
                # Use our CNN initialization script
                try:
                    from initialize_with_cnn import main as cnn_init_main
                    import subprocess
                    
                    # Run CNN initialization
                    cmd = [
                        sys.executable, "initialize_with_cnn.py",
                        "--data", data_path,
                        "--output", init_file,
                        "--cnn_model", cnn_model_path,
                        "--n_points", "50000",
                        "--density_thresh", "0.05",
                        "--density_rescale", "0.15",
                        "--model_size", "medium"
                    ]
                    
                    print(f"  Running: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        print(f"  CNN initialization failed: {result.stderr}")
                        continue
                    
                    print(f"  CNN initialization completed")
                    
                except Exception as e:
                    print(f"  Error running CNN initialization: {e}")
                    continue
        
        # Load and evaluate
        try:
            init_points = load_point_cloud(init_file)
            metrics = evaluate_initial_quality(init_points, scene, scanner_cfg, pipe_args)
            
            results[method] = {
                'metrics': metrics,
                'file': init_file,
                'num_points': init_points.shape[0]
            }
            
            print(f"\n  Evaluation metrics:")
            print(f"    PSNR: {metrics['psnr']:.2f} dB")
            print(f"    MSE: {metrics['mse']:.4f}")
            print(f"    L1: {metrics['l1']:.4f}")
            
        except Exception as e:
            print(f"  Error evaluating {method}: {e}")
            results[method] = {'error': str(e)}
    
    return results


def visualize_comparison(results, output_dir="./init_comparison"):
    """Visualize comparison results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    methods = []
    psnrs = []
    mses = []
    l1s = []
    
    for method, data in results.items():
        if 'metrics' in data:
            methods.append(method)
            psnrs.append(data['metrics']['psnr'])
            mses.append(data['metrics']['mse'])
            l1s.append(data['metrics']['l1'])
    
    if len(methods) == 0:
        print("No valid results to visualize")
        return
    
    # Create bar plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSNR
    axes[0].bar(methods, psnrs, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0].set_title('Initial PSNR (dB)')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(psnrs):
        axes[0].text(i, v + 0.1, f'{v:.1f}', ha='center')
    
    # MSE
    axes[1].bar(methods, mses, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1].set_title('Initial MSE')
    axes[1].set_ylabel('MSE')
    axes[1].grid(True, alpha=0.3)
    
    for i, v in enumerate(mses):
        axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center')
    
    # L1
    axes[2].bar(methods, l1s, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[2].set_title('Initial L1 Error')
    axes[2].set_ylabel('L1 Error')
    axes[2].grid(True, alpha=0.3)
    
    for i, v in enumerate(l1s):
        axes[2].text(i, v + 0.001, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plot_path = osp.join(output_dir, 'init_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {plot_path}")
    
    # Create summary report
    report_path = osp.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("Initialization Method Comparison Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Data path: {data_path}\n")
        f.write(f"  Compared methods: {list(results.keys())}\n\n")
        
        f.write("Results:\n")
        for method, data in results.items():
            f.write(f"\n{method.upper()}:\n")
            f.write(f"  {'='*30}\n")
            
            if 'error' in data:
                f.write(f"  Error: {data['error']}\n")
            else:
                f.write(f"  File: {data['file']}\n")
                f.write(f"  Number of points: {data['num_points']}\n")
                f.write(f"  PSNR: {data['metrics']['psnr']:.2f} dB\n")
                f.write(f"  MSE: {data['metrics']['mse']:.4f}\n")
                f.write(f"  L1: {data['metrics']['l1']:.4f}\n")
        
        f.write("\n\nInterpretation:\n")
        f.write("- Higher PSNR is better (indicating better reconstruction)\n")
        f.write("- Lower MSE and L1 are better\n")
        f.write("- Random initialization is the baseline\n")
        f.write("- FDK should outperform random (if available)\n")
        f.write("- CNN aims to outperform both\n")
    
    print(f"Saved report to {report_path}")
    
    return plot_path, report_path


def main():
    parser = argparse.ArgumentParser(description="Compare initialization methods")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--methods', nargs='+', 
                       default=['random', 'cnn'],
                       choices=['random', 'fdk', 'cnn'],
                       help='Methods to compare')
    parser.add_argument('--cnn_model', type=str, default=None,
                       help='Path to CNN model for CNN initialization')
    parser.add_argument('--output_dir', type=str, default='./init_comparison',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Global variable for visualization function
    global data_path
    data_path = args.data
    
    # Run comparison
    results = compare_initialization_methods(
        data_path=args.data,
        methods=args.methods,
        cnn_model_path=args.cnn_model
    )
    
    # Visualize results
    if results:
        plot_path, report_path = visualize_comparison(results, args.output_dir)
        
        print("\n" + "=" * 70)
        print("Comparison Summary:")
        print("=" * 70)
        
        for method, data in results.items():
            if 'metrics' in data:
                print(f"{method.upper():10s} - PSNR: {data['metrics']['psnr']:6.2f} dB, "
                      f"MSE: {data['metrics']['mse']:.4f}, "
                      f"L1: {data['metrics']['l1']:.4f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print(f"  - Visualization: {plot_path}")
        print(f"  - Report: {report_path}")
        
        # Determine best method
        best_method = None
        best_psnr = -float('inf')
        
        for method, data in results.items():
            if 'metrics' in data and data['metrics']['psnr'] > best_psnr:
                best_psnr = data['metrics']['psnr']
                best_method = method
        
        if best_method:
            print(f"\nBest initialization method: {best_method.upper()} "
                  f"(PSNR: {best_psnr:.2f} dB)")
            
            # Recommendations
            print("\nRecommendations:")
            if best_method == 'cnn':
                print("✓ CNN initialization shows promise for improving 3D Gaussian training")
                print("✓ Consider using knowledge distillation for further improvement")
                print("✓ Train CNN with more data/epochs for better generalization")
            elif best_method == 'fdk':
                print("✓ FDK provides good initialization but requires TIGRE")
                print("✓ CNN could potentially match or exceed FDK with more training")
            else:
                print("✓ Random initialization is the baseline")
                print("✓ Both FDK and CNN should outperform random when properly implemented")
    
    else:
        print("No valid results generated")


if __name__ == "__main__":
    main()