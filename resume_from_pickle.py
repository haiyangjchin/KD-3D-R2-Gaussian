#!/usr/bin/env python3
"""
Resume training from pickle file (point_cloud.pickle).
"""

import os
import os.path as osp
import torch

# Add torch lib directory to DLL search path for CUDA extensions
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    print(f"Added torch DLL directory: {torch_lib}")
# Add CUDA DLL directory for CUDA extensions
cuda_bin = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin"
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)
    print(f"Added CUDA DLL directory: {cuda_bin}")

# Add Anaconda DLL directories
anaconda_dir = os.path.dirname(
    os.path.dirname(torch.__file__)
)  # D:\Anaconda or similar
anaconda_lib_bin = os.path.join(anaconda_dir, "Library", "bin")
anaconda_dlls = os.path.join(anaconda_dir, "DLLs")

for path in [anaconda_lib_bin, anaconda_dlls]:
    if os.path.exists(path):
        os.add_dll_directory(path)
        print(f"Added DLL directory: {path}")

# Add Windows system directories (optional but may help)
sys_dirs = [
    os.environ.get("SystemRoot", "") + "\\System32",
    os.environ.get("SystemRoot", "") + "\\SysWOW64",
]
for path in sys_dirs:
    if os.path.exists(path):
        os.add_dll_directory(path)
        print(f"Added system DLL directory: {path}")

import sys
from argparse import ArgumentParser
import numpy as np
import yaml
import json

sys.path.append("./")
sys.path.append("./r2_gaussian/submodules/simple-knn")
sys.path.append("./r2_gaussian/submodules/xray-gaussian-rasterization-voxelization")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.dataset import Scene
from train_with_distillation import train_with_distillation


def main():
    parser = ArgumentParser(description="Resume training from pickle")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--resume_iteration",
        type=int,
        required=True,
        help="Iteration to resume from (e.g., 7500)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress console output"
    )

    # Extract parameters using ModelParams, OptimizationParams, PipelineParams
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Update args with configuration values
    args_dict = vars(args)
    for key in list(cfg.keys()):
        args_dict[key] = cfg[key]

    # Setup device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Initialize safe state
    safe_state(args.quiet)

    model_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    # Update model_args model_path to output_dir
    model_args.model_path = args.output_dir

    # Load scene
    scene = Scene(model_args, args.eval)

    # Get scanner configuration for scale bound calculation
    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])

    # Calculate scale bound if scale_min/max are set
    scale_bound = None
    if model_args.scale_min > 0 and model_args.scale_max > 0:
        scale_bound = (
            np.array([model_args.scale_min, model_args.scale_max]) * volume_to_world
        )

    # Initialize Gaussian model
    gaussians = GaussianModel(scale_bound)

    # Load from pickle
    print(f"Resuming from iteration {args.resume_iteration}")
    initialize_gaussian(gaussians, model_args, args.resume_iteration)

    scene.gaussians = gaussians

    # Load distillation configuration if exists
    distill_config = None
    distill_config_path = None

    # First try from config file
    if (
        "distill_config" in cfg
        and cfg["distill_config"]
        and osp.exists(cfg["distill_config"])
    ):
        distill_config_path = cfg["distill_config"]
    # Then try from command line args (if exists)
    elif (
        hasattr(args, "distill_config")
        and args.distill_config
        and osp.exists(args.distill_config)
    ):
        distill_config_path = args.distill_config

    if distill_config_path:
        with open(distill_config_path, "r") as f:
            distill_config = json.load(f)

    # Check if we should use distillation
    cnn_model_path = None
    # First try from config file
    if "cnn_model" in cfg and cfg["cnn_model"]:
        cnn_model_path = cfg["cnn_model"]
    # Then try from command line args (if exists)
    elif hasattr(args, "cnn_model") and args.cnn_model:
        cnn_model_path = args.cnn_model

    use_distillation = cnn_model_path and osp.exists(cnn_model_path)

    # Check if we should use improved training
    try:
        from train_improved import ViewImportanceSampler

        IMPROVED_TRAINING_AVAILABLE = True
    except ImportError:
        IMPROVED_TRAINING_AVAILABLE = False

    use_improvements = IMPROVED_TRAINING_AVAILABLE

    print("=" * 70)
    print("Resuming Training from Pickle")
    print("=" * 70)
    print(f"Data source: {model_args.source_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resumed from iteration: {args.resume_iteration}")
    print(f"Target iterations: {opt_args.iterations}")
    print(f"Remaining iterations: {opt_args.iterations - args.resume_iteration}")
    print("=" * 70)

    # Get checkpoint and save iterations from config
    checkpoint_iterations = cfg.get("checkpoint_iterations", [])
    save_iterations = cfg.get("save_iterations", [])

    # Run training from resumed iteration
    train_with_distillation(
        model_path=args.output_dir,
        iteration=args.resume_iteration,
        testing_iterations=args.test_iterations
        if hasattr(args, "test_iterations")
        else [],
        scene=scene,
        gaussians=gaussians,
        opt=opt_args,
        pipe=pipe_args,
        cnn_model_path=cnn_model_path if use_distillation else None,
        distillation_config=distill_config,
        use_improvements=use_improvements,
        output_path=args.output_dir,
        checkpoint_iterations=checkpoint_iterations,
        saving_iterations=save_iterations,
    )

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
