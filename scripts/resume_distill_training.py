#!/usr/bin/env python3
"""
Resume training from a checkpoint for knowledge distillation.
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

from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import json

sys.path.append(".")
from r2_gaussian import GaussianModel
from r2_gaussian.scene import Scene, GaussianModel as GaussianModelAlias
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian.initialize import initialize_gaussian
from r2_gaussian.utils.system_utils import mkdir_p, searchForMaxIteration
from r2_gaussian.utils.logging_utils import Log
from r2_gaussian.distillation.cnn_distillation import (
    CNNDistillationTeacher,
    create_distillation_loss,
    ProgressiveDistillation,
)
from r2_gaussian.distillation.improved_training import ViewImportanceSampler
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.distillation import DISTILLATION_AVAILABLE, IMPROVED_TRAINING_AVAILABLE
from r2_gaussian.utils.query import query

# Import training function
from train_with_distillation import train_with_distillation


def main():
    parser = ArgumentParser(description="Resume training from checkpoint")
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
        "--resume_checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file to resume from (e.g., chkpnt9000.pth)",
    )
    parser.add_argument(
        "--target_iterations",
        type=int,
        default=10000,
        help="Target total iterations (default: 10000)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress console output"
    )
    parser.add_argument(
        "--cnn_model",
        type=str,
        default=None,
        help="Path to pre-trained CNN model for distillation",
    )
    parser.add_argument(
        "--no_distill", action="store_true", help="Disable knowledge distillation"
    )
    parser.add_argument(
        "--distill_config",
        type=str,
        default=None,
        help="Path to distillation configuration JSON file",
    )
    parser.add_argument(
        "--no_improvements",
        action="store_true",
        help="Disable improved training techniques",
    )

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

    # Extract parameters from parsed args
    model_args = ModelParams()
    opt_args = OptimizationParams()
    pipe_args = PipelineParams()

    # Update from configuration
    for key, value in cfg.items():
        if hasattr(model_args, key):
            setattr(model_args, key, value)
        if hasattr(opt_args, key):
            setattr(opt_args, key, value)
        if hasattr(pipe_args, key):
            setattr(pipe_args, key, value)

    # Override iterations to target
    opt_args.iterations = args.target_iterations

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

    # Load checkpoint
    print(f"Loading checkpoint from {args.resume_checkpoint}")
    checkpoint = torch.load(
        args.resume_checkpoint, map_location="cpu", weights_only=False
    )

    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        model_state, loaded_iter = checkpoint
        print(f"Resuming from iteration {loaded_iter}")

        # Restore Gaussian model
        gaussians.restore(model_state, opt_args)

        # Update model_path in model_args for saving
        model_args.model_path = args.output_dir

        # Set starting iteration
        start_iteration = loaded_iter
    else:
        raise ValueError(f"Invalid checkpoint format: {type(checkpoint)}")

    scene.gaussians = gaussians

    # Load distillation configuration
    distill_config = None
    if args.distill_config and osp.exists(args.distill_config):
        with open(args.distill_config, "r") as f:
            distill_config = json.load(f)

    # Check if we should use distillation
    use_distillation = (
        not args.no_distill and args.cnn_model and osp.exists(args.cnn_model)
    )

    # Check if we should use improved training
    use_improvements = not args.no_improvements and IMPROVED_TRAINING_AVAILABLE

    print("=" * 70)
    print("Resuming Training with Knowledge Distillation")
    print("=" * 70)
    print(f"Data source: {model_args.source_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resumed from iteration: {start_iteration}")
    print(f"Target iterations: {opt_args.iterations}")
    print(f"Remaining iterations: {opt_args.iterations - start_iteration}")
    print(f"Knowledge distillation: {'ENABLED' if use_distillation else 'DISABLED'}")
    if use_distillation:
        print(f"  Teacher model: {args.cnn_model}")
    print(f"Improved training: {'ENABLED' if use_improvements else 'DISABLED'}")
    print("=" * 70)

    # Get checkpoint iterations from config
    checkpoint_iterations = cfg.get("checkpoint_iterations", [])
    save_iterations = cfg.get("save_iterations", [])

    # Run training from resumed iteration
    train_with_distillation(
        model_path=args.output_dir,
        iteration=start_iteration,
        testing_iterations=args.test_iterations
        if hasattr(args, "test_iterations")
        else [],
        scene=scene,
        gaussians=gaussians,
        opt=opt_args,
        pipe=pipe_args,
        cnn_model_path=args.cnn_model if use_distillation else None,
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
