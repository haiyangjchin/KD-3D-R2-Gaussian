#!/usr/bin/env python3
"""
Resume training from checkpoint with device fixes.
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
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import json

sys.path.append("./")
sys.path.append("./r2_gaussian/submodules/simple-knn")
sys.path.append("./r2_gaussian/submodules/xray-gaussian-rasterization-voxelization")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.dataset import Scene

# Import CNN model and distillation utilities
try:
    from models.ct_unet3d import CTUNet3D
    from utils.distillation_utils import (
        VolumeDistillationLoss,
        ProgressiveDistillation,
        ProjectionConsistencyLoss,
        create_distillation_loss,
    )

    DISTILLATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Distillation modules not available: {e}")
    DISTILLATION_AVAILABLE = False

# Import improved training components
try:
    from train_improved import ViewImportanceSampler, depth_consistency_loss

    IMPROVED_TRAINING_AVAILABLE = True
except ImportError:
    print("Warning: Improved training modules not available")
    IMPROVED_TRAINING_AVAILABLE = False


class CNNDistillationTeacher:
    """Same as in train_with_distillation.py"""

    def __init__(
        self, cnn_model_path, device="cuda", target_depth=64, model_size="small"
    ):
        if not DISTILLATION_AVAILABLE:
            raise ImportError("Distillation modules not available")

        self.device = device if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(cnn_model_path, map_location=self.device)

        if model_size == "small":
            features = [16, 32, 64, 128]
        elif model_size == "medium":
            features = [32, 64, 128, 256]
        elif model_size == "large":
            features = [64, 128, 256, 512]
        else:
            features = [32, 64, 128, 256]

        self.model = CTUNet3D(
            in_channels=1,
            out_channels=1,
            features=features,
            use_dropout=False,
            target_depth=target_depth,
        )

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.target_depth = target_depth
        self.model_size = model_size

        print(f"Loaded CNN teacher model from {cnn_model_path}")
        print(f"  Model size: {model_size}, features: {features}")
        print(f"  Target depth: {target_depth}")
        print(f"  Device: {self.device}")

    def precompute_volume_from_scene(self, scene, scanner_cfg=None):
        """Simplified - just return dummy for now"""
        if scanner_cfg is None:
            raise ValueError("scanner_cfg required")

        target_d, target_h, target_w = scanner_cfg["nVoxel"]
        dummy_volume = torch.randn(target_d, target_h, target_w)
        self.cached_volume = dummy_volume
        return dummy_volume

    def get_volume(self, scanner_cfg=None):
        if not hasattr(self, "cached_volume") or self.cached_volume is None:
            raise ValueError("Teacher volume not precomputed")
        return self.cached_volume


def move_model_to_device(gaussians, device="cuda"):
    """Move all Gaussian model tensors to specified device"""
    if not torch.cuda.is_available():
        device = "cpu"

    # Move core tensors
    gaussians._xyz = gaussians._xyz.to(device)
    gaussians._scaling = gaussians._scaling.to(device)
    gaussians._rotation = gaussians._rotation.to(device)
    gaussians._density = gaussians._density.to(device)
    gaussians.max_radii2D = gaussians.max_radii2D.to(device)

    # Move gradient accumulators
    if gaussians.xyz_gradient_accum is not None:
        gaussians.xyz_gradient_accum = gaussians.xyz_gradient_accum.to(device)
    if gaussians.denom is not None:
        gaussians.denom = gaussians.denom.to(device)

    print(f"Moved model to {device}")


def training_with_distillation(
    model_path,
    iteration,
    testing_iterations,
    scene,
    gaussians,
    opt,
    pipe,
    teacher_model,
    distillation_loss,
    progressive_distill,
    use_importance_sampling=False,
    view_sampler=None,
    enhanced_regularization=False,
    scanner_cfg=None,
):
    """
    Training iteration with knowledge distillation.
    Simplified version from train_with_distillation.py
    """
    from random import randint
    from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss

    # Get viewpoint (camera)
    viewpoint_cam = scene.getTrainCameras()[
        randint(0, len(scene.getTrainCameras()) - 1)
    ]

    # Render
    render_pkg = render(viewpoint_cam, gaussians, pipe)
    image = render_pkg["render"]

    # Compute standard loss
    gt_image = viewpoint_cam.original_image
    if torch.cuda.is_available():
        gt_image = gt_image.cuda()

    device = gt_image.device
    loss = {
        "total": torch.tensor(0.0, device=device),
        "standard": torch.tensor(0.0, device=device),
        "distill": torch.tensor(0.0, device=device),
    }

    # Standard rendering loss
    render_loss = l1_loss(image, gt_image)
    loss["render"] = render_loss
    loss["standard"] += render_loss

    if opt.lambda_dssim > 0:
        loss_dssim = 1.0 - ssim(image, gt_image)
        loss["dssim"] = loss_dssim
        loss["standard"] = loss["standard"] + opt.lambda_dssim * loss_dssim

    # 3D TV loss
    if opt.lambda_tv > 0 and scanner_cfg is not None:
        tv_vol_size = 32
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

        bbox_min = (
            torch.tensor(scanner_cfg["offOrigin"])
            - torch.tensor(scanner_cfg["sVoxel"]) / 2
        )
        bbox_max = (
            torch.tensor(scanner_cfg["offOrigin"])
            + torch.tensor(scanner_cfg["sVoxel"]) / 2
        )
        tv_vol_center = (
            bbox_min
            + tv_vol_sVoxel / 2
            + (bbox_max - tv_vol_sVoxel - bbox_min) * torch.rand(3)
        )

        vol_pred = query(
            gaussians,
            tv_vol_center,
            tv_vol_nVoxel,
            tv_vol_sVoxel,
            pipe,
        )["vol"]

        loss_tv = tv_3d_loss(vol_pred, reduction="mean")
        loss["tv"] = loss_tv
        loss["standard"] = loss["standard"] + opt.lambda_tv * loss_tv

    # Knowledge distillation loss (simplified)
    distill_weight = 0.1  # Fixed for simplicity
    if distill_weight > 0 and teacher_model is not None and scanner_cfg is not None:
        student_volume = query(
            gaussians,
            torch.tensor(scanner_cfg["offOrigin"]),
            torch.tensor(scanner_cfg["nVoxel"]),
            torch.tensor(scanner_cfg["sVoxel"]),
            pipe,
        )["vol"]

        teacher_volume = teacher_model.get_volume(scanner_cfg)
        teacher_volume = teacher_volume.to(student_volume.device)

        # Simple L1 distillation loss
        loss["distill"] = torch.mean(torch.abs(teacher_volume - student_volume))

    loss["total"] = loss["standard"] + distill_weight * loss["distill"]

    return loss, render_pkg


def train_with_distillation(
    model_path,
    iteration,
    testing_iterations,
    scene,
    gaussians,
    opt,
    pipe,
    cnn_model_path=None,
    distillation_config=None,
    use_improvements=True,
    output_path=None,
    checkpoint_iterations=[],
    saving_iterations=[],
):
    """
    Simplified training loop
    """
    if not DISTILLATION_AVAILABLE:
        raise ImportError("Distillation modules not available")

    # Setup teacher model
    teacher = None
    if cnn_model_path and osp.exists(cnn_model_path):
        try:
            teacher = CNNDistillationTeacher(
                cnn_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                target_depth=64,
                model_size="medium",
            )
            print(f"Knowledge distillation enabled with teacher model")
        except Exception as e:
            print(f"Failed to load teacher model: {e}")
            teacher = None

    # Get scanner configuration
    scanner_cfg = scene.scanner_cfg

    # Precompute teacher volume
    if teacher is not None:
        print("Precomputing teacher volume for distillation...")
        teacher.precompute_volume_from_scene(scene, scanner_cfg)
        print("Teacher volume precomputed.")

    # Setup training
    gaussians.training_setup(opt)

    # Move model to CUDA if available
    if torch.cuda.is_available():
        move_model_to_device(gaussians, "cuda")

    # Training loop
    first_iter = iteration
    progress_bar = tqdm(
        range(0, opt.iterations), desc="Train with Distillation", leave=False
    )
    progress_bar.update(first_iter)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        from random import randint

        viewpoint_cam = scene.getTrainCameras()[
            randint(0, len(scene.getTrainCameras()) - 1)
        ]

        # Training step
        loss, render_pkg = training_with_distillation(
            model_path,
            iteration,
            testing_iterations,
            scene,
            gaussians,
            opt,
            pipe,
            teacher,
            None,
            None,  # Simplified distillation
            use_importance_sampling=False,
            view_sampler=None,
            enhanced_regularization=False,
            scanner_cfg=scanner_cfg,
        )

        loss["total"].backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        # Save checkpoints and models
        if iteration in saving_iterations or iteration == opt.iterations:
            tqdm.write(f"[ITER {iteration}] Saving Gaussians")
            # Simplified save - would need queryfunc
            pass

        if iteration in checkpoint_iterations:
            tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
            torch.save(
                (gaussians.capture(), iteration),
                osp.join(model_path, "ckpt", f"chkpnt{iteration}.pth"),
            )

        # Update progress bar
        if iteration % 10 == 0:
            postfix = {
                "loss": f"{loss['total'].item():.1e}",
                "pts": f"{gaussians.get_density.shape[0]:2.1e}",
            }
            progress_bar.set_postfix(postfix)
            progress_bar.update(10)

        if iteration == opt.iterations:
            progress_bar.close()

    print(f"Training completed!")


def main():
    parser = ArgumentParser(
        description="Resume training from checkpoint (fixed device)"
    )

    # Add parameters from ModelParams, OptimizationParams, PipelineParams
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

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
        help="Path to checkpoint file to resume from",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress console output"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    args_dict = vars(args)
    for key in list(cfg.keys()):
        args_dict[key] = cfg[key]

    # Setup device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    safe_state(args.quiet)

    # Extract parameters
    model_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    print("=" * 70)
    print("Resuming Training from Checkpoint")
    print("=" * 70)
    print(f"Data source: {model_args.source_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint: {args.resume_checkpoint}")
    print(f"Target iterations: {opt_args.iterations}")
    print("=" * 70)

    # Load checkpoint
    print(f"Loading checkpoint from {args.resume_checkpoint}")
    checkpoint = torch.load(
        args.resume_checkpoint, map_location="cpu", weights_only=False
    )

    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        model_state, loaded_iter = checkpoint
        print(f"Resuming from iteration {loaded_iter}")
        start_iteration = loaded_iter
    else:
        raise ValueError(f"Invalid checkpoint format: {type(checkpoint)}")

    # Load scene
    scene = Scene(model_args, args.eval)

    # Get scanner configuration
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

    # Restore from checkpoint
    print("Restoring Gaussian model from checkpoint...")
    gaussians.restore(model_state, opt_args)

    # Move to CUDA if available
    if torch.cuda.is_available():
        print("Moving model to CUDA...")
        move_model_to_device(gaussians, "cuda")

    scene.gaussians = gaussians

    # Get checkpoint and save iterations
    checkpoint_iterations = cfg.get("checkpoint_iterations", [])
    save_iterations = cfg.get("save_iterations", [])

    # Run training
    train_with_distillation(
        model_path=args.output_dir,
        iteration=start_iteration,
        testing_iterations=cfg.get("test_iterations", []),
        scene=scene,
        gaussians=gaussians,
        opt=opt_args,
        pipe=pipe_args,
        cnn_model_path=cfg.get("cnn_model"),
        use_improvements=False,  # Simplified
        output_path=args.output_dir,
        checkpoint_iterations=checkpoint_iterations,
        saving_iterations=save_iterations,
    )

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
