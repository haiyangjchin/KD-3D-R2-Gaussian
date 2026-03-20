#!/usr/bin/env python3
"""
Resume training from checkpoint for knowledge distillation.
This script modifies the original train_with_distillation.py to support resuming from checkpoint.
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

sys.path.append("./")
sys.path.append("./r2_gaussian/submodules/simple-knn")
sys.path.append("./r2_gaussian/submodules/xray-gaussian-rasterization-voxelization")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice

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
    """
    CNN teacher model for knowledge distillation.
    Loads a pre-trained CNN and provides volume predictions.
    """

    def __init__(
        self, cnn_model_path, device="cuda", target_depth=64, model_size="small"
    ):
        """
        Args:
            cnn_model_path: Path to trained CNN model checkpoint
            device: Torch device
            target_depth: Target depth dimension for output volume
            model_size: Size of CNN model ('small', 'medium', 'large')
        """
        if not DISTILLATION_AVAILABLE:
            raise ImportError("Distillation modules not available")

        self.device = device if torch.cuda.is_available() else "cpu"

        # Load CNN model
        checkpoint = torch.load(cnn_model_path, map_location=self.device)

        # Determine feature sizes
        if model_size == "small":
            features = [16, 32, 64, 128]
        elif model_size == "medium":
            features = [32, 64, 128, 256]
        elif model_size == "large":
            features = [64, 128, 256, 512]
        else:
            features = [32, 64, 128, 256]

        # Create model
        self.model = CTUNet3D(
            in_channels=1,
            out_channels=1,
            features=features,
            use_dropout=False,
            target_depth=target_depth,
        )

        # Load weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Store configuration
        self.target_depth = target_depth
        self.model_size = model_size

        print(f"Loaded CNN teacher model from {cnn_model_path}")
        print(f"  Model size: {model_size}, features: {features}")
        print(f"  Target depth: {target_depth}")
        print(f"  Device: {self.device}")

    def predict_volume(self, projections, target_proj_size=(64, 64)):
        """
        Predict 3D volume from 2D projections.

        Args:
            projections: Projection images [num_angles, height, width]
            target_proj_size: Target projection size for CNN

        Returns:
            Predicted volume [depth, height, width]
        """
        original_height, original_width = projections.shape[1], projections.shape[2]
        target_height, target_width = target_proj_size

        # Convert to torch tensor
        projs_tensor = (
            torch.from_numpy(projections).float().unsqueeze(1)
        )  # [N, 1, H, W]

        # Resize if needed
        if (original_height, original_width) != (target_height, target_width):
            projs_resized = torch.nn.functional.interpolate(
                projs_tensor,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        else:
            projs_resized = projs_tensor

        # Prepare input: [batch, channels, depth, height, width]
        projs_input = projs_resized.unsqueeze(0)  # [1, N, 1, H, W]
        projs_input = projs_input.permute(0, 2, 1, 3, 4)  # [1, 1, N, H, W]
        projs_input = projs_input.to(self.device)

        # Forward pass
        with torch.no_grad():
            vol_pred = self.model(projs_input)

        # Remove batch and channel dimensions
        vol_pred = vol_pred.squeeze(0).squeeze(0)  # [D, H, W]

        return vol_pred.cpu().numpy()

    def precompute_volume_from_scene(self, scene, scanner_cfg=None):
        """
        Precompute teacher volume from all training views in the scene.

        Args:
            scene: Scene object containing training cameras
            scanner_cfg: Scanner configuration for volume dimensions

        Returns:
            Teacher volume tensor [D, H, W] matching scanner_cfg dimensions
        """
        # Collect all training projections
        train_cameras = scene.getTrainCameras()
        projections = []
        for cam in train_cameras:
            # Get original projection image (grayscale)
            # Assuming cam.original_image is tensor [C, H, W] or [H, W]
            img = cam.original_image
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.ndim == 3:  # [C, H, W]
                img = img.squeeze(0)  # Remove channel dimension
            projections.append(img)

        # Stack projections [num_angles, height, width]
        projections = np.stack(projections, axis=0)

        # Predict volume using teacher model
        teacher_volume_np = self.predict_volume(projections, target_proj_size=(64, 64))

        # Convert to tensor
        teacher_volume = torch.from_numpy(teacher_volume_np).float()

        # Resize to match scanner_cfg dimensions if provided
        if scanner_cfg is not None:
            target_d, target_h, target_w = scanner_cfg["nVoxel"]
            current_d, current_h, current_w = teacher_volume.shape

            if (current_d, current_h, current_w) != (target_d, target_h, target_w):
                # Use 3D interpolation to resize
                teacher_volume = teacher_volume.unsqueeze(0).unsqueeze(
                    0
                )  # [1, 1, D, H, W]
                teacher_volume = torch.nn.functional.interpolate(
                    teacher_volume,
                    size=(target_d, target_h, target_w),
                    mode="trilinear",
                    align_corners=False,
                )
                teacher_volume = teacher_volume.squeeze(0).squeeze(0)  # [D, H, W]

        # Cache the volume
        self.cached_volume = teacher_volume
        return teacher_volume

    def get_volume(self, scanner_cfg=None):
        """
        Get cached teacher volume, resized if needed.

        Args:
            scanner_cfg: Scanner configuration for resizing

        Returns:
            Teacher volume tensor [D, H, W]
        """
        if not hasattr(self, "cached_volume") or self.cached_volume is None:
            raise ValueError(
                "Teacher volume not precomputed. Call precompute_volume_from_scene first."
            )

        volume = self.cached_volume
        if scanner_cfg is not None:
            target_d, target_h, target_w = scanner_cfg["nVoxel"]
            current_d, current_h, current_w = volume.shape

            if (current_d, current_h, current_w) != (target_d, target_h, target_w):
                volume = volume.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                volume = torch.nn.functional.interpolate(
                    volume,
                    size=(target_d, target_h, target_w),
                    mode="trilinear",
                    align_corners=False,
                )
                volume = volume.squeeze(0).squeeze(0)

        return volume


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

    This function replaces the standard training iteration to include
    distillation loss from the teacher CNN.
    """

    # Get viewpoint (camera)
    viewpoint_cam = scene.getTrainCameras()[
        randint(0, len(scene.getTrainCameras()) - 1)
    ]

    # Render
    render_pkg = render(viewpoint_cam, gaussians, pipe)
    image = render_pkg["render"]

    # Compute standard loss
    gt_image = viewpoint_cam.original_image.cuda()
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
    if opt.lambda_tv > 0:
        # Use fixed volume size for TV loss
        tv_vol_size = 32
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

        # Random volume center
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

        # Depth consistency loss (if enabled)
        if enhanced_regularization and IMPROVED_TRAINING_AVAILABLE:
            loss_depth = depth_consistency_loss(vol_pred, reduction="mean")
            loss["depth"] = loss_depth
            loss["standard"] = (
                loss["standard"] + 0.02 * loss_depth
            )  # lambda_depth = 0.02

    # Knowledge distillation loss
    distill_weight = progressive_distill.get_weight(iteration)
    if distill_weight > 0 and teacher_model is not None:
        # Get teacher's volume prediction for current viewpoint
        # For distillation, we need to get volume from student at same resolution
        # and compare with teacher's prediction

        # Query student volume at full resolution
        student_volume = query(
            gaussians,
            torch.tensor(scanner_cfg["offOrigin"]),
            torch.tensor(scanner_cfg["nVoxel"]),
            torch.tensor(scanner_cfg["sVoxel"]),
            pipe,
        )["vol"]

        # Get teacher's volume prediction (precomputed)
        teacher_volume = teacher_model.get_volume(scanner_cfg)

        # Ensure teacher volume is on same device as student volume
        teacher_volume = teacher_volume.to(student_volume.device)

        # Compute distillation loss
        distill_losses = distillation_loss(teacher_volume, student_volume)
        loss["distill"] = distill_losses["total"]

    # Total loss with distillation weighting
    loss["total"] = loss["standard"] + distill_weight * loss["distill"]

    # Update view importance if using importance sampling
    if (
        use_importance_sampling
        and view_sampler is not None
        and IMPROVED_TRAINING_AVAILABLE
    ):
        # Find view index
        view_idx = None
        for i, cam in enumerate(scene.getTrainCameras()):
            if cam.image_name == viewpoint_cam.image_name:
                view_idx = i
                break

        if view_idx is not None:
            view_sampler.update_view_loss(view_idx, loss["total"].item())

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
    Main training loop with knowledge distillation.

    Args:
        model_path: Path to save models
        iteration: Current iteration
        testing_iterations: Iterations to run testing
        scene: Scene object
        gaussians: Gaussian model
        opt: Optimization parameters
        pipe: Pipeline parameters
        cnn_model_path: Path to pre-trained CNN model
        distillation_config: Configuration for distillation
        use_improvements: Use improved training techniques
        output_path: Output path for logs
        checkpoint_iterations: Iterations to save checkpoints
        saving_iterations: Iterations to save models
    """

    if not DISTILLATION_AVAILABLE:
        raise ImportError("Distillation modules not available")

    # Default distillation config
    if distillation_config is None:
        distillation_config = {
            "type": "volume",
            "temperature": 2.0,
            "alpha": 0.7,
            "use_kl": True,
            "use_l1": True,
            "use_ssim": False,
            "total_iterations": opt.iterations,
            "warmup_ratio": 0.0,
            "max_distill_weight": 0.5,
            "schedule": "linear",
        }

    # Setup teacher model
    teacher = None
    if cnn_model_path and osp.exists(cnn_model_path):
        try:
            teacher = CNNDistillationTeacher(
                cnn_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                target_depth=64,  # Should match training
                model_size="medium",
            )
            print(f"Knowledge distillation enabled with teacher model")
        except Exception as e:
            print(f"Failed to load teacher model: {e}")
            teacher = None
    else:
        print(f"CNN model not found at {cnn_model_path}")
        teacher = None

    # Setup distillation loss
    distill_loss = create_distillation_loss(distillation_config)

    # Setup progressive distillation
    progressive_distill = ProgressiveDistillation(
        total_iterations=distillation_config["total_iterations"],
        warmup_ratio=distillation_config["warmup_ratio"],
        max_distill_weight=distillation_config["max_distill_weight"],
        schedule=distillation_config["schedule"],
    )

    # Setup improved training components
    view_sampler = None
    if use_improvements and IMPROVED_TRAINING_AVAILABLE:
        num_views = len(scene.getTrainCameras())
        view_sampler = ViewImportanceSampler(
            num_views=num_views, warmup_iters=500, update_interval=100, alpha=0.9
        )
        print(f"Importance-weighted view sampling enabled")

    enhanced_regularization = use_improvements

    # Get scanner configuration
    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])
    bbox = scene.bbox
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )

    # Define query function for volume query
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Precompute teacher volume for distillation
    if teacher is not None:
        print("Precomputing teacher volume for distillation...")
        teacher.precompute_volume_from_scene(scene, scanner_cfg)
        print("Teacher volume precomputed.")

    # Setup training (optimizer, learning rate schedule, etc.)
    gaussians.training_setup(opt)

    # Training loop
    first_iter = iteration
    progress_bar = tqdm(
        range(0, opt.iterations), desc="Train with Distillation", leave=False
    )
    progress_bar.update(first_iter)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get viewpoint (with importance sampling if enabled)
        if use_improvements and view_sampler is not None:
            view_idx = view_sampler.sample_view(iteration)
            viewpoint_cam = scene.getTrainCameras()[view_idx]
        else:
            viewpoint_cam = scene.getTrainCameras()[
                randint(0, len(scene.getTrainCameras()) - 1)
            ]

        # Training step with distillation
        loss, render_pkg = training_with_distillation(
            model_path,
            iteration,
            testing_iterations,
            scene,
            gaussians,
            opt,
            pipe,
            teacher,
            distill_loss,
            progressive_distill,
            use_importance_sampling=use_improvements,
            view_sampler=view_sampler,
            enhanced_regularization=enhanced_regularization,
            scanner_cfg=scanner_cfg,
        )

        # Backward pass
        loss["total"].backward()

        # Optimizer step
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        # Densification and pruning (disabled for testing)
        if iteration < opt.densify_until_iter and False:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                gaussians.max_radii2D[render_pkg["visibility_filter"]],
                render_pkg["radii"],
            )
            gaussians.add_densification_stats(
                render_pkg["viewspace_points"], render_pkg["visibility_filter"]
            )

            if (
                iteration > opt.densify_from_iter
                and iteration % opt.densification_interval == 0
            ):
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold,
                    opt.density_min_threshold,
                    opt.max_screen_size,
                    max_scale,
                    opt.max_num_gaussians,
                    densify_scale_threshold,
                    bbox,
                )
                if gaussians.get_density.shape[0] == 0:
                    raise ValueError(
                        "No Gaussian left. Change adaptive control hyperparameters!"
                    )

            if hasattr(opt, "opacity_reset_interval"):
                if (
                    iteration % opt.opacity_reset_interval == 0
                    or iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()
            elif hasattr(opt, "densification_interval"):
                if (
                    iteration % opt.densification_interval == 0
                    or iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

        # Save checkpoints and models
        if iteration in saving_iterations or iteration == opt.iterations:
            tqdm.write(f"[ITER {iteration}] Saving Gaussians")
            scene.save(iteration, queryfunc)

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
                "standard": f"{loss['standard'].item():.1e}",
                "distill": f"{loss['distill'].item():.1e}",
                "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                "distill_w": f"{progressive_distill.get_weight(iteration):.3f}",
            }

            if (
                use_improvements
                and view_sampler is not None
                and iteration >= view_sampler.warmup_iters
            ):
                stats = view_sampler.get_importance_stats()
                postfix["imp_max"] = f"{stats['max_importance']:.3f}"

            progress_bar.set_postfix(postfix)
            progress_bar.update(10)

        if iteration == opt.iterations:
            progress_bar.close()

    print(f"Training with distillation completed!")
    if teacher is not None:
        print(f"  Teacher model: {cnn_model_path}")
    print(
        f"  Final distillation weight: {progressive_distill.get_weight(opt.iterations):.3f}"
    )


def main():
    """
    Main function for training with knowledge distillation, with resume support.
    """
    # Set up command line argument parser
    parser = ArgumentParser(
        description="Training with Knowledge Distillation (with resume support)"
    )

    # Add parameters from ModelParams, OptimizationParams, PipelineParams
    # These classes add their arguments to the parser
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # Additional training arguments
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
        "--quiet", "-q", action="store_true", help="Suppress console output"
    )
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[7_000, 30_000],
        help="Iterations at which to run testing",
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[7_000, 30_000],
        help="Iterations at which to save models",
    )
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=[],
        help="Iterations at which to save checkpoints",
    )

    # Distillation-specific arguments
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

    # Resume argument
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from (e.g., chkpnt9000.pth)",
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
    model_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

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
    print("Training with Knowledge Distillation")
    print("=" * 70)
    print(f"Data source: {model_args.source_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Iterations: {opt_args.iterations}")
    print(f"Knowledge distillation: {'ENABLED' if use_distillation else 'DISABLED'}")
    if use_distillation:
        print(f"  Teacher model: {args.cnn_model}")
    print(f"Improved training: {'ENABLED' if use_improvements else 'DISABLED'}")

    # Resume logic
    start_iteration = 0
    if args.resume_checkpoint and osp.exists(args.resume_checkpoint):
        print(f"\nResuming from checkpoint: {args.resume_checkpoint}")
        # Load checkpoint
        checkpoint = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        )
        if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
            model_state, loaded_iter = checkpoint
            start_iteration = loaded_iter
            print(f"  Loaded iteration: {start_iteration}")
        else:
            print(f"Warning: Invalid checkpoint format, starting from iteration 0")
    else:
        if args.resume_checkpoint:
            print(
                f"Warning: Checkpoint not found at {args.resume_checkpoint}, starting from iteration 0"
            )

    print("=" * 70)

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

    # Resume from checkpoint if provided
    if args.resume_checkpoint and osp.exists(args.resume_checkpoint):
        print(f"Restoring Gaussian model from checkpoint...")
        gaussians.restore(model_state, opt_args)
        print(f"Model restored successfully.")
    else:
        # Initialize Gaussian parameters from dataset
        initialize_gaussian(gaussians, model_args, None)

    scene.gaussians = gaussians

    # Run training
    train_with_distillation(
        model_path=args.output_dir,
        iteration=start_iteration,
        testing_iterations=args.test_iterations,
        scene=scene,
        gaussians=gaussians,
        opt=opt_args,
        pipe=pipe_args,
        cnn_model_path=args.cnn_model if use_distillation else None,
        distillation_config=distill_config,
        use_improvements=use_improvements,
        output_path=args.output_dir,
        checkpoint_iterations=args.checkpoint_iterations,
        saving_iterations=args.save_iterations,
    )

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
