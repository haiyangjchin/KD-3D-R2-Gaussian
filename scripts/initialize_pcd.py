import os
import numpy as np
import tigre.algorithms as algs
import open3d as o3d
import sys
import argparse
import os.path as osp
import json
import pickle
from tqdm import trange
import copy
import torch

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry_tigre, recon_volume
from r2_gaussian.arguments import ParamGroup, ModelParams, PipelineParams
from r2_gaussian.utils.plot_utils import show_one_volume, show_two_volume
from r2_gaussian.gaussian import GaussianModel, query, initialize_gaussian
from r2_gaussian.utils.image_utils import metric_vol
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.general_utils import t2a

# CNN model imports
try:
    from models.ct_unet3d import CTUNet3D
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    print("Warning: CNN model modules not available. CNN initialization will not work.")

np.random.seed(0)


def reconstruct_with_cnn(projs, angles, scanner_cfg, cnn_model_path, target_size=(64, 64, 64)):
    """
    Reconstruct 3D volume using trained CNN model.
    
    Args:
        projs: Projection images [num_angles, height, width]
        angles: Projection angles [num_angles]
        scanner_cfg: Scanner configuration dictionary
        cnn_model_path: Path to trained CNN model checkpoint
        target_size: Target volume size for CNN (depth, height, width)
    
    Returns:
        Reconstructed volume in original scanner coordinates (original size)
    """
    # Dynamic import to avoid dependency issues
    try:
        from models.ct_unet3d import CTUNet3D
    except ImportError:
        raise ImportError("CNN model modules not available. Please ensure models.ct_unet3d is accessible.")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CNN model
    checkpoint = torch.load(cnn_model_path, map_location=device)
    
    # Determine model configuration from checkpoint
    if 'args' in checkpoint:
        # Checkpoint saved from training script
        checkpoint_args = checkpoint['args']
        model_size = checkpoint_args.get('model_size', 'medium')
        target_depth = target_size[0]
    else:
        # Default configuration
        model_size = 'medium'
        target_depth = target_size[0]
    
    # Create model
    if model_size == 'small':
        features = [16, 32, 64, 128]
    elif model_size == 'medium':
        features = [32, 64, 128, 256]
    elif model_size == 'large':
        features = [64, 128, 256, 512]
    else:
        features = [32, 64, 128, 256]
    
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
    
    # Prepare input projections
    # Resize projections to target size (height, width)
    original_height, original_width = projs.shape[1], projs.shape[2]
    target_height, target_width = target_size[1], target_size[2]
    
    # Convert to torch tensor and resize
    projs_tensor = torch.from_numpy(projs).float().unsqueeze(1)  # [N, 1, H, W]
    if (original_height, original_width) != (target_height, target_width):
        projs_resized = torch.nn.functional.interpolate(
            projs_tensor, 
            size=(target_height, target_width), 
            mode='bilinear',
            align_corners=False
        )
    else:
        projs_resized = projs_tensor
    
    # Stack projections along depth dimension (angles as depth)
    # Input shape: [batch, channels, depth, height, width]
    projs_input = projs_resized.unsqueeze(0)  # [1, 1, num_angles, H, W]
    projs_input = projs_input.to(device)
    
    # Forward pass
    with torch.no_grad():
        vol_pred = model(projs_input)
    
    # vol_pred shape: [1, 1, target_depth, target_height, target_width]
    vol_pred = vol_pred.squeeze(0).squeeze(0)  # [D, H, W]
    
    # Convert to numpy
    vol_pred_np = vol_pred.cpu().numpy()
    
    # Resize back to original volume size if needed
    original_depth, original_height, original_width = (
        scanner_cfg['nVoxel'][2], scanner_cfg['nVoxel'][1], scanner_cfg['nVoxel'][0]
    )
    
    if vol_pred_np.shape != (original_depth, original_height, original_width):
        # Use trilinear interpolation to resize to original dimensions
        vol_pred_tensor = torch.from_numpy(vol_pred_np).float().unsqueeze(0).unsqueeze(0)
        vol_resized = torch.nn.functional.interpolate(
            vol_pred_tensor,
            size=(original_depth, original_height, original_width),
            mode='trilinear',
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()
        return vol_resized
    else:
        return vol_pred_np


class InitParams(ParamGroup):
    def __init__(self, parser):
        self.recon_method = "fdk"
        self.n_points = 50000
        self.density_thresh = 0.05
        self.density_rescale = 0.15
        self.random_density_max = 1.0  # Parameters for random mode
        self.cnn_model_path = None  # Path to trained CNN model
        self.cnn_target_size = (64, 64, 64)  # Target size for CNN input/output
        self.cnn_use_train_projections = True  # Use train or test projections
        super().__init__(parser, "Initialization Parameters")


def init_pcd(
    projs,
    angles,
    geo,
    scanner_cfg,
    args: InitParams,
    save_path,
):
    "Initialize Gaussians."
    recon_method = args.recon_method
    n_points = args.n_points
    assert recon_method in ["random", "fdk", "cnn"], "--recon_method not supported."
    if recon_method == "random":
        print(f"Initialize random point clouds.")
        sampled_positions = np.array(scanner_cfg["offOrigin"])[None, ...] + np.array(
            scanner_cfg["sVoxel"]
        )[None, ...] * (np.random.rand(n_points, 3) - 0.5)
        sampled_densities = (
            np.random.rand(
                n_points,
            )
            * args.random_density_max
        )
    elif recon_method == "cnn":
        # CNN-based initialization
        print(f"Initialize point clouds with CNN reconstruction.")
        if not CNN_AVAILABLE:
            raise ImportError("CNN modules not available. Cannot use CNN initialization.")
        
        # Load CNN model
        cnn_model_path = args.cnn_model_path
        if cnn_model_path is None:
            # Try to find default model path
            default_path = osp.join(osp.dirname(__file__), "cnn_output_final", "best_model.pth")
            if osp.exists(default_path):
                cnn_model_path = default_path
            else:
                raise ValueError("CNN model path not specified and default not found.")
        
        print(f"Loading CNN model from {cnn_model_path}")
        vol = reconstruct_with_cnn(projs, angles, scanner_cfg, cnn_model_path, args.cnn_target_size)
        
        # Continue with sampling (same as FDK)
        density_mask = vol > args.density_thresh
        valid_indices = np.argwhere(density_mask)
        offOrigin = np.array(scanner_cfg["offOrigin"])
        dVoxel = np.array(scanner_cfg["dVoxel"])
        sVoxel = np.array(scanner_cfg["sVoxel"])

        assert (
            valid_indices.shape[0] >= n_points
        ), "Valid voxels less than target number of sampling. Check threshold"

        sampled_indices = valid_indices[
            np.random.choice(len(valid_indices), n_points, replace=False)
        ]
        sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
        sampled_densities = vol[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]
        sampled_densities = sampled_densities * args.density_rescale
    else:
        # Use traditional algorithms for initialization
        print(
            f"Initialize point clouds with the volume reconstructed from {recon_method}."
        )
        vol = recon_volume(projs, angles, copy.deepcopy(geo), recon_method)
        # show_one_volume(vol)

        density_mask = vol > args.density_thresh
        valid_indices = np.argwhere(density_mask)
        offOrigin = np.array(scanner_cfg["offOrigin"])
        dVoxel = np.array(scanner_cfg["dVoxel"])
        sVoxel = np.array(scanner_cfg["sVoxel"])

        assert (
            valid_indices.shape[0] >= n_points
        ), "Valid voxels less than target number of sampling. Check threshold"

        sampled_indices = valid_indices[
            np.random.choice(len(valid_indices), n_points, replace=False)
        ]
        sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
        sampled_densities = vol[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]
        sampled_densities = sampled_densities * args.density_rescale

    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    np.save(save_path, out)
    print(f"Initialization saved in {save_path}.")


def main(
    args, init_args: InitParams, model_args: ModelParams, pipe_args: PipelineParams
):
    # Read scene
    data_path = args.data
    model_args.source_path = data_path
    scene = Scene(model_args, False)  #! Here we scale the scene to [-1,1]^3 space.
    train_cameras = scene.getTrainCameras()
    projs_train = np.concatenate(
        [t2a(cam.original_image) for cam in train_cameras], axis=0
    )
    angles_train = np.stack([t2a(cam.angle) for cam in train_cameras], axis=0)
    scanner_cfg = scene.scanner_cfg
    geo = get_geometry_tigre(scanner_cfg)

    save_path = args.output
    if not save_path:
        if osp.exists(osp.join(data_path, "meta_data.json")):
            save_path = osp.join(data_path, "init_" + osp.basename(data_path) + ".npy")
        elif data_path.split(".")[-1] in ["pickle", "pkl"]:
            save_path = osp.join(
                osp.dirname(data_path),
                "init_" + osp.basename(data_path).split(".")[0] + ".npy",
            )
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."

    assert not osp.exists(
        save_path
    ), f"Initialization file {save_path} exists! Delete it first."
    os.makedirs(osp.dirname(save_path), exist_ok=True)

    init_pcd(
        projs=projs_train,
        angles=angles_train,
        geo=geo,
        scanner_cfg=scanner_cfg,
        args=init_args,
        save_path=save_path,
    )

    # Evaluate using ground truth volume (for debug only)
    if args.evaluate:
        with torch.no_grad():
            model_args.ply_path = save_path
            scale_bound = None
            volume_to_world = max(scanner_cfg["sVoxel"])
            if model_args.scale_min and model_args.scale_max:
                scale_bound = (
                    np.array([model_args.scale_min, model_args.scale_max])
                    * volume_to_world
                )
            gaussians = GaussianModel(scale_bound)
            initialize_gaussian(gaussians, model_args, None)
            vol_pred = query(
                gaussians,
                scanner_cfg["offOrigin"],
                scanner_cfg["nVoxel"],
                scanner_cfg["sVoxel"],
                pipe_args,
            )["vol"]
            vol_gt = scene.vol_gt.cuda()
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            print(f"3D PSNR for initial Gaussians: {psnr_3d}")
            # show_two_volume(vol_gt, vol_pred, title1="gt", title2="init")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate initialization parameters")
    init_parser = InitParams(parser)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--data", type=str, help="Path to data.")
    parser.add_argument("--output", default=None, type=str, help="Path to output.")
    parser.add_argument("--evaluate", default=False, action="store_true", help="Add this flag to evaluate quality (given GT volume, for debug only)")
    # fmt: on

    args = parser.parse_args()
    main(args, init_parser.extract(args), lp.extract(args), pp.extract(args))
