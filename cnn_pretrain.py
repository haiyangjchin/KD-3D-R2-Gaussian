#!/usr/bin/env python3
"""
CNN Pre-training Script for CT Reconstruction
Trains a 3D U-Net to reconstruct 3D volumes from 2D projections.
"""

import os
import os.path as osp
import sys
import argparse
import logging
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# TensorBoard is optional
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
import yaml

# Add project root to path
sys.path.append("./")
# Add submodule paths for CUDA extensions
sys.path.append("./r2_gaussian/submodules/simple-knn")
sys.path.append("./r2_gaussian/submodules/xray-gaussian-rasterization-voxelization")

# Add DLL directories for CUDA extensions on Windows
if os.name == 'nt':  # Windows
    # Add torch DLL directory
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        print(f"Added torch DLL directory: {torch_lib}")
    
    # Add CUDA DLL directory
    cuda_bin = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin"
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)
        print(f"Added CUDA DLL directory: {cuda_bin}")
    
    # Add Anaconda DLL directories
    anaconda_dir = os.path.dirname(os.path.dirname(torch.__file__))
    anaconda_lib_bin = os.path.join(anaconda_dir, "Library", "bin")
    anaconda_dlls = os.path.join(anaconda_dir, "DLLs")
    
    for path in [anaconda_lib_bin, anaconda_dlls]:
        if os.path.exists(path):
            os.add_dll_directory(path)
            print(f"Added DLL directory: {path}")
    
    # Add Windows system directories
    sys_dirs = [
        os.environ.get("SystemRoot", "") + "\\System32",
        os.environ.get("SystemRoot", "") + "\\SysWOW64",
    ]
    for path in sys_dirs:
        if os.path.exists(path):
            os.add_dll_directory(path)
            print(f"Added system DLL directory: {path}")

# Import project modules
from r2_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from r2_gaussian.utils.general_utils import t2a
from r2_gaussian.utils.loss_utils import ssim
from models.ct_unet3d import CTUNet3D


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CTReconstructionDataset(Dataset):
    """Dataset for CT reconstruction from 2D projections to 3D volume.
    
    Loads all projections for a scene and stacks them along the depth dimension.
    Supports resizing to target dimensions for efficient training.
    """
    
    def __init__(self, data_path, split='train', target_proj_size=(64, 64), 
                 target_vol_size=(64, 64, 64), max_projections=None, 
                 dataset_type='Blender'):
        """
        Args:
            data_path: Path to data directory or file
            split: 'train' or 'test'
            target_proj_size: Target size for projections (height, width)
            target_vol_size: Target size for volume (depth, height, width)
            max_projections: Maximum number of projections to use (None for all)
            dataset_type: 'Blender' or 'NAF'
        """
        super().__init__()
        
        # Load scene information
        logger.info(f"Loading scene from {data_path} (type: {dataset_type})")
        eval_mode = (split == 'test')
        scene_info = sceneLoadTypeCallbacks[dataset_type](data_path, eval_mode)
        
        # Get cameras for the split
        if split == 'train':
            cameras = scene_info.train_cameras
        else:
            cameras = scene_info.test_cameras
        
        # Get ground truth volume
        self.vol_gt = scene_info.vol.cpu().numpy()  # Convert to numpy for processing
        
        # Store scanner configuration
        self.scanner_cfg = scene_info.scanner_cfg
        
        # Load all projections
        self.projections = []
        self.angles = []
        
        logger.info(f"Loading {len(cameras)} projections for {split} split")
        for cam in cameras:
            # Camera image is already loaded as numpy array
            proj = cam.image
            self.projections.append(proj)
            self.angles.append(cam.angle)
        
        # Convert to numpy arrays
        self.projections = np.stack(self.projections, axis=0)  # [N, H, W]
        self.angles = np.array(self.angles)
        
        # Limit number of projections if specified
        if max_projections is not None and len(self.projections) > max_projections:
            idx = np.linspace(0, len(self.projections) - 1, max_projections, dtype=int)
            self.projections = self.projections[idx]
            self.angles = self.angles[idx]
        
        # Store target sizes
        self.target_proj_size = target_proj_size
        self.target_vol_size = target_vol_size
        
        # Original sizes
        self.orig_proj_shape = self.projections.shape  # [N, H, W]
        self.orig_vol_shape = self.vol_gt.shape  # [D, H, W]
        
        logger.info(f"Original projection shape: {self.orig_proj_shape}")
        logger.info(f"Original volume shape: {self.orig_vol_shape}")
        logger.info(f"Target projection size: {target_proj_size}")
        logger.info(f"Target volume size: {target_vol_size}")
        
        # Store metadata
        self.metadata = {
            'angles': self.angles,
            'scanner_cfg': self.scanner_cfg,
            'orig_proj_shape': self.orig_proj_shape,
            'orig_vol_shape': self.orig_vol_shape,
            'target_proj_size': self.target_proj_size,
            'target_vol_size': self.target_vol_size
        }
    
    def __len__(self):
        return 1  # Each dataset instance contains one scene
    
    def __getitem__(self, idx):
        """Get a single scene's data.
        
        Returns:
            projections: Tensor of shape [1, num_angles, H, W]
            volume_gt: Tensor of shape [1, D, H, W]
        """
        if idx != 0:
            raise IndexError(f"Dataset contains only one scene, index {idx} out of range")
        
        # Resize projections if needed
        proj = self.projections  # [N, H, W]
        if np.shape(proj)[1:] != self.target_proj_size:
            # Simple resize using torch.nn.functional.interpolate
            proj_tensor = torch.from_numpy(proj).float().unsqueeze(1)  # [N, 1, H, W]
            proj_resized = torch.nn.functional.interpolate(
                proj_tensor, 
                size=self.target_proj_size, 
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [N, H, W]
            proj = proj_resized.numpy()
        
        # Resize volume if needed
        vol = self.vol_gt  # [D, H, W]
        if vol.shape != self.target_vol_size:
            vol_tensor = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            vol_resized = torch.nn.functional.interpolate(
                vol_tensor,
                size=self.target_vol_size,
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # [D, H, W]
            vol = vol_resized.numpy()
        
        # Stack projections along depth dimension (angles as depth)
        # Add channel dimension: [1, num_angles, H, W]
        proj_tensor = torch.from_numpy(proj).float().unsqueeze(0)
        
        # Add channel dimension to volume: [1, D, H, W]
        vol_tensor = torch.from_numpy(vol).float().unsqueeze(0)
        
        return proj_tensor, vol_tensor


def create_dataloaders(args):
    """Create train and validation dataloaders."""
    train_dataset = CTReconstructionDataset(
        data_path=args.data_path,
        split='train',
        target_proj_size=args.proj_size,
        target_vol_size=args.vol_size,
        max_projections=args.max_projections,
        dataset_type=args.dataset_type
    )
    
    val_dataset = CTReconstructionDataset(
        data_path=args.data_path,
        split='test',
        target_proj_size=args.proj_size,
        target_vol_size=args.vol_size,
        max_projections=args.max_projections,
        dataset_type=args.dataset_type
    )
    
    # Since each dataset has only one scene, we use batch_size=1
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(args):
    """Create the 3D U-Net model."""
    # Input channels = 1 (projections stacked along depth dimension)
    # Output channels = 1 (density volume)
    
    # Determine feature sizes based on model size
    if args.model_size == 'small':
        features = [16, 32, 64, 128]
    elif args.model_size == 'medium':
        features = [32, 64, 128, 256]
    elif args.model_size == 'large':
        features = [64, 128, 256, 512]
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    model = CTUNet3D(
        in_channels=1,
        out_channels=1,
        features=features,
        use_dropout=args.use_dropout,
        target_depth=args.vol_size[0]
    )
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Feature sizes: {features}")
    
    return model


def compute_loss(pred_vol, gt_vol, args):
    """Compute reconstruction loss."""
    # L1 Loss
    l1_loss_value = torch.nn.functional.l1_loss(pred_vol, gt_vol)
    
    # SSIM Loss (optional)
    if args.use_ssim:
        # Reshape 3D volume to 2D slices for SSIM computation
        # pred_vol shape: [batch, 1, depth, height, width] or [batch, depth, height, width]
        # gt_vol shape: same as pred_vol
        
        # Ensure we have channel dimension
        if pred_vol.dim() == 4:  # [batch, depth, height, width]
            pred_vol_with_channel = pred_vol.unsqueeze(1)  # [batch, 1, depth, height, width]
            gt_vol_with_channel = gt_vol.unsqueeze(1)
        else:
            pred_vol_with_channel = pred_vol
            gt_vol_with_channel = gt_vol
        
        batch_size, channels, depth, height, width = pred_vol_with_channel.shape
        
        # Compute SSIM for each depth slice and average
        ssim_loss_total = 0.0
        for d in range(depth):
            # Extract slice at depth d
            pred_slice = pred_vol_with_channel[:, :, d, :, :]  # [batch, 1, height, width]
            gt_slice = gt_vol_with_channel[:, :, d, :, :]
            
            # Compute SSIM (1.0 - ssim) since we want to minimize the loss
            ssim_value = ssim(pred_slice, gt_slice, window_size=11, size_average=True)
            ssim_loss_total += (1.0 - ssim_value)
        
        # Average across depth slices
        ssim_loss_value = ssim_loss_total / depth
        
        total_loss = l1_loss_value + args.ssim_weight * ssim_loss_value
        loss_dict = {'l1': l1_loss_value.item(), 'ssim': ssim_loss_value.item(), 'total': total_loss.item()}
    else:
        total_loss = l1_loss_value
        loss_dict = {'l1': l1_loss_value.item(), 'total': total_loss.item()}
    
    return total_loss, loss_dict


def train_epoch(model, train_loader, optimizer, device, args, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    for batch_idx, (projections, volumes) in enumerate(progress_bar):
        # Move data to device
        projections = projections.to(device)  # [1, num_angles, H, W]
        volumes = volumes.to(device)         # [1, D, H, W]
        
        # Forward pass
        optimizer.zero_grad()
        pred_volumes = model(projections)
        
        # Compute loss
        loss, loss_dict = compute_loss(pred_volumes, volumes, args)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        epoch_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'L1': f'{loss_dict.get("l1", 0):.4f}'
        })
        
        # Log to tensorboard
        if writer is not None and batch_idx % args.log_interval == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            for key, value in loss_dict.items():
                writer.add_scalar(f'Train/{key.capitalize()}', value, global_step)
    
    avg_loss = epoch_loss / num_batches
    return avg_loss


def validate(model, val_loader, device, args, epoch, writer=None):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for projections, volumes in tqdm(val_loader, desc='Validation'):
            projections = projections.to(device)
            volumes = volumes.to(device)
            
            pred_volumes = model(projections)
            loss, loss_dict = compute_loss(pred_volumes, volumes, args)
            
            val_loss += loss.item()
    
    avg_val_loss = val_loss / num_batches
    
    # Log validation loss
    if writer is not None:
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
    
    return avg_val_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, args, is_best=False):
    """Save model checkpoint with safe file writing."""
    checkpoint_dir = osp.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
        'args': vars(args)
    }
    
    # Save regular checkpoint
    checkpoint_path = osp.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
        # Try alternative: save to temporary file then move
        temp_path = checkpoint_path + '.tmp'
        try:
            torch.save(checkpoint, temp_path)
            import shutil
            shutil.move(temp_path, checkpoint_path)
            logger.info(f"Saved checkpoint via temp file to {checkpoint_path}")
        except Exception as e2:
            logger.error(f"Failed to save checkpoint via temp file: {e2}")
            raise
    
    # Save best model
    if is_best:
        best_path = osp.join(args.output_dir, 'best_model.pth')
        try:
            # Remove existing file if it exists to avoid permission issues
            if osp.exists(best_path):
                try:
                    os.remove(best_path)
                except:
                    pass  # Ignore removal errors
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        except Exception as e:
            logger.error(f"Failed to save best model to {best_path}: {e}")
            # Try alternative: save to temporary file then move
            temp_best = best_path + '.tmp'
            try:
                torch.save(checkpoint, temp_best)
                import shutil
                shutil.move(temp_best, best_path)
                logger.info(f"Saved best model via temp file to {best_path}")
            except Exception as e2:
                logger.error(f"Failed to save best model via temp file: {e2}")
                # Don't raise, just log - training can continue without saving best model


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in the given directory."""
    if not osp.exists(checkpoint_dir):
        return None
    
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
            # Extract epoch number
            try:
                epoch_num = int(f.split('_')[-1].split('.')[0])
                checkpoint_files.append((epoch_num, osp.join(checkpoint_dir, f)))
            except:
                continue
    
    if not checkpoint_files:
        return None
    
    # Sort by epoch number descending
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_files[0][1]  # Return path of latest checkpoint


def main(args):
    """Main training function."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = osp.join(args.output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    logger.info(f"Saved configuration to {config_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(args)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(args).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=args.lr_patience
    )
    
    # Resume from checkpoint if available
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_dir = osp.join(args.output_dir, 'checkpoints')
    
    # First check if --resume_from is specified
    if args.resume_from is not None and osp.exists(args.resume_from):
        try:
            logger.info(f"Loading checkpoint from {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Only load optimizer and scheduler if they exist in checkpoint
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Set start epoch
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                logger.info(f"Resuming from epoch {start_epoch}")
            
            if 'loss' in checkpoint:
                best_val_loss = checkpoint.get('loss', float('inf'))
                logger.info(f"Previous best loss: {best_val_loss:.6f}")
            
            # Restore scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {args.resume_from}: {e}")
    # Otherwise, check for checkpoints in output directory
    elif osp.exists(checkpoint_dir):
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            try:
                logger.info(f"Loading checkpoint from {latest_checkpoint}")
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                best_val_loss = checkpoint.get('loss', float('inf'))
                logger.info(f"Resumed training from epoch {start_epoch}, best loss: {best_val_loss:.6f}")
                
                # Restore scheduler state if available
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
    
    # Setup tensorboard writer
    writer = None
    if args.use_tensorboard:
        if TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            writer = SummaryWriter(log_dir=args.output_dir)
            logger.info("TensorBoard logging enabled")
        else:
            logger.warning("TensorBoard requested but not available. Install with: pip install tensorboard")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args, epoch, writer)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate(model, val_loader, device, args, epoch, writer)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss:.6f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.6f}")
        
        if (epoch + 1) % args.save_interval == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args, is_best)
    
    # Save final model
    final_model_path = osp.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Close tensorboard writer
    if writer is not None:
        writer.close()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Pre-training for CT Reconstruction")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data directory or file')
    parser.add_argument('--dataset_type', type=str, default='Blender',
                       choices=['Blender', 'NAF'],
                       help='Type of dataset')
    parser.add_argument('--proj_size', type=int, nargs=2, default=[64, 64],
                       help='Target projection size (height, width)')
    parser.add_argument('--vol_size', type=int, nargs=3, default=[64, 64, 64],
                       help='Target volume size (depth, height, width)')
    parser.add_argument('--max_projections', type=int, default=None,
                       help='Maximum number of projections to use (default: all)')
    
    # Model parameters
    parser.add_argument('--model_size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Size of the model')
    parser.add_argument('--use_dropout', action='store_true',
                       help='Use dropout for regularization')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--lr_patience', type=int, default=10,
                       help='Patience for LR scheduler')
    parser.add_argument('--use_ssim', action='store_true',
                       help='Use SSIM loss in addition to L1 loss')
    parser.add_argument('--ssim_weight', type=float, default=0.1,
                       help='Weight for SSIM loss')
    
    # Logging and saving
    parser.add_argument('--output_dir', type=str, default='./cnn_pretrain_output',
                       help='Output directory for logs and models')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log to tensorboard every N batches')
    parser.add_argument('--use_tensorboard', action='store_true',
                       help='Use tensorboard for logging')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # Run training
    main(args)