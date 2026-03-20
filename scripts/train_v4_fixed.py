#!/usr/bin/env python3
"""
Fixed training script for v4 CNN model.
Uses subprocess to call cnn_pretrain.py with proper command-line arguments.
"""

import os
import sys
import yaml
import subprocess
import argparse

def build_command_from_config(config_path):
    """Build command line arguments from YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cmd = [sys.executable, "cnn_pretrain.py"]
    
    for key, value in config.items():
        if value is None:
            continue
        
        # Handle boolean flags
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        # Handle lists (proj_size, vol_size)
        elif isinstance(value, list):
            cmd.append(f"--{key}")
            for item in value:
                cmd.append(str(item))
        # Handle other values
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description="Run CNN training from YAML config")
    parser.add_argument("--config", type=str, default="cnn_teacher_50epoch_v4_config.yaml",
                       help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print command without executing")
    args = parser.parse_args()
    
    cmd = build_command_from_config(args.config)
    
    print("Command to execute:")
    print(" ".join(cmd))
    
    if args.dry_run:
        return
    
    print("\nStarting training...")
    
    # Set environment for CUDA extensions
    env = os.environ.copy()
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        print(f"Added torch DLL directory: {torch_lib}")
    
    cuda_bin = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin"
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)
        print(f"Added CUDA DLL directory: {cuda_bin}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print("Training completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()