#!/usr/bin/env python3
"""
Script to run CNN training with configuration from YAML file.
"""

import os
import sys
import yaml
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run CNN training from YAML config")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print command without executing")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build command line arguments
    cmd_args = ["python", "cnn_pretrain.py"]
    
    for key, value in config.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd_args.append(f"--{key}")
        elif isinstance(value, list):
            # For lists like proj_size: [64, 64], need to pass multiple times
            for item in value:
                cmd_args.append(f"--{key}")
                cmd_args.append(str(item))
        else:
            cmd_args.append(f"--{key}")
            cmd_args.append(str(value))
    
    print("Command to execute:")
    print(" ".join(cmd_args))
    
    if not args.dry_run:
        print("\nStarting training...")
        # Set environment variable for CUDA extensions if needed
        env = os.environ.copy()
        # Add torch DLL directory to PATH for Windows
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib):
            # For Windows, add to DLL search path via os.add_dll_directory
            # This will be done in cnn_pretrain.py
            pass
        
        # Run the command
        result = subprocess.run(cmd_args, env=env)
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            sys.exit(result.returncode)
        else:
            print("Training completed successfully")

if __name__ == "__main__":
    main()