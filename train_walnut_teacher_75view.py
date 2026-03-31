#!/usr/bin/env python3
"""
Training script for walnut CNN teacher model (75-view version).
Uses 75-view data to train a teacher that can generalize to 25, 50, 75 views.
"""

import os
import sys
import yaml

# Add torch DLL directory to DLL search path for CUDA extensions
import torch

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

# Add submodule paths for CUDA extensions
sys.path.append("./")
sys.path.append("./r2_gaussian/submodules/simple-knn")
sys.path.append("./r2_gaussian/submodules/xray-gaussian-rasterization-voxelization")

# Now import and run training
import cnn_pretrain

# Load configuration
config_path = "cnn_teacher_walnut_75view_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# Convert config to argparse namespace
class Args:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


args = Args(config)

# Print configuration
print("=" * 60)
print("Walnut CNN Teacher Model Training (75-view)")
print("=" * 60)
print("Training configuration:")
for key, value in vars(args).items():
    print(f"  {key}: {value}")
print("=" * 60)

# Run training
cnn_pretrain.main(args)
