#!/usr/bin/env python3
"""
Test device fix for checkpoint loading.
"""

import torch
import sys

sys.path.append(".")

# Load checkpoint
checkpoint_path = "distill_student_10k_seashell_tv/ckpt/chkpnt9000.pth"
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
    model_state, iteration = checkpoint
    print(f"Iteration: {iteration}")
    print(f"Model state type: {type(model_state)}")
    print(f"Model state length: {len(model_state)}")

    # Check each tensor's device
    for i, tensor in enumerate(model_state):
        if isinstance(tensor, torch.Tensor):
            print(
                f"  Tensor {i}: shape={tensor.shape}, device={tensor.device}, dtype={tensor.dtype}"
            )
        elif tensor is None:
            print(f"  Element {i}: None")
        else:
            print(f"  Element {i}: type={type(tensor)}")

    # Check if optimizer state dict is included
    if len(model_state) >= 8:
        opt_dict = model_state[7]
        print(f"\nOptimizer state dict type: {type(opt_dict)}")
        if isinstance(opt_dict, dict):
            print(f"Optimizer keys: {list(opt_dict.keys())}")

            # Check device of optimizer tensors
            for key, value in opt_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, device={value.device}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} items")
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, torch.Tensor):
                            print(
                                f"    {subkey}: shape={subvalue.shape}, device={subvalue.device}"
                            )

print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
