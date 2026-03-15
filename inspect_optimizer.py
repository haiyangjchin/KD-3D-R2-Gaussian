#!/usr/bin/env python3
"""
Inspect optimizer state dict from checkpoint.
"""

import torch
import sys

sys.path.append(".")

checkpoint_path = "distill_student_10k_seashell_tv/ckpt/chkpnt9000.pth"
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
    model_state, iteration = checkpoint
    print(f"Iteration: {iteration}")

    # Unpack model_state
    (
        _xyz,
        _scaling,
        _rotation,
        _density,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale,
        scale_bound,
    ) = model_state

    print(f"Optimizer state dict type: {type(opt_dict)}")
    print(f"Keys: {list(opt_dict.keys())}")

    # Inspect param_groups
    param_groups = opt_dict["param_groups"]
    print(f"\nNumber of param_groups: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        print(
            f"  Group {i}: name={group.get('name', 'N/A')}, lr={group.get('lr')}, params len={len(group['params'])}"
        )

    # Inspect state
    state = opt_dict["state"]
    print(f"\nNumber of parameter states: {len(state)}")
    for param_idx, state_dict in state.items():
        print(f"  Param {param_idx}: keys={list(state_dict.keys())}")
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                print(
                    f"    {key}: shape={tensor.shape}, device={tensor.device}, dtype={tensor.dtype}"
                )
            else:
                print(f"    {key}: {type(tensor)}")
