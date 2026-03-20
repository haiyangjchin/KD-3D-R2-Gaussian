#!/usr/bin/env python3
"""
Quick script to check CNN model file.
Usage: python quick_check_model.py ./cnn_output_final/best_model.pth
"""

import torch
import sys

def quick_check(model_path):
    print(f"Quick check of: {model_path}")
    print("=" * 50)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Basic info
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Keys: {list(checkpoint.keys())}")
        
        # Check for common keys
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"Loss: {checkpoint['loss']:.6f}")
        if 'args' in checkpoint:
            print(f"Training args: {checkpoint['args'].get('model_size', 'unknown')} model")
    
    # Check if it's a state dict or full checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\nModel state dict keys: {len(state_dict)} parameters")
        
        # Show first few keys
        print("\nFirst 10 parameter names:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            tensor = state_dict[key]
            print(f"  {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    
    print("\n" + "=" * 50)
    print("Quick Python one-liners to check:")
    print(f"  torch.load('{model_path}', map_location='cpu').keys()")
    print(f"  len(torch.load('{model_path}', map_location='cpu')['model_state_dict'])")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_check_model.py <model_path>")
        sys.exit(1)
    quick_check(sys.argv[1])