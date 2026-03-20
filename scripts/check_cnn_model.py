#!/usr/bin/env python3
"""
Utility script to inspect CNN model checkpoint.
"""

import torch
import sys
import os

def inspect_model_checkpoint(model_path):
    """Inspect the contents of a PyTorch model checkpoint."""
    print(f"Checking model file: {model_path}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Try to load the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Print basic info
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Keys in checkpoint: {list(checkpoint.keys())}")
        
        # Check for common keys
        for key in checkpoint.keys():
            value = checkpoint[key]
            if hasattr(value, 'shape'):
                print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
            elif isinstance(value, (int, float, str)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: type = {type(value)}")
    
    # Try to load as a model
    print("\n" + "=" * 60)
    print("Attempting to load as a model...")
    
    try:
        from models.ct_unet3d import CTUNet3D
        
        # Try different feature sizes based on config
        # Check the config file for the correct model size
        config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            model_size = config.get('model_size', 'small')
            print(f"Model size from config: {model_size}")
            
            if model_size == 'small':
                features = [16, 32, 64, 128]
            elif model_size == 'medium':
                features = [32, 64, 128, 256]
            elif model_size == 'large':
                features = [64, 128, 256, 512]
            else:
                features = [16, 32, 64, 128]
        else:
            print("Config file not found, trying small model size")
            features = [16, 32, 64, 128]
        
        model = CTUNet3D(
            in_channels=1,
            out_channels=1,
            features=features,
            use_dropout=False,
            target_depth=64
        )
        
        # Try to load state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Check for key mismatches
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        print(f"\nModel has {len(model_keys)} parameters")
        print(f"Checkpoint has {len(checkpoint_keys)} parameters")
        
        missing_in_model = checkpoint_keys - model_keys
        missing_in_checkpoint = model_keys - checkpoint_keys
        
        if missing_in_model:
            print(f"\nKeys in checkpoint but not in model: {len(missing_in_model)}")
            for key in sorted(list(missing_in_model))[:10]:
                print(f"  - {key}")
            if len(missing_in_model) > 10:
                print(f"  ... and {len(missing_in_model) - 10} more")
        
        if missing_in_checkpoint:
            print(f"\nKeys in model but not in checkpoint: {len(missing_in_checkpoint)}")
            for key in sorted(list(missing_in_checkpoint))[:10]:
                print(f"  - {key}")
            if len(missing_in_checkpoint) > 10:
                print(f"  ... and {len(missing_in_checkpoint) - 10} more")
        
        # Try to load with strict=False to see what happens
        print("\n" + "=" * 60)
        print("Loading model with strict=False...")
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully (with some mismatches allowed)")
            
            # Print model summary
            print("\n" + "=" * 60)
            print("Model architecture:")
            print(model)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            print(f"Error loading state dict: {e}")
    
    except Exception as e:
        print(f"Error creating or loading model: {e}")
        import traceback
        traceback.print_exc()
    
    # Try a simple torch.load with print_all option
    print("\n" + "=" * 60)
    print("Detailed checkpoint structure:")
    try:
        # Recursively print checkpoint structure
        def print_structure(obj, indent=0, max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                print("  " * indent + "...")
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    print("  " * indent + f"{key}: ", end="")
                    if isinstance(value, (dict, list, tuple)):
                        print()
                        if isinstance(value, dict):
                            print_structure(value, indent+1, max_depth, current_depth+1)
                        elif isinstance(value, (list, tuple)):
                            if len(value) > 0:
                                print_structure(value[0] if isinstance(value[0], (dict, list, tuple)) else type(value[0]), 
                                              indent+1, max_depth, current_depth+1)
                            else:
                                print(f"empty {type(value).__name__}")
                    else:
                        if hasattr(value, 'shape'):
                            print(f"Tensor shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"{type(value).__name__}")
            elif isinstance(obj, (list, tuple)):
                print(f"{type(obj).__name__} of length {len(obj)}")
                if len(obj) > 0:
                    print_structure(obj[0], indent, max_depth, current_depth+1)
            else:
                if hasattr(obj, 'shape'):
                    print(f"Tensor shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"{type(obj).__name__}")
        
        print_structure(checkpoint)
        
    except Exception as e:
        print(f"Error printing structure: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_cnn_model.py <model_path>")
        print("Example: python check_cnn_model.py ./cnn_output_final/best_model.pth")
        sys.exit(1)
    
    model_path = sys.argv[1]
    inspect_model_checkpoint(model_path)

if __name__ == "__main__":
    main()