#!/usr/bin/env python3
"""Test script for CNN dataset loading."""

import sys
sys.path.append("./")

import numpy as np
import torch
from cnn_pretrain import CTReconstructionDataset

def test_dataset():
    # Test with a small dataset
    data_path = "./data/real_dataset/cone_ntrain_25_angle_360/pine"
    
    try:
        dataset = CTReconstructionDataset(
            data_path=data_path,
            split='train',
            target_proj_size=(64, 64),
            target_vol_size=(64, 64, 64),
            max_projections=25,
            dataset_type='Blender'
        )
        
        print("Dataset created successfully!")
        print(f"Number of scenes: {len(dataset)}")
        
        # Get one item
        projections, volume = dataset[0]
        
        print(f"Projections shape: {projections.shape}")
        print(f"Volume shape: {volume.shape}")
        print(f"Number of projections: {len(metadata['angles'])}")
        print(f"Projections dtype: {projections.dtype}")
        print(f"Volume dtype: {volume.dtype}")
        
        # Check ranges
        print(f"Projections min/max: {projections.min():.3f}, {projections.max():.3f}")
        print(f"Volume min/max: {volume.min():.3f}, {volume.max():.3f}")
        
        # Test with batch
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        for batch in loader:
            proj_batch, vol_batch, meta_batch = batch
            print(f"Batch projections shape: {proj_batch.shape}")
            print(f"Batch volume shape: {vol_batch.shape}")
            break
            
        print("Test passed!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dataset()