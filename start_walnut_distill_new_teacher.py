#!/usr/bin/env python3
"""
Script to start walnut distillation training with new walnut 75-view teacher model.
This script can be run in terminal to avoid timeout issues.
"""

import os
import sys
import subprocess

# Change to project directory
os.chdir("E:\\r2_gaussian")

# Build the command
command = [
    sys.executable,
    "train_with_distillation.py",
    "--config",
    "experiments/distill/distill_student_10k_walnut_tv.yaml",
    "--output_dir",
    "experiments/distill/distill_student_10k_walnut_tv_new",
    "--resume",
    "experiments/distill/distill_student_10k_walnut_tv_new/ckpt/chkpnt3000.pth",
]

print("=" * 70)
print("Starting Walnut Distillation Training with New Teacher Model")
print("=" * 70)
print("Command:", " ".join(command))
print("Output directory: experiments/distill/distill_student_10k_walnut_tv_new")
print("Log file: experiments/logs/distill/walnut_distill_new_teacher.log")
print("=" * 70)
print("Press Ctrl+C to stop training (it will continue from last checkpoint)")
print("=" * 70)

try:
    # Run the command
    subprocess.run(command, check=True)
except KeyboardInterrupt:
    print("\nTraining stopped by user. Last checkpoint will be saved.")
except Exception as e:
    print(f"\nError occurred: {e}")
    sys.exit(1)
