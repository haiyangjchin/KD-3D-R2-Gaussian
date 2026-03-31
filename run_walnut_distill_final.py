#!/usr/bin/env python3
"""
Final 1000 iterations for walnut distillation training.
From iteration 9000 to 10000.
"""

import os
import sys
import subprocess

os.chdir("E:\\r2_gaussian")

cmd = [
    sys.executable,
    "train_with_distillation.py",
    "--config",
    "experiments/distill/distill_student_10k_walnut_tv.yaml",
    "--output_dir",
    "experiments/distill/distill_student_10k_walnut_tv_new",
    "--resume",
    "experiments/distill/distill_student_10k_walnut_tv_new/ckpt/chkpnt9000.pth",
]

print("=" * 70)
print("Final Stage: Iterations 9000 → 10000")
print("Starting from checkpoint: chkpnt9000.pth")
print("=" * 70)

subprocess.run(cmd)
