#!/usr/bin/env python3
"""
Script to run walnut distillation training in background.
This script will keep running until training completes or is interrupted.
"""

import os
import sys
import subprocess
import time
from datetime import datetime


def main():
    print("=" * 70)
    print("Walnut Distillation Training - Background Mode")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Change to project directory
    project_dir = "E:\\r2_gaussian"
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")

    # Training configuration
    config_file = "experiments/distill/distill_student_10k_walnut_tv.yaml"
    output_dir = "experiments/distill/distill_student_10k_walnut_tv_new"
    checkpoint = f"{output_dir}/ckpt/chkpnt3000.pth"
    log_file = "experiments/logs/distill/walnut_distill_new_teacher.log"

    print(f"Config: {config_file}")
    print(f"Output: {output_dir}")
    print(f"Resume from: {checkpoint}")
    print(f"Log file: {log_file}")
    print("=" * 70)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        return 1

    # Build command
    cmd = [
        sys.executable,
        "train_with_distillation.py",
        "--config",
        config_file,
        "--output_dir",
        output_dir,
        "--resume",
        checkpoint,
    ]

    print("Command:", " ".join(cmd))
    print("=" * 70)
    print("Training started. Press Ctrl+C to stop.")
    print("=" * 70)

    try:
        # Run training with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Real-time output to console and file
        with open(log_file, "a") as log:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()

        # Wait for process to complete
        return_code = process.wait()
        print(f"\nTraining completed with return code: {return_code}")
        return return_code

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if process:
            process.terminate()
            process.wait()
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
