#!/usr/bin/env python3
"""
Launch script for walnut 50-view baseline training.
This will start the training in the background.
"""

import os
import sys
import subprocess
import time


def main():
    os.chdir("E:\\r2_gaussian")

    print("Starting Walnut 50-view baseline training...")
    print("This will run for approximately 30-40 minutes.")
    print("Training output will be saved to: experiments/logs/baseline/")
    print("=" * 60)

    # Create log directory if it doesn't exist
    os.makedirs("experiments/logs/baseline", exist_ok=True)

    # Command to run training
    cmd = [
        sys.executable,
        "train_with_distillation.py",
        "--config",
        "experiments/baseline/baseline_50view_walnut.yaml",
        "--output_dir",
        "baseline_50view_walnut",
        "--no_distill",
    ]

    # Open log file
    log_file = open("experiments/logs/baseline/walnut_50view_baseline.log", "a")

    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        print(f"Training started with PID: {process.pid}")
        print(f"Log file: experiments/logs/baseline/walnut_50view_baseline.log")
        print("=" * 60)
        print("Training is running in the background.")
        print("You can check progress with:")
        print("  dir E:\\r2_gaussian\\baseline_50view_walnut\\ckpt")
        print("  or check the log file")
        print("=" * 60)

        # Wait a bit to ensure it starts properly
        time.sleep(5)

        # Check if process is still running
        if process.poll() is None:
            print("✅ Training process is running successfully!")
            print("You can close this window - training will continue in background.")
        else:
            print("❌ Training process terminated early.")
            return 1

    except Exception as e:
        print(f"Error starting training: {e}")
        return 1
    finally:
        log_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
