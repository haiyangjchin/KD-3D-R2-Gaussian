#!/usr/bin/env python3
"""
Start walnut training in background.
"""

import subprocess
import os
import sys
import time


def main():
    cmd = [
        "python",
        "train_with_distillation.py",
        "--config",
        "distill_student_10k_walnut_tv.yaml",
        "--output_dir",
        "distill_student_10k_walnut_tv",
        "--distill_config",
        "distill_loss_config_10k.json",
    ]

    log_file = "distill_10k_walnut_tv_training_new.log"
    print(f"Starting training command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")

    with open(log_file, "a") as log:
        log.write(f"\n{'=' * 80}\n")
        log.write(f"Training started at {time.ctime()}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write(f"{'=' * 80}\n\n")
        log.flush()

        # Start process
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)

        pid = proc.pid
        with open("walnut_training.pid", "w") as f:
            f.write(str(pid))

        print(f"Training started with PID: {pid}")
        print(f"Process running in background.")
        print(f"To check if it's running: tasklist | findstr {pid}")
        print(f"To monitor logs: tail -f {log_file}")

        # Wait a moment to check if process starts successfully
        time.sleep(5)

        if proc.poll() is None:
            print("Training process is running successfully!")
            return 0
        else:
            print(f"Training process exited with code: {proc.poll()}")
            print("Check the log file for error messages.")
            return 1


if __name__ == "__main__":
    sys.exit(main())
