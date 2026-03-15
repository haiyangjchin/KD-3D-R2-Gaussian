#!/usr/bin/env python3
"""
Start walnut training with monitoring to ensure no interruption.
"""

import subprocess
import os
import sys
import time
import signal


def start_training():
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
        # Write separator
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
        print(f"Process will continue running in background.")
        print(f"To check if it's running: tasklist | findstr {pid}")
        print(f"To monitor logs: tail -f {log_file}")

        # Wait a few seconds to ensure process starts successfully
        time.sleep(5)

        if proc.poll() is None:
            print("Training process is running successfully!")
            return proc
        else:
            print(f"Training process exited with code: {proc.poll()}")
            print("Check the log file for error messages.")
            return None


def monitor():
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        proc = start_training()
        if proc is None:
            retry_count += 1
            print(f"Startup failed, retry {retry_count}/{max_retries}")
            time.sleep(10)
            continue

        # Monitor process
        try:
            while True:
                time.sleep(60)  # Check every minute
                if proc.poll() is not None:
                    exit_code = proc.poll()
                    print(f"Training process exited with code: {exit_code}")
                    print("Checking if training completed normally...")
                    # Check log for completion message
                    with open("distill_10k_walnut_tv_training_new.log", "r") as f:
                        content = f.read()
                        if "Training completed" in content:
                            print("Training completed normally.")
                            return 0
                    print("Training did not complete normally, restarting...")
                    retry_count += 1
                    break
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, terminating training...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 1

    print(f"Failed after {max_retries} retries.")
    return 1


if __name__ == "__main__":
    sys.exit(monitor())
