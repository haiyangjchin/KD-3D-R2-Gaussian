import subprocess
import sys
import os
import time

# Change to project directory
os.chdir("E:\\r2_gaussian")

print("Starting 50-view pine baseline training in background...")
print("This script will start the training and then exit.")
print("Training will continue even after this script exits.")
print()
print("Configuration:")
print("- Data: 50 views (cone_ntrain_50_angle_360/pine)")
print("- Iterations: 10000")
print("- Checkpoints: 2500, 5000, 7500, 10000")
print("- Method: Original R2-Gaussian (no distillation)")
print()

# Command to run training
cmd = [
    "D:\\Anaconda\\python.exe",
    "E:\\r2_gaussian\\train.py",
    "--config",
    "E:\\r2_gaussian\\experiments\\baseline\\baseline_50view_pine.yaml",
]

# Output log file
log_file = "E:\\r2_gaussian\\experiments\\baseline\\baseline_50view_pine.log"

try:
    # Open log file for writing
    with open(log_file, "w") as f:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS,
        )

    print(f"Training started successfully!")
    print(f"Process ID: {process.pid}")
    print(f"Log file: {log_file}")
    print(
        f"Output directory: E:\\r2_gaussian\\experiments\\baseline\\baseline_50view_pine"
    )
    print()
    print("This script will now exit. Training continues in background.")
    print("You can check the log file for progress.")

    # Give it a moment to start
    time.sleep(2)

except Exception as e:
    print(f"Error starting training: {e}")
    import traceback

    traceback.print_exc()

print("Script exiting...")
sys.exit(0)
