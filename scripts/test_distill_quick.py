import subprocess
import sys
import os

# Test command for distillation with quick config
cmd = [
    "python",
    "train_with_distillation.py",
    "--config",
    "distill_tv0.05_quick.yaml",
    "--output_dir",
    "test_distill_tv0.05_quick",
    "--quiet",
]

print(f"Running command: {' '.join(cmd)}")
print(f"Current directory: {os.getcwd()}")

# Run with timeout
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
except subprocess.TimeoutExpired:
    print("Command timed out after 5 minutes")
except Exception as e:
    print(f"Error: {e}")
