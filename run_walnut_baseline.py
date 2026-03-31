#!/usr/bin/env python3
"""
Script to run walnut baseline training for 50-view and 75-view.
This script runs training sequentially.
"""

import os
import sys
import subprocess
import time


def run_training(config_file, output_dir):
    """Run training for a specific configuration"""
    print(f"\n{'=' * 70}")
    print(f"Starting training: {config_file}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 70}\n")

    cmd = [
        sys.executable,
        "train_with_distillation.py",
        "--config",
        config_file,
        "--output_dir",
        output_dir,
        "--no_distill",  # Run as baseline (no distillation)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Training completed: {config_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {config_file}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted: {config_file}")
        return False


def main():
    os.chdir("E:\\r2_gaussian")

    print("=" * 70)
    print("Walnut Baseline Training - 50-view and 75-view")
    print("=" * 70)

    # Training configurations
    experiments = [
        {
            "config": "experiments/baseline/baseline_50view_walnut.yaml",
            "output": "baseline_50view_walnut",
            "description": "Walnut 50-view baseline",
        },
        {
            "config": "experiments/baseline/baseline_75view_walnut.yaml",
            "output": "baseline_75view_walnut",
            "description": "Walnut 75-view baseline",
        },
    ]

    results = []

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp['description']}")

        success = run_training(exp["config"], exp["output"])
        results.append({"experiment": exp["description"], "success": success})

        if not success:
            print(f"\n⚠️ Do you want to continue with next experiment? (y/n)")
            response = input().strip().lower()
            if response != "y":
                break

    # Summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    for result in results:
        status = "✅ Completed" if result["success"] else "❌ Failed"
        print(f"{result['experiment']}: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
