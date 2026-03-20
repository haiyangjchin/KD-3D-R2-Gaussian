#!/usr/bin/env python3
"""
Script to run baseline comparison experiments for R²-Gaussian vs distillation.
Runs experiments with different lambda_tv values and collects PSNR/SSIM metrics.
"""

import os
import subprocess
import sys
import time
import yaml
import json
import shutil
from datetime import datetime


def run_experiment(config_path, script_type="distill", log_suffix=""):
    """
    Run a single experiment with given configuration.

    Args:
        config_path: Path to YAML configuration file
        script_type: Type of script to run - "original" for train.py or "distill" for train_with_distillation.py
        log_suffix: Suffix for log file name
    """
    # Extract experiment name from config path
    exp_name = os.path.splitext(os.path.basename(config_path))[0]

    # Load configuration to get model_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = config.get("model_path", "./output")

    # Determine which script to use
    if script_type == "original":
        script = "train.py"
        cmd = ["python", script, "--config", config_path]
    elif script_type == "distill":
        script = "train_with_distillation.py"
        cmd = ["python", script, "--config", config_path, "--output_dir", model_path]
    else:
        raise ValueError(f"Unknown script_type: {script_type}")

    # Create log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{exp_name}_{timestamp}.log"
    if log_suffix:
        log_file = f"{exp_name}_{log_suffix}.log"

    print(f"\n{'=' * 80}")
    print(f"Starting experiment: {exp_name}")
    print(f"Script: {script}")
    print(f"Configuration: {config_path}")
    print(f"Log file: {log_file}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}")

    # Check if output directory already exists
    if os.path.exists(model_path):
        print(f"Warning: Output directory {model_path} already exists.")
        # Create backup directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{model_path}_backup_{timestamp}"
        print(f"Backing up existing directory to {backup_path}")
        shutil.move(model_path, backup_path)
        print(f"Original directory backed up, proceeding with experiment.")

    # Run the command
    try:
        with open(log_file, "w") as log:
            # Write header to log
            log.write(f"Experiment: {exp_name}\n")
            log.write(f"Start time: {datetime.now()}\n")
            log.write(f"Command: {' '.join(cmd)}\n")
            log.write(f"{'=' * 80}\n\n")
            log.flush()

            # Start process
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, text=True, bufsize=1
            )

            print(f"Process started with PID: {proc.pid}")
            print(f"Training in progress... Check {log_file} for details.")

            # Wait for completion
            return_code = proc.wait()

            if return_code == 0:
                print(f"Experiment {exp_name} completed successfully!")
                return True
            else:
                print(f"Experiment {exp_name} failed with return code: {return_code}")
                return False

    except KeyboardInterrupt:
        print(f"\nExperiment {exp_name} interrupted by user.")
        return False
    except Exception as e:
        print(f"Error running experiment {exp_name}: {e}")
        return False


def evaluate_experiment(config_path):
    """
    Evaluate an experiment after training completes.
    Extracts PSNR/SSIM metrics from evaluation files.
    """
    exp_name = os.path.splitext(os.path.basename(config_path))[0]

    # Load config to get model_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = config.get("model_path", "./output")

    # Check for evaluation results
    eval_dir = os.path.join(model_path, "eval")
    if not os.path.exists(eval_dir):
        print(f"No evaluation directory found at {eval_dir}")
        return None

    # Look for the final iteration evaluation
    final_iter_dir = os.path.join(eval_dir, "iteration_10000")
    if not os.path.exists(final_iter_dir):
        # Try to find the latest iteration
        iter_dirs = [
            d
            for d in os.listdir(eval_dir)
            if d.startswith("iteration_") or d.startswith("iter_")
        ]
        print(f"Debug: Found iteration directories: {iter_dirs}")
        if not iter_dirs:
            print(f"No iteration directories found in {eval_dir}")
            return None
        # Sort by iteration number
        iter_dirs.sort(key=lambda x: int(x.split("_")[1]))
        final_iter_dir = os.path.join(eval_dir, iter_dirs[-1])

    # Look for 3D evaluation file
    eval_3d_file = os.path.join(final_iter_dir, "eval3d.yml")
    if os.path.exists(eval_3d_file):
        with open(eval_3d_file, "r") as f:
            eval_data = yaml.safe_load(f)

        # Extract metrics
        metrics = {
            "experiment": exp_name,
            "iteration": os.path.basename(final_iter_dir),
            "psnr_3d": eval_data.get("psnr_3d", 0),
            "ssim_3d": eval_data.get("ssim_3d", 0),
        }

        # Also look for 2D evaluation
        eval_2d_file = os.path.join(final_iter_dir, "eval2d_render_test.yml")
        if os.path.exists(eval_2d_file):
            with open(eval_2d_file, "r") as f:
                eval_2d_data = yaml.safe_load(f)
            metrics["psnr_2d"] = eval_2d_data.get("psnr_2d", 0)
            metrics["ssim_2d"] = eval_2d_data.get("ssim_2d", 0)

        return metrics
    else:
        print(f"No eval3d.yml found in {final_iter_dir}")
        return None


def main():
    """Main function to run all experiments."""
    print("R2-Gaussian Baseline Comparison Experiments")
    print("=" * 80)

    # Define experiments to run - all baseline and distillation experiments
    experiments = [
        {
            "name": "Original R2-Gaussian (lambda_tv=0.05)",
            "config": "baseline_r2_gaussian_tv0.05.yaml",
            "script": "original",
            "skip_training": True,  # Skip training since already partially trained
        },
        {
            "name": "Distillation (lambda_tv=0.01) - Absolute",
            "config": "distill_tv0.01_absolute.yaml",
            "script": "distill",
        },
        {
            "name": "Distillation (lambda_tv=0.05) - Absolute",
            "config": "distill_tv0.05_absolute.yaml",
            "script": "distill",
        },
        {
            "name": "Distillation (λ_tv=0.2) - Absolute",
            "config": "distill_tv0.2_absolute.yaml",
            "script": "distill",
        },
    ]

    results = []

    # Run experiments
    for exp in experiments:
        print(f"\n{'-' * 80}")
        print(f"Experiment: {exp['name']}")
        print(f"{'-' * 80}")

        config_path = exp["config"]
        if not os.path.exists(config_path):
            print(f"Configuration file not found: {config_path}")
            continue

        # Skip training if already done
        if exp.get("skip_training", False):
            print(f"Skipping training (already exists).")
        else:
            # Auto-run without confirmation
            print(f"Starting experiment automatically...")

            # Check if output directory already exists
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_path = config.get("model_path", "./output")
            if os.path.exists(model_path):
                print(f"Warning: Output directory {model_path} already exists.")
                # Create backup directory with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{model_path}_backup_{timestamp}"
                print(f"Backing up existing directory to {backup_path}")
                shutil.move(model_path, backup_path)

            # Run training
            success = run_experiment(
                config_path, exp["script"], log_suffix=exp["name"].replace(" ", "_")
            )
            if not success:
                print(f"Training failed for {exp['name']}")
                continue

        # Evaluate results
        print(f"Evaluating results for {exp['name']}...")
        metrics = evaluate_experiment(config_path)
        if metrics:
            results.append(metrics)
            print(
                f"Metrics collected: PSNR_3D={metrics.get('psnr_3d', 'N/A'):.2f}, SSIM_3D={metrics.get('ssim_3d', 'N/A'):.4f}"
            )
        else:
            print(f"Could not collect metrics for {exp['name']}")

    # Save results
    if results:
        results_file = "baseline_comparison_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Also save as CSV
        csv_file = "baseline_comparison_results.csv"
        with open(csv_file, "w") as f:
            # Header
            f.write("experiment,iteration,psnr_3d,ssim_3d,psnr_2d,ssim_2d\n")
            for r in results:
                f.write(
                    f"{r['experiment']},{r['iteration']},{r.get('psnr_3d', 0):.4f},{r.get('ssim_3d', 0):.6f},{r.get('psnr_2d', 0):.4f},{r.get('ssim_2d', 0):.6f}\n"
                )

        print(f"\n{'=' * 80}")
        print(f"Results saved to:")
        print(f"  JSON: {results_file}")
        print(f"  CSV: {csv_file}")
        print(f"{'=' * 80}")

        # Print summary table
        print("\nSummary of Results:")
        print(
            f"{'Experiment':<40} {'PSNR_3D':>10} {'SSIM_3D':>10} {'PSNR_2D':>10} {'SSIM_2D':>10}"
        )
        print(f"{'-' * 90}")
        for r in results:
            print(
                f"{r['experiment'][:38]:<40} {r.get('psnr_3d', 0):>10.2f} {r.get('ssim_3d', 0):>10.4f} {r.get('psnr_2d', 0):>10.2f} {r.get('ssim_2d', 0):>10.4f}"
            )

    print(f"\nBaseline comparison experiments completed!")


if __name__ == "__main__":
    main()
