#!/usr/bin/env python3
"""
Collect baseline comparison results from existing experiments.
"""

import os
import yaml
import json
import csv
from pathlib import Path


def get_experiment_results(exp_dir, exp_name, lambda_tv):
    """Extract results from experiment directory."""
    results = []

    # Check if directory exists
    if not os.path.exists(exp_dir):
        return results

    # Look for evaluation directories
    eval_dir = os.path.join(exp_dir, "eval")
    if not os.path.exists(eval_dir):
        # Try point_cloud directory for volume results
        point_cloud_dir = os.path.join(exp_dir, "point_cloud")
        if os.path.exists(point_cloud_dir):
            # Use evaluate_distill_results module
            try:
                from evaluate_distill_results import evaluate_all_iterations

                eval_results = evaluate_all_iterations(exp_dir)
                if eval_results:
                    for iter_name, metrics in eval_results.items():
                        iteration = int(iter_name.replace("iteration_", ""))
                        results.append(
                            {
                                "experiment": exp_name,
                                "dataset": get_dataset_from_path(exp_dir),
                                "lambda_tv": lambda_tv,
                                "iteration": iteration,
                                "psnr_3d": metrics.get("psnr", 0),
                                "ssim_3d": metrics.get("ssim", 0),
                                "mae": metrics.get("mae", 0),
                                "rmse": metrics.get("rmse", 0),
                                "type": "distill"
                                if "distill" in exp_name.lower()
                                else "original",
                            }
                        )
            except ImportError:
                print(f"Could not import evaluation module for {exp_dir}")
        return results

    # Iterate through iteration directories
    for iter_dir in os.listdir(eval_dir):
        if iter_dir.startswith("iteration_"):
            iteration = int(iter_dir.split("_")[1])
            iter_path = os.path.join(eval_dir, iter_dir)

            # Look for 3D evaluation file
            eval_3d_file = os.path.join(iter_path, "eval3d.yml")
            if os.path.exists(eval_3d_file):
                with open(eval_3d_file, "r") as f:
                    eval_data = yaml.safe_load(f)

                result = {
                    "experiment": exp_name,
                    "dataset": get_dataset_from_path(exp_dir),
                    "lambda_tv": lambda_tv,
                    "iteration": iteration,
                    "psnr_3d": eval_data.get("psnr_3d", 0),
                    "ssim_3d": eval_data.get("ssim_3d", 0),
                    "type": "distill" if "distill" in exp_name.lower() else "original",
                }

                # Look for 2D evaluation
                eval_2d_file = os.path.join(iter_path, "eval2d_render_test.yml")
                if os.path.exists(eval_2d_file):
                    with open(eval_2d_file, "r") as f:
                        eval_2d_data = yaml.safe_load(f)
                    result["psnr_2d"] = eval_2d_data.get("psnr_2d", 0)
                    result["ssim_2d"] = eval_2d_data.get("ssim_2d", 0)

                results.append(result)

    return results


def get_dataset_from_path(path):
    """Extract dataset name from path."""
    path_str = str(path)
    if "pine" in path_str.lower():
        return "pine"
    elif "seashell" in path_str.lower():
        return "seashell"
    elif "walnut" in path_str.lower():
        return "walnut"
    else:
        # Try to extract from source_path in config
        config_files = [
            f for f in os.listdir(path) if f.endswith(".yaml") or f.endswith(".yml")
        ]
        for config_file in config_files:
            try:
                with open(os.path.join(path, config_file), "r") as f:
                    config = yaml.safe_load(f)
                    source_path = config.get("source_path", "")
                    if "pine" in source_path.lower():
                        return "pine"
                    elif "seashell" in source_path.lower():
                        return "seashell"
                    elif "walnut" in source_path.lower():
                        return "walnut"
            except:
                pass
        return "unknown"


def main():
    """Main function to collect all results."""
    experiments = [
        {
            "name": "Distillation (λ_tv=0.1) - Pine",
            "dir": "./distill_student_10k_pine_tv",
            "lambda_tv": 0.1,
        },
        {
            "name": "Distillation (λ_tv=0.1) - Seashell",
            "dir": "./distill_student_10k_seashell_tv",
            "lambda_tv": 0.1,
        },
        {
            "name": "Distillation (λ_tv=0.1) - Walnut",
            "dir": "./distill_student_10k_walnut_tv",
            "lambda_tv": 0.1,
        },
    ]

    all_results = []

    print("Collecting baseline comparison results...")
    print("=" * 80)

    for exp in experiments:
        print(f"\nProcessing: {exp['name']}")
        results = get_experiment_results(exp["dir"], exp["name"], exp["lambda_tv"])

        if results:
            # Sort by iteration
            results.sort(key=lambda x: x["iteration"])

            # Take the final iteration result
            final_result = results[-1] if results else None
            if final_result:
                print(
                    f"  Final iteration {final_result['iteration']}: "
                    f"PSNR_3D={final_result['psnr_3d']:.2f} dB, "
                    f"SSIM_3D={final_result['ssim_3d']:.4f}"
                )
                all_results.append(final_result)
            else:
                print(f"  No results found")
        else:
            print(f"  No evaluation data found")

    # Also check for existing evaluation CSV files
    csv_files = [
        "distill_evaluation_results_tv.csv",
        "distill_evaluation_results_seashell_tv.csv",
        "distill_evaluation_results_walnut_tv.csv",
    ]

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"\nReading existing results from {csv_file}")
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try to determine dataset from filename
                    dataset = "pine"
                    if "seashell" in csv_file:
                        dataset = "seashell"
                    elif "walnut" in csv_file:
                        dataset = "walnut"

                    all_results.append(
                        {
                            "experiment": f"Distillation (λ_tv=0.1) - {dataset.capitalize()}",
                            "dataset": dataset,
                            "lambda_tv": 0.1,
                            "iteration": int(
                                row.get("iteration", "0").replace("iteration_", "")
                            ),
                            "psnr_3d": float(row.get("psnr", 0)),
                            "ssim_3d": float(row.get("ssim", 0)),
                            "mae": float(row.get("mae", 0)),
                            "rmse": float(row.get("rmse", 0)),
                            "type": "distill",
                        }
                    )

    # Save results
    if all_results:
        # Save as JSON
        with open("baseline_comparison_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Save as CSV
        with open("baseline_comparison_summary.csv", "w", newline="") as f:
            fieldnames = [
                "experiment",
                "dataset",
                "lambda_tv",
                "iteration",
                "psnr_3d",
                "ssim_3d",
                "psnr_2d",
                "ssim_2d",
                "mae",
                "rmse",
                "type",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)

        print(f"\n{'=' * 80}")
        print(f"Results saved to:")
        print(f"  baseline_comparison_summary.json")
        print(f"  baseline_comparison_summary.csv")
        print(f"{'=' * 80}")

        # Print summary table
        print("\nBaseline Comparison Summary (Final Iteration):")
        print(
            f"{'Experiment':<50} {'Dataset':<10} {'λ_tv':<8} {'PSNR_3D':<10} {'SSIM_3D':<10}"
        )
        print(f"{'-' * 90}")
        for result in all_results:
            print(
                f"{result['experiment'][:48]:<50} {result['dataset']:<10} {result['lambda_tv']:<8.2f} {result['psnr_3d']:<10.2f} {result['ssim_3d']:<10.4f}"
            )

        # Note about missing baselines
        print(f"\n{'=' * 80}")
        print(
            "NOTE: Original R²-Gaussian baseline (λ_tv=0.05) results are not available"
        )
        print("due to CUDA extension compilation issues (ModuleNotFoundError: ")
        print("No module named 'xray_gaussian_rasterization_voxelization').")
        print("To complete the baseline comparison, you need to:")
        print(
            "1. Compile the CUDA extensions in r2_gaussian/submodules/xray-gaussian-rasterization-voxelization"
        )
        print("2. Run train.py with λ_tv=0.05 configuration")
        print(f"{'=' * 80}")
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    main()
