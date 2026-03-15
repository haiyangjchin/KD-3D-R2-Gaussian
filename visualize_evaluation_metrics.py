#!/usr/bin/env python3
"""
Visualize evaluation metrics for pine, seashell, and walnut datasets.
Generates separate line charts for each dataset showing MAE, RMSE, PSNR, SSIM across iterations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for better visual appearance
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = [14, 10]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16

# File paths
data_files = {
    "pine": "distill_evaluation_results_tv.csv",
    "seashell": "distill_evaluation_results_seashell_tv.csv",
    "walnut": "distill_evaluation_results_walnut_tv.csv",
}


def load_and_prepare_data(file_path):
    """Load CSV file and prepare data for visualization."""
    df = pd.read_csv(file_path)

    # Extract iteration number from 'iteration_2500' format
    df["iteration_num"] = df["iteration"].str.replace("iteration_", "").astype(int)
    df = df.sort_values("iteration_num")

    return df


def create_individual_plot(df, dataset_name, output_path):
    """Create a 2x2 subplot figure with all four metrics for a single dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Evaluation Metrics - {dataset_name.capitalize()} Dataset",
        fontsize=16,
        fontweight="bold",
    )

    # Colors for each metric
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Plot 1: MAE
    ax1 = axes[0, 0]
    ax1.plot(
        df["iteration_num"],
        df["mae"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        color=colors[0],
        label="MAE",
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("MAE (Mean Absolute Error)")
    ax1.set_title("Mean Absolute Error (MAE)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add value annotations
    for i, (x, y) in enumerate(zip(df["iteration_num"], df["mae"])):
        ax1.annotate(
            f"{y:.6f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    # Plot 2: RMSE
    ax2 = axes[0, 1]
    ax2.plot(
        df["iteration_num"],
        df["rmse"],
        marker="s",
        linewidth=2.5,
        markersize=8,
        color=colors[1],
        label="RMSE",
    )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("RMSE (Root Mean Square Error)")
    ax2.set_title("Root Mean Square Error (RMSE)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    for i, (x, y) in enumerate(zip(df["iteration_num"], df["rmse"])):
        ax2.annotate(
            f"{y:.6f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    # Plot 3: PSNR
    ax3 = axes[1, 0]
    ax3.plot(
        df["iteration_num"],
        df["psnr"],
        marker="^",
        linewidth=2.5,
        markersize=8,
        color=colors[2],
        label="PSNR",
    )
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("PSNR (dB)")
    ax3.set_title("Peak Signal-to-Noise Ratio (PSNR)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    for i, (x, y) in enumerate(zip(df["iteration_num"], df["psnr"])):
        ax3.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    # Plot 4: SSIM
    ax4 = axes[1, 1]
    ax4.plot(
        df["iteration_num"],
        df["ssim"],
        marker="d",
        linewidth=2.5,
        markersize=8,
        color=colors[3],
        label="SSIM",
    )
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("SSIM")
    ax4.set_title("Structural Similarity Index (SSIM)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    for i, (x, y) in enumerate(zip(df["iteration_num"], df["ssim"])):
        ax4.annotate(
            f"{y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close()


def create_comparison_plot(all_data, output_path):
    """Create a comparison plot showing all datasets together."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Evaluation Metrics Comparison Across Datasets", fontsize=18, fontweight="bold"
    )

    colors = {"pine": "#1f77b4", "seashell": "#ff7f0e", "walnut": "#2ca02c"}
    markers = {"pine": "o", "seashell": "s", "walnut": "^"}

    # Plot 1: MAE Comparison
    ax1 = axes[0, 0]
    for dataset_name, df in all_data.items():
        ax1.plot(
            df["iteration_num"],
            df["mae"],
            marker=markers[dataset_name],
            linewidth=2.5,
            markersize=8,
            color=colors[dataset_name],
            label=dataset_name.capitalize(),
        )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("MAE")
    ax1.set_title("MAE Comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: RMSE Comparison
    ax2 = axes[0, 1]
    for dataset_name, df in all_data.items():
        ax2.plot(
            df["iteration_num"],
            df["rmse"],
            marker=markers[dataset_name],
            linewidth=2.5,
            markersize=8,
            color=colors[dataset_name],
            label=dataset_name.capitalize(),
        )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE Comparison")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: PSNR Comparison
    ax3 = axes[1, 0]
    for dataset_name, df in all_data.items():
        ax3.plot(
            df["iteration_num"],
            df["psnr"],
            marker=markers[dataset_name],
            linewidth=2.5,
            markersize=8,
            color=colors[dataset_name],
            label=dataset_name.capitalize(),
        )
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("PSNR (dB)")
    ax3.set_title("PSNR Comparison")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: SSIM Comparison
    ax4 = axes[1, 1]
    for dataset_name, df in all_data.items():
        ax4.plot(
            df["iteration_num"],
            df["ssim"],
            marker=markers[dataset_name],
            linewidth=2.5,
            markersize=8,
            color=colors[dataset_name],
            label=dataset_name.capitalize(),
        )
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("SSIM")
    ax4.set_title("SSIM Comparison")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def main():
    """Main function to generate all visualizations."""
    print("Loading evaluation data...")

    # Load all datasets
    all_data = {}
    for dataset_name, file_path in data_files.items():
        if os.path.exists(file_path):
            df = load_and_prepare_data(file_path)
            all_data[dataset_name] = df
            print(f"  Loaded {dataset_name}: {len(df)} iterations")
        else:
            print(f"  Warning: File not found - {file_path}")

    if not all_data:
        print("No data files found. Please check file paths.")
        return

    # Create individual plots for each dataset
    print("\nGenerating individual plots...")
    for dataset_name, df in all_data.items():
        output_path = f"evaluation_metrics_{dataset_name}.png"
        create_individual_plot(df, dataset_name, output_path)

    # Create comparison plot
    print("\nGenerating comparison plot...")
    create_comparison_plot(all_data, "evaluation_metrics_comparison.png")

    print("\nVisualization completed successfully!")
    print("Generated files:")
    print("  - evaluation_metrics_pine.png")
    print("  - evaluation_metrics_seashell.png")
    print("  - evaluation_metrics_walnut.png")
    print("  - evaluation_metrics_comparison.png")


if __name__ == "__main__":
    main()
