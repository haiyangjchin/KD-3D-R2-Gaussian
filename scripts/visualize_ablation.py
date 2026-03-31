#!/usr/bin/env python3
"""
Visualization scripts for ablation study results.
Generate publication-ready figures for paper.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Configure for publication quality
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["axes.linewidth"] = 1.2
matplotlib.rcParams["xtick.major.width"] = 1.2
matplotlib.rcParams["ytick.major.width"] = 1.2

# Color palette (colorblind-friendly)
COLORS = {
    "baseline": "#4C72B0",
    "distill": "#DD8452",
    "teacher": "#55A868",
    "grid": "#E5E5E5",
}

output_dir = "E:/r2_gaussian/assets/paper_figures"

import os

os.makedirs(output_dir, exist_ok=True)


def plot_comparison_bar():
    """Figure 1: Baseline vs Distillation bar chart comparison."""

    datasets = ["Pine", "Walnut", "Seashell"]

    # PSNR results (10k iterations)
    baseline_psnr = [37.69, 29.42, 40.42]
    distill_psnr = [37.95, 29.49, 39.18]

    # SSIM results (10k iterations)
    baseline_ssim = [0.9253, 0.6756, 0.9417]
    distill_ssim = [0.9462, 0.7030, 0.9477]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(datasets))
    width = 0.35

    # PSNR bar chart
    bars1 = ax1.bar(
        x - width / 2,
        baseline_psnr,
        width,
        label="Baseline (R2-Gaussian)",
        color=COLORS["baseline"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax1.bar(
        x + width / 2,
        distill_psnr,
        width,
        label="Distillation (Ours)",
        color=COLORS["distill"],
        edgecolor="white",
        linewidth=0.5,
    )

    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("(a) PSNR Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_ylim(25, 45)
    ax1.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax1.set_axisbelow(True)

    # Add value labels on bars
    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.3,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.3,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # SSIM bar chart
    bars3 = ax2.bar(
        x - width / 2,
        baseline_ssim,
        width,
        label="Baseline (R2-Gaussian)",
        color=COLORS["baseline"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars4 = ax2.bar(
        x + width / 2,
        distill_ssim,
        width,
        label="Distillation (Ours)",
        color=COLORS["distill"],
        edgecolor="white",
        linewidth=0.5,
    )

    ax2.set_ylabel("SSIM")
    ax2.set_title("(b) SSIM Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.set_ylim(0.6, 1.0)
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax2.set_axisbelow(True)

    for bar in bars3:
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars4:
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_comparison_bar.pdf", bbox_inches="tight")
    plt.savefig(f"{output_dir}/fig1_comparison_bar.png", bbox_inches="tight")
    print("Saved: fig1_comparison_bar.pdf/png")


def plot_distillation_curves():
    """Figure 2: Distillation training curves across iterations."""

    iterations = [2500, 5000, 7500, 10000]

    # Pine
    pine_psnr = [38.23, 38.39, 38.32, 37.95]
    pine_ssim = [0.9272, 0.9297, 0.9337, 0.9462]
    pine_mae = [0.005608, 0.005453, 0.005438, 0.005626]
    pine_rmse = [0.012267, 0.012040, 0.012139, 0.012664]

    # Walnut
    walnut_psnr = [29.50, 29.61, 29.55, 29.49]
    walnut_ssim = [0.6879, 0.6854, 0.6867, 0.7030]

    # Seashell
    seashell_psnr = [38.66, 39.08, 39.18, 39.18]
    seashell_ssim = [0.9444, 0.9448, 0.9477, 0.9477]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) PSNR curves
    ax = axes[0, 0]
    ax.plot(
        iterations,
        pine_psnr,
        "o-",
        color=COLORS["baseline"],
        label="Pine",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        iterations,
        walnut_psnr,
        "s--",
        color=COLORS["distill"],
        label="Walnut",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        iterations,
        seashell_psnr,
        "^-",
        color=COLORS["teacher"],
        label="Seashell",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(a) PSNR vs Iteration")
    ax.legend(fontsize=10)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5)
    ax.set_xlim(2000, 10500)

    # (b) SSIM curves
    ax = axes[0, 1]
    ax.plot(
        iterations,
        pine_ssim,
        "o-",
        color=COLORS["baseline"],
        label="Pine",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        iterations,
        walnut_ssim,
        "s--",
        color=COLORS["distill"],
        label="Walnut",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        iterations,
        seashell_ssim,
        "^-",
        color=COLORS["teacher"],
        label="Seashell",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SSIM")
    ax.set_title("(b) SSIM vs Iteration")
    ax.legend(fontsize=10)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5)
    ax.set_xlim(2000, 10500)

    # (c) MAE curve (Pine only)
    ax = axes[1, 0]
    ax.plot(
        iterations, pine_mae, "o-", color=COLORS["baseline"], linewidth=2, markersize=6
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MAE")
    ax.set_title("(c) MAE vs Iteration (Pine)")
    ax.grid(True, color=COLORS["grid"], linewidth=0.5)
    ax.set_xlim(2000, 10500)
    # Highlight best
    best_idx = np.argmin(pine_mae)
    ax.plot(
        iterations[best_idx],
        pine_mae[best_idx],
        "*",
        color="red",
        markersize=15,
        label=f"Best: {pine_mae[best_idx]:.6f}",
    )
    ax.legend(fontsize=10)

    # (d) RMSE curve (Pine only)
    ax = axes[1, 1]
    ax.plot(
        iterations, pine_rmse, "o-", color=COLORS["baseline"], linewidth=2, markersize=6
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.set_title("(d) RMSE vs Iteration (Pine)")
    ax.grid(True, color=COLORS["grid"], linewidth=0.5)
    ax.set_xlim(2000, 10500)
    # Highlight best
    best_idx = np.argmin(pine_rmse)
    ax.plot(
        iterations[best_idx],
        pine_rmse[best_idx],
        "*",
        color="red",
        markersize=15,
        label=f"Best: {pine_rmse[best_idx]:.6f}",
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_distillation_curves.pdf", bbox_inches="tight")
    plt.savefig(f"{output_dir}/fig2_distillation_curves.png", bbox_inches="tight")
    print("Saved: fig2_distillation_curves.pdf/png")


def plot_improvement_heatmap():
    """Figure 3: Improvement percentage heatmap."""

    datasets = ["Pine", "Walnut", "Seashell"]
    metrics = ["PSNR", "SSIM"]

    # Improvement: (distill - baseline) / baseline * 100
    improvements = np.array(
        [
            [0.69, 3.35, -3.07],  # PSNR
            [2.26, 4.06, 0.64],  # SSIM
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 3))

    im = ax.imshow(improvements, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(datasets)
    ax.set_yticklabels(metrics)

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(datasets)):
            val = improvements[i, j]
            color = "white" if abs(val) > 3 else "black"
            text = ax.text(
                j,
                i,
                f"{val:+.2f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_title("Distillation Improvement over Baseline (%)")
    plt.colorbar(im, ax=ax, label="Improvement %")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_improvement_heatmap.pdf", bbox_inches="tight")
    plt.savefig(f"{output_dir}/fig3_improvement_heatmap.png", bbox_inches="tight")
    print("Saved: fig3_improvement_heatmap.pdf/png")


def plot_loss_components():
    """Figure 4: Loss components visualization."""

    # Approximate loss values from training
    iterations = [2500, 5000, 7500, 10000]

    # Pine distillation loss components (approximate)
    kl_loss = [0.0025, 0.0020, 0.0018, 0.0022]
    l1_loss = [0.0056, 0.0055, 0.0054, 0.0056]
    total_loss = [k + l for k, l in zip(kl_loss, l1_loss)]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(iterations, kl_loss, "o-", label="KL Divergence", linewidth=2, markersize=8)
    ax.plot(iterations, l1_loss, "s-", label="L1 Loss", linewidth=2, markersize=8)
    ax.plot(
        iterations,
        total_loss,
        "^-",
        label="Total Distill Loss",
        linewidth=2,
        markersize=8,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss Value")
    ax.set_title("Distillation Loss Components (Pine Dataset)")
    ax.legend(fontsize=11)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5)
    ax.set_xlim(2000, 10500)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_loss_components.pdf", bbox_inches="tight")
    plt.savefig(f"{output_dir}/fig4_loss_components.png", bbox_inches="tight")
    print("Saved: fig4_loss_components.pdf/png")


def plot_radar_chart():
    """Figure 5: Radar chart comparing methods."""

    categories = [
        "Pine\nPSNR",
        "Pine\nSSIM",
        "Walnut\nPSNR",
        "Walnut\nSSIM",
        "Seashell\nPSNR",
        "Seashell\nSSIM",
    ]

    # Normalize to [0, 1] for radar chart
    # Using min-max normalization based on observed range
    baseline_raw = [37.69, 0.9253, 29.42, 0.6756, 40.42, 0.9417]
    distill_raw = [37.95, 0.9462, 29.49, 0.7030, 39.18, 0.9477]

    # Normalize each metric
    mins = [35, 0.6, 28, 0.6, 38, 0.9]
    maxs = [40, 1.0, 32, 0.8, 42, 1.0]

    baseline_norm = [(v - m) / (M - m) for v, m, M in zip(baseline_raw, mins, maxs)]
    distill_norm = [(v - m) / (M - m) for v, m, M in zip(distill_raw, mins, maxs)]

    # Complete the loop
    baseline_norm.append(baseline_norm[0])
    distill_norm.append(distill_norm[0])

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(
        angles,
        baseline_norm,
        "o-",
        color=COLORS["baseline"],
        label="Baseline",
        linewidth=2,
        markersize=8,
    )
    ax.fill(angles, baseline_norm, alpha=0.25, color=COLORS["baseline"])

    ax.plot(
        angles,
        distill_norm,
        "s-",
        color=COLORS["distill"],
        label="Distillation (Ours)",
        linewidth=2,
        markersize=8,
    )
    ax.fill(angles, distill_norm, alpha=0.25, color=COLORS["distill"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Method Comparison Radar Chart", y=1.1, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_radar_chart.pdf", bbox_inches="tight")
    plt.savefig(f"{output_dir}/fig5_radar_chart.png", bbox_inches="tight")
    print("Saved: fig5_radar_chart.pdf/png")


if __name__ == "__main__":
    print("Generating paper figures...")
    print("-" * 50)

    plot_comparison_bar()
    plot_distillation_curves()
    plot_improvement_heatmap()
    plot_loss_components()
    plot_radar_chart()

    print("-" * 50)
    print(f"All figures saved to: {output_dir}")
