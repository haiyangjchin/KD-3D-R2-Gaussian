import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob


def analyze_error_distribution(base_path="./distill_student_10k_pine"):
    """
    分析误差分布，解释为什么RMSE上升而SSIM改善
    """
    # 查找所有迭代目录
    iter_dirs = glob.glob(os.path.join(base_path, "point_cloud", "iteration_*"))
    iter_dirs.sort(key=lambda x: int(os.path.basename(x).replace("iteration_", "")))

    print("=" * 80)
    print("误差分布分析")
    print("=" * 80)

    results = []

    for iter_dir in iter_dirs:
        iter_name = os.path.basename(iter_dir)
        vol_pred_path = os.path.join(iter_dir, "vol_pred.npy")
        vol_gt_path = os.path.join(iter_dir, "vol_gt.npy")

        if not (os.path.exists(vol_pred_path) and os.path.exists(vol_gt_path)):
            continue

        # 加载体积数据
        vol_pred = np.load(vol_pred_path)
        vol_gt = np.load(vol_gt_path)

        # 归一化
        if vol_gt.max() > 1.0:
            vol_gt_norm = vol_gt / vol_gt.max()
            vol_pred_norm = vol_pred / vol_gt.max()
        else:
            vol_gt_norm = vol_gt
            vol_pred_norm = vol_pred

        # 计算绝对误差
        abs_error = np.abs(vol_pred_norm - vol_gt_norm)

        # 计算各种统计量
        mae = np.mean(abs_error)
        mse = np.mean(abs_error**2)
        rmse = np.sqrt(mse)

        # 误差分布统计
        error_stats = {
            "iteration": iter_name,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "max_error": abs_error.max(),
            "p95_error": np.percentile(abs_error, 95),  # 95分位误差
            "p99_error": np.percentile(abs_error, 99),  # 99分位误差
            "std_error": abs_error.std(),  # 误差标准差
            "skewness": calculate_skewness(abs_error),  # 偏度
            "error_above_0.01": np.mean(abs_error > 0.01),  # 大误差比例
            "error_above_0.05": np.mean(abs_error > 0.05),  # 很大误差比例
        }

        results.append(error_stats)

        print(f"\n迭代 {iter_name}:")
        print(f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        print(f"  最大误差: {abs_error.max():.6f}")
        print(f"  95分位误差: {np.percentile(abs_error, 95):.6f}")
        print(f"  99分位误差: {np.percentile(abs_error, 99):.6f}")
        print(f"  误差标准差: {abs_error.std():.6f}")
        print(f"  误差>0.01的比例: {np.mean(abs_error > 0.01):.4%}")
        print(f"  误差>0.05的比例: {np.mean(abs_error > 0.05):.4%}")

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 绘制误差分布变化图
    plot_error_analysis(df)

    # 分析关键发现
    analyze_findings(df)

    return df


def calculate_skewness(data):
    """计算偏度"""
    data_flat = data.flatten()
    mean = np.mean(data_flat)
    std = np.std(data_flat)
    if std == 0:
        return 0
    skew = np.mean(((data_flat - mean) / std) ** 3)
    return skew


def plot_error_analysis(df):
    """绘制误差分析图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 提取迭代次数作为x轴
    iterations = df["iteration"].str.replace("iteration_", "").astype(int)

    # 1. RMSE和MAE对比
    ax = axes[0, 0]
    ax.plot(iterations, df["rmse"], "b-o", label="RMSE", linewidth=2, markersize=6)
    ax.plot(iterations, df["mae"], "r-s", label="MAE", linewidth=2, markersize=6)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Error")
    ax.set_title("RMSE vs MAE over Iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 最大误差和95分位误差
    ax = axes[0, 1]
    ax.plot(
        iterations, df["max_error"], "b-o", label="Max Error", linewidth=2, markersize=6
    )
    ax.plot(
        iterations,
        df["p95_error"],
        "r-s",
        label="95th Percentile",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Error")
    ax.set_title("Max and 95th Percentile Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 大误差比例
    ax = axes[0, 2]
    ax.plot(
        iterations,
        df["error_above_0.01"] * 100,
        "b-o",
        label="Error > 0.01",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        iterations,
        df["error_above_0.05"] * 100,
        "r-s",
        label="Error > 0.05",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Percentage of Large Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 误差标准差
    ax = axes[1, 0]
    ax.plot(iterations, df["std_error"], "b-o", linewidth=2, markersize=6)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Error Standard Deviation")
    ax.grid(True, alpha=0.3)

    # 5. 误差偏度
    ax = axes[1, 1]
    ax.plot(iterations, df["skewness"], "b-o", linewidth=2, markersize=6)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Skewness")
    ax.set_title("Error Distribution Skewness")
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 6. RMSE-MAE比值（异常值敏感度）
    ax = axes[1, 2]
    rmse_mae_ratio = df["rmse"] / df["mae"]
    ax.plot(iterations, rmse_mae_ratio, "b-o", linewidth=2, markersize=6)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("RMSE/MAE Ratio")
    ax.set_title("RMSE/MAE Ratio (Outlier Sensitivity)")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Error Distribution Analysis for Knowledge Distillation", fontsize=16, y=0.98
    )
    plt.tight_layout()

    output_dir = "./error_analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "error_distribution_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n误差分析图已保存: {output_path}")


def analyze_findings(df):
    """分析关键发现并给出解释"""
    print("\n" + "=" * 80)
    print("关键发现和解释")
    print("=" * 80)

    # 计算变化
    early_iter = df.iloc[0]  # 2500次迭代
    best_rmse_iter = df.loc[df["rmse"].idxmin()]  # RMSE最低的迭代
    final_iter = df.iloc[-1]  # 10000次迭代

    print(f"早期迭代 ({early_iter['iteration']}):")
    print(f"  RMSE: {early_iter['rmse']:.6f}, MAE: {early_iter['mae']:.6f}")
    print(f"  最大误差: {early_iter['max_error']:.6f}")
    print(f"  大误差(>0.01)比例: {early_iter['error_above_0.01']:.4%}")

    print(f"\n最佳RMSE迭代 ({best_rmse_iter['iteration']}):")
    print(f"  RMSE: {best_rmse_iter['rmse']:.6f}, MAE: {best_rmse_iter['mae']:.6f}")
    print(f"  最大误差: {best_rmse_iter['max_error']:.6f}")
    print(f"  大误差(>0.01)比例: {best_rmse_iter['error_above_0.01']:.4%}")

    print(f"\n最终迭代 ({final_iter['iteration']}):")
    print(f"  RMSE: {final_iter['rmse']:.6f}, MAE: {final_iter['mae']:.6f}")
    print(f"  最大误差: {final_iter['max_error']:.6f}")
    print(f"  大误差(>0.01)比例: {final_iter['error_above_0.01']:.4%}")

    # 计算变化百分比
    rmse_change = (
        (final_iter["rmse"] - best_rmse_iter["rmse"]) / best_rmse_iter["rmse"] * 100
    )
    mae_change = (
        (final_iter["mae"] - best_rmse_iter["mae"]) / best_rmse_iter["mae"] * 100
    )
    max_error_change = (
        (final_iter["max_error"] - best_rmse_iter["max_error"])
        / best_rmse_iter["max_error"]
        * 100
    )
    large_error_change = (
        (final_iter["error_above_0.01"] - best_rmse_iter["error_above_0.01"])
        / best_rmse_iter["error_above_0.01"]
        * 100
    )

    print(f"\n从最佳RMSE迭代到最终迭代的变化:")
    print(f"  RMSE增加: {rmse_change:+.2f}%")
    print(f"  MAE增加: {mae_change:+.2f}%")
    print(f"  最大误差增加: {max_error_change:+.2f}%")
    print(f"  大误差比例变化: {large_error_change:+.2f}%")

    # 解释可能的原因
    print(f"\n" + "-" * 80)
    print("可能的原因分析:")
    print("-" * 80)

    if final_iter["max_error"] > best_rmse_iter["max_error"]:
        print("1. **存在异常大误差**: 最终迭代出现了更大的最大误差")
        print("   - RMSE对异常值敏感，少数大误差会显著提高RMSE")
        print("   - MAE受异常值影响较小，因此增加幅度较小")

    if final_iter["error_above_0.01"] > best_rmse_iter["error_above_0.01"]:
        print("2. **大误差数量增加**: 最终迭代有更多的大误差像素")
        print("   - 虽然整体结构可能更好（SSIM提高），但局部区域可能产生更大误差")

    if final_iter["std_error"] > best_rmse_iter["std_error"]:
        print("3. **误差分布更分散**: 误差的标准差增加")
        print("   - 误差分布更加不均匀，某些区域误差变大")

    rmse_mae_ratio_final = final_iter["rmse"] / final_iter["mae"]
    rmse_mae_ratio_best = best_rmse_iter["rmse"] / best_rmse_iter["mae"]

    if rmse_mae_ratio_final > rmse_mae_ratio_best:
        print("4. **误差分布右偏**: RMSE/MAE比值增加表明误差分布更右偏")
        print("   - 存在更多的大误差（正偏度），导致RMSE相对MAE增加更多")

    print("\n5. **SSIM与RMSE的不同敏感性**:")
    print("   - SSIM关注结构相似性，对亮度/对比度变化不敏感")
    print("   - RMSE关注像素级精度，对任何偏差都敏感")
    print("   - 模型可能在保持整体结构（高SSIM）的同时，在某些区域产生较大偏差")

    print("\n6. **可能的过拟合迹象**:")
    if (
        final_iter["mae"] > best_rmse_iter["mae"]
        and final_iter["rmse"] > best_rmse_iter["rmse"]
    ):
        print("   - 验证集误差（MAE/RMSE）在后期增加，是过拟合的典型表现")
        print("   - 模型可能过度适应训练数据的特定模式")
    else:
        print("   - MAE变化不大，可能不是典型过拟合")
        print("   - 可能是模型在优化过程中产生的trade-off")

    # 保存分析报告
    save_analysis_report(df, early_iter, best_rmse_iter, final_iter)


def save_analysis_report(df, early_iter, best_rmse_iter, final_iter):
    """保存分析报告"""
    output_dir = "./error_analysis"
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "error_analysis_report.md")

    with open(report_path, "w") as f:
        f.write("# 误差分布分析报告\n\n")
        f.write("## 问题描述\n\n")
        f.write("知识蒸馏训练中观察到：RMSE在5000次迭代后上升，而SSIM持续改善。\n\n")

        f.write("## 数据分析\n\n")
        f.write("### 误差统计汇总\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        f.write("### 关键指标变化\n\n")
        f.write(
            f"- **最佳RMSE迭代**: {best_rmse_iter['iteration']} (RMSE={best_rmse_iter['rmse']:.6f})\n"
        )
        f.write(
            f"- **最终迭代**: {final_iter['iteration']} (RMSE={final_iter['rmse']:.6f})\n"
        )
        f.write(
            f"- **RMSE增加**: {((final_iter['rmse'] - best_rmse_iter['rmse']) / best_rmse_iter['rmse'] * 100):+.2f}%\n"
        )
        f.write(
            f"- **MAE增加**: {((final_iter['mae'] - best_rmse_iter['mae']) / best_rmse_iter['mae'] * 100):+.2f}%\n"
        )
        f.write(
            f"- **最大误差增加**: {((final_iter['max_error'] - best_rmse_iter['max_error']) / best_rmse_iter['max_error'] * 100):+.2f}%\n"
        )
        f.write(
            f"- **大误差比例增加**: {((final_iter['error_above_0.01'] - best_rmse_iter['error_above_0.01']) / best_rmse_iter['error_above_0.01'] * 100):+.2f}%\n\n"
        )

        f.write("## 原因分析\n\n")
        f.write(
            "1. **RMSE对异常值敏感**: RMSE对较大误差进行平方惩罚，少数大误差会显著提高RMSE\n"
        )
        f.write("2. **误差分布变化**: 后期迭代可能产生更多的大误差像素\n")
        f.write("3. **SSIM-RMSE权衡**: SSIM关注结构相似性，RMSE关注像素级精度\n")
        f.write("4. **可能的局部过拟合**: 模型在某些区域过度适应，产生较大偏差\n\n")

        f.write("## 建议\n\n")
        f.write("1. 检查误差分布图，识别大误差区域\n")
        f.write("2. 考虑早停策略，在RMSE开始上升时停止训练\n")
        f.write("3. 增加正则化，防止过拟合\n")
        f.write("4. 分析SSIM和RMSE的权衡，根据应用需求选择合适指标\n")

    print(f"\n分析报告已保存: {report_path}")


if __name__ == "__main__":
    # 设置matplotlib参数
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 150,
        }
    )

    df = analyze_error_distribution()
