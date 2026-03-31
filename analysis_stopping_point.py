#!/usr/bin/env python3
"""
分析蒸馏训练的最佳停止点
"""

import pandas as pd
import numpy as np


def load_data():
    """加载25视图和50视图蒸馏实验数据"""
    # 25视图数据
    df_25 = pd.read_csv("E:\\r2_gaussian\\docs\\distill_evaluation_results.csv")
    df_25["iteration_num"] = df_25["iteration"].str.extract("(\d+)").astype(int)
    df_25["view"] = 25

    # 50视图数据
    df_50 = pd.read_csv("E:\\r2_gaussian\\docs\\distill_50view_pine_results.csv")
    df_50["iteration_num"] = df_50["iteration"].str.extract("(\d+)").astype(int)
    df_50["view"] = 50

    return df_25, df_50


def calculate_marginal_gains(df, name):
    """计算边际收益"""
    print(f"\n{'=' * 80}")
    print(f"{name} 蒸馏训练边际收益分析")
    print(f"{'=' * 80}")

    # 按迭代排序
    df_sorted = df.sort_values("iteration_num").reset_index(drop=True)

    # 计算PSNR和SSIM的变化
    df_sorted["psnr_change"] = df_sorted["psnr"].diff()
    df_sorted["ssim_change"] = df_sorted["ssim"].diff()

    # 计算相对变化率（百分比）
    df_sorted["psnr_change_pct"] = df_sorted["psnr"].pct_change() * 100
    df_sorted["ssim_change_pct"] = df_sorted["ssim"].pct_change() * 100

    # 显示结果
    print("\n边际收益（每次迭代间隔）：")
    print(
        f"{'迭代':<15} {'PSNR (dB)':<12} {'PSNR变化':<12} {'PSNR变化%':<12} {'SSIM':<12} {'SSIM变化':<12} {'SSIM变化%':<12}"
    )
    print("-" * 90)

    for _, row in df_sorted.iterrows():
        if pd.isna(row["psnr_change"]):
            psnr_change_str = "-"
            psnr_change_pct_str = "-"
        else:
            psnr_change_str = f"{row['psnr_change']:+.3f}"
            psnr_change_pct_str = f"{row['psnr_change_pct']:+.2f}%"

        if pd.isna(row["ssim_change"]):
            ssim_change_str = "-"
            ssim_change_pct_str = "-"
        else:
            ssim_change_str = f"{row['ssim_change']:+.6f}"
            ssim_change_pct_str = f"{row['ssim_change_pct']:+.2f}%"

        print(
            f"{int(row['iteration_num']):<15} {row['psnr']:<12.4f} {psnr_change_str:<12} {psnr_change_pct_str:<12} {row['ssim']:<12.6f} {ssim_change_str:<12} {ssim_change_pct_str:<12}"
        )

    # 找到PSNR峰值和SSIM峰值
    psnr_peak_idx = df_sorted["psnr"].idxmax()
    ssim_peak_idx = df_sorted["ssim"].idxmax()

    psnr_peak_iter = df_sorted.loc[psnr_peak_idx, "iteration_num"]
    ssim_peak_iter = df_sorted.loc[ssim_peak_idx, "iteration_num"]

    print(
        f"\nPSNR峰值：迭代 {psnr_peak_iter}，PSNR = {df_sorted.loc[psnr_peak_idx, 'psnr']:.4f} dB"
    )
    print(
        f"SSIM峰值：迭代 {ssim_peak_iter}，SSIM = {df_sorted.loc[ssim_peak_idx, 'ssim']:.6f}"
    )

    return df_sorted


def analyze_stopping_point(df_25, df_50):
    """分析最佳停止点"""
    print("\n" + "=" * 80)
    print("最佳停止点分析")
    print("=" * 80)

    # 合并数据
    df_25_sorted = df_25.sort_values("iteration_num")
    df_50_sorted = df_50.sort_values("iteration_num")

    # 计算PSNR和SSIM的改进幅度
    print("\n1. PSNR改进幅度（相对于前一次迭代）：")
    for view, df in [(25, df_25_sorted), (50, df_50_sorted)]:
        print(f"\n{view}视图：")
        for i in range(1, len(df)):
            prev_iter = df.iloc[i - 1]["iteration_num"]
            curr_iter = df.iloc[i]["iteration_num"]
            prev_psnr = df.iloc[i - 1]["psnr"]
            curr_psnr = df.iloc[i]["psnr"]
            improvement = curr_psnr - prev_psnr
            print(
                f"  {prev_iter}→{curr_iter}: PSNR变化 {improvement:+.3f} dB ({improvement / prev_psnr * 100:+.2f}%)"
            )

    print("\n2. SSIM改进幅度（相对于前一次迭代）：")
    for view, df in [(25, df_25_sorted), (50, df_50_sorted)]:
        print(f"\n{view}视图：")
        for i in range(1, len(df)):
            prev_iter = df.iloc[i - 1]["iteration_num"]
            curr_iter = df.iloc[i]["iteration_num"]
            prev_ssim = df.iloc[i - 1]["ssim"]
            curr_ssim = df.iloc[i]["ssim"]
            improvement = curr_ssim - prev_ssim
            print(
                f"  {prev_iter}→{curr_iter}: SSIM变化 {improvement:+.6f} ({improvement / prev_ssim * 100:+.2f}%)"
            )

    # 定义停止标准
    print("\n3. 停止标准分析：")
    print("   a) PSNR停止点：当PSNR开始下降时")
    print("   b) SSIM停止点：当SSIM改善小于阈值时（如0.005）")
    print("   c) 边际收益停止点：当边际收益低于初始收益的10%时")

    # 计算边际收益阈值
    print("\n4. 建议停止点：")

    # 对于25视图
    df_25_sorted = df_25.sort_values("iteration_num").reset_index(drop=True)
    # 找到PSNR开始下降的点
    psnr_declining_25 = None
    for i in range(1, len(df_25_sorted)):
        if df_25_sorted.iloc[i]["psnr"] < df_25_sorted.iloc[i - 1]["psnr"]:
            psnr_declining_25 = df_25_sorted.iloc[i - 1]["iteration_num"]
            break

    # 对于50视图
    df_50_sorted = df_50.sort_values("iteration_num").reset_index(drop=True)
    psnr_declining_50 = None
    for i in range(1, len(df_50_sorted)):
        if df_50_sorted.iloc[i]["psnr"] < df_50_sorted.iloc[i - 1]["psnr"]:
            psnr_declining_50 = df_50_sorted.iloc[i - 1]["iteration_num"]
            break

    print(f"\n25视图：")
    if psnr_declining_25:
        print(
            f"  PSNR在迭代{psnr_declining_25}后开始下降，建议停止点：{psnr_declining_25}"
        )
    else:
        print(f"  PSNR持续上升，建议停止点：{df_25_sorted.iloc[-1]['iteration_num']}")

    # 检查SSIM改善是否小于阈值
    for i in range(1, len(df_25_sorted)):
        improvement = df_25_sorted.iloc[i]["ssim"] - df_25_sorted.iloc[i - 1]["ssim"]
        if improvement < 0.005:
            print(
                f"  SSIM在迭代{df_25_sorted.iloc[i]['iteration_num']}后改善小于0.005，建议停止点：{df_25_sorted.iloc[i - 1]['iteration_num']}"
            )
            break

    print(f"\n50视图：")
    if psnr_declining_50:
        print(
            f"  PSNR在迭代{psnr_declining_50}后开始下降，建议停止点：{psnr_declining_50}"
        )
    else:
        print(f"  PSNR持续上升，建议停止点：{df_50_sorted.iloc[-1]['iteration_num']}")

    for i in range(1, len(df_50_sorted)):
        improvement = df_50_sorted.iloc[i]["ssim"] - df_50_sorted.iloc[i - 1]["ssim"]
        if improvement < 0.005:
            print(
                f"  SSIM在迭代{df_50_sorted.iloc[i]['iteration_num']}后改善小于0.005，建议停止点：{df_50_sorted.iloc[i - 1]['iteration_num']}"
            )
            break


def main():
    print("蒸馏训练最佳停止点分析")
    print("=" * 80)

    # 加载数据
    df_25, df_50 = load_data()

    # 计算边际收益
    df_25_sorted = calculate_marginal_gains(df_25, "25视图")
    df_50_sorted = calculate_marginal_gains(df_50, "50视图")

    # 分析停止点
    analyze_stopping_point(df_25, df_50)

    # 综合建议
    print("\n" + "=" * 80)
    print("综合建议")
    print("=" * 80)

    print("\n基于现有数据，蒸馏训练的最佳停止点建议：")
    print("1. 25视图数据集：")
    print("   - 如果优先PSNR：5000次迭代（PSNR峰值）")
    print("   - 如果优先SSIM：10000次迭代（SSIM持续改善）")
    print("   - 平衡建议：7500次迭代（PSNR接近峰值，SSIM较高）")

    print("\n2. 50视图数据集：")
    print("   - 如果优先PSNR：5000次迭代（PSNR峰值）")
    print("   - 如果优先SSIM：10000次迭代（SSIM持续改善）")
    print("   - 平衡建议：7500次迭代（PSNR接近峰值，SSIM较高）")

    print("\n3. 通用建议：")
    print("   - 蒸馏训练相比基线，收敛速度更快")
    print("   - 5000-7500次迭代是性价比最高的停止点")
    print("   - 超过10000次迭代的边际收益较低")

    print("\n4. 后续验证：")
    print("   - 建议在75视图数据集上验证停止点")
    print("   - 可以尝试早停策略：当PSNR连续3次迭代下降时停止")
    print("   - 结合验证集性能确定最终停止点")


if __name__ == "__main__":
    main()
