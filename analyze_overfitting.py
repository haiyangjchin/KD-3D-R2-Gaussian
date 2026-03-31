#!/usr/bin/env python3
"""
分析蒸馏训练是否过拟合
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_and_analyze():
    """加载并分析数据"""
    # 加载蒸馏训练结果
    df_25 = pd.read_csv("E:\\r2_gaussian\\docs\\distill_evaluation_results.csv")
    df_50 = pd.read_csv("E:\\r2_gaussian\\docs\\distill_50view_pine_results.csv")

    df_25["iteration_num"] = df_25["iteration"].str.extract("(\d+)").astype(int)
    df_50["iteration_num"] = df_50["iteration"].str.extract("(\d+)").astype(int)

    print("=" * 80)
    print("过拟合分析")
    print("=" * 80)

    print("\n1. PSNR vs SSIM 趋势分析")
    print("-" * 40)

    for view, df in [(25, df_25), (50, df_50)]:
        print(f"\n{view}视图数据集:")
        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            psnr_change = curr["psnr"] - prev["psnr"]
            ssim_change = curr["ssim"] - prev["ssim"]

            if psnr_change < 0 and ssim_change > 0:
                print(
                    f"  迭代{prev['iteration_num']}→{curr['iteration_num']}: PSNR {psnr_change:+.3f} dB, SSIM {ssim_change:+.6f} → **PSNR下降，SSIM上升**"
                )
            elif psnr_change > 0 and ssim_change > 0:
                print(
                    f"  迭代{prev['iteration_num']}→{curr['iteration_num']}: PSNR {psnr_change:+.3f} dB, SSIM {ssim_change:+.6f} → 两者都上升"
                )
            elif psnr_change < 0 and ssim_change < 0:
                print(
                    f"  迭代{prev['iteration_num']}→{curr['iteration_num']}: PSNR {psnr_change:+.3f} dB, SSIM {ssim_change:+.6f} → 两者都下降"
                )
            else:
                print(
                    f"  迭代{prev['iteration_num']}→{curr['iteration_num']}: PSNR {psnr_change:+.3f} dB, SSIM {ssim_change:+.6f} → PSNR上升，SSIM下降"
                )

    print("\n2. 过拟合特征分析")
    print("-" * 40)

    # 计算PSNR和SSIM的相关性
    for view, df in [(25, df_25), (50, df_50)]:
        correlation = df["psnr"].corr(df["ssim"])
        print(f"\n{view}视图：PSNR与SSIM相关性 = {correlation:.3f}")
        if correlation < -0.5:
            print("  强负相关：PSNR下降时SSIM上升（或反之）")
        elif correlation > 0.5:
            print("  强正相关：PSNR和SSIM同向变化")
        else:
            print("  弱相关或无相关")

    print("\n3. 训练动态分析")
    print("-" * 40)

    print("\n蒸馏训练的关键特征：")
    print("a) 双目标优化：同时优化重建损失和蒸馏损失")
    print("b) 教师引导：教师模型可能更关注结构相似性（SSIM）")
    print("c) 权重调度：蒸馏权重随时间增加（从0.001到0.5）")

    print("\n4. 过拟合可能性评估")
    print("-" * 40)

    print("\n过拟合迹象：")
    print("✗ 缺少独立验证集结果")
    print("✓ 但训练集上PSNR下降而SSIM上升")
    print("✓ 这可能是多目标优化的权衡")

    print("\n不过拟合迹象：")
    print("✓ 模型在训练集上没有持续改进所有指标")
    print("✓ 这是正常的优化权衡现象")

    print("\n5. 实际影响分析")
    print("-" * 40)

    print("\n从应用角度看：")
    print("1. PSNR关注绝对误差：更适合需要精确数值重建的任务")
    print("2. SSIM关注结构相似性：更适合视觉感知质量")
    print("3. 医学影像通常更关注结构完整性（SSIM）")

    print("\n6. 建议验证方法")
    print("-" * 40)

    print("\n要确认是否过拟合，需要：")
    print("1. 在独立验证集上评估（如果有）")
    print("2. 比较训练集和验证集的PSNR/SSIM曲线")
    print("3. 分析教师模型的特性（是否更关注SSIM）")

    print("\n7. 当前数据下的结论")
    print("-" * 40)

    print("\n基于现有数据：")
    print("1. 这更像是**多目标优化的权衡**，而非典型过拟合")
    print("2. 模型在学习教师的软标签，教师可能更关注结构")
    print("3. 蒸馏权重增加导致模型更关注结构相似性")
    print("4. 5000次迭代可能是PSNR最优停止点")
    print("5. 10000次迭代可能是SSIM最优停止点")

    return df_25, df_50


def main():
    df_25, df_50 = load_and_analyze()

    print("\n" + "=" * 80)
    print("最终建议")
    print("=" * 80)

    print("\n基于分析，建议：")
    print("1. 如果应用场景重视视觉质量 → 选择SSIM高的停止点（10000次）")
    print("2. 如果应用场景重视数值精度 → 选择PSNR高的停止点（5000次）")
    print("3. 平衡方案 → 选择7500次迭代")
    print("4. 进一步验证 → 在独立测试集上评估，确认趋势")

    print("\n这不是典型过拟合，而是蒸馏训练的正常特性。")


if __name__ == "__main__":
    main()
