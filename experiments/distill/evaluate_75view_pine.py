#!/usr/bin/env python3
"""
评估75视图pine数据集的蒸馏训练结果
"""

import sys
import os

# 添加项目路径
sys.path.append("E:\\r2_gaussian")

# 导入评估函数
from scripts.evaluate_distill_results import (
    evaluate_all_iterations,
    save_results_to_csv,
)


def main():
    # 设置基础路径
    base_path = "E:\\r2_gaussian\\experiments\\distill\\distill_75view_pine"

    print("=" * 80)
    print("评估75视图pine数据集的蒸馏训练结果")
    print(f"基础路径: {base_path}")
    print("=" * 80)

    # 评估所有迭代
    results = evaluate_all_iterations(base_path)

    # 保存结果
    if results:
        output_csv = "E:\\r2_gaussian\\docs\\distill_75view_pine_results.csv"
        output_md = "E:\\r2_gaussian\\docs\\distill_75view_pine_results.md"

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        # 保存CSV
        save_results_to_csv(results, output_csv)

        # 保存Markdown表格
        import pandas as pd

        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = "iteration"
        df.reset_index(inplace=True)

        with open(output_md, "w", encoding="utf-8") as f:
            f.write("# 75视图Pine数据集蒸馏训练评估结果\n\n")
            f.write("## 实验配置\n")
            f.write("- **数据集**: pine (75视图)\n")
            f.write("- **方法**: R²-Gaussian + CNN蒸馏\n")
            f.write("- **教师模型**: checkpoint_epoch_034.pth\n")
            f.write("- **迭代次数**: 10000\n")
            f.write("- **检查点**: 2500, 5000, 5500, 6000, 6500, 7000, 7500, 10000\n\n")
            f.write("## 评估结果\n\n")
            f.write(df.to_markdown(index=False))
            f.write(f"\n\n结果保存于: {output_csv}\n")

        print(f"\n结果已保存到:")
        print(f"  CSV: {output_csv}")
        print(f"  Markdown: {output_md}")
    else:
        print("未找到评估结果")


if __name__ == "__main__":
    main()
