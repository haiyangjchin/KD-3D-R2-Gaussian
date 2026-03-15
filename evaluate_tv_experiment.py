import sys

sys.path.append(".")
from evaluate_distill_results import evaluate_all_iterations, save_results_to_csv

# 评估TV增强实验
print("评估3D TV正则化增强实验 (distill_student_10k_pine_tv)")
results = evaluate_all_iterations(base_path="./distill_student_10k_pine_tv")

# 保存结果到单独的文件
if results:
    save_results_to_csv(results, output_path="distill_evaluation_results_tv.csv")
    print("\n评估完成！结果已保存到 distill_evaluation_results_tv.csv")
