"""
在 real_dataset/cone_ntrain_25_angle_360 数据集上测试改进效果
自动运行基准和改进版本的对比训练
"""
import os
import subprocess
import time
import yaml
from pathlib import Path

def run_training(script_name, source_path, output_path, extra_args=""):
    """运行训练并记录时间"""
    cmd = f'python {script_name} -s "{source_path}" -m "{output_path}" {extra_args}'

    print("=" * 80)
    print(f"运行命令: {cmd}")
    print("=" * 80)

    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ 训练失败！返回码: {result.returncode}")
        return None

    print(f"✅ 训练完成，用时: {elapsed_time/60:.2f} 分钟")
    return elapsed_time

def compare_results(baseline_path, improved_path, output_file="comparison_results.txt"):
    """对比两个版本的结果"""
    print("\n" + "=" * 80)
    print("对比训练结果")
    print("=" * 80)

    results = {}

    # 读取两个版本的评估结果
    for name, path in [("baseline", baseline_path), ("improved", improved_path)]:
        eval_path = Path(path) / "eval"

        # 找到最后一次迭代的结果
        iter_dirs = sorted(eval_path.glob("iter_*"))
        if not iter_dirs:
            print(f"⚠️ 未找到 {name} 的评估结果")
            continue

        last_iter = iter_dirs[-1]

        # 读取3D评估结果
        eval3d_file = last_iter / "eval3d.yml"
        if eval3d_file.exists():
            with open(eval3d_file, 'r') as f:
                eval3d = yaml.safe_load(f)
                results[name] = {
                    'psnr_3d': eval3d.get('psnr_3d', 0),
                    'ssim_3d': eval3d.get('ssim_3d', 0),
                }

        # 读取2D测试集结果
        eval2d_test_file = last_iter / "eval2d_render_test.yml"
        if eval2d_test_file.exists():
            with open(eval2d_test_file, 'r') as f:
                eval2d = yaml.safe_load(f)
                results[name].update({
                    'psnr_2d': eval2d.get('psnr_2d', 0),
                    'ssim_2d': eval2d.get('ssim_2d', 0),
                })

    # 打印对比结果
    if 'baseline' in results and 'improved' in results:
        print("\n指标对比:")
        print("-" * 80)
        print(f"{'指标':<15} {'基准版本':<15} {'改进版本':<15} {'提升':<15}")
        print("-" * 80)

        for metric in ['psnr_3d', 'ssim_3d', 'psnr_2d', 'ssim_2d']:
            baseline_val = results['baseline'].get(metric, 0)
            improved_val = results['improved'].get(metric, 0)
            improvement = improved_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0

            print(f"{metric:<15} {baseline_val:<15.4f} {improved_val:<15.4f} {improvement:>+7.4f} ({improvement_pct:>+6.2f}%)")

        print("-" * 80)

        # 保存对比结果到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("训练结果对比 - real_dataset/cone_ntrain_25_angle_360\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'指标':<15} {'基准版本':<15} {'改进版本':<15} {'提升':<15}\n")
            f.write("-" * 80 + "\n")
            for metric in ['psnr_3d', 'ssim_3d', 'psnr_2d', 'ssim_2d']:
                baseline_val = results['baseline'].get(metric, 0)
                improved_val = results['improved'].get(metric, 0)
                improvement = improved_val - baseline_val
                improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                f.write(f"{metric:<15} {baseline_val:<15.4f} {improved_val:<15.4f} {improvement:>+7.4f} ({improvement_pct:>+6.2f}%)\n")

        print(f"\n对比结果已保存到: {output_file}")
    else:
        print("⚠️ 无法对比结果，某些版本的评估数据缺失")

def main():
    # 数据集路径
    data_root = "data/real_dataset/cone_ntrain_25_angle_360"
    output_root = "output/real_dataset/cone_ntrain_25_angle_360"

    # 三个测试案例
    cases = ["pine", "seashell", "walnut"]

    print("\n" + "=" * 80)
    print("开始在 real_dataset/cone_ntrain_25_angle_360 上测试改进效果")
    print("=" * 80)
    print(f"数据集: {data_root}")
    print(f"测试案例: {', '.join(cases)}")
    print(f"输出目录: {output_root}")
    print("=" * 80 + "\n")

    user_input = input("是否继续? (y/n): ")
    if user_input.lower() != 'y':
        print("已取消")
        return

    # 对每个案例进行训练
    for case_name in cases:
        print("\n" + "=" * 80)
        print(f"处理案例: {case_name}")
        print("=" * 80)

        source_path = f"{data_root}/{case_name}"

        # 1. 训练基准版本（原始 train.py）
        print(f"\n[1/2] 训练基准版本 - {case_name}")
        baseline_output = f"{output_root}/baseline/{case_name}"
        baseline_time = run_training(
            "train.py",
            source_path,
            baseline_output
        )

        # 2. 训练改进版本（train_improved.py）
        print(f"\n[2/2] 训练改进版本 - {case_name}")
        improved_output = f"{output_root}/improved/{case_name}"
        improved_time = run_training(
            "train_improved.py",
            source_path,
            improved_output
        )

        # 3. 对比结果
        if baseline_time and improved_time:
            compare_results(
                baseline_output,
                improved_output,
                f"{output_root}/comparison_{case_name}.txt"
            )

            print(f"\n训练时间对比:")
            print(f"  基准版本: {baseline_time/60:.2f} 分钟")
            print(f"  改进版本: {improved_time/60:.2f} 分钟")
            print(f"  时间差异: {(improved_time - baseline_time)/60:+.2f} 分钟")

    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)
    print(f"结果保存在: {output_root}/")
    print("可以查看以下文件:")
    for case_name in cases:
        print(f"  - comparison_{case_name}.txt")

if __name__ == "__main__":
    main()
