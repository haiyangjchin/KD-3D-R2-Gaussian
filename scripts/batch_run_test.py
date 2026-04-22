#!/usr/bin/env python3
"""
批量运行 test.py 生成所有模型的体积重建结果
用于后续可视化对比图生成
"""

import os
import subprocess
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# 模型配置列表
# 格式: (dataset, views, method, model_path)
MODELS = [
    # Pine 25 views
    ("pine", 25, "Baseline", "experiments/baseline/baseline_25view_pine"),
    ("pine", 25, "Full-distill", "experiments/distill/pine_ablation/pine_25v_full_distill_A"),
    ("pine", 25, "Masked-Self-distill", "experiments/distill/pine_ablation/pine_25v_self_masked_mw05"),
    # Pine 50 views
    ("pine", 50, "Baseline", "experiments/baseline/baseline_50view_pine"),
    ("pine", 50, "Full-distill", "experiments/distill/pine_ablation/pine_50v_full_distill_A"),
    ("pine", 50, "Masked-Self-distill", "experiments/distill/pine_ablation/pine_50v_self_masked_mw05"),
    # Pine 75 views
    ("pine", 75, "Baseline", "experiments/baseline/baseline_75view_pine"),
    ("pine", 75, "Full-distill", "experiments/distill/pine_ablation/pine_75v_full_distill_A"),
    ("pine", 75, "Masked-Self-distill", "experiments/distill/pine_ablation/pine_75v_self_masked_mw05"),
    
    # Seashell 25 views
    ("seashell", 25, "Baseline", "experiments/baseline/baseline_25view_seashell"),
    ("seashell", 25, "Full-distill", "experiments/distill/seashell_ablation/seashell_25v_full_distill_A"),
    ("seashell", 25, "Delayed-distill", "experiments/distill/seashell_ablation/seashell_25v_delayed_distill_I"),
    ("seashell", 25, "Masked-Self-distill", "experiments/distill/seashell_ablation/seashell_25v_self_masked_mw05_best"),
    # Seashell 50 views
    ("seashell", 50, "Baseline", "experiments/baseline/baseline_50view_seashell"),
    ("seashell", 50, "Full-distill", "experiments/distill/seashell_ablation/seashell_50v_full_distill_A"),
    ("seashell", 50, "Delayed-distill", "experiments/distill/seashell_ablation/seashell_50v_delayed_distill_I"),
    ("seashell", 50, "Masked-Self-distill", "experiments/distill/seashell_ablation/seashell_50v_self_masked_mw05"),
    # Seashell 75 views
    ("seashell", 75, "Baseline", "experiments/baseline/baseline_75view_seashell"),
    ("seashell", 75, "Full-distill", "experiments/distill/seashell_ablation/seashell_75v_full_distill_A"),
    ("seashell", 75, "Delayed-distill", "experiments/distill/seashell_ablation/seashell_75v_delayed_distill_I"),
    ("seashell", 75, "Masked-Self-distill", "experiments/distill/seashell_ablation/seashell_75v_self_masked_mw05"),
    
    # Walnut 25 views
    ("walnut", 25, "Baseline", "experiments/baseline/baseline_25view_walnut"),
    ("walnut", 25, "Full-distill", "experiments/distill/walnut_ablation/walnut_25v_full_distill_A"),
    ("walnut", 25, "Masked-Self-distill", "experiments/distill/walnut_ablation/walnut_25v_self_masked_mw05"),
    # Walnut 50 views
    ("walnut", 50, "Baseline", "experiments/baseline/baseline_50view_walnut"),
    ("walnut", 50, "Full-distill", "experiments/distill/walnut_ablation/walnut_50v_full_distill_A"),
    ("walnut", 50, "Masked-Self-distill", "experiments/distill/walnut_ablation/walnut_50v_self_masked_mw05"),
    # Walnut 75 views
    ("walnut", 75, "Baseline", "experiments/baseline/baseline_75view_walnut"),
    ("walnut", 75, "Full-distill", "experiments/distill/walnut_ablation/walnut_75v_full_distill_A"),
    ("walnut", 75, "Masked-Self-distill", "experiments/distill/walnut_ablation/walnut_75v_self_masked_mw05"),
]


def check_result_exists(model_path):
    """检查模型是否已有体积重建结果"""
    # 查找最新的 iteration 目录
    point_cloud_dir = Path(model_path) / "point_cloud"
    if not point_cloud_dir.exists():
        return False, None
    
    iterations = sorted([d.name for d in point_cloud_dir.iterdir() if d.is_dir()])
    if not iterations:
        return False, None
    
    latest_iter = iterations[-1]
    iter_num = latest_iter.replace("iteration_", "")
    
    # 检查重建结果是否存在
    recon_path = Path(model_path) / "test" / f"iter_{iter_num}" / "reconstruction"
    if recon_path.exists() and (recon_path / "vol_pred.npy").exists():
        return True, iter_num
    
    return False, iter_num


def run_test(model_path, iter_num=None):
    """运行 test.py 生成体积重建结果"""
    cmd = [
        sys.executable, "test.py",
        "-m", model_path,
        "--skip_render_train",
        "--skip_render_test",
        "--quiet"
    ]
    
    if iter_num:
        cmd.extend(["--iteration", iter_num])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("批量生成体积重建结果")
    print("=" * 60)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for dataset, views, method, model_path in MODELS:
        print(f"\n[{dataset.upper()} {views}v - {method}]")
        print(f"  Model: {model_path}")
        
        # 检查模型目录是否存在
        if not Path(model_path).exists():
            print(f"  SKIP: 模型目录不存在")
            skip_count += 1
            continue
        
        # 检查结果是否已存在
        exists, iter_num = check_result_exists(model_path)
        if exists:
            print(f"  SKIP: 结果已存在 (iter_{iter_num})")
            skip_count += 1
            continue
        
        # 运行测试
        print(f"  Running test (iter={iter_num})...")
        if run_test(model_path, iter_num):
            print(f"  OK: 测试完成")
            success_count += 1
        else:
            print(f"  FAIL: 测试失败")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"完成! 成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
