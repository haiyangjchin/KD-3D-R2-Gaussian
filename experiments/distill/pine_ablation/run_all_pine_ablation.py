#!/usr/bin/env python3
"""
Pine延迟蒸馏消融实验 - 批量顺序运行B/C/D/E四组对照
每组实验完成后自动评估并记录结果

实验设计:
  A (已完成): warmup=1000, interval=4, max_weight=0.5  -> PSNR=38.05 (3h18min)
  B: warmup=3000, interval=8,  max_weight=0.3  -> 预计 ~20min
  C: warmup=5000, interval=8,  max_weight=0.3  -> 预计 ~15min
  D: warmup=5000, interval=16, max_weight=0.3  -> 预计 ~12min
  E: warmup=7000, interval=8,  max_weight=0.2  -> 预计 ~10min

Baseline (已完成): 无蒸馏 -> PSNR=37.69 (7min)
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

# 项目根目录
PROJECT_DIR = r"E:\r2_gaussian"
os.chdir(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

EXPERIMENTS = [
    {
        "name": "pine_exp_B",
        "config": "experiments/distill/pine_ablation/pine_exp_B.yaml",
        "output": "experiments/distill/pine_ablation/pine_exp_B",
        "desc": "warmup=3000, interval=8, max_weight=0.3",
    },
    {
        "name": "pine_exp_C",
        "config": "experiments/distill/pine_ablation/pine_exp_C.yaml",
        "output": "experiments/distill/pine_ablation/pine_exp_C",
        "desc": "warmup=5000, interval=8, max_weight=0.3",
    },
    {
        "name": "pine_exp_D",
        "config": "experiments/distill/pine_ablation/pine_exp_D.yaml",
        "output": "experiments/distill/pine_ablation/pine_exp_D",
        "desc": "warmup=5000, interval=16, max_weight=0.3",
    },
    {
        "name": "pine_exp_E",
        "config": "experiments/distill/pine_ablation/pine_exp_E.yaml",
        "output": "experiments/distill/pine_ablation/pine_exp_E",
        "desc": "warmup=7000, interval=8, max_weight=0.2",
    },
]


def evaluate_experiment(output_dir, iter_n=10000):
    """评估某个实验的PSNR/SSIM"""
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    vol_pred_path = os.path.join(
        output_dir, "point_cloud", f"iteration_{iter_n}", "vol_pred.npy"
    )
    vol_gt_path = os.path.join(
        output_dir, "point_cloud", f"iteration_{iter_n}", "vol_gt.npy"
    )

    if not os.path.exists(vol_pred_path) or not os.path.exists(vol_gt_path):
        return None

    vol_pred = np.load(vol_pred_path)
    vol_gt = np.load(vol_gt_path)

    if vol_gt.max() > 1.0:
        vol_gt_n = vol_gt / vol_gt.max()
        vol_pred_n = vol_pred / vol_gt.max()
    else:
        vol_gt_n = vol_gt
        vol_pred_n = vol_pred

    psnr = peak_signal_noise_ratio(vol_gt_n, vol_pred_n, data_range=1.0)
    ssim_vals = [
        structural_similarity(vol_gt_n[i], vol_pred_n[i], data_range=1.0)
        for i in range(vol_gt_n.shape[0])
    ]
    ssim = float(np.mean(ssim_vals))

    return {"psnr": float(psnr), "ssim": ssim}


def run_experiment(exp):
    """运行单个实验"""
    name = exp["name"]
    config = exp["config"]
    output = exp["output"]

    os.makedirs(output, exist_ok=True)
    log_file = os.path.join("experiments", "logs", "distill", f"{name}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    cmd = [
        sys.executable,
        "train_with_distillation.py",
        "--config", config,
        "--output_dir", output,
    ]

    print(f"\n{'='*70}")
    print(f"开始实验: {name}")
    print(f"描述: {exp['desc']}")
    print(f"命令: {' '.join(cmd)}")
    print(f"日志: {log_file}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    start_time = time.time()

    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"实验: {name}\n")
        log.write(f"描述: {exp['desc']}\n")
        log.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"{'='*70}\n")
        log.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        return_code = process.wait()

        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60

        log.write(f"\n{'='*70}\n")
        log.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"训练时长: {elapsed_min:.1f} 分钟\n")
        log.write(f"返回代码: {return_code}\n")

    print(f"\n训练完成! 用时: {elapsed_min:.1f} 分钟, 返回代码: {return_code}")

    return return_code, elapsed_min


def main():
    print("=" * 70)
    print("Pine 延迟蒸馏消融实验")
    print(f"总共 {len(EXPERIMENTS)} 组实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}

    # 参考值
    results["baseline"] = {
        "psnr": 37.69, "ssim": 0.9276,
        "time_min": 7.0, "desc": "无蒸馏 baseline"
    }
    results["exp_A"] = {
        "psnr": 38.05, "ssim": 0.9467,
        "time_min": 198.0, "desc": "warmup=1000, interval=4, max_weight=0.5 (原始)"
    }

    total_start = time.time()

    for exp in EXPERIMENTS:
        return_code, elapsed_min = run_experiment(exp)

        # 评估各checkpoint
        metrics_all = {}
        for iter_n in [2500, 5000, 7500, 10000]:
            m = evaluate_experiment(exp["output"], iter_n)
            if m:
                metrics_all[iter_n] = m
                print(f"  Iter {iter_n}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.6f}")

        final_metrics = metrics_all.get(10000, {})
        results[exp["name"]] = {
            "psnr": final_metrics.get("psnr", 0),
            "ssim": final_metrics.get("ssim", 0),
            "time_min": elapsed_min,
            "desc": exp["desc"],
            "all_iters": metrics_all,
            "return_code": return_code,
        }

    total_elapsed = (time.time() - total_start) / 60

    # 打印汇总
    print("\n" + "=" * 90)
    print("Pine 延迟蒸馏消融实验 - 结果汇总")
    print("=" * 90)
    print(f"{'实验':<15} {'PSNR':>8} {'SSIM':>10} {'时间(min)':>10} {'描述'}")
    print("-" * 90)
    for name, r in sorted(results.items()):
        print(
            f"{name:<15} {r['psnr']:>8.2f} {r['ssim']:>10.6f} "
            f"{r['time_min']:>10.1f}   {r['desc']}"
        )
    print("-" * 90)
    print(f"总用时: {total_elapsed:.1f} 分钟")

    # 保存结果到JSON
    result_file = "experiments/distill/pine_ablation/results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
