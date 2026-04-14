#!/usr/bin/env python3
"""
Pine H/I实验 - 在F(PSNR最优)基础上增加蒸馏强度提升SSIM
H: interval=8, max_weight=0.3, lambda_tv=0.2
I: interval=8, max_weight=0.5, lambda_tv=0.2
"""

import os
import sys
import subprocess
import time
import json
import numpy as np
from datetime import datetime

PROJECT_DIR = r"E:\r2_gaussian"
os.chdir(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

EXPERIMENTS = [
    {
        "name": "pine_exp_H",
        "config": "experiments/distill/pine_ablation/pine_exp_H.yaml",
        "output": "experiments/distill/pine_ablation/pine_exp_H",
        "desc": "F + interval=8 (蒸馏步数x2)",
    },
    {
        "name": "pine_exp_I",
        "config": "experiments/distill/pine_ablation/pine_exp_I.yaml",
        "output": "experiments/distill/pine_ablation/pine_exp_I",
        "desc": "F + interval=8 + max_weight=0.5",
    },
]


def evaluate_experiment(output_dir, iter_n=10000):
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
    return {"psnr": float(psnr), "ssim": float(np.mean(ssim_vals))}


def run_experiment(exp):
    name, config, output = exp["name"], exp["config"], exp["output"]
    os.makedirs(output, exist_ok=True)
    log_file = os.path.join("experiments", "logs", "distill", f"{name}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    cmd = [
        sys.executable,
        "train_with_distillation.py",
        "--config",
        config,
        "--output_dir",
        output,
    ]
    print(f"\n{'=' * 70}\n开始: {name} ({exp['desc']})\n{'=' * 70}")
    start_time = time.time()
    with open(log_file, "w", encoding="utf-8") as log:
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
    elapsed_min = (time.time() - start_time) / 60
    print(f"\n完成! 用时: {elapsed_min:.1f}min, 返回码: {return_code}")
    return return_code, elapsed_min


def main():
    results = {}
    for exp in EXPERIMENTS:
        rc, elapsed = run_experiment(exp)
        metrics = {}
        for it in [2500, 5000, 7500, 10000]:
            m = evaluate_experiment(exp["output"], it)
            if m:
                metrics[it] = m
                print(f"  Iter {it}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.6f}")
        final = metrics.get(10000, {})
        results[exp["name"]] = {
            "psnr": final.get("psnr", 0),
            "ssim": final.get("ssim", 0),
            "time_min": elapsed,
            "desc": exp["desc"],
            "all_iters": metrics,
        }

    # 汇总
    ref = [
        ("Baseline", 37.69, 0.9276, 7.0, "无蒸馏, tv=0.05"),
        ("A(原始)", 38.05, 0.9467, 198.0, "w=1k,i=4,mw=0.5,tv=0.1"),
        ("D", 38.50, 0.9266, 6.0, "w=5k,i=16,mw=0.3,tv=0.1"),
        ("F(PSNR最优)", 38.72, 0.9277, 6.1, "w=5k,i=16,mw=0.3,tv=0.2"),
    ]
    print(f"\n{'=' * 95}")
    print("Pine全部实验汇总")
    print(f"{'=' * 95}")
    print(f"{'实验':<18} {'PSNR':>8} {'SSIM':>10} {'时间(min)':>10}   {'配置'}")
    print("-" * 95)
    for n, p, s, t, d in ref:
        print(f"{n:<18} {p:>8.2f} {s:>10.4f} {t:>10.1f}   {d}")
    for n, r in results.items():
        print(
            f"{n:<18} {r['psnr']:>8.2f} {r['ssim']:>10.4f} {r['time_min']:>10.1f}   {r['desc']}"
        )
    print("-" * 95)

    with open("experiments/distill/pine_ablation/results_HI.json", "w") as f:
        json.dump(results, f, indent=2)
    print("结果已保存到 experiments/distill/pine_ablation/results_HI.json")


if __name__ == "__main__":
    main()
