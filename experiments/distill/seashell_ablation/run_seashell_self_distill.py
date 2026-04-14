#!/usr/bin/env python3
"""
Seashell自蒸馏实验
用75view baseline的vol_pred作为教师，指导25view训练
教师PSNR=42.55 (远强于CNN的22.75)

实验:
  Q: warmup=5000, interval=8, mw=0.5, tv=0.05 (I配置+原生TV+自蒸馏)
  R: warmup=5000, interval=16, mw=0.3, tv=0.05 (更轻蒸馏)
"""

import os, sys, subprocess, time, json, yaml
import numpy as np

PROJECT_DIR = r"E:\r2_gaussian"
os.chdir(PROJECT_DIR)

TEACHER_VOL = (
    "experiments/baseline/baseline_75view_seashell/test/iter_10000/vol_pred.npy"
)

EXPERIMENTS = [
    {
        "name": "seashell_self_Q",
        "distill_warmup_iters": 5000,
        "distill_interval": 8,
        "max_distill_weight": 0.5,
        "lambda_tv": 0.05,
        "desc": "self-distill: mw=0.5, tv=0.05, w=5k, i=8",
    },
    {
        "name": "seashell_self_R",
        "distill_warmup_iters": 5000,
        "distill_interval": 16,
        "max_distill_weight": 0.3,
        "lambda_tv": 0.05,
        "desc": "self-distill: mw=0.3, tv=0.05, w=5k, i=16",
    },
]

BASE_CFG = {
    "source_path": "./data/real_dataset/cone_ntrain_25_angle_360/seashell",
    "data_device": "cuda",
    "ply_path": "",
    "scale_min": 0.0005,
    "scale_max": 0.5,
    "eval": True,
    "iterations": 10000,
    "test_iterations": [2500, 5000, 7500, 10000],
    "save_iterations": [2500, 5000, 7500, 10000],
    "checkpoint_iterations": [2500, 5000, 7500, 10000],
    "quiet": False,
    "detect_anomaly": False,
    "position_lr_init": 0.0002,
    "position_lr_final": 2.0e-05,
    "position_lr_max_steps": 30000,
    "density_lr_init": 0.01,
    "density_lr_final": 0.001,
    "density_lr_max_steps": 30000,
    "scaling_lr_init": 0.005,
    "scaling_lr_final": 0.0005,
    "scaling_lr_max_steps": 30000,
    "rotation_lr_init": 0.001,
    "rotation_lr_final": 0.0001,
    "rotation_lr_max_steps": 30000,
    "lambda_dssim": 0.25,
    "tv_vol_size": 32,
    "density_min_threshold": 1.0e-05,
    "densification_interval": 100,
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densify_grad_threshold": 5.0e-05,
    "densify_scale_threshold": 0.1,
    "max_screen_size": None,
    "max_scale": None,
    "max_num_gaussians": 500000,
    "compute_cov3D_python": False,
    "debug": False,
    "static_volume_path": TEACHER_VOL,
    "no_distill": False,
    "no_improvements": False,
    "use_importance_sampling": False,
    "enhanced_regularization": False,
}


def evaluate(output_dir, iter_n):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    vp = np.load(
        os.path.join(output_dir, "point_cloud", f"iteration_{iter_n}", "vol_pred.npy")
    )
    vg = np.load(
        os.path.join(output_dir, "point_cloud", f"iteration_{iter_n}", "vol_gt.npy")
    )
    if vg.max() > 1.0:
        vp, vg = vp / vg.max(), vg / vg.max()
    psnr = peak_signal_noise_ratio(vg, vp, data_range=1.0)
    ssim = float(
        np.mean(
            [
                structural_similarity(vg[i], vp[i], data_range=1.0)
                for i in range(vg.shape[0])
            ]
        )
    )
    return {"psnr": float(psnr), "ssim": ssim}


def run_one(exp):
    name = exp["name"]
    output = f"experiments/distill/seashell_ablation/{name}"
    config_path = f"experiments/distill/seashell_ablation/{name}.yaml"

    cfg = dict(BASE_CFG)
    cfg["model_path"] = output
    cfg["lambda_tv"] = exp["lambda_tv"]
    cfg["distill_warmup_iters"] = exp["distill_warmup_iters"]
    cfg["distill_interval"] = exp["distill_interval"]
    cfg["max_distill_weight"] = exp["max_distill_weight"]
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    os.makedirs(output, exist_ok=True)
    log_file = f"experiments/logs/distill/{name}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    cmd = [
        sys.executable,
        "train_with_distillation.py",
        "--config",
        config_path,
        "--output_dir",
        output,
    ]
    print(f"\n{'=' * 70}\n开始: {name} ({exp['desc']})\n{'=' * 70}")
    start = time.time()
    with open(log_file, "w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        p.wait()
    elapsed = (time.time() - start) / 60
    print(f"\n完成! 用时: {elapsed:.1f}min")

    metrics = {}
    for it in [2500, 5000, 7500, 10000]:
        try:
            m = evaluate(output, it)
            metrics[it] = m
            print(f"  Iter {it}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.6f}")
        except:
            pass
    final = metrics.get(10000, {})
    return {
        "psnr": final.get("psnr", 0),
        "ssim": final.get("ssim", 0),
        "time_min": elapsed,
        "desc": exp["desc"],
        "all_iters": metrics,
    }


def main():
    results = {}
    for exp in EXPERIMENTS:
        results[exp["name"]] = run_one(exp)

    print(f"\n{'=' * 90}")
    print("Seashell 自蒸馏结果")
    print(f"{'=' * 90}")
    print(f"{'实验':<25} {'PSNR':>8} {'SSIM':>10} {'时间':>8}   {'说明'}")
    print("-" * 90)
    print(
        f"{'Baseline(25v)':<25} {'40.42':>8} {'0.9428':>10} {'8':>8}   无蒸馏, tv=0.05"
    )
    print(
        f"{'CNN蒸馏(最好)':<25} {'39.39':>8} {'0.9442':>10} {'15':>8}   CNN教师PSNR=22.75"
    )
    print(
        f"{'教师(75v baseline)':<25} {'42.55':>8} {'0.9575':>10} {'':>8}   自蒸馏教师质量"
    )
    for n, r in results.items():
        print(
            f"{n:<25} {r['psnr']:>8.2f} {r['ssim']:>10.4f} {r['time_min']:>7.1f}m   {r['desc']}"
        )
    print("-" * 90)

    with open(
        "experiments/distill/seashell_ablation/results_self_distill.json", "w"
    ) as f:
        json.dump(results, f, indent=2)
    print("结果已保存")


if __name__ == "__main__":
    main()
