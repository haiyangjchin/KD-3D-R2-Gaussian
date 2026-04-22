#!/usr/bin/env python3
"""
Walnut 25/50/75view Masked Self-Distill (mw=0.5, tv=0.05)
顺序执行: 先25v，再50v，最后75v
使用75v baseline作为教师
"""

import os, sys, subprocess, time, json, yaml
import numpy as np

PROJECT_DIR = r"E:\r2_gaussian"
os.chdir(PROJECT_DIR)

# 顺序执行: 先25v，再50v，最后75v
EXPERIMENTS = [
    {"name": "walnut_25v_self_masked_mw05", "views": 25,
     "desc": "25view Masked Self-Distill: mw=0.5, tv=0.05, 使用75v baseline教师"},
    {"name": "walnut_50v_self_masked_mw05", "views": 50,
     "desc": "50view Masked Self-Distill: mw=0.5, tv=0.05, 使用75v baseline教师"},
    {"name": "walnut_75v_self_masked_mw05", "views": 75,
     "desc": "75view Masked Self-Distill: mw=0.5, tv=0.05, 使用75v baseline教师"},
]

BASE_CFG = {
    "data_device": "cuda", "ply_path": "", "scale_min": 0.0005, "scale_max": 0.5, "eval": True,
    "iterations": 10000,
    "test_iterations": [2500, 5000, 7500, 10000],
    "save_iterations": [2500, 5000, 7500, 10000],
    "checkpoint_iterations": [2500, 5000, 7500, 10000],
    "quiet": False, "detect_anomaly": False,
    "position_lr_init": 0.0002, "position_lr_final": 2.0e-05, "position_lr_max_steps": 30000,
    "density_lr_init": 0.01, "density_lr_final": 0.001, "density_lr_max_steps": 30000,
    "scaling_lr_init": 0.005, "scaling_lr_final": 0.0005, "scaling_lr_max_steps": 30000,
    "rotation_lr_init": 0.001, "rotation_lr_final": 0.0001, "rotation_lr_max_steps": 30000,
    "lambda_dssim": 0.25, "lambda_tv": 0.05, "tv_vol_size": 32,
    "density_min_threshold": 1.0e-05, "densification_interval": 100,
    "densify_from_iter": 500, "densify_until_iter": 15000,
    "densify_grad_threshold": 5.0e-05, "densify_scale_threshold": 0.1,
    "max_screen_size": None, "max_scale": None, "max_num_gaussians": 500000,
    "compute_cov3D_python": False, "debug": False,
    # Masked Self-Distill配置
    "distill_warmup_iters": 5000,
    "distill_interval": 8,
    "max_distill_weight": 0.5,
    "mask_threshold": 0.01,
    "use_mask": True,
    "use_l1": True,
    "use_kl": False,
    "use_mse": False,
    "no_distill": False, "no_improvements": False,
    "use_importance_sampling": False, "enhanced_regularization": False,
}

TEACHER_PATH = "experiments/baseline/baseline_75view_walnut/test/iter_10000/vol_pred.npy"


def evaluate(output_dir, iter_n):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    vp = np.load(os.path.join(output_dir, "point_cloud", f"iteration_{iter_n}", "vol_pred.npy"))
    vg = np.load(os.path.join(output_dir, "point_cloud", f"iteration_{iter_n}", "vol_gt.npy"))
    if vg.max() > 1.0:
        vp, vg = vp / vg.max(), vg / vg.max()
    psnr = peak_signal_noise_ratio(vg, vp, data_range=1.0)
    ssim = float(np.mean([structural_similarity(vg[i], vp[i], data_range=1.0) for i in range(vg.shape[0])]))
    return {"psnr": float(psnr), "ssim": ssim}


def run_one(exp):
    name = exp["name"]
    views = exp["views"]
    output = f"experiments/distill/walnut_ablation/{name}"
    config_path = f"experiments/distill/walnut_ablation/{name}.yaml"
    cfg = dict(BASE_CFG)
    cfg["source_path"] = f"./data/real_dataset/cone_ntrain_{views}_angle_360/walnut"
    cfg["model_path"] = output
    cfg["static_volume_path"] = TEACHER_PATH
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    os.makedirs(output, exist_ok=True)
    log_file = f"experiments/logs/distill/{name}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    cmd = [sys.executable, "train_with_distillation.py", "--config", config_path, "--output_dir", output]
    print(f"\n{'='*70}\n开始: {name}\n{exp['desc']}\n{'='*70}")
    start = time.time()
    with open(log_file, "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
        for line in p.stdout:
            print(line, end="")
            log.write(line); log.flush()
        p.wait()
    elapsed = (time.time() - start) / 60
    print(f"\n完成! 用时: {elapsed:.1f}min")
    metrics = {}
    for it in [2500, 5000, 7500, 10000]:
        try:
            m = evaluate(output, it)
            metrics[it] = m
            print(f"  Iter {it}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.6f}")
        except: pass
    final = metrics.get(10000, {})
    return {"psnr": final.get("psnr", 0), "ssim": final.get("ssim", 0), "time_min": elapsed, "desc": exp["desc"], "all_iters": metrics}


def main():
    results = {}
    for exp in EXPERIMENTS:
        results[exp["name"]] = run_one(exp)

    print(f"\n{'='*90}")
    print("Walnut 25/50/75v Masked Self-Distill 实验结果")
    print(f"{'='*90}")
    print(f"{'实验':<35} {'PSNR':>8} {'SSIM':>10} {'时间':>8}")
    print("-" * 70)
    # 参考值
    refs = [
        ("Walnut 25v Baseline",        29.42, 0.6764, 7),
        ("Walnut 25v Full-Distill",    29.40, 0.6758, 24.5),
        ("Walnut 25v Delayed-Distill", 29.78, 0.6772, 9),
        ("Walnut 50v Baseline",        32.48, 0.7188, 7),
        ("Walnut 50v Full-Distill",    32.28, 0.7206, 25.9),
        ("Walnut 50v Delayed-Distill", 32.34, 0.7198, 9),
        ("Walnut 75v Baseline",        33.49, 0.7297, 7),
        ("Walnut 75v Full-Distill",    33.15, 0.7329, 26.6),
        ("Walnut 75v Delayed-Distill", 32.99, 0.7299, 9),
    ]
    for n, p, s, t in refs:
        print(f"{n:<35} {p:>8.2f} {s:>10.4f} {t:>7}m")
    for n, r in results.items():
        print(f"{n:<35} {r['psnr']:>8.2f} {r['ssim']:>10.4f} {r['time_min']:>7.1f}m")
    print("-" * 70)

    with open("experiments/distill/walnut_ablation/results_masked_self_distill_25_50_75.json", "w") as f:
        json.dump(results, f, indent=2)
    print("结果已保存")


if __name__ == "__main__":
    main()