#!/usr/bin/env python3
"""
Pine/Walnut 50view + 75view 实验 - I配置
warmup=5000, interval=8, max_weight=0.5, lambda_tv=0.2
"""

import os, sys, subprocess, time, json, yaml
import numpy as np

PROJECT_DIR = r"E:\r2_gaussian"
os.chdir(PROJECT_DIR)

# I配置模板 (只需替换 source_path, model_path, cnn_model)
TEMPLATE = {
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
    "lambda_tv": 0.2,
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
    "distill_warmup_iters": 5000,
    "distill_interval": 8,
    "max_distill_weight": 0.5,
    "no_distill": False,
    "no_improvements": False,
    "use_importance_sampling": False,
    "enhanced_regularization": False,
}

EXPERIMENTS = [
    {
        "name": "pine_50view_I",
        "source_path": "./data/real_dataset/cone_ntrain_50_angle_360/pine",
        "cnn_model": "./cnn_teacher_50epoch_v4/checkpoints/checkpoint_epoch_034.pth",
    },
    {
        "name": "pine_75view_I",
        "source_path": "./data/real_dataset/cone_ntrain_75_angle_360/pine",
        "cnn_model": "./cnn_teacher_50epoch_v4/checkpoints/checkpoint_epoch_034.pth",
    },
    {
        "name": "walnut_50view_I",
        "source_path": "./data/real_dataset/cone_ntrain_50_angle_360/walnut",
        "cnn_model": "./cnn_teacher_walnut_75view/best_model.pth",
    },
    {
        "name": "walnut_75view_I",
        "source_path": "./data/real_dataset/cone_ntrain_75_angle_360/walnut",
        "cnn_model": "./cnn_teacher_walnut_75view/best_model.pth",
    },
]


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
    output = f"experiments/distill/pine_ablation/{name}"
    config_path = f"experiments/distill/pine_ablation/{name}.yaml"

    # 生成yaml
    cfg = dict(TEMPLATE)
    cfg["source_path"] = exp["source_path"]
    cfg["model_path"] = output
    cfg["cnn_model"] = exp["cnn_model"]
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
    print(
        f"\n{'=' * 70}\n开始: {name}\n  数据: {exp['source_path']}\n  教师: {exp['cnn_model']}\n{'=' * 70}"
    )
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
        "all_iters": metrics,
    }


def main():
    results = {}
    for exp in EXPERIMENTS:
        results[exp["name"]] = run_one(exp)

    # Baseline参考值
    baselines = {
        "pine_50view": {"psnr": 39.13, "ssim": 0.9418},
        "pine_75view": {"psnr": 39.05, "ssim": 0.9435},
        "walnut_50view": {"psnr": 32.48, "ssim": 0.7188},
        "walnut_75view": {"psnr": 33.49, "ssim": 0.7297},
    }

    print(f"\n{'=' * 95}")
    print("50/75 view 实验汇总")
    print(f"{'=' * 95}")
    print(
        f"{'实验':<20} {'PSNR':>8} {'SSIM':>10} {'时间':>8}   {'vs Baseline PSNR':>16} {'vs Baseline SSIM':>16}"
    )
    print("-" * 95)
    for name, r in results.items():
        bkey = name.replace("_I", "")
        b = baselines.get(bkey, {})
        dp = r["psnr"] - b.get("psnr", 0)
        ds = r["ssim"] - b.get("ssim", 0)
        print(
            f"{name:<20} {r['psnr']:>8.2f} {r['ssim']:>10.4f} {r['time_min']:>7.1f}m   {dp:>+16.2f} {ds:>+16.4f}"
        )
    print("-" * 95)

    with open("experiments/distill/pine_ablation/results_50_75view.json", "w") as f:
        json.dump(results, f, indent=2)
    print("结果已保存到 experiments/distill/pine_ablation/results_50_75view.json")


if __name__ == "__main__":
    main()
