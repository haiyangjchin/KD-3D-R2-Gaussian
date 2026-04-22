#!/usr/bin/env python3
"""
Optuna TPE超参搜索 - 自动寻找最优蒸馏参数
支持: CNN蒸馏 / Masked自蒸馏
使用HyperbandPruner在iter 2500/5000时剪枝差的trial
"""

import os
import sys
import argparse
import json
import time
import subprocess
import yaml
import numpy as np
from datetime import datetime

try:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("请安装optuna: pip install optuna")
    sys.exit(1)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)


def evaluate_volume(output_dir, iter_n):
    """评估某个checkpoint的PSNR/SSIM"""
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
        vol_pred = vol_pred / vol_gt.max()
        vol_gt = vol_gt / vol_gt.max()

    psnr = peak_signal_noise_ratio(vol_gt, vol_pred, data_range=1.0)
    ssim_vals = [
        structural_similarity(vol_gt[i], vol_pred[i], data_range=1.0)
        for i in range(vol_gt.shape[0])
    ]
    return {"psnr": float(psnr), "ssim": float(np.mean(ssim_vals))}


def build_config(trial, args):
    """根据trial采样构建训练配置"""

    # TPE采样超参
    warmup = trial.suggest_int("distill_warmup_iters", 2000, 7000, step=1000)
    interval = trial.suggest_categorical("distill_interval", [4, 8, 12, 16])
    max_weight = trial.suggest_float("max_distill_weight", 0.05, 0.5, step=0.05)
    lambda_tv = trial.suggest_float("lambda_tv", 0.05, 0.2, step=0.05)

    cfg = {
        "source_path": args.data_path,
        "data_device": "cuda",
        "ply_path": "",
        "scale_min": 0.0005,
        "scale_max": 0.5,
        "eval": True,
        "iterations": 10000,
        "test_iterations": [2500, 5000, 7500, 10000],
        "save_iterations": [2500, 5000, 7500, 10000],
        "checkpoint_iterations": [2500, 5000, 7500, 10000],
        "quiet": True,
        "detect_anomaly": False,
        # 学习率（固定）
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
        # 搜索参数
        "lambda_dssim": 0.25,
        "lambda_tv": lambda_tv,
        "tv_vol_size": 32,
        # Density control（固定）
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
        # 蒸馏调度参数（搜索）
        "distill_warmup_iters": warmup,
        "distill_interval": interval,
        "max_distill_weight": max_weight,
        # 通用
        "no_distill": False,
        "no_improvements": False,
        "use_importance_sampling": False,
        "enhanced_regularization": False,
    }

    # 蒸馏类型
    if args.distill_mode == "cnn":
        cfg["cnn_model"] = args.teacher_path
    elif args.distill_mode == "self_masked":
        cfg["static_volume_path"] = args.teacher_path
        cfg["use_kl"] = False
        cfg["use_mse"] = False
        cfg["use_l1"] = True
        cfg["use_mask"] = True
        cfg["mask_threshold"] = 0.01

    return cfg


def objective(trial, args):
    """Optuna目标函数"""
    trial_name = f"trial_{trial.number:03d}"
    output_dir = os.path.join(args.output_dir, trial_name)
    config_path = os.path.join(args.output_dir, f"{trial_name}.yaml")

    # 构建配置
    cfg = build_config(trial, args)
    cfg["model_path"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # 运行训练
    cmd = [
        sys.executable,
        "train_with_distillation.py",
        "--config",
        config_path,
        "--output_dir",
        output_dir,
        "--quiet",
    ]

    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    process.wait()
    elapsed = (time.time() - start_time) / 60

    # 中间指标报告 + 剪枝
    for step_idx, iter_n in enumerate([2500, 5000, 7500, 10000]):
        metrics = evaluate_volume(output_dir, iter_n)
        if metrics is None:
            raise optuna.TrialPruned()

        # 报告中间值（用PSNR作为优化目标）
        trial.report(metrics["psnr"], step_idx)

        # 检查是否应该剪枝
        if trial.should_prune():
            print(
                f"  Trial {trial.number} pruned at iter {iter_n} (PSNR={metrics['psnr']:.2f})"
            )
            raise optuna.TrialPruned()

    # 最终结果
    final = evaluate_volume(output_dir, 10000)
    if final is None:
        raise optuna.TrialPruned()

    # 记录额外信息
    trial.set_user_attr("ssim", final["ssim"])
    trial.set_user_attr("time_min", elapsed)
    trial.set_user_attr(
        "config",
        {
            "warmup": cfg["distill_warmup_iters"],
            "interval": cfg["distill_interval"],
            "max_weight": cfg["max_distill_weight"],
            "lambda_tv": cfg["lambda_tv"],
        },
    )

    print(
        f"  Trial {trial.number}: PSNR={final['psnr']:.2f}, SSIM={final['ssim']:.4f}, Time={elapsed:.1f}min"
    )
    return final["psnr"]


def main():
    parser = argparse.ArgumentParser(description="Optuna TPE Hyperparameter Search")
    parser.add_argument("--data_path", type=str, required=True, help="数据集路径")
    parser.add_argument(
        "--teacher_path",
        type=str,
        required=True,
        help="教师模型路径（.pth或vol_pred.npy）",
    )
    parser.add_argument(
        "--distill_mode",
        type=str,
        choices=["cnn", "self_masked"],
        default="cnn",
        help="蒸馏模式: cnn=CNN教师, self_masked=masked自蒸馏",
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/optuna_search", help="输出目录"
    )
    parser.add_argument("--n_trials", type=int, default=25, help="搜索次数")
    parser.add_argument(
        "--study_name", type=str, default="distill_hpo", help="Study名称"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Optuna TPE Hyperparameter Search")
    print("=" * 70)
    print(f"数据集: {args.data_path}")
    print(f"教师: {args.teacher_path}")
    print(f"模式: {args.distill_mode}")
    print(f"搜索次数: {args.n_trials}")
    print(f"输出: {args.output_dir}")
    print("=" * 70)

    # 创建study
    sampler = TPESampler(seed=42, multivariate=True)
    pruner = HyperbandPruner(
        min_resource=1,  # 最早在step 1 (iter 2500) 剪枝
        max_resource=4,  # 最多4个step (2500, 5000, 7500, 10000)
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",  # 最大化PSNR
        sampler=sampler,
        pruner=pruner,
    )

    # 开始搜索
    start_time = time.time()
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    total_time = (time.time() - start_time) / 60

    # 输出结果
    print("\n" + "=" * 70)
    print("搜索完成!")
    print("=" * 70)
    print(f"总用时: {total_time:.1f} 分钟")
    print(f"完成trials: {len(study.trials)}")
    print(
        f"剪枝trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )

    # 最优结果
    best = study.best_trial
    print(f"\n最优Trial #{best.number}:")
    print(f"  PSNR: {best.value:.2f}")
    print(f"  SSIM: {best.user_attrs.get('ssim', 'N/A')}")
    print(f"  Time: {best.user_attrs.get('time_min', 'N/A'):.1f} min")
    print(f"  参数:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    # Top 5 结果
    print(f"\nTop 5 Trials:")
    trials_sorted = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True,
    )
    for i, t in enumerate(trials_sorted[:5]):
        config = t.user_attrs.get("config", {})
        print(
            f"  #{t.number}: PSNR={t.value:.2f}, SSIM={t.user_attrs.get('ssim', 0):.4f}, "
            f"w={config.get('warmup', '?')}, i={config.get('interval', '?')}, "
            f"mw={config.get('max_weight', '?')}, tv={config.get('lambda_tv', '?')}"
        )

    # 保存结果
    results = {
        "best_trial": {
            "number": best.number,
            "psnr": best.value,
            "ssim": best.user_attrs.get("ssim"),
            "time_min": best.user_attrs.get("time_min"),
            "params": best.params,
        },
        "all_trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "psnr": t.value
                if t.state == optuna.trial.TrialState.COMPLETE
                else None,
                "ssim": t.user_attrs.get("ssim"),
                "time_min": t.user_attrs.get("time_min"),
                "params": t.params,
            }
            for t in study.trials
        ],
        "search_config": {
            "data_path": args.data_path,
            "teacher_path": args.teacher_path,
            "distill_mode": args.distill_mode,
            "n_trials": args.n_trials,
            "total_time_min": total_time,
        },
    }

    result_path = os.path.join(args.output_dir, "optuna_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {result_path}")

    # 生成参数重要性图
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(args.output_dir, "param_importances.png"))
        print("参数重要性图已保存")
    except Exception:
        pass

    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(args.output_dir, "optimization_history.png"))
        print("优化历史图已保存")
    except Exception:
        pass


if __name__ == "__main__":
    main()
