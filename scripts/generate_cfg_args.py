#!/usr/bin/env python3
"""
为 distill 模型生成 cfg_args 文件
从 yaml 配置文件转换为 test.py 需要的 Namespace 格式
"""

import os
import yaml
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# 模型和对应的 yaml 配置映射
MODEL_CONFIGS = [
    # Pine 25 views
    ("experiments/distill/pine_ablation/pine_25v_full_distill_A", "experiments/distill/pine_ablation/pine_25v_full_distill_A.yaml"),
    ("experiments/distill/pine_ablation/pine_25v_self_masked_mw05", "experiments/distill/pine_ablation/pine_25v_self_masked_mw05.yaml"),
    # Pine 50 views
    ("experiments/distill/pine_ablation/pine_50v_full_distill_A", "experiments/distill/pine_ablation/pine_50v_full_distill_A.yaml"),
    ("experiments/distill/pine_ablation/pine_50v_self_masked_mw05", "experiments/distill/pine_ablation/pine_50v_self_masked_mw05.yaml"),
    # Pine 75 views
    ("experiments/distill/pine_ablation/pine_75v_full_distill_A", "experiments/distill/pine_ablation/pine_75v_full_distill_A.yaml"),
    ("experiments/distill/pine_ablation/pine_75v_self_masked_mw05", "experiments/distill/pine_ablation/pine_75v_self_masked_mw05.yaml"),
    
    # Seashell 25 views
    ("experiments/distill/seashell_ablation/seashell_25v_full_distill_A", "experiments/distill/seashell_ablation/seashell_25v_full_distill_A.yaml"),
    ("experiments/distill/seashell_ablation/seashell_25v_delayed_distill_I", "experiments/distill/seashell_ablation/seashell_25v_delayed_distill_I.yaml"),
    ("experiments/distill/seashell_ablation/seashell_25v_self_masked_mw05_best", "experiments/distill/seashell_ablation/seashell_25v_self_masked_mw05_best.yaml"),
    # Seashell 50 views
    ("experiments/distill/seashell_ablation/seashell_50v_full_distill_A", "experiments/distill/seashell_ablation/seashell_50v_full_distill_A.yaml"),
    ("experiments/distill/seashell_ablation/seashell_50v_delayed_distill_I", "experiments/distill/seashell_ablation/seashell_50v_delayed_distill_I.yaml"),
    ("experiments/distill/seashell_ablation/seashell_50v_self_masked_mw05", "experiments/distill/seashell_ablation/seashell_50v_self_masked_mw05.yaml"),
    # Seashell 75 views
    ("experiments/distill/seashell_ablation/seashell_75v_full_distill_A", "experiments/distill/seashell_ablation/seashell_75v_full_distill_A.yaml"),
    ("experiments/distill/seashell_ablation/seashell_75v_delayed_distill_I", "experiments/distill/seashell_ablation/seashell_75v_delayed_distill_I.yaml"),
    ("experiments/distill/seashell_ablation/seashell_75v_self_masked_mw05", "experiments/distill/seashell_ablation/seashell_75v_self_masked_mw05.yaml"),
    
    # Walnut 25 views
    ("experiments/distill/walnut_ablation/walnut_25v_full_distill_A", "experiments/distill/walnut_ablation/walnut_25v_full_distill_A.yaml"),
    ("experiments/distill/walnut_ablation/walnut_25v_self_masked_mw05", "experiments/distill/walnut_ablation/walnut_25v_self_masked_mw05.yaml"),
    # Walnut 50 views
    ("experiments/distill/walnut_ablation/walnut_50v_full_distill_A", "experiments/distill/walnut_ablation/walnut_50v_full_distill_A.yaml"),
    ("experiments/distill/walnut_ablation/walnut_50v_self_masked_mw05", "experiments/distill/walnut_ablation/walnut_50v_self_masked_mw05.yaml"),
    # Walnut 75 views
    ("experiments/distill/walnut_ablation/walnut_75v_full_distill_A", "experiments/distill/walnut_ablation/walnut_75v_full_distill_A.yaml"),
    ("experiments/distill/walnut_ablation/walnut_75v_self_masked_mw05", "experiments/distill/walnut_ablation/walnut_75v_self_masked_mw05.yaml"),
]


def generate_cfg_args(model_path, yaml_path):
    """从 yaml 生成 cfg_args 文件"""
    if not Path(yaml_path).exists():
        print(f"  WARN: yaml 配置文件不存在: {yaml_path}")
        return False
    
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # 创建 Namespace 对象
    # 只包含 test.py 需要的参数
    args_dict = {
        'source_path': config.get('source_path', ''),
        'model_path': config.get('model_path', model_path),
        'data_device': config.get('data_device', 'cuda'),
        'ply_path': config.get('ply_path', ''),
        'scale_min': config.get('scale_min', 0.0005),
        'scale_max': config.get('scale_max', 0.5),
        'eval': config.get('eval', True),
        'compute_cov3D_python': config.get('compute_cov3D_python', False),
        'debug': config.get('debug', False),
        'iterations': config.get('iterations', 10000),
        'position_lr_init': config.get('position_lr_init', 0.0002),
        'position_lr_final': config.get('position_lr_final', 2.0e-05),
        'position_lr_max_steps': config.get('position_lr_max_steps', 30000),
        'density_lr_init': config.get('density_lr_init', 0.01),
        'density_lr_final': config.get('density_lr_final', 0.001),
        'density_lr_max_steps': config.get('density_lr_max_steps', 30000),
        'scaling_lr_init': config.get('scaling_lr_init', 0.005),
        'scaling_lr_final': config.get('scaling_lr_final', 0.0005),
        'scaling_lr_max_steps': config.get('scaling_lr_max_steps', 30000),
        'rotation_lr_init': config.get('rotation_lr_init', 0.001),
        'rotation_lr_final': config.get('rotation_lr_final', 0.0001),
        'rotation_lr_max_steps': config.get('rotation_lr_max_steps', 30000),
        'lambda_dssim': config.get('lambda_dssim', 0.25),
        'lambda_tv': config.get('lambda_tv', 0.05),
        'tv_vol_size': config.get('tv_vol_size', 32),
        'density_min_threshold': config.get('density_min_threshold', 1.0e-05),
        'densification_interval': config.get('densification_interval', 100),
        'densify_from_iter': config.get('densify_from_iter', 500),
        'densify_until_iter': config.get('densify_until_iter', 15000),
        'densify_grad_threshold': config.get('densify_grad_threshold', 5.0e-05),
        'densify_scale_threshold': config.get('densify_scale_threshold', 0.1),
        'max_screen_size': config.get('max_screen_size', None),
        'max_scale': config.get('max_scale', None),
        'max_num_gaussians': config.get('max_num_gaussians', 500000),
        'quiet': config.get('quiet', False),
        'detect_anomaly': config.get('detect_anomaly', False),
        'test_iterations': config.get('test_iterations', [2500, 5000, 7500, 10000]),
        'save_iterations': config.get('save_iterations', [2500, 5000, 7500, 10000]),
        'checkpoint_iterations': config.get('checkpoint_iterations', [2500, 5000, 7500, 10000]),
    }
    
    cfg_args = Namespace(**args_dict)
    
    # 写入 cfg_args 文件
    cfg_args_path = Path(model_path) / "cfg_args"
    with open(cfg_args_path, 'w') as f:
        f.write(repr(cfg_args))
    
    print(f"  OK: 生成 {cfg_args_path}")
    return True


def main():
    print("=" * 60)
    print("生成 cfg_args 文件")
    print("=" * 60)
    
    success = 0
    skip = 0
    fail = 0
    
    for model_path, yaml_path in MODEL_CONFIGS:
        print(f"\n{model_path}")
        
        cfg_args_path = Path(model_path) / "cfg_args"
        if cfg_args_path.exists():
            print(f"  SKIP: cfg_args 已存在")
            skip += 1
            continue
        
        if generate_cfg_args(model_path, yaml_path):
            success += 1
        else:
            fail += 1
    
    print("\n" + "=" * 60)
    print(f"完成! 成功: {success}, 跳过: {skip}, 失败: {fail}")
    print("=" * 60)


if __name__ == "__main__":
    main()
