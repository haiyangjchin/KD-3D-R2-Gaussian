#!/usr/bin/env python3
"""
生成消融实验可视化对比图
每张图 3行×5列：
- 行: 25 views, 50 views, 75 views
- 列: Ground Truth, Baseline, Full-distill, Delayed-distill, Masked-Self distill
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型路径配置
# 格式: {dataset: {views: {method: model_path}}}
MODEL_PATHS = {
    'pine': {
        25: {
            'Baseline': 'experiments/baseline/baseline_25view_pine',
            'Full-distill': 'experiments/distill/pine_ablation/pine_25v_full_distill_A',
            'Masked-Self-distill': 'experiments/distill/pine_ablation/pine_25v_self_masked_mw05',
        },
        50: {
            'Baseline': 'experiments/baseline/baseline_50view_pine',
            'Full-distill': 'experiments/distill/pine_ablation/pine_50v_full_distill_A',
            'Masked-Self-distill': 'experiments/distill/pine_ablation/pine_50v_self_masked_mw05',
        },
        75: {
            'Baseline': 'experiments/baseline/baseline_75view_pine',
            'Full-distill': 'experiments/distill/pine_ablation/pine_75v_full_distill_A',
            'Masked-Self-distill': 'experiments/distill/pine_ablation/pine_75v_self_masked_mw05',
        },
    },
    'seashell': {
        25: {
            'Baseline': 'experiments/baseline/baseline_25view_seashell',
            'Full-distill': 'experiments/distill/seashell_ablation/seashell_25v_full_distill_A',
            'Delayed-distill': 'experiments/distill/seashell_ablation/seashell_25v_delayed_distill_I',
            'Masked-Self-distill': 'experiments/distill/seashell_ablation/seashell_25v_self_masked_mw05_best',
        },
        50: {
            'Baseline': 'experiments/baseline/baseline_50view_seashell',
            'Full-distill': 'experiments/distill/seashell_ablation/seashell_50v_full_distill_A',
            'Delayed-distill': 'experiments/distill/seashell_ablation/seashell_50v_delayed_distill_I',
            'Masked-Self-distill': 'experiments/distill/seashell_ablation/seashell_50v_self_masked_mw05',
        },
        75: {
            'Baseline': 'experiments/baseline/baseline_75view_seashell',
            'Full-distill': 'experiments/distill/seashell_ablation/seashell_75v_full_distill_A',
            'Delayed-distill': 'experiments/distill/seashell_ablation/seashell_75v_delayed_distill_I',
            'Masked-Self-distill': 'experiments/distill/seashell_ablation/seashell_75v_self_masked_mw05',
        },
    },
    'walnut': {
        25: {
            'Baseline': 'experiments/baseline/baseline_25view_walnut',
            'Full-distill': 'experiments/distill/walnut_ablation/walnut_25v_full_distill_A',
            'Masked-Self-distill': 'experiments/distill/walnut_ablation/walnut_25v_self_masked_mw05',
        },
        50: {
            'Baseline': 'experiments/baseline/baseline_50view_walnut',
            'Full-distill': 'experiments/distill/walnut_ablation/walnut_50v_full_distill_A',
            'Masked-Self-distill': 'experiments/distill/walnut_ablation/walnut_50v_self_masked_mw05',
        },
        75: {
            'Baseline': 'experiments/baseline/baseline_75view_walnut',
            'Full-distill': 'experiments/distill/walnut_ablation/walnut_75v_full_distill_A',
            'Masked-Self-distill': 'experiments/distill/walnut_ablation/walnut_75v_self_masked_mw05',
        },
    },
}

# 列配置（按数据集可能有所不同）
COLUMNS_PINE_WALNUT = ['Ground Truth', 'Baseline', 'Full-distill', 'Masked-Self-distill']
COLUMNS_SEASHELL = ['Ground Truth', 'Baseline', 'Full-distill', 'Delayed-distill', 'Masked-Self-distill']

# 迭代次数
ITERATION = 7500

# 切片索引（体积中心）
SLICE_RATIO = 0.5  # 取中间切片


def load_volume(model_path, vol_type='pred'):
    """加载体积数据"""
    vol_path = os.path.join(
        PROJECT_ROOT, model_path, 
        'test', f'iter_{ITERATION}',
        f'vol_{vol_type}.npy'
    )
    if os.path.exists(vol_path):
        return np.load(vol_path)
    return None


def get_slice(volume, axis=2, ratio=0.5):
    """获取体积的某个轴向切片"""
    if volume is None:
        return None
    idx = int(volume.shape[axis] * ratio)
    if axis == 0:
        return volume[idx, :, :]
    elif axis == 1:
        return volume[:, idx, :]
    else:
        return volume[:, :, idx]


def normalize_slice(slice_data):
    """归一化切片数据到 0-1"""
    if slice_data is None:
        return None
    vmin = slice_data.min()
    vmax = slice_data.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(slice_data)
    return (slice_data - vmin) / (vmax - vmin)


def generate_comparison_figure(dataset, save_path):
    """生成单个数据集的对比图"""
    print(f"\n生成 {dataset.upper()} 对比图...")
    
    views_list = [25, 50, 75]
    
    # 确定列配置
    if dataset == 'seashell':
        columns = COLUMNS_SEASHELL
    else:
        columns = COLUMNS_PINE_WALNUT
    
    n_rows = len(views_list)
    n_cols = len(columns)
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    # 确保 axes 是 2D 数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 加载 Ground Truth（从任意模型加载，GT 是相同的）
    gt_volume = None
    for views in views_list:
        for method in columns[1:]:  # 跳过 GT
            model_path = MODEL_PATHS[dataset][views].get(method)
            if model_path:
                gt_volume = load_volume(model_path, 'gt')
                if gt_volume is not None:
                    break
        if gt_volume is not None:
            break
    
    if gt_volume is None:
        print(f"  错误: 无法加载 {dataset} 的 Ground Truth")
        return False
    
    # 获取切片
    gt_slice = get_slice(gt_volume, axis=2, ratio=SLICE_RATIO)
    gt_slice_norm = normalize_slice(gt_slice)
    
    # 填充每一行每一列
    for row_idx, views in enumerate(views_list):
        for col_idx, col_name in enumerate(columns):
            ax = axes[row_idx, col_idx]
            
            if col_name == 'Ground Truth':
                # 显示 GT
                slice_data = gt_slice_norm
                title = 'Ground Truth'
            else:
                # 加载预测体积
                model_path = MODEL_PATHS[dataset][views].get(col_name)
                if model_path:
                    pred_volume = load_volume(model_path, 'pred')
                    if pred_volume is not None:
                        pred_slice = get_slice(pred_volume, axis=2, ratio=SLICE_RATIO)
                        # 使用 GT 的归一化参数
                        vmin = gt_slice.min()
                        vmax = gt_slice.max()
                        slice_data = np.clip((pred_slice - vmin) / (vmax - vmin), 0, 1)
                        title = col_name
                    else:
                        slice_data = None
                        title = f'{col_name}\n(Not Found)'
                else:
                    slice_data = None
                    title = f'{col_name}\n(N/A)'
            
            if slice_data is not None:
                im = ax.imshow(slice_data, cmap='gray', vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
            
            # 设置标题和样式
            if row_idx == 0:
                ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
            if col_idx == 0:
                ax.set_ylabel(f'{views} Views', fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
    
    # 添加总标题
    fig.suptitle(f'{dataset.capitalize()} - Ablation Study Comparison', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  已保存: {save_path}")
    return True


def main():
    print("=" * 60)
    print("生成消融实验可视化对比图")
    print("=" * 60)
    
    save_dir = os.path.join(PROJECT_ROOT, 'experiments', 'visualizations', 'ablation_comparison')
    
    datasets = ['pine', 'seashell', 'walnut']
    
    for dataset in datasets:
        save_path = os.path.join(save_dir, f'{dataset}_comparison.png')
        generate_comparison_figure(dataset, save_path)
    
    print("\n" + "=" * 60)
    print("所有对比图已生成！")
    print(f"保存位置: {os.path.abspath(save_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
