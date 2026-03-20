#!/usr/bin/env python3
"""
计算指定目录下所有场景的3D和2D PSNR、SSIM平均值。
"""

import os
import yaml
import numpy as np
from pathlib import Path

def load_yaml(filepath):
    """加载YAML文件并返回字典。"""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def main():
    base_dir = Path("output/synthetic_dataset/cone_ntrain_25_angle_360")
    if not base_dir.exists():
        print(f"错误：目录不存在 {base_dir}")
        return

    # 收集所有场景目录（以数字开头并以'_cone'结尾的目录）
    scenes = [d for d in base_dir.iterdir() if d.is_dir() and d.name[0].isdigit() and d.name.endswith('_cone')]
    print(f"找到 {len(scenes)} 个场景")
    
    # 存储每个场景的指标
    psnr_2d_list = []
    ssim_2d_list = []
    psnr_3d_list = []
    ssim_3d_list = []
    # 用于跨所有投影计算平均值
    all_psnr_projs = []
    all_ssim_projs = []
    
    for scene in scenes:
        eval2d_path = scene / "test" / "iter_30000" / "eval2d_render_test.yml"
        eval3d_path = scene / "test" / "iter_30000" / "eval3d.yml"
        
        if not eval2d_path.exists():
            print(f"警告：{eval2d_path} 不存在，跳过场景 {scene.name}")
            continue
        if not eval3d_path.exists():
            print(f"警告：{eval3d_path} 不存在，跳过场景 {scene.name}")
            continue
        
        # 加载2D评估
        data2d = load_yaml(eval2d_path)
        psnr_2d = data2d.get('psnr_2d')
        ssim_2d = data2d.get('ssim_2d')
        psnr_projs = data2d.get('psnr_2d_projs', [])
        ssim_projs = data2d.get('ssim_2d_projs', [])
        
        # 加载3D评估
        data3d = load_yaml(eval3d_path)
        psnr_3d = data3d.get('psnr_3d')
        ssim_3d = data3d.get('ssim_3d')
        
        if psnr_2d is None or ssim_2d is None or psnr_3d is None or ssim_3d is None:
            print(f"警告：场景 {scene.name} 中缺少某些指标")
            continue
        
        psnr_2d_list.append(psnr_2d)
        ssim_2d_list.append(ssim_2d)
        psnr_3d_list.append(psnr_3d)
        ssim_3d_list.append(ssim_3d)
        all_psnr_projs.extend(psnr_projs)
        all_ssim_projs.extend(ssim_projs)
        
        print(f"{scene.name}: psnr_2d={psnr_2d:.2f}, ssim_2d={ssim_2d:.4f}, psnr_3d={psnr_3d:.2f}, ssim_3d={ssim_3d:.4f}")
    
    if not psnr_2d_list:
        print("没有找到有效数据。")
        return
    
    # 计算平均值
    avg_psnr_2d = np.mean(psnr_2d_list)
    avg_ssim_2d = np.mean(ssim_2d_list)
    avg_psnr_3d = np.mean(psnr_3d_list)
    avg_ssim_3d = np.mean(ssim_3d_list)
    
    # 计算跨所有投影的平均值
    avg_psnr_projs = np.mean(all_psnr_projs) if all_psnr_projs else None
    avg_ssim_projs = np.mean(all_ssim_projs) if all_ssim_projs else None
    
    print("\n=== 结果 ===")
    print(f"场景数量: {len(psnr_2d_list)}")
    print(f"平均 PSNR 2D (每个场景): {avg_psnr_2d:.4f}")
    print(f"平均 SSIM 2D (每个场景): {avg_ssim_2d:.6f}")
    print(f"平均 PSNR 3D (每个场景): {avg_psnr_3d:.4f}")
    print(f"平均 SSIM 3D (每个场景): {avg_ssim_3d:.6f}")
    if avg_psnr_projs is not None:
        print(f"平均 PSNR 2D (所有投影): {avg_psnr_projs:.4f}")
        print(f"平均 SSIM 2D (所有投影): {avg_ssim_projs:.6f}")
    
    # 可选：保存到文件
    output_path = base_dir / "average_metrics.txt"
    with open(output_path, 'w') as f:
        f.write(f"场景数量: {len(psnr_2d_list)}\n")
        f.write(f"平均 PSNR 2D (每个场景): {avg_psnr_2d:.4f}\n")
        f.write(f"平均 SSIM 2D (每个场景): {avg_ssim_2d:.6f}\n")
        f.write(f"平均 PSNR 3D (每个场景): {avg_psnr_3d:.4f}\n")
        f.write(f"平均 SSIM 3D (每个场景): {avg_ssim_3d:.6f}\n")
        if avg_psnr_projs is not None:
            f.write(f"平均 PSNR 2D (所有投影): {avg_psnr_projs:.4f}\n")
            f.write(f"平均 SSIM 2D (所有投影): {avg_ssim_projs:.6f}\n")
    print(f"\n结果已保存至 {output_path}")

if __name__ == "__main__":
    main()