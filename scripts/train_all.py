# Script to train all cases (Windows 适配版)
import os
import os.path as osp
import glob
import subprocess
import argparse


def main(args):
    source_path = args.source
    output_path = args.output
    device = args.device
    config_path = args.config

    # 获取所有案例文件夹（如 0_chest_cone、1_beetle_cone 等）
    case_paths = sorted(glob.glob(osp.join(source_path, "*")))

    if len(case_paths) == 0:
        raise ValueError(f"在 {source_path} 下未找到任何案例文件夹！")

    # 遍历每个案例训练
    for idx, case_path in enumerate(case_paths):
        case_name = osp.basename(case_path)
        case_output_path = osp.join(output_path, case_name)  # 用 osp.join 适配 Windows 路径
        
        # 跳过已训练的案例（可选，避免重复训练）
        if osp.exists(case_output_path):
            print(f"[{idx+1}/{len(case_paths)}] 案例 {case_name} 已存在，跳过...")
            continue

        # Windows 下设置 CUDA 设备的方式
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)

        # 构建训练命令
        cmd = [
            "python", "train.py", 
            "-s", case_path, 
            "-m", case_output_path
        ]
        if config_path:
            cmd += ["--config", config_path]

        # 执行命令（更安全的 subprocess 方式）
        print(f"\n[{idx+1}/{len(case_paths)}] 开始训练案例: {case_name}")
        result = subprocess.run(cmd, env=env, shell=True)
        
        if result.returncode != 0:
            print(f"案例 {case_name} 训练失败！返回码: {result.returncode}")
            break  # 可选：失败时停止，或继续下一个案例


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/synthetic_dataset/cone_ntrain_50_angle_360", type=str, help="数据集根路径（如 synthetic_dataset/cone_ntrain_25_angle_360）")
    parser.add_argument("--output", default="output/synthetic_dataset/cone_ntrain_50_angle_360", type=str, help="输出根路径")
    parser.add_argument("--config", default=None, type=str, help="配置文件路径")
    parser.add_argument("--device", default=0, type=int, help="GPU 设备编号")
    args = parser.parse_args()
    main(args)