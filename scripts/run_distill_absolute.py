#!/usr/bin/env python3
"""
使用绝对路径运行蒸馏实验，避免路径问题。
"""

import os
import sys
import subprocess
import datetime

# 获取当前工作目录的绝对路径
current_dir = os.path.abspath(os.getcwd())
print(f"当前目录: {current_dir}")

# 配置文件路径（绝对路径）
config_file = os.path.join(current_dir, "distill_tv0.01.yaml")

# 输出目录路径（绝对路径）
output_dir = os.path.join(current_dir, "distill_tv0.01_absolute")
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# 教师模型路径（检查是否存在）
teacher_model = os.path.join(
    current_dir, "cnn_teacher_50epoch_v4/checkpoints/checkpoint_epoch_034.pth"
)
if not os.path.exists(teacher_model):
    print(f"警告: 教师模型文件不存在: {teacher_model}")
    # 尝试其他可能的路径
    teacher_model = "./cnn_teacher_50epoch_v4/checkpoints/checkpoint_epoch_034.pth"
    print(f"使用相对路径: {teacher_model}")

# 数据路径（检查是否存在）
data_path = os.path.join(current_dir, "data/real_dataset/cone_ntrain_25_angle_360/pine")
if not os.path.exists(data_path):
    print(f"错误: 数据目录不存在: {data_path}")
    sys.exit(1)
print(f"数据目录: {data_path}")

# 创建日志文件
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(current_dir, f"distill_tv0.01_absolute_{timestamp}.log")
print(f"日志文件: {log_file}")

# 训练命令
cmd = [
    "python",
    "train_with_distillation.py",
    "--config",
    config_file,
    "--output_dir",
    output_dir,
]

print("=" * 80)
print("启动蒸馏训练 (使用绝对路径)")
print(f"命令: {' '.join(cmd)}")
print("=" * 80)

try:
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"绝对路径训练启动 - {datetime.datetime.now()}\n")
        log.write(f"当前目录: {current_dir}\n")
        log.write(f"配置文件: {config_file}\n")
        log.write(f"输出目录: {output_dir}\n")
        log.write(f"数据目录: {data_path}\n")
        log.write(f"教师模型: {teacher_model}\n")
        log.write(f"命令: {' '.join(cmd)}\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        # 启动进程
        process = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, text=True, bufsize=1, shell=True
        )

        print(f"训练进程启动，PID: {process.pid}")
        print(f"训练进行中... 查看日志文件: {log_file}")

        # 等待进程完成
        return_code = process.wait()

        print(f"\n训练完成，返回码: {return_code}")
        if return_code == 0:
            print("训练成功!")
        else:
            print("训练失败!")

except KeyboardInterrupt:
    print("\n训练被用户中断")
except Exception as e:
    print(f"运行训练时出错: {e}")

print(f"\n训练结束时间: {datetime.datetime.now()}")
