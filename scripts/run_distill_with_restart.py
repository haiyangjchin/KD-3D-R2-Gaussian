#!/usr/bin/env python3
"""
简单重启脚本：运行蒸馏训练，如果失败则重启。
"""

import os
import sys
import time
import subprocess
import datetime


def run_training():
    """运行一次训练"""
    config_file = "distill_tv0.01.yaml"
    output_dir = "distill_tv0.01_fixed"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"distill_tv0.01_run_{timestamp}.log"

    cmd = [
        "python",
        "train_with_distillation.py",
        "--config",
        config_file,
        "--output_dir",
        output_dir,
    ]

    print(f"[{datetime.datetime.now()}] 启动训练")
    print(f"命令: {' '.join(cmd)}")
    print(f"日志文件: {log_file}")

    try:
        with open(log_file, "w", encoding="utf-8") as log:
            log.write(f"训练启动 - {datetime.datetime.now()}\n")
            log.write(f"命令: {' '.join(cmd)}\n")
            log.write("=" * 80 + "\n\n")
            log.flush()

            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=True,
            )

            print(f"进程PID: {process.pid}")

            # 等待进程完成
            return_code = process.wait()

            return return_code

    except Exception as e:
        print(f"运行训练时出错: {e}")
        return -1


def main():
    """主函数"""
    print("=" * 80)
    print("蒸馏训练重启脚本")
    print(f"开始时间: {datetime.datetime.now()}")
    print("=" * 80)

    restart_count = 0
    max_restarts = 50

    while restart_count < max_restarts:
        restart_count += 1
        print(f"\n第 {restart_count} 次尝试...")

        return_code = run_training()

        if return_code == 0:
            print(f"[{datetime.datetime.now()}] 训练成功完成!")
            break
        else:
            print(f"[{datetime.datetime.now()}] 训练失败，返回码: {return_code}")

            if restart_count < max_restarts:
                # 等待一段时间后重启
                wait_time = min(300, restart_count * 30)  # 最多等待5分钟
                print(f"等待 {wait_time} 秒后重启...")
                time.sleep(wait_time)
            else:
                print(f"已达到最大重启次数 {max_restarts}，停止尝试")

    print(f"\n[{datetime.datetime.now()}] 脚本结束")
    print(f"总计尝试次数: {restart_count}")


if __name__ == "__main__":
    main()
