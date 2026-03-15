#!/usr/bin/env python3
"""
简单检查训练进程，如果停止则重启。
每10分钟检查一次。
"""

import os
import sys
import time
import subprocess
import datetime


def is_training_running():
    """检查训练进程是否在运行"""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            shell=True,
        )
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if "train_with_distillation" in line or "distill_tv0.01" in line:
                return True
    except Exception as e:
        print(f"检查进程时出错: {e}")
    return False


def restart_training():
    """重启训练进程"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"distill_tv0.01_restart_{timestamp}.log"

    cmd = [
        "python",
        "train_with_distillation.py",
        "--config",
        "distill_tv0.01.yaml",
        "--output_dir",
        "distill_tv0.01",
    ]

    print(f"[{datetime.datetime.now()}] 重启训练...")
    print(f"命令: {' '.join(cmd)}")
    print(f"日志文件: {log_file}")

    try:
        with open(log_file, "w") as log:
            log.write(f"训练重启 - {datetime.datetime.now()}\n")
            log.write(f"命令: {' '.join(cmd)}\n")
            log.write("=" * 80 + "\n\n")
            log.flush()

            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=True,
            )

            print(f"新进程PID: {process.pid}")
            return True

    except Exception as e:
        print(f"重启训练时出错: {e}")
        return False


def main():
    """主检查循环"""
    print("=" * 80)
    print("训练进程检查器启动")
    print(f"开始时间: {datetime.datetime.now()}")
    print("每10分钟检查一次训练进程")
    print("=" * 80)

    check_interval = 600  # 10分钟
    last_check = datetime.datetime.now()

    try:
        while True:
            now = datetime.datetime.now()

            if not is_training_running():
                print(f"[{now}] 训练进程未运行，尝试重启...")
                success = restart_training()
                if not success:
                    print(f"[{now}] 重启失败，将在{check_interval}秒后重试")
            else:
                # 计算距离上次检查的时间
                elapsed = (now - last_check).total_seconds()
                if elapsed >= 3600:  # 每1小时报告一次
                    print(f"[{now}] 训练进程正常运行 (已运行{elapsed / 3600:.1f}小时)")
                    last_check = now

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print(f"\n[{datetime.datetime.now()}] 检查器被用户中断")
        print("训练进程可能仍在运行，请手动检查")


if __name__ == "__main__":
    main()
