#!/usr/bin/env python3
"""
监控训练进程，确保蒸馏实验持续运行。
如果进程停止，自动重启训练。
"""

import os
import sys
import time
import subprocess
import signal
import datetime

# 训练配置
CONFIG_FILE = "distill_tv0.01.yaml"
OUTPUT_DIR = "distill_tv0.01"
LOG_FILE = "distill_tv0.01_monitored.log"
CHECK_INTERVAL = 60  # 检查间隔（秒）
STALE_THRESHOLD = 300  # 日志停滞阈值（秒）


def get_process_pid():
    """获取训练进程的PID"""
    try:
        # Windows下使用tasklist命令
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            shell=True,
        )
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if "train_with_distillation" in line or CONFIG_FILE in line:
                # CSV格式: "python.exe","1234","Session Name","Session#","Mem Usage"
                parts = line.split(",")
                if len(parts) >= 2:
                    pid = parts[1].strip('"')
                    return int(pid)
    except Exception as e:
        print(f"获取进程PID时出错: {e}")
    return None


def is_log_active(log_file, threshold=STALE_THRESHOLD):
    """检查日志文件是否在最近有更新"""
    if not os.path.exists(log_file):
        return False

    try:
        mtime = os.path.getmtime(log_file)
        current_time = time.time()
        return (current_time - mtime) < threshold
    except Exception as e:
        print(f"检查日志文件时出错: {e}")
        return False


def start_training():
    """启动训练进程"""
    cmd = [
        "python",
        "train_with_distillation.py",
        "--config",
        CONFIG_FILE,
        "--output_dir",
        OUTPUT_DIR,
    ]

    # 创建日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"distill_tv0.01_auto_{timestamp}.log"

    print(f"[{datetime.datetime.now()}] 启动训练进程...")
    print(f"命令: {' '.join(cmd)}")
    print(f"日志文件: {log_file}")

    try:
        with open(log_file, "w") as log:
            # 写入启动信息
            log.write(f"自动重启训练 - {datetime.datetime.now()}\n")
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

            print(f"进程启动，PID: {process.pid}")
            return process.pid, log_file

    except Exception as e:
        print(f"启动训练进程时出错: {e}")
        return None, None


def main():
    """主监控循环"""
    print("=" * 80)
    print("蒸馏训练监控器启动")
    print(f"配置文件: {CONFIG_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"检查间隔: {CHECK_INTERVAL}秒")
    print("=" * 80)

    current_pid = None
    current_log = None
    restart_count = 0

    # 检查是否已有进程在运行
    pid = get_process_pid()
    if pid:
        print(f"[{datetime.datetime.now()}] 发现已有训练进程，PID: {pid}")
        current_pid = pid
        # 查找最近的日志文件
        log_files = [
            f
            for f in os.listdir(".")
            if f.startswith("distill_tv0.01") and f.endswith(".log")
        ]
        if log_files:
            current_log = max(log_files, key=os.path.getmtime)
            print(f"使用现有日志文件: {current_log}")

    try:
        while True:
            now = datetime.datetime.now()

            # 检查进程是否存在
            if current_pid:
                pid = get_process_pid()
                if not pid or pid != current_pid:
                    print(f"[{now}] 训练进程已停止 (原PID: {current_pid})")
                    current_pid = None

            # 检查日志是否活跃
            if current_pid and current_log:
                if not is_log_active(current_log):
                    print(
                        f"[{now}] 日志文件已停滞超过{STALE_THRESHOLD}秒，可能进程挂起"
                    )
                    # 强制终止进程
                    try:
                        os.kill(current_pid, signal.SIGTERM)
                        print(f"已终止挂起的进程 {current_pid}")
                    except Exception as e:
                        print(f"终止进程时出错: {e}")
                    current_pid = None

            # 如果需要重启
            if not current_pid:
                restart_count += 1
                print(f"[{now}] 重启训练 (第{restart_count}次重启)")
                new_pid, new_log = start_training()
                if new_pid:
                    current_pid = new_pid
                    current_log = new_log
                else:
                    print(f"[{now}] 启动失败，等待{CHECK_INTERVAL}秒后重试")

            # 等待下一个检查周期
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n[{datetime.datetime.now()}] 监控器被用户中断")
        print(f"总计重启次数: {restart_count}")
        if current_pid:
            print(f"当前训练进程PID: {current_pid}")
            print("警告: 训练进程仍在运行，请手动终止")


if __name__ == "__main__":
    main()
