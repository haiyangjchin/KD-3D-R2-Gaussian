#!/usr/bin/env python3
"""
Start training process in background and record PID.
"""

import subprocess
import os
import sys


def main():
    cmd = [
        "python",
        "resume_from_pickle.py",
        "--config",
        "distill_student_10k_seashell_tv.yaml",
        "--output_dir",
        "distill_student_10k_seashell_tv",
        "--resume_iteration",
        "7500",
    ]

    print(f"Starting training command: {' '.join(cmd)}")

    # 打开日志文件
    log_file = "distill_10k_seashell_tv_resume_final.log"
    print(f"Log file: {log_file}")

    try:
        with open(log_file, "w") as log:
            # 启动进程
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, text=True
            )

            # 记录PID
            pid = proc.pid
            with open("training.pid", "w") as f:
                f.write(str(pid))

            print(f"Training started with PID: {pid}")
            print(f"Process will continue running in background.")
            print(f"To check if it's running: tasklist | findstr {pid}")
            print(f"To monitor logs: tail -f {log_file}")

            # 等待几秒确保进程启动成功
            import time

            time.sleep(2)

            # 检查进程是否仍在运行
            if proc.poll() is None:
                print("Training process is running successfully!")
                return 0
            else:
                print(f"Training process exited with code: {proc.poll()}")
                print("Check the log file for error messages.")
                return 1

    except Exception as e:
        print(f"Failed to start training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
