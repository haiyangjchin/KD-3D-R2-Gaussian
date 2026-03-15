#!/usr/bin/env python3
"""
连续运行蒸馏实验，确保训练不中断。
如果训练失败，自动重启。
"""

import os
import sys
import time
import subprocess
import datetime
import signal
import threading


class ContinuousTrainer:
    def __init__(self, config_file, output_dir):
        self.config_file = config_file
        self.output_dir = output_dir
        self.process = None
        self.restart_count = 0
        self.max_restarts = 100  # 最大重启次数
        self.check_interval = 30  # 检查间隔（秒）
        self.running = True

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir} (已创建)")

    def start_training(self):
        """启动训练进程"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{self.output_dir}_continuous_{timestamp}.log"

        cmd = [
            "python",
            "train_with_distillation.py",
            "--config",
            self.config_file,
            "--output_dir",
            self.output_dir,
        ]

        self.restart_count += 1
        print(f"[{datetime.datetime.now()}] 启动训练 (第{self.restart_count}次)")
        print(f"命令: {' '.join(cmd)}")
        print(f"日志文件: {log_file}")

        try:
            with open(log_file, "w", encoding="utf-8") as log:
                # 写入启动信息
                log.write(f"连续训练启动 - {datetime.datetime.now()}\n")
                log.write(f"重启次数: {self.restart_count}\n")
                log.write(f"命令: {' '.join(cmd)}\n")
                log.write("=" * 80 + "\n\n")
                log.flush()

                # 启动进程
                self.process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    shell=True,
                )

                print(f"进程PID: {self.process.pid}")
                return True

        except Exception as e:
            print(f"启动训练进程时出错: {e}")
            return False

    def check_process(self):
        """检查进程状态"""
        if self.process is None:
            return False

        # 检查进程是否仍在运行
        return_code = self.process.poll()
        if return_code is not None:
            print(f"[{datetime.datetime.now()}] 进程已结束，返回码: {return_code}")
            self.process = None
            return False

        return True

    def monitor_logs(self):
        """监控日志文件状态（简化版）"""
        # 查找最新的日志文件
        log_files = []
        for f in os.listdir("."):
            if f.startswith(self.output_dir) and f.endswith(".log"):
                log_files.append(f)

        if not log_files:
            return True  # 没有日志文件，可能是刚开始

        # 找到最新的日志文件
        latest_log = max(log_files, key=lambda x: os.path.getmtime(x))

        # 检查文件是否最近有更新
        try:
            mtime = os.path.getmtime(latest_log)
            current_time = time.time()

            # 如果超过10分钟没有更新，可能进程挂起了
            if (current_time - mtime) > 600:  # 10分钟
                print(
                    f"[{datetime.datetime.now()}] 日志文件 {latest_log} 超过10分钟未更新，进程可能挂起"
                )
                return False
        except Exception as e:
            print(f"检查日志文件时出错: {e}")

        return True

    def stop(self):
        """停止训练和监控"""
        self.running = False
        if self.process:
            print(f"[{datetime.datetime.now()}] 终止训练进程...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("进程未响应，强制终止")
                self.process.kill()

    def run(self):
        """主运行循环"""
        print("=" * 80)
        print("连续蒸馏训练启动")
        print(f"配置文件: {self.config_file}")
        print(f"输出目录: {self.output_dir}")
        print(f"开始时间: {datetime.datetime.now()}")
        print("=" * 80)

        # 初始启动
        if not self.start_training():
            print("初始启动失败，等待后重试...")
            time.sleep(60)

        last_check = time.time()

        try:
            while self.running and self.restart_count <= self.max_restarts:
                current_time = time.time()

                # 定期检查进程状态
                if current_time - last_check >= self.check_interval:
                    process_ok = self.check_process()
                    logs_ok = self.monitor_logs()

                    if not process_ok or not logs_ok:
                        print(f"[{datetime.datetime.now()}] 检测到问题，重启训练...")

                        # 终止可能挂起的进程
                        if self.process:
                            try:
                                self.process.terminate()
                                self.process.wait(timeout=5)
                            except:
                                pass
                            self.process = None

                        # 等待一段时间再重启
                        wait_time = min(60, self.restart_count * 10)
                        print(f"等待{wait_time}秒后重启...")
                        time.sleep(wait_time)

                        # 重启训练
                        self.start_training()

                    last_check = current_time

                # 短暂休眠避免CPU占用过高
                time.sleep(5)

        except KeyboardInterrupt:
            print(f"\n[{datetime.datetime.now()}] 用户中断")
        except Exception as e:
            print(f"\n[{datetime.datetime.now()}] 发生错误: {e}")
        finally:
            self.stop()
            print(f"\n[{datetime.datetime.now()}] 连续训练结束")
            print(f"总计重启次数: {self.restart_count}")


def main():
    # 使用distill_tv0.01配置
    config_file = "distill_tv0.01.yaml"
    output_dir = "distill_tv0.01_continuous"

    trainer = ContinuousTrainer(config_file, output_dir)

    # 设置信号处理
    import signal as sig

    def signal_handler(signum, frame):
        print(f"\n接收到信号 {signum}，正在停止...")
        trainer.stop()
        sys.exit(0)

    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)

    # 运行训练
    trainer.run()


if __name__ == "__main__":
    main()
