#!/usr/bin/env python3
"""
增强版tv0.01消融实验训练脚本，确保训练不中断。
如果训练失败，自动重启，并提供详细的训练进度监控。
"""

import os
import sys
import time
import subprocess
import datetime
import signal
import threading
import json


class EnhancedContinuousTrainer:
    def __init__(self, config_file, output_dir):
        self.config_file = config_file
        self.output_dir = output_dir
        self.process = None
        self.restart_count = 0
        self.max_restarts = 100  # 最大重启次数
        self.check_interval = 30  # 检查间隔（秒）
        self.running = True
        self.training_start_time = None
        self.total_training_time = 0
        self.last_iteration = 0
        self.iteration_speed = 0
        self.current_loss = 0
        self.log_file = None

        # 设置环境变量以减少CUDA内存碎片
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir} (已创建)")

        # 创建训练状态文件
        self.status_file = os.path.join(output_dir, "training_status.json")
        self.save_training_status()

    def save_training_status(self):
        """保存训练状态到JSON文件"""
        status = {
            "config_file": self.config_file,
            "output_dir": self.output_dir,
            "restart_count": self.restart_count,
            "last_iteration": self.last_iteration,
            "total_training_time": self.total_training_time,
            "current_loss": self.current_loss,
            "iteration_speed": self.iteration_speed,
            "last_update": datetime.datetime.now().isoformat(),
            "training_start_time": self.training_start_time.isoformat()
            if self.training_start_time
            else None,
        }
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存训练状态时出错: {e}")

    def start_training(self):
        """启动训练进程"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{self.output_dir}_continuous_{timestamp}.log"

        if self.training_start_time is None:
            self.training_start_time = datetime.datetime.now()

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
        print(f"日志文件: {self.log_file}")
        print(f"状态文件: {self.status_file}")

        try:
            with open(self.log_file, "w", encoding="utf-8") as log:
                # 写入启动信息
                log.write(f"增强版消融实验训练启动 - {datetime.datetime.now()}\n")
                log.write(f"配置文件: {self.config_file}\n")
                log.write(f"输出目录: {self.output_dir}\n")
                log.write(f"重启次数: {self.restart_count}\n")
                log.write(f"命令: {' '.join(cmd)}\n")
                log.write(f"开始时间: {self.training_start_time}\n")
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
                print(f"训练已启动，可通过以下命令实时查看进度:")
                print(f"  tail -f {self.log_file}")
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
        """增强版日志监控，提取训练进度信息"""
        if not self.log_file or not os.path.exists(self.log_file):
            return True  # 没有日志文件，可能是刚开始

        try:
            # 检查文件是否最近有更新
            mtime = os.path.getmtime(self.log_file)
            current_time = time.time()

            # 如果超过10分钟没有更新，可能进程挂起了
            if (current_time - mtime) > 600:  # 10分钟
                print(
                    f"[{datetime.datetime.now()}] 日志文件 {self.log_file} 超过10分钟未更新，进程可能挂起"
                )
                return False

            # 尝试读取最后几行日志，提取训练进度
            with open(self.log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if not lines:
                return True

            # 从最后100行中查找训练进度信息
            search_lines = lines[-100:] if len(lines) > 100 else lines
            for line in reversed(search_lines):
                # 查找迭代信息
                if "Train with Distillation:" in line:
                    # 提取迭代进度
                    try:
                        # 查找类似 "| 1234/10000" 的格式
                        import re

                        match = re.search(r"(\d+)/(\d+)", line)
                        if match:
                            current = int(match.group(1))
                            total = int(match.group(2))
                            self.last_iteration = current

                            # 提取损失值
                            loss_match = re.search(r"loss=([\d\.e\+\-]+)", line)
                            if loss_match:
                                self.current_loss = float(loss_match.group(1))

                            # 提取迭代速度
                            speed_match = re.search(r"(\d+\.?\d*)\s*it/s", line)
                            if speed_match:
                                self.iteration_speed = float(speed_match.group(1))

                            # 更新总训练时间
                            if self.training_start_time:
                                elapsed = (
                                    datetime.datetime.now() - self.training_start_time
                                ).total_seconds()
                                self.total_training_time = elapsed

                            # 保存状态
                            self.save_training_status()

                            # 打印进度摘要
                            if current % 100 == 0:  # 每100次迭代打印一次
                                progress = current / total * 100
                                eta = "N/A"
                                if self.iteration_speed > 0:
                                    remaining = (total - current) / self.iteration_speed
                                    eta = str(
                                        datetime.timedelta(seconds=int(remaining))
                                    )

                                print(
                                    f"[{datetime.datetime.now()}] 进度: {progress:.1f}% ({current}/{total}), "
                                    f"损失: {self.current_loss:.2e}, 速度: {self.iteration_speed:.1f}it/s, "
                                    f"ETA: {eta}"
                                )
                    except Exception as e:
                        # 解析失败不影响监控
                        pass

                    break  # 找到最新进度后退出

        except Exception as e:
            print(f"监控日志文件时出错: {e}")

        return True

    def print_training_summary(self):
        """打印训练摘要"""
        if self.last_iteration > 0:
            print("\n" + "=" * 80)
            print("当前训练摘要:")
            print(f"  最新迭代: {self.last_iteration}")
            print(f"  当前损失: {self.current_loss:.2e}")
            print(f"  迭代速度: {self.iteration_speed:.1f} it/s")
            if self.total_training_time > 0:
                print(
                    f"  总训练时间: {str(datetime.timedelta(seconds=int(self.total_training_time)))}"
                )
            print("=" * 80)

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

        # 打印最终摘要
        self.print_training_summary()

    def run(self):
        """主运行循环"""
        print("=" * 80)
        print("增强版tv0.01消融实验训练启动")
        print(f"配置文件: {self.config_file}")
        print(f"输出目录: {self.output_dir}")
        print(f"开始时间: {datetime.datetime.now()}")
        print(f"最大重启次数: {self.max_restarts}")
        print(f"检查间隔: {self.check_interval}秒")
        print("=" * 80)

        # 显示实时监控提示
        print("\n实时监控提示:")
        print(f"1. 查看实时日志: tail -f {self.output_dir}_continuous_*.log")
        print(f"2. 查看训练状态: cat {self.status_file} | python -m json.tool")
        print(f"3. 查看输出目录: ls -la {self.output_dir}/")
        print("=" * 80 + "\n")

        # 初始启动
        if not self.start_training():
            print("初始启动失败，等待后重试...")
            time.sleep(60)

        last_check = time.time()
        last_summary = time.time()

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

                # 每5分钟打印一次训练摘要
                if current_time - last_summary >= 300:  # 5分钟
                    self.print_training_summary()
                    last_summary = current_time

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
            print(f"最终迭代: {self.last_iteration}")
            print(f"最终损失: {self.current_loss:.2e}")


def main():
    # 使用distill_tv0.01.yaml配置文件
    config_file = "distill_tv0.01.yaml"
    # 使用新的输出目录避免冲突
    output_dir = "distill_tv0.01_new"

    # 检查配置文件是否存在
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在: {config_file}")
        print(f"当前目录: {os.getcwd()}")
        sys.exit(1)

    trainer = EnhancedContinuousTrainer(config_file, output_dir)

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
