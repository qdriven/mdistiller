
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np
from .DKD import DKD, dkd_loss

class CTDKD(DKD):
    """Curriculum Temperature for Decoupled Knowledge Distillation"""

    def __init__(self, student, teacher, cfg):
        super(CTDKD, self).__init__(student, teacher, cfg)
        # 保存配置对象
        self.cfg = cfg  # 添加这一行
        
        # 初始化温度参数
        try:
            self.current_temperature = nn.Parameter(
                torch.tensor(cfg.CTDKD.INIT_TEMPERATURE, dtype=torch.float32),
                requires_grad=True
            )
            self.min_temperature = cfg.CTDKD.MIN_TEMPERATURE
            self.max_temperature = cfg.CTDKD.MAX_TEMPERATURE
            self.temp_optimizer = torch.optim.Adam(
                [self.current_temperature],
                lr=cfg.CTDKD.LEARNING_RATE,
                weight_decay=0.0001
            )
            print(f"CTDKD initialized with temperature: {self.current_temperature.item()}")
            print(f"Temperature bounds: [{self.min_temperature}, {self.max_temperature}]")
            print(f"Learning rate: {cfg.CTDKD.LEARNING_RATE}")
        except AttributeError as e:
            print(f"Error initializing CTDKD: {e}")
            print("Please ensure CTDKD configuration is defined in cfg.py")
            self.current_temperature = nn.Parameter(
                torch.tensor(cfg.DKD.T, dtype=torch.float32),
                requires_grad=True
            )
            self.min_temperature = 1.0
            self.max_temperature = 10.0
            self.temp_optimizer = torch.optim.Adam(
                [self.current_temperature],
                lr=0.001,
                weight_decay=0.0001
            )
            print(f"Using default temperature: {self.current_temperature.item()}")
        
        # 获取总训练轮数，如果配置中没有则使用默认值
        self.total_epochs = getattr(cfg.SOLVER, 'EPOCHS', 200)  # 默认200轮
        
        # 其他初始化
        self.warmup = 5  # 温度warmup轮数
        self.step_count = 0
        self.temp_update_freq = 5
        self.temp_log = []
        self.temp_log.append(self.current_temperature.item())
        
        # 设置日志路径
        if hasattr(cfg, 'LOG') and hasattr(cfg.LOG, 'PREFIX'):
            self.temp_log_path = os.path.join(cfg.LOG.PREFIX, "temperature_log.txt")
        else:
            self.temp_log_path = "temperature_log.txt"

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # 数值稳定性处理
        logits_student = torch.nan_to_num(logits_student, nan=0.0, posinf=1e5, neginf=-1e5)
        logits_teacher = torch.nan_to_num(logits_teacher, nan=0.0, posinf=1e5, neginf=-1e5)

        # 计算CE损失
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # 获取当前epoch，默认为1
        epoch = kwargs.get("epoch", 1)
        
        # 使用类变量中的总轮数
        progress = epoch / self.total_epochs
        
        # 动态调整温度范围
        dynamic_max_temp = self.max_temperature * (1.0 - 0.5 * progress)
        dynamic_min_temp = self.min_temperature + progress

        # 计算性能差距
        with torch.no_grad():
            student_ce = F.cross_entropy(logits_student, target)
            teacher_ce = F.cross_entropy(logits_teacher, target)
            perf_gap = torch.clamp(student_ce - teacher_ce, 0.0, 10.0)

        # 基于性能差距计算目标温度
        target_temp = dynamic_min_temp + (dynamic_max_temp - dynamic_min_temp) * (perf_gap / 5.0)
        
        # 更新温度
        if self.step_count % self.temp_update_freq == 0:
            temp_grad = (target_temp - self.current_temperature.detach()) * 0.1
            self.temp_optimizer.zero_grad()
            self.current_temperature.grad = temp_grad
            self.temp_optimizer.step()
            
            with torch.no_grad():
                self.current_temperature.copy_(
                    torch.clamp(self.current_temperature, 
                              min=dynamic_min_temp,
                              max=dynamic_max_temp)
                )

        # 使用当前温度计算KD损失
        temperature = self.current_temperature.detach()
        
        # 应用warmup
        warmup_factor = min(epoch / self.warmup, 1.0)
        
        loss_dkd = warmup_factor * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            temperature,
        )

        # 记录当前温度到日志
        if self.step_count % 10 == 0 and hasattr(self.cfg, 'LOG') and hasattr(self.cfg.LOG, 'PREFIX'):
            with open(os.path.join(self.cfg.LOG.PREFIX, "worklog.txt"), "a") as f:
                f.write(f"[TEMP] Step={self.step_count}, T={temperature.item():.4f}\n")

        self.step_count += 1
        
        # 确保温度被记录到temp_log
        self.temp_log.append(temperature.item())

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
            "temperature": temperature.item(),  # 确保这是一个Python标量
            "current_temperature": temperature.item(),  # 添加两种形式以确保兼容性
        }

        return logits_student, losses_dict

    def update_temperature(self):
        """
        Update temperature using the optimizer.
        Called by trainer after each iteration.
        """
        # Skip if too early or not time to update
        if self.step_count < 100 or self.step_count % self.temp_update_freq != 0:
            return
            
        # Clamp temperature to ensure it stays within bounds
        with torch.no_grad():
            self.current_temperature.copy_(
                torch.clamp(self.current_temperature, self.min_temperature, self.max_temperature)
            )
        
        # Cyclical temperature adjustment based on epoch
        if not self.adaptive_update and self.epoch_count > 0:
            # Implement cyclical temperature schedule
            cycle_length = 40  # epochs per cycle
            cycle_position = (self.epoch_count % cycle_length) / cycle_length
            
            # Calculate target temperature based on position in cycle
            # Start high, go low, then back to high
            if cycle_position < 0.5:  # First half: decrease temperature
                target_temp = self.max_temperature - (self.max_temperature - self.min_temperature) * (cycle_position * 2)
            else:  # Second half: increase temperature
                target_temp = self.min_temperature + (self.max_temperature - self.min_temperature) * ((cycle_position - 0.5) * 2)
            
            # Apply new temperature
            with torch.no_grad():
                self.current_temperature.copy_(torch.tensor(target_temp, dtype=torch.float32))
            
            temp_value = self.current_temperature.item()
            if self.step_count % 100 == 0:
                print(f"Cyclical temperature update: {temp_value:.4f}, epoch: {self.epoch_count}, position: {cycle_position:.2f}")
                
                if hasattr(self.cfg, 'LOG') and hasattr(self.cfg.LOG, 'PREFIX'):
                    log_dir = self.cfg.LOG.PREFIX
                    with open(os.path.join(log_dir, "worklog.txt"), "a") as f:
                        f.write(f"Cyclical temperature update: {temp_value:.4f}, epoch: {self.epoch_count}\n")

    def _save_temp_log(self):
        """Save temperature log to JSON file"""
        try:
            # 确定保存路径
            save_path = "temperature_log_CTDKD_.json"  # 默认路径
            
            # 如果有配置的日志路径，使用配置的路径
            if hasattr(self, 'cfg') and hasattr(self.cfg, 'LOG') and hasattr(self.cfg.LOG, 'PREFIX'):
                log_dir = self.cfg.LOG.PREFIX
                save_path = os.path.join(log_dir, "temperature_log_CTDKD_.json")
                
            # 保存日志
            if self.temp_log:  # 只在有数据时保存
                with open(save_path, 'w') as f:
                    json.dump(self.temp_log, f)
                print(f"Temperature log saved to {save_path}")
        except Exception as e:
            print(f"Error saving temperature log: {e}")

    def forward_test(self, image):
        # Save final temperature log before testing
        if self.temp_log:
            self._save_temp_log()
            
        return super().forward_test(image)

    def get_current_temperature(self):
        """返回当前温度值"""
        return self.current_temperature.item()
