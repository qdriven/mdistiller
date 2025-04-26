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
        # 初始化温度参数
        try:
            self.current_temperature = nn.Parameter(
                torch.tensor(cfg.CTDKD.INIT_TEMPERATURE, dtype=torch.float32),
                requires_grad=True  # Changed to True to enable direct optimization
            )
            self.min_temperature = cfg.CTDKD.MIN_TEMPERATURE
            self.max_temperature = cfg.CTDKD.MAX_TEMPERATURE
            self.temp_optimizer = torch.optim.Adam(
                [self.current_temperature],
                lr=cfg.CTDKD.LEARNING_RATE,
                weight_decay=0.0001  # Added weight decay for stability
            )
            print(f"CTDKD initialized with temperature: {self.current_temperature.item()}")
            print(f"Temperature bounds: [{self.min_temperature}, {self.max_temperature}]")
            print(f"Learning rate: {cfg.CTDKD.LEARNING_RATE}")
        except AttributeError as e:
            print(f"Error initializing CTDKD: {e}")
            print("Please ensure CTDKD configuration is defined in cfg.py")
            # 使用DKD的默认温度作为备选
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
        
        # 保存上一次的loss_dkd用于更新温度
        self.last_loss_dkd = None
        self.last_logits_student = None
        self.last_logits_teacher = None
        self.last_target = None
        
        # 用于温度记录
        self.cfg = cfg
        self.step_count = 0
        self.epoch_count = 0
        self.temp_log = []
        self.temp_log.append(self.current_temperature.item())  # 记录初始温度
        
        # 防止梯度爆炸
        self.max_grad_norm = 1.0
        
        # 温度调整策略参数
        self.temp_update_freq = 5  # Update temperature every 5 steps
        self.adaptive_update = True  # Use adaptive temperature adjustment
        
    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        # 检查输入是否包含 nan 或 inf 值
        logits_student = torch.nan_to_num(logits_student, nan=0.0, posinf=1e5, neginf=-1e5)
        logits_teacher = torch.nan_to_num(logits_teacher, nan=0.0, posinf=1e5, neginf=-1e5)

        # Store current epoch for adaptive temperature adjustment
        self.epoch_count = kwargs.get("epoch", 1)
        
        # 使用当前温度计算损失
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        # Ensure temperature is valid
        with torch.no_grad():
            if not torch.isfinite(self.current_temperature):
                print("Warning: Temperature is NaN or Inf, resetting to initial value")
                self.current_temperature.copy_(torch.tensor(self.cfg.CTDKD.INIT_TEMPERATURE, 
                                                          dtype=torch.float32))
        
        # Clamp temperature and use the clamped value for training
        temperature = torch.clamp(
            self.current_temperature,
            self.min_temperature,
            self.max_temperature
        )
        
        # Calculate KD loss with warmup
        warmup_factor = min(self.epoch_count / self.warmup, 1.0)
        
        loss_dkd = warmup_factor * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            temperature,
        )
        
        # Safety check for NaN losses
        if not torch.isfinite(loss_ce):
            loss_ce = torch.tensor(1.0, device=logits_student.device)
        if not torch.isfinite(loss_dkd):
            loss_dkd = torch.tensor(1.0, device=logits_student.device)
        
        # Temperature optimization loss - based on student-teacher performance gap
        student_ce = F.cross_entropy(logits_student, target)
        with torch.no_grad():
            teacher_ce = F.cross_entropy(logits_teacher, target)
        
        # Calculate performance gap
        perf_gap = torch.clamp(student_ce - teacher_ce, 0.0, 10.0)
        
        # Create temperature optimization loss
        # If student is far from teacher (large gap), increase temperature
        # If student is close to teacher (small gap), decrease temperature
        # This creates a curriculum that adapts to student's progress
        temp_loss = torch.tensor(0.0, device=logits_student.device)
        
        if self.adaptive_update and self.step_count > 100:  # Start adaptive updates after 100 steps
            # Target temperature: higher when gap is large, lower when gap is small
            # Scaled between min_temp and max_temp
            target_temp = self.min_temperature + (self.max_temperature - self.min_temperature) * (perf_gap / 2.0)
            target_temp = torch.clamp(target_temp, self.min_temperature, self.max_temperature)
            
            # Loss to push temperature toward target
            temp_loss = 0.1 * torch.abs(temperature - target_temp)
        
        # Record temperature
        temp_value = temperature.item()
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}, Temperature: {temp_value:.4f}, Gap: {perf_gap.item():.4f}")
            
            # Log temperature value
            try:
                if hasattr(self.cfg, 'LOG') and hasattr(self.cfg.LOG, 'PREFIX'):
                    log_dir = self.cfg.LOG.PREFIX
                    with open(os.path.join(log_dir, "worklog.txt"), "a") as f:
                        f.write(f"Step {self.step_count}, Temperature: {temp_value:.4f}, Gap: {perf_gap.item():.4f}\n")
            except Exception as e:
                print(f"Error writing to worklog: {e}")
        
        # Record temperature value for logging
        self.step_count += 1
        if self.step_count % 50 == 0:
            # Log every 50 steps
            self.temp_log.append(temp_value)
            self._save_temp_log()
        
        # Add to losses_dict
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
            "loss_temp_opt": temp_loss,  # Add temperature optimization loss
            "temperature": torch.tensor(temp_value).to(loss_ce.device)
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
            # Determine the save path
            if hasattr(self.cfg, 'LOG') and hasattr(self.cfg.LOG, 'PREFIX'):
                log_dir = self.cfg.LOG.PREFIX
                save_path = os.path.join(log_dir, "temperature_log_CTDKD_.json")
            else:
                save_path = "temperature_log_CTDKD_.json"
                
            # Save the log
            with open(save_path, 'w') as f:
                json.dump(self.temp_log, f)
        except Exception as e:
            print(f"Error saving temperature log: {e}")

    def forward_test(self, image):
        # Save final temperature log before testing
        if self.temp_log:
            self._save_temp_log()
            
        return super().forward_test(image) 