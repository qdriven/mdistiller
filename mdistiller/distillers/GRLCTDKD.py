import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np

from ._base import Distiller
from .DKD import DKD, dkd_loss
from .grl_temperature import GlobalTemperature # Import the new module

def stable_kd_loss(logits_student, logits_teacher, temperature, eps=1e-8):
    """一个更稳定的知识蒸馏损失函数，专用于温度优化"""
    # 数值稳定性处理：缩放和裁剪输入
    max_student = torch.max(torch.abs(logits_student))
    max_teacher = torch.max(torch.abs(logits_teacher))
    
    if max_student > 1e4:
        logits_student = logits_student / max_student * 1e4
    if max_teacher > 1e4:
        logits_teacher = logits_teacher / max_teacher * 1e4
    
    # 应用温度缩放
    scaled_student = logits_student / temperature
    scaled_teacher = logits_teacher / temperature
    
    # 数值稳定性：减去每行的最大值
    scaled_student -= torch.max(scaled_student, dim=1, keepdim=True)[0]
    scaled_teacher -= torch.max(scaled_teacher, dim=1, keepdim=True)[0]
    
    # 计算 softmax 概率
    p_student = F.softmax(scaled_student, dim=1)
    with torch.no_grad():
        p_teacher = F.softmax(scaled_teacher, dim=1)
    
    # 使用 KL 散度计算损失，添加 eps 避免数值问题
    loss = p_teacher * (torch.log(p_teacher + eps) - torch.log(p_student + eps))
    
    # 沿着类别维度求和，然后取批次平均值
    loss = loss.sum(dim=1).mean()
    
    # 乘以温度的平方，并放大系数以产生更强的梯度
    loss = loss * (temperature ** 2) * 8.0  # Increased from 5.0 to 8.0 for stronger optimization
    
    return loss

class GRLCTDKD(DKD):
    """ Curriculum Temperature KD using Gradient Reversal Layer """

    def __init__(self, student, teacher, cfg):
        super(GRLCTDKD, self).__init__(student, teacher, cfg)
        self.cfg = cfg
        
        # Default values
        init_temp = 4.0
        self.grl_lambda = 0.2
        self.min_temp = 1.0
        self.max_temp = 20.0
        lr_temp = 0.0005
        
        # Read configuration
        try:
            if hasattr(cfg.GRLCTDKD, 'INIT_TEMPERATURE'):
                init_temp = cfg.GRLCTDKD.INIT_TEMPERATURE
            
            if hasattr(cfg.GRLCTDKD, 'GRL_LAMBDA'):
                self.grl_lambda = cfg.GRLCTDKD.GRL_LAMBDA
            
            if hasattr(cfg.GRLCTDKD, 'MIN_TEMPERATURE'):
                self.min_temp = cfg.GRLCTDKD.MIN_TEMPERATURE
            
            if hasattr(cfg.GRLCTDKD, 'MAX_TEMPERATURE'):
                self.max_temp = cfg.GRLCTDKD.MAX_TEMPERATURE
            
            if hasattr(cfg.GRLCTDKD, 'LEARNING_RATE'):
                lr_temp = cfg.GRLCTDKD.LEARNING_RATE
            
            # Create temperature module with gradient reversal
            self.temp_module = GlobalTemperature(initial_temperature=init_temp)
            
            # Ensure requires_grad is True
            self.temp_module.global_t.requires_grad_(True)
                
            # Create a separate optimizer for the temperature parameter
            self.temp_optimizer = torch.optim.Adam([self.temp_module.global_t], lr=lr_temp)
            
            print(f"GRLCTDKD initialized with the following parameters:")
            print(f"- Initial temperature: {init_temp}")
            print(f"- GRL lambda: {self.grl_lambda}")
            print(f"- Temperature learning rate: {lr_temp}")
            print(f"- Min temperature: {self.min_temp}")
            print(f"- Max temperature: {self.max_temp}")
            
        except Exception as e:
            print(f"Error initializing GRLCTDKD: {e}")
            print("Falling back to fixed DKD temperature.")
            self.temp_module = None
            self.temp_optimizer = None

        self.temp_log = []
        if self.temp_module:
            # Log initial temperature
            self.temp_log.append(self.temp_module.global_t.item())
        
        self.step_count = 0
        self.current_temperature = init_temp if self.temp_module else self.temperature
        self.warmup_steps = 200  # Reduced warmup steps to start temperature optimization earlier

    def get_temperature_optimizer(self):
        return self.temp_optimizer
        
    def get_current_temperature(self):
        """Return the current temperature value for logging"""
        if self.temp_module:
            return self.current_temperature
        return self.temperature

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # Clean inputs for stability
        logits_student = torch.nan_to_num(logits_student, nan=0.0, posinf=1e5, neginf=-1e5)
        logits_teacher = torch.nan_to_num(logits_teacher, nan=0.0, posinf=1e5, neginf=-1e5)

        # Get temperature
        if self.temp_module:
            # For main training loss - use a stable, clamped detached temperature
            temperature = self.temp_module.global_t.detach().clone()
            temperature = torch.clamp(temperature, min=self.min_temp, max=self.max_temp)
            self.current_temperature = temperature.item()
        else:
            # Fallback to fixed temperature
            temperature = torch.tensor(self.temperature).to(logits_student.device)

        # Calculate losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        # Apply warmup to the KD loss
        warmup_factor = min(kwargs.get("epoch", 1) / self.warmup, 1.0)
        
        # Calculate KD loss with more stability
        loss_kd = warmup_factor * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            temperature,
        )
        
        # Initialize temperature optimization loss
        loss_temp_opt = torch.tensor(0.0, device=logits_student.device)
        
        # Only apply temperature optimization after warmup steps
        if self.temp_module and self.step_count >= self.warmup_steps:
            try:
                # Make sure the global temperature requires gradients
                if not self.temp_module.global_t.requires_grad:
                    self.temp_module.global_t.requires_grad_(True)
                
                # Apply gradient reversal
                lambda_factor = torch.tensor(self.grl_lambda, device=logits_student.device)
                
                # Change lambda over time for better convergence
                # Gradually increase GRL strength with training steps
                epoch = kwargs.get("epoch", 1)
                if epoch > 150:
                    # Increase GRL strength in later epochs
                    lambda_factor = lambda_factor * 1.5
                
                temp_with_grl = self.temp_module.grl(self.temp_module.global_t, lambda_factor)
                
                # Clamp temperature for stability
                temp_clamped = torch.clamp(temp_with_grl, min=self.min_temp, max=self.max_temp)
                
                # Create clones for stability
                student_logits = logits_student.detach().clone()
                teacher_logits = logits_teacher.detach().clone()
                
                # Calculate temperature optimization loss
                loss_temp_opt = stable_kd_loss(
                    student_logits,
                    teacher_logits,
                    temp_clamped,
                )
                
                # Check for NaN/Inf
                if not torch.isfinite(loss_temp_opt):
                    print(f"Warning: Temperature loss is not finite, using simpler alternative")
                    # Fallback: simple loss that pushes temperature towards optimal range
                    # Optimal temp should increase over training
                    optimal_temp = min(10.0 + 0.05 * epoch, self.max_temp)
                    loss_temp_opt = 0.02 * torch.abs(temp_clamped - optimal_temp)
            
            except Exception as e:
                print(f"Error in temperature optimization: {e}")
                loss_temp_opt = torch.tensor(0.0, device=logits_student.device)
        
        # Log temperature periodically
        if self.temp_module and self.step_count % 50 == 0:  # More frequent logging
            current_temp = self.temp_module.global_t.item()
            self.temp_log.append(current_temp)
            print(f"Step {self.step_count}, Temperature: {current_temp:.4f}")
            
            # Log to worklog.txt
            try:
                if hasattr(self.cfg, 'LOG') and hasattr(self.cfg.LOG, 'PREFIX'):
                    log_dir = self.cfg.LOG.PREFIX
                    with open(os.path.join(log_dir, "worklog.txt"), "a") as f:
                        f.write(f"Step {self.step_count}, Temperature: {current_temp:.4f}\n")
            except Exception as e:
                print(f"Error writing temperature to worklog: {e}")
        
        self.step_count += 1

        # Save temperature to JSON file periodically
        if self.temp_module and self.step_count % 250 == 0:  # More frequent saving
            self._save_temp_log()

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_temp_opt": loss_temp_opt,
            "temperature": torch.tensor(self.current_temperature).to(logits_student.device)
        }
        
        return logits_student, losses_dict

    def _save_temp_log(self):
        """Save temperature log to JSON file"""
        if not self.temp_log:
            return
            
        try:
            # Determine the save path
            save_path = None
            if hasattr(self.cfg, 'LOG') and hasattr(self.cfg.LOG, 'PREFIX'):
                log_dir = self.cfg.LOG.PREFIX
                save_path = os.path.join(log_dir, "temperature_log_GRLCTDKD_.json")
            else:
                save_path = "temperature_log_GRLCTDKD_.json"
                
            # Save the log
            with open(save_path, 'w') as f:
                json.dump(self.temp_log, f)
            
            if self.step_count % 5000 == 0:  # Only print occasionally
                print(f"Temperature log saved to {save_path}")
        except Exception as e:
            print(f"Error saving temperature log: {e}")

    def forward_test(self, image):
        # Ensure final log is saved before testing/exiting
        if self.temp_module and self.temp_log:
            self._save_temp_log()
            
        return super().forward_test(image) 