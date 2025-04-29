
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
    
    # 添加温度正则化项
    temp_reg = 0.01 * torch.abs(temperature - 4.0)  # 促使温度向合理值靠拢
    
    # 最终损失
    loss = loss * (temperature ** 2) * 8.0 + temp_reg
    
    return loss

class GRLCTDKD(DKD):
    """ Curriculum Temperature KD using Gradient Reversal Layer """

    def __init__(self, student, teacher, cfg):
        super(GRLCTDKD, self).__init__(student, teacher, cfg)
        self.cfg = cfg
        
        # 修改默认值
        init_temp = 2.0  # 降低初始温度（原为4.0）
        self.grl_lambda = 0.1  # 降低GRL强度（原为0.2）
        self.min_temp = 1.0
        self.max_temp = 8.0  # 降低最大温度（原为20.0）
        lr_temp = 0.0001  # 降低温度学习率（原为0.0005）
        
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

        # 获取总训练轮数，如果配置中没有则使用默认值
        self.total_epochs = getattr(cfg.SOLVER, 'EPOCHS', 200)  # 默认200轮
        self.warmup = getattr(cfg.SOLVER, 'WARMUP', 5)  # 默认5轮warmup

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

        # 数值稳定性处理
        logits_student = torch.nan_to_num(logits_student, nan=0.0, posinf=1e5, neginf=-1e5)
        logits_teacher = torch.nan_to_num(logits_teacher, nan=0.0, posinf=1e5, neginf=-1e5)

        # 计算CE损失
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # 获取当前epoch，默认为1
        epoch = kwargs.get("epoch", 1)
        
        # 使用正确的配置路径计算进度
        progress = epoch / self.total_epochs  # 使用初始化时存储的total_epochs
        
        # 计算性能差距
        with torch.no_grad():
            student_ce = F.cross_entropy(logits_student, target)
            teacher_ce = F.cross_entropy(logits_teacher, target)
            perf_gap = torch.clamp(student_ce - teacher_ce, 0.0, 10.0) * self.gap_scale

        # 动态调整GRL强度
        current_grl_lambda = self.grl_lambda * (1.0 + progress)  # 随训练进度增加GRL强度

        # 温度优化
        if self.temp_module and self.step_count >= self.warmup_steps:
            # 确保温度模块启用梯度
            self.temp_module.global_t.requires_grad_(True)
            
            # 动态调整温度范围
            current_max_temp = max(
                self.min_temp + 2.0,
                self.max_temp * (1.0 - 0.5 * progress)  # 随训练进度降低最大温度
            )
            current_min_temp = self.min_temp + progress  # 随训练进度提高最小温度
            
            # 计算目标温度
            target_temp = current_min_temp + (current_max_temp - current_min_temp) * (perf_gap / 5.0)
            
            # 应用GRL并计算温度损失
            temp_with_grl = self.temp_module.grl(self.temp_module.global_t, current_grl_lambda)
            
            # 使用改进的稳定KD损失
            loss_temp_opt = stable_kd_loss(
                logits_student.detach(),
                logits_teacher.detach(),
                temp_with_grl,
            )
            
            # 添加温度约束损失
            temp_constraint = 0.1 * torch.abs(temp_with_grl - target_temp)
            loss_temp_opt = loss_temp_opt + temp_constraint

            # 更新温度
            if self.step_count % self.temp_update_freq == 0:
                self.temp_optimizer.zero_grad()
                loss_temp_opt.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_([self.temp_module.global_t], 1.0)
                
                self.temp_optimizer.step()
                
                # 确保温度在有效范围内
                with torch.no_grad():
                    self.temp_module.global_t.copy_(
                        torch.clamp(self.temp_module.global_t,
                                  min=current_min_temp,
                                  max=current_max_temp)
                    )
        else:
            loss_temp_opt = torch.tensor(0.0, device=logits_student.device)

        # 获取当前温度用于KD损失
        temperature = (self.temp_module.global_t.detach() if self.temp_module 
                      else torch.tensor(self.temperature).to(logits_student.device))
        self.current_temperature = temperature.item()

        # 应用warmup
        warmup_factor = min(epoch / self.warmup, 1.0)
        
        # 计算KD损失
        loss_kd = warmup_factor * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            temperature,
        )

        # 记录温度和性能差距
        if self.step_count % 10 == 0:
            if hasattr(self.cfg, 'LOG') and hasattr(self.cfg.LOG, 'PREFIX'):
                with open(os.path.join(self.cfg.LOG.PREFIX, "worklog.txt"), "a") as f:
                    f.write(f"[TEMP] Step={self.step_count}, T={temperature.item():.4f}, "
                           f"Gap={perf_gap.item():.4f}, GRL={current_grl_lambda:.4f}\n")

        self.step_count += 1

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_temp_opt": loss_temp_opt,
            "current_temperature": self.current_temperature,
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
