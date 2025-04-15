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
                requires_grad=False  # 不参与主网络的梯度计算
            )
            self.min_temperature = cfg.CTDKD.MIN_TEMPERATURE
            self.max_temperature = cfg.CTDKD.MAX_TEMPERATURE
            self.temp_optimizer = torch.optim.Adam(
                [self.current_temperature],
                lr=cfg.CTDKD.LEARNING_RATE
            )
            print(f"CTDKD initialized with temperature: {self.current_temperature.item()}")
        except AttributeError as e:
            print(f"Error initializing CTDKD: {e}")
            print("Please ensure CTDKD configuration is defined in cfg.py")
            # 使用DKD的默认温度作为备选
            self.current_temperature = nn.Parameter(
                torch.tensor(cfg.DKD.T, dtype=torch.float32),
                requires_grad=False  # 不参与主网络的梯度计算
            )
            self.min_temperature = 1.0
            self.max_temperature = 10.0
            self.temp_optimizer = torch.optim.Adam(
                [self.current_temperature],
                lr=0.001
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
        self.temp_log = []
        self.temp_log.append(self.current_temperature.item())  # 记录初始温度
        
        # 防止梯度爆炸
        self.max_grad_norm = 1.0
        
    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        # 使用当前温度计算损失
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        # 确保温度值有效，如果无效则重置为初始值
        if torch.isnan(self.current_temperature) or torch.isinf(self.current_temperature):
            print("Warning: Temperature is NaN or Inf, resetting to initial value")
            with torch.no_grad():
                self.current_temperature.copy_(torch.tensor(self.cfg.CTDKD.INIT_TEMPERATURE, 
                                                          dtype=torch.float32))
        
        # 约束温度在合理范围内
        temperature = torch.clamp(
            self.current_temperature.detach(),  # 使用detach避免梯度传递到主网络
            self.min_temperature,
            self.max_temperature
        )
        
        # 防止温度为0或极小值
        temperature = torch.max(temperature, torch.tensor(0.1).to(temperature.device))
        
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            temperature,
        )
        
        # 保存当前批次的信息以便后续更新温度
        self.last_loss_dkd = loss_dkd.detach()
        self.last_logits_student = logits_student.detach()
        self.last_logits_teacher = logits_teacher.detach()
        self.last_target = target.detach()
        
        # 记录当前温度值
        temp_value = temperature.item()
        if self.step_count % 10 == 0:  # 每10次迭代打印一次
            print(f"Step {self.step_count}, Temperature: {temp_value:.4f}")
            # 记录到日志中
            with open(os.path.join(self.cfg.LOG.PREFIX, 'worklog.txt'), 'a') as f:
                f.write(f"temperature: {temp_value:.4f}\n")
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        
        return logits_student, losses_dict
    
    def update_temperature(self):
        """在训练迭代结束后调用此方法来更新温度参数"""
        if self.last_loss_dkd is None:
            return
            
        # 只在每10步更新一次温度，减少不稳定性
        if self.step_count % 10 != 0:
            self.step_count += 1
            self.last_loss_dkd = None
            self.last_logits_student = None
            self.last_logits_teacher = None
            self.last_target = None
            return
        
        # 让温度参数梯度可计算
        self.current_temperature.requires_grad = True
        
        # 重新计算loss_dkd用于温度优化
        temperature = torch.clamp(
            self.current_temperature,
            self.min_temperature,
            self.max_temperature
        )
        
        # 防止温度为0或极小值
        temperature = torch.max(temperature, torch.tensor(0.1).to(temperature.device))
        
        try:
            # 计算损失，用于更新温度
            loss_for_temp = dkd_loss(
                self.last_logits_student,
                self.last_logits_teacher,
                self.last_target,
                self.alpha,
                self.beta,
                temperature,
            )
            
            # 检查损失是否为有效值
            if torch.isnan(loss_for_temp) or torch.isinf(loss_for_temp):
                print("Warning: Loss for temperature update is NaN or Inf, skipping update")
                self.current_temperature.requires_grad = False
                return
            
            # 使用一个更简单的逻辑：基于当前loss自适应调整温度
            # 这样可以避免不稳定的梯度计算
            with torch.no_grad():
                # 记录当前温度
                old_temp = self.current_temperature.item()
                
                # 计算学生和教师的交叉熵损失
                student_ce = F.cross_entropy(self.last_logits_student, self.last_target)
                teacher_ce = F.cross_entropy(self.last_logits_teacher, self.last_target)
                
                # 如果学生比教师差很多，增加温度使蒸馏信号更软
                # 如果学生接近教师，降低温度使蒸馏信号更尖锐
                diff = (student_ce - teacher_ce).item()
                
                # 自适应调整温度：根据性能差距调整
                if diff > 0.5:  # 学生比教师差很多
                    new_temp = min(old_temp + 0.05, self.max_temperature)
                elif diff < 0.1:  # 学生接近教师
                    new_temp = max(old_temp - 0.05, self.min_temperature)
                else:
                    new_temp = old_temp  # 保持不变
                
                # 应用新温度
                self.current_temperature.copy_(torch.tensor(new_temp, dtype=torch.float32))
            
            # 更新完成后再次设置为不参与主网络的梯度计算
            self.current_temperature.requires_grad = False
            
            # 记录温度值
            temp_value = self.current_temperature.item()
            
            # 确保记录的温度是有效值
            if not np.isnan(temp_value) and not np.isinf(temp_value):
                self.temp_log.append(temp_value)
                print(f"Temperature updated: {old_temp:.4f} -> {temp_value:.4f}, diff: {diff:.4f}")
            else:
                print("Warning: Temperature value is NaN or Inf after update")
                # 重置为有效值
                with torch.no_grad():
                    self.current_temperature.copy_(torch.tensor(self.cfg.CTDKD.INIT_TEMPERATURE, 
                                                            dtype=torch.float32))
                self.temp_log.append(self.cfg.CTDKD.INIT_TEMPERATURE)
                
            self.step_count += 1
            
            # 每50步保存一次温度日志，确保更频繁地保存
            if self.step_count % 50 == 0:
                self._save_temp_log()
        
        except Exception as e:
            print(f"Error in temperature update: {e}")
            # 出错时重置温度
            with torch.no_grad():
                self.current_temperature.copy_(torch.tensor(self.cfg.CTDKD.INIT_TEMPERATURE, 
                                                        dtype=torch.float32))
        
        # 清除保存的数据
        self.last_loss_dkd = None
        self.last_logits_student = None
        self.last_logits_teacher = None
        self.last_target = None
    
    def _save_temp_log(self):
        """保存温度日志到文件"""
        if not hasattr(self.cfg.LOG, 'PREFIX'):
            print("Warning: LOG.PREFIX not found in config, cannot save temperature log")
            return
            
        log_dir = self.cfg.LOG.PREFIX
        os.makedirs(log_dir, exist_ok=True)
        
        # 从日志目录中提取任务标识符（目录名）
        task_id = os.path.basename(os.path.normpath(log_dir))
        
        # 使用任务标识符作为文件名后缀
        log_filename = f'temperature_log_{task_id}.json'
        log_path = os.path.join(log_dir, log_filename)
        
        try:
            with open(log_path, 'w') as f:
                json.dump(self.temp_log, f)
            print(f"Temperature log saved to {log_path}, {len(self.temp_log)} records")
        except Exception as e:
            print(f"Error saving temperature log: {e}")
    
    def forward_test(self, image):
        # 在测试之前保存温度日志，确保最终的温度记录被保存
        self._save_temp_log()
        return super().forward_test(image) 