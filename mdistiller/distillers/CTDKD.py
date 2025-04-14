import torch
import torch.nn as nn
import torch.nn.functional as F
from .DKD import DKD, dkd_loss

class CTDKD(DKD):
    """Curriculum Temperature for Decoupled Knowledge Distillation"""

    def __init__(self, student, teacher, cfg):
        super(CTDKD, self).__init__(student, teacher, cfg)
        # 初始化温度参数
        self.current_temperature = nn.Parameter(
            torch.tensor(cfg.CTDKD.INIT_TEMPERATURE)
        )
        self.min_temperature = cfg.CTDKD.MIN_TEMPERATURE
        self.max_temperature = cfg.CTDKD.MAX_TEMPERATURE
        self.temp_optimizer = torch.optim.Adam(
            [self.current_temperature],
            lr=cfg.CTDKD.LEARNING_RATE
        )
        
    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        # 使用当前温度计算损失
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        # 约束温度在合理范围内
        temperature = torch.clamp(
            self.current_temperature, 
            self.min_temperature,
            self.max_temperature
        )
        
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            temperature,
        )

        # 更新温度参数
        self.temp_optimizer.zero_grad()
        loss_dkd.backward(retain_graph=True)
        self.temp_optimizer.step()

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
            "temperature": temperature.item(),
        }
        return logits_student, losses_dict, features_student 