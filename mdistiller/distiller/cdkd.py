import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class TemperatureModule(nn.Module):
    def __init__(self, temp_type="instance"):
        super().__init__()
        self.temp_type = temp_type
        if temp_type == "global":
            self.temperature = nn.Parameter(torch.ones(1) * 4.0)
        else:  # instance-level temperature
            self.temperature_predictor = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
            self.grl = GradientReversalLayer.apply

    def forward(self, features, alpha=1.0):
        if self.temp_type == "global":
            return self.temperature
        else:
            # Apply GRL and predict temperature
            reversed_features = self.grl(features, alpha)
            temp = self.temperature_predictor(reversed_features)
            return 1.0 + temp * 8.0  # Scale temperature to [1, 9]

class CDKD(Distiller):
    """Curriculum Decoupled Knowledge Distillation"""
    
    def __init__(self, student, teacher, cfg):
        super(CDKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CDKD.CE_WEIGHT
        self.alpha = cfg.CDKD.ALPHA
        self.beta = cfg.CDKD.BETA
        self.warmup_epochs = cfg.CDKD.WARMUP
        self.temp_module = TemperatureModule(cfg.CDKD.TEMP_TYPE)
        self.current_epoch = 0

    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        # Get adaptive temperature
        alpha = min(1.0, self.current_epoch / self.warmup_epochs)
        temperature = self.temp_module(features_student, alpha)
        
        # Calculate losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        # KL divergence with adaptive temperature
        soft_student = F.log_softmax(logits_student / temperature, dim=1)
        soft_teacher = F.softmax(logits_teacher / temperature, dim=1)
        loss_kd = (temperature ** 2) * F.kl_div(
            soft_student, soft_teacher, reduction="batchmean"
        )
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def train_step(self, image, target, epoch, **kwargs):
        self.current_epoch = epoch
        return super().train_step(image, target, **kwargs) 