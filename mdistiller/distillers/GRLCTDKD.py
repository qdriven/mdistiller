import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np

from ._base import Distiller
from .DKD import DKD, dkd_loss
from .grl_temperature import GlobalTemperature # Import the new module

class GRLCTDKD(DKD):
    """ Curriculum Temperature KD using Gradient Reversal Layer """

    def __init__(self, student, teacher, cfg):
        super(GRLCTDKD, self).__init__(student, teacher, cfg)
        self.cfg = cfg
        
        try:
            init_temp = cfg.GRLCTDKD.INIT_TEMPERATURE
            self.grl_lambda = cfg.GRLCTDKD.GRL_LAMBDA
            lr_temp = cfg.GRLCTDKD.LEARNING_RATE
            
            self.temp_module = GlobalTemperature(initial_temperature=init_temp)
            self.temp_optimizer = torch.optim.Adam(self.temp_module.parameters(), lr=lr_temp)
            
            print(f"GRLCTDKD initialized. Init Temp: {init_temp}, Lambda: {self.grl_lambda}, Temp LR: {lr_temp}")
        except AttributeError as e:
            print(f"Error initializing GRLCTDKD: {e}")
            print("Ensure GRLCTDKD config (INIT_TEMPERATURE, GRL_LAMBDA, LEARNING_RATE) exists.")
            print(f"Warning: Falling back to fixed DKD temperature {self.temperature}")
            self.temp_module = None
            self.temp_optimizer = None

        self.temp_log = []
        if self.temp_module:
             # Log initial temperature
             self.temp_log.append(self.temp_module.global_t.item())
        self.step_count = 0

    def get_temperature_optimizer(self):
        return self.temp_optimizer

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        if self.temp_module:
            # Forward pass gets the current temp value. GRL is applied during backward.
            temperature = self.temp_module(self.grl_lambda)
        else:
            # Fallback to fixed temperature from DKD config
            temperature = torch.tensor(self.temperature).to(logits_student.device)

        # Calculate standard losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # Use detached temperature for the main KD loss calculation
        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            temperature.detach(),
        )
        
        # --- Temperature Optimization Loss ---
        loss_temp_opt = torch.tensor(0.0).to(logits_student.device)
        if self.temp_module and self.temp_optimizer:
            # Recompute DKD loss *with attached temperature* for gradient calculation
            # The forward call here ensures GRL is applied to the temperature parameter
            # for the backward pass of this specific loss.
            temperature_for_grad = self.temp_module(self.grl_lambda)
            loss_temp_opt = dkd_loss(
                                logits_student.detach(), 
                                logits_teacher.detach(),
                                target,
                                self.alpha,
                                self.beta,
                                temperature_for_grad, # Attached temperature
                            )
            # Minimizing this loss maximizes DKD w.r.t temperature due to GRL
        
        # Log temperature periodically
        if self.temp_module and self.step_count % 50 == 0: # Log every 50 steps
            current_temp_value = self.temp_module.global_t.item()
            self.temp_log.append(current_temp_value)
            print(f"Step {self.step_count}, GRL-Temp: {current_temp_value:.4f}")
        self.step_count += 1

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_temp_opt": loss_temp_opt # Add this loss for the temp optimizer step
        }
        return logits_student, losses_dict

    def _save_temp_log(self):
        """Saves the temperature log to a file."""
        if not self.temp_module or not self.temp_log:
            return
        if not hasattr(self.cfg.LOG, 'PREFIX') or not self.cfg.LOG.PREFIX:
            print("Warning: LOG.PREFIX not defined in config, cannot save GRL temperature log.")
            return
            
        try:
            log_dir = self.cfg.LOG.PREFIX
            os.makedirs(log_dir, exist_ok=True)
            
            task_id = os.path.basename(os.path.normpath(log_dir)) # Get task ID from dir name
            log_filename = f'temperature_log_GRL_{task_id}.json'
            log_path = os.path.join(log_dir, log_filename)
            
            with open(log_path, 'w') as f:
                # Ensure log contains initial temp + logged values
                full_log = [self.temp_log[0]] + self.temp_log[1:] 
                json.dump(full_log, f, indent=4) # Use indent for readability
            print(f"GRL Temperature log saved to {log_path}, {len(full_log)} records")
        except Exception as e:
            print(f"Error saving GRL temperature log: {e}")

    def forward_test(self, image):
        # Ensure final log is saved before testing/exiting
        self._save_temp_log()
        # Call the parent's forward_test method
        return super(GRLCTDKD, self).forward_test(image) 