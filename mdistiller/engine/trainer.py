
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import getpass
from tensorboardX import SummaryWriter
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)
from .dot import DistillationOrientedTrainer
import numpy as np


class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1

        # Get the temperature optimizer if the distiller provides one
        self.temp_optimizer = None
        if hasattr(self.distiller.module, 'get_temperature_optimizer') and callable(getattr(self.distiller.module, 'get_temperature_optimizer')):
            self.temp_optimizer = self.distiller.module.get_temperature_optimizer()
            if self.temp_optimizer:
                print("Found separate temperature optimizer.")
            
        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def init_optimizer(self, cfg):
        # Initialize optimizer only for non-temperature parameters
        params_to_optimize = []
        if hasattr(self.distiller.module, 'get_learnable_parameters'):
            params_to_optimize = self.distiller.module.get_learnable_parameters()
        elif hasattr(self.distiller.module, 'student'): # Fallback for simple models
            params_to_optimize = self.distiller.module.student.parameters()
        else:
             raise ValueError("Could not get learnable parameters from distiller")
             
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                params_to_optimize,
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def log(self, lr, epoch, log_dict):
        """统一的日志记录格式"""
        # 准备日志数据
        sanitized_log_dict = {}
        for k, v in log_dict.items():
            if isinstance(v, (int, float)):
                sanitized_log_dict[k] = float(v)
            else:
                sanitized_log_dict[k] = v

        # tensorboard log
        for k, v in sanitized_log_dict.items():
            if k == "train_loss_temp_opt": 
                self.tf_writer.add_scalar(k, v * 100, epoch)
            else:
                self.tf_writer.add_scalar(k, v, epoch)
        self.tf_writer.flush()

        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb
            wandb.log({"current lr": lr})
            wandb.log(sanitized_log_dict)

        # 更新最佳准确率
        if "test_acc" in sanitized_log_dict and not np.isnan(sanitized_log_dict["test_acc"]):
            if sanitized_log_dict["test_acc"] > self.best_acc:
                self.best_acc = sanitized_log_dict["test_acc"]
                if self.cfg.LOG.WANDB:
                    wandb.run.summary["best_acc"] = self.best_acc

        # 统一的worklog.txt格式
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("-" * 25 + os.linesep)
            writer.write(f"epoch: {epoch}" + os.linesep)
            writer.write(f"lr: {lr:.2f}" + os.linesep)
            
            # 按固定顺序写入关键指标
            metrics_order = [
                "train_acc", "train_loss", "test_acc", "test_acc_top5", 
                "test_loss", "train_loss_temp_opt"
            ]
            
            for metric in metrics_order:
                if metric in sanitized_log_dict:
                    writer.write(f"{metric}: {sanitized_log_dict[metric]:.2f}" + os.linesep)
            
            # 确保温度信息被记录
            if "temperature" in sanitized_log_dict:
                writer.write(f"temperature: {sanitized_log_dict['temperature']:.4f}" + os.linesep)
            elif "current_temperature" in sanitized_log_dict:
                writer.write(f"temperature: {sanitized_log_dict['current_temperature']:.4f}" + os.linesep)
            
            writer.write("-" * 25 + os.linesep)

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
            # Load temp optimizer state if resuming and it exists
            if self.temp_optimizer and 'temp_optimizer' in state:
                 print("Loading temperature optimizer state.")
                 self.temp_optimizer.load_state_dict(state['temp_optimizer'])
                 
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy: {}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}\n".format(float(self.best_acc)))
        # Save temperature log at the end of training
        if hasattr(self.distiller.module, '_save_temp_log'):
            print("Saving final temperature log...")
            self.distiller.module._save_temp_log()

    def train_epoch(self, epoch):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(), # Main loss
            "top1": AverageMeter(),
            "top5": AverageMeter(),
            "temperature": AverageMeter(),  # 添加温度计量器
        }
        # Add meter for temperature optimization loss if applicable
        if self.temp_optimizer:
             train_meters["loss_temp_opt"] = AverageMeter()
             train_meters["temperature"] = AverageMeter()
        # 添加温度监控
        if hasattr(self.distiller.module, 'get_current_temperature'):
            train_meters["temperature"] = AverageMeter()
             
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()

        # validate - 确保返回值是浮点数，不是张量
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller)
        
        # 确保返回值是 Python 标量，而不是张量
        if isinstance(test_acc, torch.Tensor):
            test_acc = test_acc.item()
        if isinstance(test_acc_top5, torch.Tensor):
            test_acc_top5 = test_acc_top5.item()
        if isinstance(test_loss, torch.Tensor):
            test_loss = test_loss.item()

        # 获取当前温度
        current_temp = None
        if hasattr(self.distiller.module, 'get_current_temperature'):
            current_temp = self.distiller.module.get_current_temperature()

        # log - 确保只记录一次
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss,
            }
        )
        
        # 添加温度到日志
        if current_temp is not None:
            log_dict["temperature"] = current_temp
        elif "temperature" in train_meters:
            log_dict["temperature"] = train_meters["temperature"].avg

        # 只在有温度优化器时添加温度相关的日志
        if self.temp_optimizer:
            log_dict["train_loss_temp_opt"] = train_meters["loss_temp_opt"].avg
            if hasattr(self.distiller.module, 'get_current_temperature'):
                log_dict["temperature"] = self.distiller.module.get_current_temperature()
                
        self.log(lr, epoch, log_dict)
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        # Save temp optimizer state if exists
        if self.temp_optimizer:
             state['temp_optimizer'] = self.temp_optimizer.state_dict()
             
        student_state = {"model": self.distiller.module.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        if self.temp_optimizer:
            self.temp_optimizer.zero_grad()
             
        train_start_time = time.time()
        # Adapt data unpacking based on loader format
        try:
            if len(data) == 2: # Standard image, target
                image, target = data
            elif len(data) == 3: # Usually image, target, index
                image, target, _ = data
            elif len(data) == 4: # Usually CRD sample: image, target, index, contrastive_idx
                image, target, _, _ = data
            else:
                raise ValueError(f"Unexpected data format length: {len(data)}")
        except (ValueError, TypeError) as e:
            print(f"Error unpacking data: {e}. Data: {data}")
            return "Error unpacking data"
            
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # Initialize loss_main
        loss_main = 0.0
        
        # Calculate main loss, excluding temperature-related losses
        try:
            main_losses = [l for k, l in losses_dict.items() 
                         if k not in ['loss_temp_opt', 'temperature', 'current_temperature']]
            if main_losses:
                loss_main = sum(l.mean() if torch.is_tensor(l) else l for l in main_losses)
                if not torch.is_tensor(loss_main):
                    loss_main = torch.tensor(loss_main, device=image.device, requires_grad=True)
                loss_main.backward(retain_graph=True if self.temp_optimizer else False)
                self.optimizer.step()
        except Exception as e:
            print(f"Error in main backward pass: {e}")
            return "Error in backward pass"

        # Handle temperature optimization if applicable
        if self.temp_optimizer and 'loss_temp_opt' in losses_dict:
            loss_temp = losses_dict['loss_temp_opt']
            if torch.is_tensor(loss_temp) and loss_temp.requires_grad:
                try:
                    loss_temp.backward()
                    self.temp_optimizer.step()
                except Exception as e:
                    print(f"Error in temperature optimization: {e}")
        # Update temperature for CTDKD method
        elif hasattr(self.distiller.module, 'update_temperature'):
            try:
                print("[Temp update] Calling update_temperature()...")
                self.distiller.module.update_temperature()
            except Exception as e:
                print(f"Error in update_temperature: {e}")

        train_meters["training_time"].update(time.time() - train_start_time)
        
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        
        # Ensure acc1 and acc5 are scalar values
        acc1_val = acc1[0].item() if torch.is_tensor(acc1[0]) else acc1[0]
        acc5_val = acc5[0].item() if torch.is_tensor(acc5[0]) else acc5[0]
        
        # Update meters
        if torch.is_tensor(loss_main):
            train_meters["losses"].update(loss_main.item(), batch_size)
        else:
            train_meters["losses"].update(float(loss_main), batch_size)
            
        train_meters["top1"].update(acc1_val, batch_size)
        train_meters["top5"].update(acc5_val, batch_size)

        # Log temperature-related metrics if available
        if self.temp_optimizer and 'loss_temp_opt' in losses_dict:
            try:
                loss_temp_val = losses_dict['loss_temp_opt'].item() if torch.is_tensor(losses_dict['loss_temp_opt']) else float(losses_dict['loss_temp_opt'])
                if not (np.isnan(loss_temp_val) or np.isinf(loss_temp_val)):
                    train_meters["loss_temp_opt"].update(loss_temp_val, batch_size)
            except Exception as e:
                print(f"Error logging temperature loss: {e}")

        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDTrainer(BaseTrainer):
    # Note: CRDTrainer needs to be adapted similarly to BaseTrainer if GRLCTDKD is used with CRD.
    # The current implementation below assumes the BaseTrainer modifications are sufficient,
    # but careful testing is needed if combining CRD and GRLCTDKD.
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        if self.temp_optimizer:
             self.temp_optimizer.zero_grad()

        train_start_time = time.time()
        image, target, index, contrastive_index = data # CRD data format
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index, epoch=epoch
        )

        # backward for main optimizer
        loss_main = sum([l.mean() for k, l in losses_dict.items() if k != 'loss_temp_opt'])
        loss_main.backward(retain_graph=True if self.temp_optimizer else False)
        self.optimizer.step()
        
        # backward for temperature optimizer (if applicable)
        if self.temp_optimizer and 'loss_temp_opt' in losses_dict and losses_dict['loss_temp_opt'] is not None:
            loss_temp = losses_dict['loss_temp_opt'].mean()
            if torch.is_tensor(loss_temp) and not torch.isnan(loss_temp) and not torch.isinf(loss_temp):
                loss_temp.backward()
                self.temp_optimizer.step()
            else:
                 print(f"Warning: Skipping temp_optimizer step in CRDTrainer due to invalid loss_temp: {loss_temp}")
            
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        
        # 确保 acc1 和 acc5 是 CPU 张量或标量
        if isinstance(acc1[0], torch.Tensor):
            acc1_val = acc1[0].detach().cpu().item()
        else:
            acc1_val = acc1[0]
            
        if isinstance(acc5[0], torch.Tensor):
            acc5_val = acc5[0].detach().cpu().item()
        else:
            acc5_val = acc5[0]
            
        train_meters["losses"].update(loss_main.item(), batch_size)
        train_meters["top1"].update(acc1_val, batch_size)
        train_meters["top5"].update(acc5_val, batch_size)
        if self.temp_optimizer and 'loss_temp_opt' in losses_dict and losses_dict['loss_temp_opt'] is not None:
             try:
                loss_temp_val = losses_dict['loss_temp_opt'].item()
                if not np.isnan(loss_temp_val) and not np.isinf(loss_temp_val):
                    train_meters["loss_temp_opt"].update(loss_temp_val, batch_size)
                else:
                    train_meters["loss_temp_opt"].update(0, batch_size)
             except Exception as e:
                print(f"Error logging temperature loss in CRDTrainer: {e}")
                train_meters["loss_temp_opt"].update(0, batch_size)
             
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        if self.temp_optimizer:
             temp_loss_avg = train_meters["loss_temp_opt"].avg
             msg += "| Loss(TempOpt):{:.4f}".format(temp_loss_avg)
             
        return msg


class DOT(BaseTrainer):
    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        # Same as BaseTrainer.train, no specific changes needed for DOT
        super(DOT, self).train(resume=resume)
        
    def train_epoch(self, epoch):
         # Same as BaseTrainer.train_epoch
         super(DOT, self).train_epoch(epoch)
         
    def train_iter(self, data, epoch, train_meters):
        # Same as BaseTrainer.train_iter, assuming DOT optimizer handles its logic internally
        # Note: DOT optimizer is not compatible with the separate temp_optimizer logic.
        # If using GRLCTDKD with DOT, the temperature won't be optimized.
        if self.temp_optimizer:
            print("Warning: DOT optimizer is not compatible with GRL temperature optimizer. Temperature will not be learned.")
            
        self.optimizer.zero_grad()
        train_start_time = time.time()
        # DOT assumes specific data format (image, target, index)
        try:
             image, target, index = data 
        except ValueError:
             print("Error: DOT expects data format (image, target, index)")
             return "Error unpacking data for DOT"
             
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # index = index.cuda(non_blocking=True) # Index needed?

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # backward
        loss_main = sum([l.mean() for k, l in losses_dict.items()]) # DOT uses all losses
        loss_main.backward()
        # DOT optimizer step handles task and kd gradients differently
        try:
             # Check if losses_dict contains expected keys for DOT
             ce_loss = losses_dict.get("loss_ce", torch.tensor(0.0).to(loss_main.device))
             kd_loss = losses_dict.get("loss_kd", torch.tensor(0.0).to(loss_main.device))
             self.optimizer.step(ce_loss, kd_loss) 
        except KeyError as e:
             print(f"Error: DOT optimizer expects 'loss_ce' and 'loss_kd' in losses_dict. Found: {losses_dict.keys()}. Skipping step.")
        
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        
        # 确保 acc1 和 acc5 是 CPU 张量或标量
        if isinstance(acc1[0], torch.Tensor):
            acc1_val = acc1[0].detach().cpu().item()
        else:
            acc1_val = acc1[0]
            
        if isinstance(acc5[0], torch.Tensor):
            acc5_val = acc5[0].detach().cpu().item()
        else:
            acc5_val = acc5[0]
            
        train_meters["losses"].update(loss_main.item(), batch_size)
        train_meters["top1"].update(acc1_val, batch_size)
        train_meters["top5"].update(acc5_val, batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDDOT(BaseTrainer):
    # Note: CRDDOT needs similar checks/adaptations as DOT if used with GRLCTDKD
    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        # Same as BaseTrainer.train
        super(CRDDOT, self).train(resume=resume)
        
    def train_epoch(self, epoch):
         # Same as BaseTrainer.train_epoch
         super(CRDDOT, self).train_epoch(epoch)
         
    def train_iter(self, data, epoch, train_meters):
        if self.temp_optimizer:
            print("Warning: CRDDOT optimizer is not compatible with GRL temperature optimizer. Temperature will not be learned.")
            
        self.optimizer.zero_grad()
        train_start_time = time.time()
        # CRDDOT uses CRD data format
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index, epoch=epoch
        )

        # dot backward for CRD
        loss_ce, loss_kd = losses_dict['loss_ce'].mean(), losses_dict['loss_kd'].mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        
        # 确保 acc1 和 acc5 是 CPU 张量或标量
        if isinstance(acc1[0], torch.Tensor):
            acc1_val = acc1[0].detach().cpu().item()
        else:
            acc1_val = acc1[0]
            
        if isinstance(acc5[0], torch.Tensor):
            acc5_val = acc5[0].detach().cpu().item()
        else:
            acc5_val = acc5[0]
            
        train_meters["losses"].update((loss_ce + loss_kd).item(), batch_size)
        train_meters["top1"].update(acc1_val, batch_size)
        train_meters["top5"].update(acc5_val, batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CTDKDTrainer(BaseTrainer):
    """Trainer for the original performance-gap based CTDKD"""
    # This trainer remains unchanged as it uses the 'update_temperature' callback
    # which is handled by the BaseTrainer's train_iter logic already.
    pass
