import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mdistiller.models import model_dict
from mdistiller.dataset.cifar import get_dataset
from mdistiller.distiller import distiller_dict
from mdistiller.engine.utils import load_checkpoint, save_checkpoint
from mdistiller.engine.cfg import CFG as cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg.merge_from_file(args.cfg)
    
    # Get dataset
    train_loader, val_loader, num_classes = get_dataset(cfg)
    
    # Create teacher and student models
    teacher = model_dict[cfg.MODEL.TEACHER](num_classes=num_classes)
    student = model_dict[cfg.MODEL.STUDENT](num_classes=num_classes)
    
    # Load teacher checkpoint
    teacher_checkpoint = load_checkpoint(cfg.MODEL.TEACHER_CHECKPOINT)
    teacher.load_state_dict(teacher_checkpoint['model'])
    teacher = teacher.cuda().eval()
    student = student.cuda()
    
    # Create distiller
    distiller = distiller_dict[cfg.DISTILLER.TYPE](student, teacher, cfg)
    distiller = distiller.cuda()
    
    # Setup training
    optimizer = optim.SGD(
        distiller.get_learnable_parameters(),
        lr=cfg.TRAIN.OPTIMIZER.LR,
        momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
        weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
    )
    
    # Learning rate scheduler
    if cfg.TRAIN.LR_SCHEDULER.TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.TRAIN.EPOCHS, eta_min=cfg.TRAIN.LR_SCHEDULER.MIN_LR
        )
    else:
        raise NotImplementedError(cfg.TRAIN.LR_SCHEDULER.TYPE)
    
    # Tensorboard
    writer = SummaryWriter(os.path.join("runs", cfg.EXPERIMENT.TAG))
    
    best_acc = 0
    for epoch in range(cfg.TRAIN.EPOCHS):
        # Train
        distiller.train()
        for i, (images, targets) in enumerate(train_loader):
            images = images.cuda()
            targets = targets.cuda()
            
            # Forward and backward
            output_student, losses_dict = distiller.train_step(images, targets, epoch=epoch)
            
            loss = sum([l.mean() for l in losses_dict.values()])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                loss_str = " ".join([f"{k}: {v.item():.4f}" for k, v in losses_dict.items()])
                print(f"Epoch: [{epoch}/{cfg.TRAIN.EPOCHS}][{i}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} ({loss_str})")
                
        # Validate
        distiller.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.cuda()
                targets = targets.cuda()
                outputs = distiller(images)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        writer.add_scalar("Accuracy/val", acc, epoch)
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            save_checkpoint({
                'epoch': epoch,
                'model': student.state_dict(),
                'accuracy': acc,
                'optimizer': optimizer.state_dict(),
            }, is_best=True, filename=f'checkpoints/students/{cfg.DATASET.TYPE}_{cfg.MODEL.STUDENT}_cdkd.pth')
        
        scheduler.step()
    
    writer.close()
    print(f"Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main() 