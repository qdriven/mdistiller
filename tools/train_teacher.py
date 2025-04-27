import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from mdistiller.models import cifar_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint
from mdistiller.engine.cfg import CFG as cfg

def train_teacher():
    # 基本设置
    cudnn.benchmark = True
    
    # 配置参数
    cfg.DATASET.TYPE = "cifar100"
    cfg.DATASET.TEST.BATCH_SIZE = 64
    cfg.SOLVER.BATCH_SIZE = 64
    cfg.SOLVER.EPOCHS = 240
    cfg.SOLVER.LR = 0.05
    cfg.SOLVER.LR_DECAY_STAGES = [150, 180, 210]
    cfg.SOLVER.LR_DECAY_RATE = 0.1
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.MOMENTUM = 0.9
    
    # 获取数据集
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    
    # 创建教师模型 (ResNet110)
    net, _ = cifar_model_dict["resnet110"]
    model = net(num_classes=num_classes).cuda()
    
    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                         lr=cfg.SOLVER.LR,
                         momentum=cfg.SOLVER.MOMENTUM,
                         weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                             milestones=cfg.SOLVER.LR_DECAY_STAGES,
                                             gamma=cfg.SOLVER.LR_DECAY_RATE)
    
    # 创建保存目录
    save_dir = "download_ckpts/cifar_teachers/resnet110_vanilla"
    os.makedirs(save_dir, exist_ok=True)
    
    best_acc = 0.0
    
    # 训练循环
    for epoch in range(cfg.SOLVER.EPOCHS):
        # 训练阶段
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f'Epoch: {epoch} | Accuracy: {acc:.2f}%')
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, f'{save_dir}/ckpt_best.pth')
        
        # 定期保存检查点
        if (epoch + 1) % 40 == 0:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, f'{save_dir}/ckpt_epoch_{epoch+1}.pth')
        
        scheduler.step()
    
    print(f'Best accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    train_teacher()