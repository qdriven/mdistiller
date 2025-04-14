import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mdistiller.models import cifar_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import load_checkpoint

def extract_features(model, data_loader, num_samples=1000):
    features = []
    labels = []
    count = 0
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            if count >= num_samples:
                break
            images = images.cuda()
            _, feat = model(images)
            features.append(feat.cpu().numpy())
            labels.append(targets.numpy())
            count += images.size(0)
    return np.vstack(features), np.concatenate(labels)

def plot_tsne(features, labels, save_path, title):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()
    
    cfg.merge_from_file(args.cfg)
    
    # 加载数据集
    _, val_loader, _, num_classes = get_dataset(cfg)
    
    # 加载模型
    if cfg.DATASET.TYPE == "cifar100":
        model = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
    model.cuda()
    model.load_state_dict(load_checkpoint(args.ckpt)['model'])
    
    # 提取特征
    features, labels = extract_features(model, val_loader)
    
    # 绘制t-SNE
    save_dir = os.path.join(cfg.LOG.PREFIX, 'tsne')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{cfg.DISTILLER.TYPE}_tsne.png")
    plot_tsne(features, labels, save_path, f"{cfg.DISTILLER.TYPE} t-SNE Visualization")

if __name__ == '__main__':
    main() 