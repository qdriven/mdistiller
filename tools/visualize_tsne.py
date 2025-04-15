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
            outputs = model(images)
            
            # 根据调试信息，模型返回 (logits, features_dict) 
            # 其中 features_dict 包含 'feats', 'pooled_feat', 'preact_feats'
            if isinstance(outputs, tuple) and len(outputs) == 2:
                _, feat_dict = outputs
                if isinstance(feat_dict, dict) and 'pooled_feat' in feat_dict:
                    # 使用pooled_feat，这通常是最终的特征表示
                    features.append(feat_dict['pooled_feat'].cpu().numpy())
                elif isinstance(feat_dict, dict) and 'feats' in feat_dict:
                    # 如果没有pooled_feat，则使用feats
                    feats = feat_dict['feats']
                    if isinstance(feats, list) and feats:
                        # 使用列表中的最后一项作为特征
                        last_feat = feats[-1]
                        if isinstance(last_feat, torch.Tensor):
                            # 如果需要，对特征进行全局平均池化
                            if len(last_feat.shape) > 2:
                                last_feat = last_feat.mean([2, 3])
                            features.append(last_feat.cpu().numpy())
                        else:
                            print(f"Warning: Expected tensor in feats list, got {type(last_feat)}")
                            continue
                    elif isinstance(feats, torch.Tensor):
                        # 如果feats直接是张量
                        if len(feats.shape) > 2:
                            feats = feats.mean([2, 3])
                        features.append(feats.cpu().numpy())
                    else:
                        print(f"Warning: Unexpected feats type: {type(feats)}")
                        continue
                else:
                    print(f"Warning: Unexpected feature dictionary structure")
                    continue
            else:
                print(f"Warning: Unexpected model output structure")
                continue
                
            labels.append(targets.numpy())
            count += images.size(0)
    
    if not features:
        raise ValueError("No features were extracted. Check model output format.")
        
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

def remove_module_prefix(state_dict, prefix='module.student.'):
    """移除状态字典中的模块前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
    return new_state_dict

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
    
    # 加载检查点并处理权重
    checkpoint = load_checkpoint(args.ckpt)
    
    # 检查是否需要处理前缀
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 检查是否有模块前缀，如果有则移除
    if any(k.startswith('module.student.') for k in state_dict.keys()):
        print("检测到 'module.student.' 前缀，将进行处理...")
        state_dict = remove_module_prefix(state_dict)
    
    # 加载处理后的权重
    try:
        model.load_state_dict(state_dict)
        print("成功加载模型权重")
    except Exception as e:
        print(f"加载权重时出错: {e}")
        # 打印找到的键和模型期望的键
        print("检查点中的键:")
        for key in list(state_dict.keys())[:10]:  # 只打印前10个键
            print(f"  {key}")
        print(f"  ... (共 {len(state_dict)} 个键)")
        
        print("\n模型期望的键:")
        for key in list(model.state_dict().keys())[:10]:
            print(f"  {key}")
        print(f"  ... (共 {len(model.state_dict())} 个键)")
        
        raise e
    
    # 提取特征
    features, labels = extract_features(model, val_loader)
    
    # 绘制t-SNE
    save_dir = os.path.join(cfg.LOG.PREFIX, 'tsne')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{cfg.DISTILLER.TYPE}_tsne.png")
    plot_tsne(features, labels, save_path, f"{cfg.DISTILLER.TYPE} t-SNE Visualization")

if __name__ == '__main__':
    main() 