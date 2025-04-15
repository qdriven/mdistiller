import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import accuracy_score

def load_results(log_file):
    """加载训练日志中的精度和损失"""
    epochs = []
    accuracies = []
    losses = []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
            current_epoch = None
            current_test_acc = None
            current_test_loss = None
            
            for line in lines:
                line = line.strip()
                
                # 检查是否是 epoch 行
                if line.startswith('epoch:'):
                    if current_epoch is not None and current_test_acc is not None:
                        epochs.append(current_epoch)
                        accuracies.append(current_test_acc)
                        losses.append(current_test_loss if current_test_loss is not None else 0)
                    
                    current_epoch = int(line.split(':')[1].strip())
                    current_test_acc = None
                    current_test_loss = None
                
                # 检查是否是 test_acc 行
                elif line.startswith('test_acc:'):
                    current_test_acc = float(line.split(':')[1].strip())
                
                # 检查是否是 test_loss 行
                elif line.startswith('test_loss:'):
                    current_test_loss = float(line.split(':')[1].strip())
                
                # 检查最后一个 epoch 数据
                elif line == 'best_acc' and current_epoch is not None and current_test_acc is not None:
                    epochs.append(current_epoch)
                    accuracies.append(current_test_acc)
                    losses.append(current_test_loss if current_test_loss is not None else 0)
            
            # 处理最后一个 epoch (如果日志没有以 'best_acc' 结束)
            if current_epoch is not None and current_test_acc is not None:
                epochs.append(current_epoch)
                accuracies.append(current_test_acc)
                losses.append(current_test_loss if current_test_loss is not None else 0)
    except Exception as e:
        print(f"Error loading log file {log_file}: {e}")
    
    return epochs, accuracies, losses

def plot_comparison(methods, log_files, metric='accuracy', save_path='comparison.png'):
    """绘制不同方法的性能比较图"""
    plt.figure(figsize=(12, 8))
    
    for method, log_file in zip(methods, log_files):
        epochs, accuracies, losses = load_results(log_file)
        if metric == 'accuracy':
            plt.plot(epochs, accuracies, label=method)
        else:
            plt.plot(epochs, losses, label=method)
    
    plt.xlabel('Epoch')
    plt.ylabel('Test ' + metric.capitalize())
    plt.title(f'Test {metric.capitalize()} Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def print_final_results(methods, log_files):
    """打印最终结果"""
    print("="*50)
    print("Final Results:")
    print("="*50)
    print(f"{'Method':<20} {'Best Accuracy':<15} {'Best Epoch':<10}")
    print("-"*50)
    
    for method, log_file in zip(methods, log_files):
        epochs, accuracies, _ = load_results(log_file)
        if accuracies:
            best_acc = max(accuracies)
            best_epoch = epochs[accuracies.index(best_acc)]
            print(f"{method:<20} {best_acc:.2f}%{'':<10} {best_epoch:<10}")
    
    print("="*50)

def create_ablation_table(configs, log_files, output_file='ablation_results.md'):
    """创建消融实验结果表格"""
    results = []
    
    for config, log_file in zip(configs, log_files):
        epochs, accuracies, _ = load_results(log_file)
        if accuracies:
            best_acc = max(accuracies)
            best_epoch = epochs[accuracies.index(best_acc)]
            
            # 从配置中提取关键参数
            with open(config, 'r') as f:
                method_type = None
                temperature = None
                for line in f:
                    if "TYPE:" in line and ("DKD" in line or "CTDKD" in line):
                        method_type = line.strip().split(":")[-1].strip().strip('"')
                    if "T:" in line and "TYPE" not in line:
                        temperature = line.strip().split(":")[-1].strip()
                    if "INIT_TEMPERATURE:" in line:
                        init_temp = line.strip().split(":")[-1].strip()
            
            results.append({
                'method': method_type,
                'temperature': temperature if method_type == "DKD" else init_temp,
                'best_acc': best_acc,
                'best_epoch': best_epoch
            })
    
    # 写入Markdown表格
    with open(output_file, 'w') as f:
        f.write("# 消融实验结果\n\n")
        f.write("| 方法 | 温度设置 | 最佳精度 | 最佳轮次 |\n")
        f.write("|------|---------|----------|----------|\n")
        
        for r in results:
            f.write(f"| {r['method']} | {r['temperature']} | {r['best_acc']:.2f}% | {r['best_epoch']} |\n")

def main():
    parser = argparse.ArgumentParser(description="Compare results of different KD methods")
    parser.add_argument('--methods', nargs='+', default=['DKD', 'CTDKD'],
                        help='Names of methods to compare')
    parser.add_argument('--logs', nargs='+', required=True,
                        help='Paths to log files')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Paths to config files for ablation study')
    parser.add_argument('--output_dir', default='./output/comparison',
                        help='Directory to save comparison results')
    parser.add_argument('--metric', choices=['accuracy', 'loss'], default='accuracy',
                        help='Metric to plot')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 绘制性能比较图
    plot_comparison(
        args.methods, 
        args.logs, 
        metric=args.metric,
        save_path=os.path.join(args.output_dir, f'{args.metric}_comparison.png')
    )
    
    # 打印最终结果
    print_final_results(args.methods, args.logs)
    
    # 如果提供了配置文件，创建消融实验表格
    if args.configs:
        create_ablation_table(
            args.configs, 
            args.logs,
            output_file=os.path.join(args.output_dir, 'ablation_results.md')
        )

if __name__ == '__main__':
    main() 