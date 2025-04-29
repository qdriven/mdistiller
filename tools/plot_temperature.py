import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def parse_worklog_acc(worklog_path):
    """解析精度相关的worklog"""
    data = {
        'epochs': [],
        'train_accs': [],
        'test_accs': [],
        'train_losses': [],
        'test_losses': [],
    }
    
    current_epoch_data = {}
    
    with open(worklog_path, 'r') as f:
        current_epoch = None
        for line in f:
            line = line.strip()
            
            if line.startswith('-' * 25):
                continue
                
            if line.startswith('epoch:'):
                if current_epoch is not None:
                    for key in data.keys():
                        if key != 'epochs' and key in current_epoch_data:
                            data[key].append(current_epoch_data[key])
                
                current_epoch = int(line.split(':')[1].strip())
                data['epochs'].append(current_epoch)
                current_epoch_data = {}
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                try:
                    value = float(value.strip())
                    if key == 'train_acc':
                        current_epoch_data['train_accs'] = value
                    elif key == 'test_acc':
                        current_epoch_data['test_accs'] = value
                    elif key == 'train_loss':
                        current_epoch_data['train_losses'] = value
                    elif key == 'test_loss':
                        current_epoch_data['test_losses'] = value
                except ValueError:
                    continue

    if current_epoch is not None:
        for key in data.keys():
            if key != 'epochs' and key in current_epoch_data:
                data[key].append(current_epoch_data[key])

    return data

def parse_temp_worklog(worklog_path, method):
    """解析温度相关的worklog"""
    data = {
        'steps': [],
        'temperatures': [],
        'gaps': [],
        'grl_values': []
    }
    
    # CTDKD格式: [TEMP] Step=100, T=3.9980
    temp_pattern_ctdkd = r'\[TEMP\] Step=(\d+), T=([\d.]+)'
    # GRLCTDKD格式: [TEMP] Step=X, T=X, Gap=X, GRL=X
    temp_pattern_grl = r'\[TEMP\] Step=(\d+), T=([\d.]+), Gap=([\d.]+), GRL=([\d.]+)'
    
    with open(worklog_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if method == 'ctdkd':
                match = re.search(temp_pattern_ctdkd, line)
                if match:
                    step = int(match.group(1))
                    temp = float(match.group(2))
                    data['steps'].append(step)
                    data['temperatures'].append(temp)
            
            elif method == 'grlctdkd':
                match = re.search(temp_pattern_grl, line)
                if match:
                    step = int(match.group(1))
                    temp = float(match.group(2))
                    gap = float(match.group(3))
                    grl = float(match.group(4))
                    
                    data['steps'].append(step)
                    data['temperatures'].append(temp)
                    data['gaps'].append(gap)
                    data['grl_values'].append(grl)
    
    return data

def plot_accuracy_comparison(methods):
    """绘制不同方法的精度对比图"""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(methods)))
    
    for method, color in zip(methods, colors):
        worklog_path = Path(f'output/{method}/cifar100_baselines/cifar100_res110_res20/worklog.txt')
        if worklog_path.exists():
            data = parse_worklog_acc(worklog_path)
            plt.plot(data['epochs'], data['train_accs'], '--', color=color, 
                    label=f'{method} Train Acc')
            plt.plot(data['epochs'], data['test_accs'], '-', color=color, 
                    label=f'{method} Test Acc')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_comparison.png')
    plt.close()

def plot_temperature_comparison(methods):
    """绘制不同方法的温度对比图"""
    plt.figure(figsize=(12, 6))
    
    # 为每个方法设置固定的颜色和样式
    method_styles = {
        'dkd': {'color': 'blue', 'linestyle': '--', 'label': 'DKD (Fixed T=4.0)'},
        'ctdkd': {'color': 'green', 'linestyle': '-', 'label': 'CTDKD Temperature'},
        'grlctdkd': {'color': 'red', 'linestyle': '-', 'label': 'GRLCTDKD Temperature'}
    }
    
    # 绘制DKD的固定温度
    plt.axhline(y=4.0, **method_styles['dkd'])
    
    # 遍历所有方法
    for method in methods:
        if method == 'dkd':
            continue  # 已经画了DKD的线
        
        worklog_path = Path(f'output/{method}/cifar100_baselines/cifar100_res110_res20/worklog.txt')
        print(f"Processing {method} from {worklog_path}")
        
        if worklog_path.exists():
            data = parse_temp_worklog(worklog_path, method)
            
            if data['temperatures']:
                print(f"Found {len(data['temperatures'])} temperature records for {method}")
                
                # 绘制温度演变
                plt.plot(data['steps'], data['temperatures'], 
                        color=method_styles[method]['color'],
                        linestyle=method_styles[method]['linestyle'],
                        label=method_styles[method]['label'])
                
                # 为GRLCTDKD添加额外信息
                if method == 'grlctdkd' and data['gaps']:
                    plt.plot(data['steps'], data['gaps'], 
                            color='red', linestyle=':', 
                            label='GRLCTDKD Performance Gap')
                    plt.plot(data['steps'], data['grl_values'], 
                            color='red', linestyle='-.', 
                            label='GRLCTDKD GRL Lambda')
        else:
            print(f"Warning: Worklog not found for {method}")
    
    plt.xlabel('Training Steps')
    plt.ylabel('Value')
    plt.title('Temperature Evolution During Training')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 8)
    plt.tight_layout()
    plt.savefig('temperature_comparison.png')
    plt.close()

def main():
    methods = ['dkd', 'ctdkd', 'grlctdkd']
    plot_accuracy_comparison(methods)
    plot_temperature_comparison(methods)

if __name__ == '__main__':
    main()
