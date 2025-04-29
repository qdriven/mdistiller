import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import re

def parse_worklog(worklog_path):
    """统一的worklog解析方法"""
    data = {
        'epochs': [],
        'train_accs': [],
        'test_accs': [],
        'temperatures': [],
        'train_losses': [],
        'test_losses': [],
        'test_acc_top5s': []
    }
    
    current_epoch_data = {}
    
    with open(worklog_path, 'r') as f:
        current_epoch = None
        for line in f:
            line = line.strip()
            
            # 跳过分隔线
            if line.startswith('-' * 25):
                continue
                
            # 解析各个指标
            if line.startswith('epoch:'):
                if current_epoch is not None:
                    # 保存前一个epoch的数据
                    for key in data.keys():
                        if key != 'epochs' and key in current_epoch_data:
                            data[key].append(current_epoch_data[key])
                
                current_epoch = int(line.split(':')[1].strip())
                data['epochs'].append(current_epoch)
                current_epoch_data = {}
                continue
            
            # 解析其他指标
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = float(value.strip())
                
                if key == 'train_acc':
                    current_epoch_data['train_accs'] = value
                elif key == 'test_acc':
                    current_epoch_data['test_accs'] = value
                elif key == 'train_loss':
                    current_epoch_data['train_losses'] = value
                elif key == 'test_loss':
                    current_epoch_data['test_losses'] = value
                elif key == 'test_acc_top5':
                    current_epoch_data['test_acc_top5s'] = value
            
            # 解析温度信息
            elif '[TEMP]' in line:
                temp = float(line.split('=')[1].strip())
                current_epoch_data['temperatures'] = temp

    # 保存最后一个epoch的数据
    if current_epoch is not None:
        for key in data.keys():
            if key != 'epochs' and key in current_epoch_data:
                data[key].append(current_epoch_data[key])

    return data

def plot_acc_comparison(dkd_data, ctdkd_data, grlctdkd_data):
    """绘制准确率对比图"""
    plt.figure(figsize=(12, 8))
    
    methods = {
        'DKD': dkd_data,
        'CTDKD': ctdkd_data,
        'GRLCTDKD': grlctdkd_data
    }
    
    colors = {'DKD': 'b', 'CTDKD': 'g', 'GRLCTDKD': 'm'}
    
    for method_name, data in methods.items():
        if data:
            color = colors[method_name]
            plt.plot(data['epochs'], data['train_accs'], f'{color}--', 
                    label=f'{method_name} Train Acc')
            plt.plot(data['epochs'], data['test_accs'], f'{color}-', 
                    label=f'{method_name} Test Acc')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_comparison.png')
    plt.close()

def plot_temp_comparison(ctdkd_data, grlctdkd_data):
    """绘制温度对比图"""
    plt.figure(figsize=(12, 8))
    
    methods = {
        'CTDKD': ctdkd_data,
        'GRLCTDKD': grlctdkd_data
    }
    
    colors = {'CTDKD': 'g', 'GRLCTDKD': 'm'}
    
    for method_name, data in methods.items():
        if data and 'temperatures' in data:
            temps = data['temperatures']
            if temps:
                plt.plot(data['epochs'], temps, f'{colors[method_name]}-', 
                        label=f'{method_name} Temperature')
    
    plt.axhline(y=4.0, color='r', linestyle='--', label='DKD (Fixed Temperature)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Temperature')
    plt.title('Temperature Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig('temperature_comparison.png')
    plt.close()

def plot_all_comparisons():
    """主函数：读取并绘制所有对比图"""
    # 定义数据路径
    paths = {
        'dkd': {
            'acc_worklog': Path('output/dkd/cifar100_baselines/cifar100_res110_res20/worklog.txt'),
        },
        'ctdkd': {
            'acc_worklog': Path('output/ctdkd/cifar100_baselines/cifar100_res110_res20/worklog.txt'),
        },
        'grlctdkd': {
            'acc_worklog': Path('output/grlctdkd/cifar100_baselines/cifar100_res110_res20/worklog.txt'),
        }
    }
    
    # 读取并处理数据
    data = {}
    for method, method_paths in paths.items():
        if method_paths['acc_worklog'].exists():
            data[method] = parse_worklog(method_paths['acc_worklog'])
        else:
            print(f"Warning: Worklog not found for {method}")
    
    # 绘制对比图
    if data:
        plot_acc_comparison(
            data.get('dkd'), 
            data.get('ctdkd'), 
            data.get('grlctdkd')
        )
        plot_temp_comparison(
            data.get('ctdkd'),
            data.get('grlctdkd')
        )
    else:
        print("No data available for plotting")

if __name__ == '__main__':
    plot_all_comparisons()
