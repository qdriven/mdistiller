import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mdistiller.engine.cfg import CFG as cfg

def extract_temps_from_worklog(worklog_file):
    """从worklog文件中提取温度记录"""
    if not os.path.exists(worklog_file):
        print(f"Warning: Worklog file {worklog_file} does not exist")
        return []
    
    temperatures = []
    try:
        with open(worklog_file, 'r') as f:
            for line in f:
                if 'temperature:' in line:
                    match = re.search(r'temperature:\s*(\d+\.\d+)', line)
                    if match:
                        temp = float(match.group(1))
                        temperatures.append(temp)
        
        if temperatures:
            print(f"Extracted {len(temperatures)} temperature records from {worklog_file}")
        else:
            print(f"No temperature records found in {worklog_file}")
    except Exception as e:
        print(f"Error reading worklog file: {e}")
    
    return temperatures

def load_log(log_file):
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} does not exist")
        return []
        
    temperatures = []
    with open(log_file, 'r') as f:
        # 首先尝试将整个文件作为一个JSON数组加载（CTDKD格式）
        try:
            content = f.read()
            if content.strip().startswith('[') and content.strip().endswith(']'):
                temps_data = json.loads(content)
                if isinstance(temps_data, list):
                    # 直接使用整个数组
                    temperatures = temps_data
                    print(f"Loaded {len(temperatures)} temperature records from {log_file} as JSON array")
                    return temperatures
        except json.JSONDecodeError:
            # 如果不是整个文件的JSON数组，回到文件开头重新按行解析
            f.seek(0)
            
        # 尝试每行解析一个JSON记录（传统JSON行格式）
        for line in f:
            try:
                log_data = json.loads(line)
                if 'temperature' in log_data:
                    temperatures.append(log_data['temperature'])
            except:
                continue
        
        if temperatures:
            print(f"Loaded {len(temperatures)} temperature records from {log_file} as JSON lines")
    
    return temperatures

def get_constant_temperature(config_file=None, length=100):
    """从配置文件中获取DKD的固定温度值，或使用默认值"""
    if config_file and os.path.exists(config_file):
        try:
            # 加载配置文件
            cfg.merge_from_file(config_file)
            temperature = cfg.DKD.T
            print(f"Loaded DKD temperature {temperature} from config file {config_file}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            temperature = 4.0  # 默认值
            print(f"Using default DKD temperature: {temperature}")
    else:
        temperature = 4.0  # 默认值
        print(f"Using default DKD temperature: {temperature}")
    
    # 创建固定值的温度数组
    return [temperature] * length

def create_synthetic_temperature_curve(initial_temp=4.0, min_temp=1.0, max_temp=10.0, length=240):
    """创建一个合成的温度曲线，用于在没有实际数据时展示CTDKD的行为"""
    # 创建一个先上升后下降的曲线，模拟CTDKD的温度变化
    temps = []
    
    # 前30%：温度上升
    rise_phase = int(length * 0.3)
    for i in range(rise_phase):
        progress = i / rise_phase
        temp = initial_temp + progress * (max_temp - initial_temp) * 0.8
        temps.append(temp)
    
    # 中间40%：高温阶段，有波动
    high_phase = int(length * 0.4)
    high_temp = initial_temp + (max_temp - initial_temp) * 0.8
    for i in range(high_phase):
        # 添加一些波动
        noise = np.sin(i/5) * 0.5
        temp = high_temp + noise
        temp = min(max(temp, min_temp), max_temp)  # 确保在范围内
        temps.append(temp)
    
    # 后30%：温度下降
    drop_phase = length - len(temps)
    start_temp = temps[-1]
    for i in range(drop_phase):
        progress = i / drop_phase
        temp = start_temp - progress * (start_temp - min_temp) * 0.9
        temps.append(max(temp, min_temp))
    
    print(f"Created synthetic temperature curve with {length} points")
    return temps

def plot_temperature_curve(temp_data, save_path, title):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(temp_data) + 1)
    plt.plot(epochs, temp_data, 'b-', label='Temperature')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dkd_log', type=str, help='Path to DKD log file or None')
    parser.add_argument('--ctdkd_log', type=str, help='Path to CTDKD temperature_log.json')
    parser.add_argument('--ctdkd_worklog', type=str, help='Path to CTDKD worklog.txt as fallback')
    parser.add_argument('--dkd_config', type=str, help='Path to DKD config file')
    parser.add_argument('--save_dir', type=str, default='./output/curves')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data if no real data available')
    parser.add_argument('--num_epochs', type=int, default=240, help='Number of training epochs (for synthetic data)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 尝试加载CTDKD温度数据，优先使用temperature_log.json，然后尝试从worklog.txt提取
    ctdkd_temps = []
    if args.ctdkd_log and os.path.exists(args.ctdkd_log):
        ctdkd_temps = load_log(args.ctdkd_log)
    
    # 如果没有找到temperature_log.json或其为空，尝试从worklog.txt提取
    if not ctdkd_temps and args.ctdkd_worklog:
        ctdkd_temps = extract_temps_from_worklog(args.ctdkd_worklog)
    
    # 如果仍然没有数据，但用户要求使用合成数据，则生成一个合成的温度曲线
    if not ctdkd_temps and args.synthetic:
        print("No CTDKD temperature data found, creating synthetic data")
        ctdkd_temps = create_synthetic_temperature_curve(length=args.num_epochs)
    
    # 绘制CTDKD温度曲线（如果有数据）
    if ctdkd_temps:
        plot_temperature_curve(
            ctdkd_temps,
            os.path.join(args.save_dir, 'temperature_curve_ctdkd.png'),
            'Temperature Learning Curve (CTDKD)'
        )
    else:
        print("No CTDKD temperature data available, skipping individual curve")

    # 加载或创建DKD温度数据
    dkd_temps = []
    if args.dkd_log:
        dkd_temps = load_log(args.dkd_log)
    
    # 如果没有DKD温度数据，创建固定值数组
    if not dkd_temps:
        # 使用CTDKD数据的长度或者默认的epoch数量
        length = len(ctdkd_temps) if ctdkd_temps else args.num_epochs
        dkd_temps = get_constant_temperature(args.dkd_config, length)
    
    # 绘制DKD温度曲线
    if dkd_temps:
        plot_temperature_curve(
            dkd_temps,
            os.path.join(args.save_dir, 'temperature_curve_dkd.png'),
            'Temperature Learning Curve (DKD)'
        )
    
    # 创建对比图（如果至少有一种方法的数据）
    if dkd_temps or ctdkd_temps:
        plt.figure(figsize=(12, 8))
        
        # 确定x轴范围
        max_length = max(len(dkd_temps) if dkd_temps else 0, 
                         len(ctdkd_temps) if ctdkd_temps else 0)
        
        if max_length == 0:
            print("No temperature data available for comparison")
            return
            
        epochs = range(1, max_length + 1)
        
        # 绘制DKD温度曲线（如果有数据）
        if dkd_temps:
            plt.plot(epochs[:len(dkd_temps)], dkd_temps, 'r-', linewidth=2, label='DKD')
            # 添加辅助线
            plt.axhline(y=dkd_temps[0], color='r', linestyle='--', alpha=0.5)
            plt.text(max_length * 0.8, dkd_temps[0] * 1.05, f'DKD: {dkd_temps[0]}', color='r')
        
        # 绘制CTDKD温度曲线（如果有数据）
        if ctdkd_temps:
            plt.plot(epochs[:len(ctdkd_temps)], ctdkd_temps, 'b-', linewidth=2, label='CTDKD')
        
        # 如果只有一种方法有数据，添加注释
        if not dkd_temps:
            plt.text(max_length * 0.5, np.mean(ctdkd_temps), 'DKD: Fixed temperature (not shown)', color='r')
        if not ctdkd_temps:
            plt.text(max_length * 0.5, dkd_temps[0] * 0.8, 'CTDKD: No data available', color='b')
            
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Temperature', fontsize=14)
        
        # 设置标题并标明数据来源
        title = 'Temperature Comparison (DKD vs CTDKD)'
        if args.synthetic and not (args.ctdkd_log and os.path.exists(args.ctdkd_log)):
            title += ' - Using synthetic data'
        plt.title(title, fontsize=16)
        
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # 美化图表
        plt.tight_layout()
        
        # 保存高分辨率图片
        plt.savefig(os.path.join(args.save_dir, 'temperature_comparison.png'), dpi=300)
        plt.close()
        
        print(f"Temperature comparison chart saved to {os.path.join(args.save_dir, 'temperature_comparison.png')}")
    else:
        print("Could not create comparison chart - no temperature data available")

if __name__ == '__main__':
    main() 