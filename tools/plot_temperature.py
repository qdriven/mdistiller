import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse

def load_log(log_file):
    temperatures = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                log_data = json.loads(line)
                if 'temperature' in log_data:
                    temperatures.append(log_data['temperature'])
            except:
                continue
    return temperatures

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
    parser.add_argument('--dkd_log', type=str, help='Path to DKD log file')
    parser.add_argument('--ctdkd_log', type=str, help='Path to CTDKD log file')
    parser.add_argument('--save_dir', type=str, default='./output/curves')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载温度数据
    if args.ctdkd_log:
        ctdkd_temps = load_log(args.ctdkd_log)
        plot_temperature_curve(
            ctdkd_temps,
            os.path.join(args.save_dir, 'temperature_curve.png'),
            'Temperature Learning Curve (CTDKD)'
        )

    # 如果有两种方法的数据，绘制对比图
    if args.dkd_log and args.ctdkd_log:
        dkd_temps = load_log(args.dkd_log)
        plt.figure(figsize=(10, 6))
        epochs = range(1, max(len(dkd_temps), len(ctdkd_temps)) + 1)
        if dkd_temps:
            plt.plot(epochs[:len(dkd_temps)], dkd_temps, 'r-', label='DKD')
        plt.plot(epochs[:len(ctdkd_temps)], ctdkd_temps, 'b-', label='CTDKD')
        plt.xlabel('Iteration')
        plt.ylabel('Temperature')
        plt.title('Temperature Comparison (DKD vs CTDKD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.save_dir, 'temperature_comparison.png'))
        plt.close()

if __name__ == '__main__':
    main() 