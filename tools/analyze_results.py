import os
import re
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

# 添加项目路径，以便导入相关模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mdistiller.engine.cfg import CFG as cfg

def find_log_files(base_dir, methods=None, model_pattern=None):
    """查找各种方法的日志文件"""
    log_files = {}
    
    # 如果没有指定方法，则使用默认方法
    if not methods:
        methods = ["DKD", "CTDKD", "GRLCTDKD"]
    
    # 列出目录中的所有子目录
    try:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Found subdirectories: {subdirs}")
    except Exception as e:
        print(f"Error listing directory {base_dir}: {e}")
        subdirs = []
    
    # 如果指定了模型模式，先过滤与模型匹配的目录
    if model_pattern:
        filtered_subdirs = []
        for subdir in subdirs:
            if model_pattern.lower() in subdir.lower():
                filtered_subdirs.append(subdir)
        if filtered_subdirs:
            print(f"Filtered subdirectories for model pattern '{model_pattern}': {filtered_subdirs}")
            subdirs = filtered_subdirs
    
    for method in methods:
        method_lower = method.lower()
        # 尝试匹配目录名中以逗号分隔的方法名
        matched_dirs = []
        for subdir in subdirs:
            # 分割目录名，通常格式为 "method,model1,model2"
            parts = subdir.split(',')
            if parts and parts[0] == method_lower:
                matched_dirs.append(subdir)
        
        # 如果没找到精确匹配，尝试模糊匹配
        if not matched_dirs:
            for subdir in subdirs:
                if method_lower in subdir.lower():
                    matched_dirs.append(subdir)
        
        # 查找工作日志
        for matched_dir in matched_dirs:
            log_path = os.path.join(base_dir, matched_dir, "worklog.txt")
            if os.path.exists(log_path):
                log_files[method] = log_path
                print(f"Found log file for {method}: {log_path}")
                break
        
        if method not in log_files:
            print(f"Warning: No log file found for {method}")
    
    return log_files

def find_temperature_logs(base_dir, methods=None, model_pattern=None):
    """查找温度日志文件"""
    temp_log_files = {}
    
    # 如果没有指定方法，则只查找支持温度优化的方法
    if not methods:
        methods = ["CTDKD", "GRLCTDKD"]
    
    # 列出目录中的所有子目录
    try:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    except Exception as e:
        print(f"Error listing directory {base_dir}: {e}")
        subdirs = []
    
    # 如果指定了模型模式，先过滤与模型匹配的目录
    if model_pattern:
        filtered_subdirs = []
        for subdir in subdirs:
            if model_pattern.lower() in subdir.lower():
                filtered_subdirs.append(subdir)
        if filtered_subdirs:
            subdirs = filtered_subdirs
    
    # 获取每个方法对应的目录
    method_dirs = {}
    for method in methods:
        method_lower = method.lower()
        for subdir in subdirs:
            parts = subdir.split(',')
            if parts and parts[0] == method_lower:
                method_dirs[method] = os.path.join(base_dir, subdir)
                break
        
        # 如果没找到精确匹配，尝试模糊匹配
        if method not in method_dirs:
            for subdir in subdirs:
                if method_lower in subdir.lower():
                    method_dirs[method] = os.path.join(base_dir, subdir)
                    break
    
    # 先在父目录（base_dir）中查找带有方法名称的温度日志
    for method in methods:
        method_upper = method.upper()
        # 尝试在父目录查找含有方法名的温度日志文件
        parent_dir_patterns = [
            os.path.join(base_dir, f"temperature_log_{method_upper}*.json"),
            os.path.join(base_dir, f"temperature_log_{method}*.json"),
            os.path.join(base_dir, f"temperature_*{method_upper}*.json"),
            os.path.join(base_dir, f"temperature_*{method}*.json")
        ]
        
        parent_logs = []
        for pattern in parent_dir_patterns:
            parent_logs.extend(glob.glob(pattern))
        
        if parent_logs:
            temp_log_files[method] = parent_logs[0]
            print(f"Found temperature log for {method} in parent directory: {temp_log_files[method]}")
            continue
    
    # 如果在父目录没有找到，再在对应的方法目录中查找
    for method, method_dir in method_dirs.items():
        if method in temp_log_files:
            continue  # 已经在父目录找到，跳过
            
        # 多种可能的文件名模式
        patterns = [
            os.path.join(method_dir, "temperature_log_*.json"),
            os.path.join(method_dir, "temperature_*.json"),
            os.path.join(method_dir, "*.json"),  # 尝试查找任何JSON文件
        ]
        
        method_logs = []
        for pattern in patterns:
            method_logs.extend(glob.glob(pattern))
        
        if method_logs:
            # 优先使用包含 "temperature" 的日志文件
            temperature_logs = [log for log in method_logs if "temperature" in os.path.basename(log).lower()]
            if temperature_logs:
                temp_log_files[method] = temperature_logs[0]
            else:
                temp_log_files[method] = method_logs[0]
            print(f"Found temperature log for {method}: {temp_log_files[method]}")
        else:
            print(f"Warning: No temperature log found for {method}")
    
    return temp_log_files

def extract_data_from_log(log_file, pattern, group_index=1):
    """从日志文件提取匹配特定模式的数据"""
    values = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    try:
                        values.append(float(match.group(group_index)))
                    except:
                        pass
        if values:
            print(f"Extracted {len(values)} values for pattern '{pattern}' from {log_file}")
        else:
            print(f"Warning: No values found for pattern '{pattern}' in {log_file}")
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return values

def load_temperature_log(log_file):
    """加载温度日志文件"""
    temperatures = []
    try:
        with open(log_file, 'r') as f:
            temps_data = json.load(f)
            if isinstance(temps_data, list):
                temperatures = [float(t) for t in temps_data]
                print(f"Loaded {len(temperatures)} temperature records from {log_file}")
            else:
                print(f"Warning: Expected JSON array in {log_file}, got {type(temps_data)}")
    except Exception as e:
        print(f"Error loading temperature log {log_file}: {e}")
    
    return temperatures

def extract_temps_from_worklog(worklog_file):
    """从工作日志中提取温度数据"""
    temperatures = []
    # 正则表达式匹配温度行（各种可能的格式）
    patterns = [
        r'temperature:\s*([\d.]+)',
        r'Temperature:\s*([\d.]+)',
        r'GRL-Temp:\s*([\d.]+)',
        r'GRL[_ ]?Temp[_ ]?[=:]?\s*([\d.]+)',
        r'Temp[_ ]?[=:]?\s*([\d.]+)',
        r'T[_ ]?=\s*([\d.]+)',  # For example "T = 10.0" or "T=10.0"
        r't[_ ]?=\s*([\d.]+)',  # For example "t = 10.0" or "t=10.0"
    ]
    
    try:
        with open(worklog_file, 'r') as f:
            lines = f.readlines()
            print(f"Read {len(lines)} lines from worklog {worklog_file}")
            
            # 先简单搜索一些可能的温度相关关键词，以辅助调试
            temp_keywords = ['temperature', 'temp', 'grl', 't=', 'T=']
            sample_lines = []
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in temp_keywords):
                    sample_lines.append(f"Line {i+1}: {line.strip()}")
                    if len(sample_lines) >= 5:  # 最多显示5行样本
                        break
            
            if sample_lines:
                print(f"Sample lines containing temperature-related keywords:")
                for line in sample_lines:
                    print(f"  {line}")
            
            # 正常处理提取温度值
            for line in lines:
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        try:
                            temp = float(match.group(1))
                            temperatures.append(temp)
                            break  # 找到一个匹配就跳出内层循环
                        except ValueError:
                            pass
        
        if temperatures:
            print(f"Extracted {len(temperatures)} temperature records from worklog {worklog_file}")
            print(f"Temperature values: min={min(temperatures)}, max={max(temperatures)}, avg={sum(temperatures)/len(temperatures):.2f}")
        else:
            print(f"No temperature records found in worklog {worklog_file}")
    except Exception as e:
        print(f"Error reading worklog file {worklog_file}: {e}")
    
    return temperatures

def get_dkd_temperature(config_file=None):
    """获取DKD的固定温度值"""
    temperature = 4.0  # 默认值
    if config_file and os.path.exists(config_file):
        try:
            temp_cfg = cfg.clone()
            temp_cfg.merge_from_file(config_file)
            if hasattr(temp_cfg, 'DKD') and hasattr(temp_cfg.DKD, 'T'):
                temperature = temp_cfg.DKD.T
                print(f"Loaded DKD temperature {temperature} from config {config_file}")
            else:
                print(f"Warning: DKD.T not found in {config_file}, using default {temperature}")
        except Exception as e:
            print(f"Error loading DKD config: {e}")
    else:
        print(f"Using default DKD temperature: {temperature}")
    
    return temperature

def plot_temperature_comparison(temp_data, save_path, title):
    """绘制温度比较图"""
    plt.figure(figsize=(12, 7))
    
    # 为每个方法绘制温度曲线
    for method, temps in temp_data.items():
        if isinstance(temps, (int, float)):
            # 对于单个值（例如DKD的固定温度），绘制一条水平线
            plt.axhline(y=temps, label=f"{method} (T={temps})", linestyle='--', color='green' if method == 'DKD' else 'gray')
        elif isinstance(temps, list) and len(temps) > 0:
            # 对于大量数据点，需要进行采样以避免图形过于密集
            max_points = 1000  # 最多显示1000个点
            
            if len(temps) > max_points:
                # 采样间隔
                step = len(temps) // max_points
                # 采样数据
                sampled_temps = temps[::step]
                # 对应的x轴数据点
                steps = list(range(0, len(temps), step))
                
                plt.plot(steps, sampled_temps, label=f"{method} (Initial:{temps[0]:.1f}, Final:{temps[-1]:.1f})")
                
                # 添加起点和终点标记
                plt.scatter([0], [temps[0]], color='red', zorder=5)
                plt.scatter([len(temps)-1], [temps[-1]], color='blue', zorder=5)
                
                # 标记最大值和最小值
                min_val = min(temps)
                max_val = max(temps)
                min_idx = temps.index(min_val)
                max_idx = temps.index(max_val)
                
                # 只有当最大值和最小值不是初始值或最终值时才标记
                if min_idx != 0 and min_idx != len(temps)-1:
                    plt.scatter([min_idx], [min_val], color='purple', zorder=5, marker='v')
                    plt.annotate(f"Min: {min_val:.1f}", (min_idx, min_val), 
                                 textcoords="offset points", xytext=(0,-15), ha='center')
                
                if max_idx != 0 and max_idx != len(temps)-1:
                    plt.scatter([max_idx], [max_val], color='orange', zorder=5, marker='^')
                    plt.annotate(f"Max: {max_val:.1f}", (max_idx, max_val), 
                                 textcoords="offset points", xytext=(0,10), ha='center')
            else:
                # 数据点较少，直接绘制
                steps = range(len(temps))
                plt.plot(steps, temps, label=f"{method} (Initial:{temps[0]:.1f}, Final:{temps[-1]:.1f})")
    
    plt.xlabel('Training Steps')
    plt.ylabel('Temperature')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # 添加注释说明图形内容
    plt.figtext(0.5, 0.01, 
                "Note: This graph shows temperature evolution for different methods.\n"
                "DKD uses a fixed temperature; CTDKD and GRLCTDKD use learnable temperatures.\n"
                "Red dot: initial value, Blue dot: final value. Purple triangle: minimum, Orange triangle: maximum.", 
                ha='center', fontsize=9, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部注释留出空间
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved temperature comparison plot to {save_path}")

def plot_metric_comparison(data, metric_name, save_path, title):
    """绘制指标比较图（准确率或损失）"""
    plt.figure(figsize=(12, 7))
    
    for method, values in data.items():
        if values:
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=method)
    
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {metric_name} comparison plot to {save_path}")

def plot_all_in_one(train_acc_data, test_acc_data, train_loss_data, temp_data, save_path, title):
    """绘制4合1综合比较图"""
    plt.figure(figsize=(20, 15))
    
    # 1. 训练准确率
    plt.subplot(2, 2, 1)
    for method, values in train_acc_data.items():
        if values:
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=method)
    plt.title("Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    # 2. 测试准确率
    plt.subplot(2, 2, 2)
    for method, values in test_acc_data.items():
        if values:
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=method)
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    # 3. 训练损失
    plt.subplot(2, 2, 3)
    for method, values in train_loss_data.items():
        if values:
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=method)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # 4. 温度变化
    plt.subplot(2, 2, 4)
    for method, temps in temp_data.items():
        if isinstance(temps, (int, float)):
            # 对于单个值（例如DKD的固定温度），绘制一条水平线
            plt.axhline(y=temps, label=f"{method} (T={temps})", linestyle='--', color='green' if method == 'DKD' else 'gray')
        elif isinstance(temps, list) and len(temps) > 0:
            # 对于大量数据点，需要进行采样以避免图形过于密集
            max_points = 500  # 最多显示500个点
            
            if len(temps) > max_points:
                # 采样间隔
                step = len(temps) // max_points
                # 采样数据
                sampled_temps = temps[::step]
                # 对应的x轴数据点
                steps = list(range(0, len(temps), step))
                
                plt.plot(steps, sampled_temps, label=f"{method} ({temps[0]:.1f}->{temps[-1]:.1f})")
            else:
                # 数据点较少，直接绘制
                steps = range(len(temps))
                plt.plot(steps, temps, label=f"{method} ({temps[0]:.1f}->{temps[-1]:.1f})")
    
    plt.title("Temperature Evolution")
    plt.xlabel("Training Steps")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给总标题留出空间
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comprehensive comparison plot to {save_path}")

def print_final_results(metrics_data):
    """打印最终结果表格"""
    print("\n" + "="*50)
    print("Final Results:")
    print("="*50)
    print(f"{'Method':<20} {'Best Accuracy':<15} {'Best Epoch':<15}")
    print("-"*50)
    
    for method, data in metrics_data.items():
        if 'test_acc' in data and data['test_acc']:
            test_acc = data['test_acc']
            best_acc = max(test_acc)
            best_epoch = test_acc.index(best_acc) + 1
            print(f"{method:<20} {best_acc:.2f}%{' '*10} {best_epoch:<15}")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize distillation results")
    parser.add_argument("--output_dir", type=str, default="./result_analysis", help="Directory to save plots")
    parser.add_argument("--logs_dir", type=str, default="./output", help="Directory containing logs")
    parser.add_argument("--methods", type=str, nargs="+", default=["DKD", "CTDKD", "GRLCTDKD"], help="Methods to compare")
    parser.add_argument("--dkd_config", type=str, help="Path to DKD config file (for temperature)")
    parser.add_argument("--model_name", type=str, default="comparison", help="Name for output files")
    parser.add_argument("--model_pattern", type=str, help="Pattern to filter log directories by model (e.g., 'res110_res32')")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 使用模型名称作为过滤模式，如果未指定独立过滤模式
    model_pattern = args.model_pattern if args.model_pattern else args.model_name
    
    # 查找日志文件
    worklog_files = find_log_files(args.logs_dir, args.methods, model_pattern)
    temp_log_files = find_temperature_logs(args.logs_dir, [m for m in args.methods if m != "DKD"], model_pattern)
    
    # 从日志文件提取数据
    train_acc_data = {}
    test_acc_data = {}
    train_loss_data = {}
    test_loss_data = {}
    temperature_data = {}
    
    for method, log_file in worklog_files.items():
        train_acc = extract_data_from_log(log_file, r"train_acc: ([0-9.]+)")
        test_acc = extract_data_from_log(log_file, r"test_acc: ([0-9.]+)")
        train_loss = extract_data_from_log(log_file, r"train_loss: ([0-9.]+)")
        test_loss = extract_data_from_log(log_file, r"test_loss: ([0-9.]+)")
        
        train_acc_data[method] = train_acc
        test_acc_data[method] = test_acc
        train_loss_data[method] = train_loss
        test_loss_data[method] = test_loss
    
    # 处理温度数据
    for method, temp_file in temp_log_files.items():
        temperature_data[method] = load_temperature_log(temp_file)
        print(f"Loaded temperature data for {method} from JSON file - {len(temperature_data[method])} values")
    
    # 如果没有找到温度日志文件，尝试从工作日志中提取
    print("\n====== Attempting to extract temperature data from worklogs ======")
    for method in [m for m in args.methods if m != "DKD"]:
        if method not in temperature_data or not temperature_data[method]:
            if method in worklog_files:
                print(f"\nExtracting temperature for {method} from worklog:")
                temps = extract_temps_from_worklog(worklog_files[method])
                if temps:
                    temperature_data[method] = temps
                    print(f"Successfully extracted {len(temps)} temperature values for {method}")
                else:
                    # 如果没有找到温度数据，设置一个默认值以便于可视化
                    if method == "CTDKD":
                        default_temp = 10.0
                    else:  # GRLCTDKD
                        default_temp = 20.5
                    print(f"No temperature data found for {method}, using default value: {default_temp}")
                    temperature_data[method] = [default_temp] * 3  # 使用三个点而不是一个常量值
    
    # 获取DKD温度
    if "DKD" in args.methods:
        dkd_temp = get_dkd_temperature(args.dkd_config)
        temperature_data["DKD"] = [dkd_temp] * 3  # 使用三个点而不是一个常量值
        print(f"Using DKD temperature: {dkd_temp}")
    
    # 展示找到的温度数据信息
    print("\n====== Temperature Data Summary ======")
    for method, temps in temperature_data.items():
        if isinstance(temps, list) and len(temps) > 0:
            print(f"{method}: {len(temps)} values, range: {min(temps):.2f} - {max(temps):.2f}")
        elif isinstance(temps, (int, float)):
            print(f"{method}: Constant value {temps:.2f}")
        else:
            print(f"{method}: No temperature data")
    
    # 生成图表
    # 1. 温度比较图
    if temperature_data:
        plot_temperature_comparison(
            temperature_data,
            os.path.join(args.output_dir, f"temperature_{args.model_name}.png"),
            "Temperature Comparison"
        )
    
    # 2. 准确率比较图
    plot_metric_comparison(
        train_acc_data,
        "Training Accuracy (%)",
        os.path.join(args.output_dir, f"train_acc_{args.model_name}.png"),
        "Training Accuracy Comparison"
    )
    
    plot_metric_comparison(
        test_acc_data,
        "Test Accuracy (%)",
        os.path.join(args.output_dir, f"test_acc_{args.model_name}.png"),
        "Test Accuracy Comparison"
    )
    
    # 3. 损失比较图
    plot_metric_comparison(
        train_loss_data,
        "Training Loss",
        os.path.join(args.output_dir, f"train_loss_{args.model_name}.png"),
        "Training Loss Comparison"
    )
    
    plot_metric_comparison(
        test_loss_data,
        "Test Loss",
        os.path.join(args.output_dir, f"test_loss_{args.model_name}.png"),
        "Test Loss Comparison"
    )
    
    # 4. 综合比较图
    plot_all_in_one(
        train_acc_data,
        test_acc_data,
        train_loss_data,
        temperature_data,
        os.path.join(args.output_dir, f"comprehensive_{args.model_name}.png"),
        f"Comprehensive Comparison ({args.model_name})"
    )
    
    # 5. 打印最终结果
    metrics_data = {}
    for method in args.methods:
        metrics_data[method] = {
            'train_acc': train_acc_data.get(method, []),
            'test_acc': test_acc_data.get(method, []),
            'train_loss': train_loss_data.get(method, []),
            'test_loss': test_loss_data.get(method, [])
        }
    
    print_final_results(metrics_data)
    
    # 保存最终结果到文件
    with open(os.path.join(args.output_dir, f"final_results_{args.model_name}.txt"), "w") as f:
        f.write("="*50 + "\n")
        f.write("Final Results:\n")
        f.write("="*50 + "\n")
        f.write(f"{'Method':<20} {'Best Accuracy':<15} {'Best Epoch':<15}\n")
        f.write("-"*50 + "\n")
        
        for method, data in metrics_data.items():
            if 'test_acc' in data and data['test_acc']:
                test_acc = data['test_acc']
                best_acc = max(test_acc)
                best_epoch = test_acc.index(best_acc) + 1
                f.write(f"{method:<20} {best_acc:.2f}%{' '*10} {best_epoch:<15}\n")
        
        f.write("="*50 + "\n")
    
    print(f"\nAll analysis results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()