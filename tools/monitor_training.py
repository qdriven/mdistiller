import os
import re
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import defaultdict

def extract_temperature_data(log_file):
    """Extract temperature values from worklog.txt file"""
    temp_data = []
    step_data = []
    
    # Regex patterns to match temperature logs
    patterns = [
        r'Step (\d+), Temperature:\s*([\d.]+)',
        r'Step (\d+), GRL-Temp:\s*([\d.]+)',
        r'Step (\d+), Temp:\s*([\d.]+)',
    ]
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        step = int(match.group(1))
                        temp = float(match.group(2))
                        step_data.append(step)
                        temp_data.append(temp)
                        break
    except Exception as e:
        print(f"Error extracting temperature data: {e}")
    
    return step_data, temp_data

def extract_loss_data(log_file):
    """Extract loss component values from worklog.txt file"""
    data = defaultdict(list)
    epoch_data = []
    
    # Regex patterns to match different loss components
    patterns = {
        'ce_loss': r'Epoch[^,]+, lr:[^,]+, ce_loss:\s*([\d.]+)',
        'kd_loss': r'Epoch[^,]+, lr:[^,]+,[^,]+, kd_loss:\s*([\d.]+)',
        'temp_loss': r'Epoch[^,]+, lr:[^,]+,[^,]+,[^,]+, temp_loss:\s*([\d.]+)',
        'epoch': r'Epoch\s*(\d+)'
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Extract epoch number
                epoch_match = re.search(patterns['epoch'], line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    epoch_data.append(epoch)
                
                # Extract loss components
                for loss_type, pattern in patterns.items():
                    if loss_type == 'epoch':
                        continue
                    match = re.search(pattern, line)
                    if match:
                        loss_val = float(match.group(1))
                        data[loss_type].append(loss_val)
    except Exception as e:
        print(f"Error extracting loss data: {e}")
    
    return epoch_data, data

def plot_temperature_evolution(step_data, temp_data, save_path):
    """Plot temperature changes over training steps"""
    plt.figure(figsize=(10, 6))
    plt.plot(step_data, temp_data, '-o', markersize=2)
    plt.xlabel('Training Step')
    plt.ylabel('Temperature')
    plt.title('Temperature Evolution During Training')
    plt.grid(True)
    
    # Highlight important values
    if temp_data:
        initial_temp = temp_data[0]
        final_temp = temp_data[-1]
        max_temp = max(temp_data)
        min_temp = min(temp_data)
        
        plt.axhline(y=initial_temp, color='g', linestyle='--', alpha=0.5, label=f'Initial: {initial_temp:.2f}')
        plt.axhline(y=final_temp, color='r', linestyle='--', alpha=0.5, label=f'Final: {final_temp:.2f}')
        
        # Add annotations for max and min
        max_idx = temp_data.index(max_temp)
        min_idx = temp_data.index(min_temp)
        
        plt.annotate(f'Max: {max_temp:.2f}', 
                     xy=(step_data[max_idx], max_temp),
                     xytext=(step_data[max_idx], max_temp + 0.5),
                     arrowprops=dict(facecolor='orange', shrink=0.05),
                     horizontalalignment='center')
        
        plt.annotate(f'Min: {min_temp:.2f}', 
                     xy=(step_data[min_idx], min_temp),
                     xytext=(step_data[min_idx], min_temp - 0.5),
                     arrowprops=dict(facecolor='blue', shrink=0.05),
                     horizontalalignment='center')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved temperature evolution plot to {save_path}")

def plot_loss_components(epoch_data, loss_data, save_path):
    """Plot different loss components over epochs"""
    plt.figure(figsize=(12, 8))
    
    for loss_type, values in loss_data.items():
        if values and len(values) > 0:
            # Ensure epoch_data and values have the same length
            epochs = epoch_data[:len(values)] if len(epoch_data) >= len(values) else epoch_data
            
            if len(epochs) < len(values):
                # If we have more values than epochs, generate sequential indices
                epochs = list(range(1, len(values) + 1))
            
            plt.plot(epochs, values, '-o', markersize=2, label=f'{loss_type} (final: {values[-1]:.4f})')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Components During Training')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss components plot to {save_path}")

def analyze_model_performance(log_dir, method_name, output_dir):
    """Analyze model performance based on worklog and temperature data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find worklog.txt file
    worklog_path = os.path.join(log_dir, "worklog.txt")
    if not os.path.exists(worklog_path):
        print(f"Warning: No worklog.txt found at {worklog_path}")
        return
    
    # Extract temperature and loss data
    step_data, temp_data = extract_temperature_data(worklog_path)
    epoch_data, loss_data = extract_loss_data(worklog_path)
    
    # Print summary statistics
    print(f"\n===== {method_name} Training Analysis =====")
    
    if temp_data:
        print(f"Temperature Stats:")
        print(f"  Initial: {temp_data[0]:.2f}")
        print(f"  Final:   {temp_data[-1]:.2f}")
        print(f"  Min:     {min(temp_data):.2f}")
        print(f"  Max:     {max(temp_data):.2f}")
        print(f"  Range:   {max(temp_data) - min(temp_data):.2f}")
        
        # Save temperature plot
        temp_plot_path = os.path.join(output_dir, f"{method_name}_temperature.png")
        plot_temperature_evolution(step_data, temp_data, temp_plot_path)
    else:
        print("No temperature data found")
    
    if loss_data:
        print(f"\nLoss Component Stats:")
        for loss_type, values in loss_data.items():
            if values:
                print(f"  {loss_type}:")
                print(f"    Initial: {values[0]:.4f}")
                print(f"    Final:   {values[-1]:.4f}")
                print(f"    Min:     {min(values):.4f}")
                print(f"    Max:     {max(values):.4f}")
        
        # Save loss components plot
        loss_plot_path = os.path.join(output_dir, f"{method_name}_losses.png")
        plot_loss_components(epoch_data, loss_data, loss_plot_path)
    else:
        print("No loss component data found")
    
    print(f"===== Analysis Complete =====\n")
    print(f"Diagnostic plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Monitor and analyze distillation training')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory containing worklog.txt file')
    parser.add_argument('--method', type=str, required=True, help='Method name for labeling output files')
    parser.add_argument('--output_dir', type=str, default='./training_analysis', help='Output directory for plots')
    args = parser.parse_args()
    
    analyze_model_performance(args.log_dir, args.method, args.output_dir)

if __name__ == "__main__":
    main() 