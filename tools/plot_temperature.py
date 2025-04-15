import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
import sys
import re
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mdistiller.engine.cfg import CFG as cfg

def find_temperature_logs(directory, pattern="temperature_log_*.json"):
    """Finds all temperature log files matching specific patterns in a directory."""
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return []
    
    # Patterns for CTDKD (original) and GRLCTDKD
    patterns_to_check = ["temperature_log_*.json", "temperature_log_GRL_*.json"]
    log_files = []
    for p in patterns_to_check:
         search_path = os.path.join(directory, p)
         found_files = glob.glob(search_path)
         if found_files:
             print(f"Found files matching {p}: {found_files}")
             log_files.extend(found_files)
         else:
             print(f"No files found matching {p} in {directory}")
             
    # Remove duplicates if any
    log_files = sorted(list(set(log_files))) # Sort for consistency
    
    if log_files:
        print(f"Found {len(log_files)} unique temperature log files in {directory}:")
        for f in log_files:
            print(f"  - {os.path.basename(f)}")
    else:
        print(f"No temperature log files found in {directory} matching expected patterns.")
    
    return log_files

def extract_temps_from_worklog(worklog_file):
    """Extracts temperature records from a worklog file."""
    if not os.path.exists(worklog_file):
        print(f"Warning: Worklog file {worklog_file} does not exist")
        return []
    
    temperatures = []
    # Regex to find temperature lines (adapt if format changes)
    temp_pattern = re.compile(r'(?:temperature|GRL-Temp):\s*(\d+\.\d+)')
    try:
        with open(worklog_file, 'r') as f:
            for line in f:
                match = temp_pattern.search(line)
                if match:
                    try:
                        temp = float(match.group(1))
                        temperatures.append(temp)
                    except ValueError:
                        print(f"Warning: Could not parse temperature value from line: {line.strip()}")
        
        if temperatures:
            print(f"Extracted {len(temperatures)} temperature records from {worklog_file}")
        else:
            print(f"No temperature records found in {worklog_file}")
    except Exception as e:
        print(f"Error reading worklog file {worklog_file}: {e}")
    
    return temperatures

def load_log(log_file):
    """Loads temperatures from a JSON log file."""
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} does not exist")
        return []
        
    temperatures = []
    try:
        with open(log_file, 'r') as f:
            # Assume the file contains a single JSON array
            temps_data = json.load(f)
            if isinstance(temps_data, list):
                temperatures = [float(t) for t in temps_data] # Ensure float type
                print(f"Loaded {len(temperatures)} records from {log_file} as JSON array")
            else:
                 print(f"Warning: Expected JSON array in {log_file}, got {type(temps_data)}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {log_file}: {e}")
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        
    return temperatures

def get_constant_temperature(config_file=None, length=100):
    """Gets the fixed DKD temperature from config or uses default."""
    temperature = 4.0 # Default DKD temperature
    if config_file and os.path.exists(config_file):
        try:
            temp_cfg = cfg.clone() # Use a clone to avoid modifying global CFG
            temp_cfg.merge_from_file(config_file)
            if hasattr(temp_cfg, 'DKD') and hasattr(temp_cfg.DKD, 'T'):
                temperature = temp_cfg.DKD.T
                print(f"Loaded DKD temperature {temperature} from config: {config_file}")
            else:
                 print(f"Warning: DKD.T not found in {config_file}. Using default {temperature}")
        except Exception as e:
            print(f"Error loading config {config_file}: {e}. Using default {temperature}")
    else:
        print(f"DKD config file not found or specified. Using default DKD temperature: {temperature}")
        
    return [temperature] * length

def create_synthetic_temperature_curve(initial_temp=4.0, min_temp=1.0, max_temp=10.0, length=240):
    """Creates a synthetic temperature curve mimicking potential CTDKD behavior."""
    temps = []
    # Simple rise-plateau-fall pattern
    phase1 = int(length * 0.3)
    phase2 = int(length * 0.7)
    for i in range(length):
        if i < phase1:
            # Rise phase
            progress = i / phase1
            temp = initial_temp + progress * (max_temp - initial_temp) * 0.6
        elif i < phase2:
            # Plateau phase (with slight noise)
            temp = max_temp * 0.8 + np.sin(i/5) * 0.3
        else:
            # Fall phase
            progress = (i - phase2) / (length - phase2)
            start_temp = max_temp * 0.8
            temp = start_temp - progress * (start_temp - min_temp) * 0.7
            
        # Clamp within bounds
        temp = min(max(temp, min_temp), max_temp)
        temps.append(temp)
        
    print(f"Created synthetic temperature curve with {length} points")
    return temps

def plot_temperature_curve(temp_data, save_path, title):
    """Plots a single temperature curve."""
    if not temp_data:
        print(f"No data provided for plotting: {title}")
        return
    plt.figure(figsize=(10, 6))
    # Assume each record corresponds to a step/iteration, not necessarily epoch
    iterations = range(1, len(temp_data) + 1)
    plt.plot(iterations, temp_data, 'b-', label='Temperature')
    plt.xlabel('Training Step/Iteration') # More accurate label
    plt.ylabel('Temperature')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved single curve plot to: {save_path}")

def extract_task_id_from_filename(filename):
    """Extracts task ID and distiller type from filename."""
    basename = os.path.basename(filename)
    match_grl = re.search(r'temperature_log_GRL_(.+?)\.json', basename)
    if match_grl:
        return match_grl.group(1), "GRLCTDKD"
    match_ctdkd = re.search(r'temperature_log_(.+?)\.json', basename)
    if match_ctdkd:
        # Avoid matching the GRL pattern again if filename is ambiguous
        if f"GRL_{match_ctdkd.group(1)}" != os.path.splitext(basename)[0].replace("temperature_log_", ""):
             return match_ctdkd.group(1), "CTDKD"
    # Fallback if no pattern matches or it's a worklog
    return os.path.splitext(basename)[0], "unknown"

def extract_task_id_from_path(path):
    """Extracts task ID from a directory path."""
    return os.path.basename(os.path.normpath(path))

def main():
    parser = argparse.ArgumentParser(description="Plot temperature curves for distillation methods.")
    parser.add_argument('--dkd_config', type=str, required=True, help='Path to DKD config file (for fixed temperature baseline)')
    parser.add_argument('--ctdkd_log', type=str, help='Path to specific CTDKD/GRLCTDKD temperature_log*.json file')
    parser.add_argument('--ctdkd_dir', type=str, help='Directory containing CTDKD/GRLCTDKD temperature log files (will use first found)')
    parser.add_argument('--ctdkd_worklog', type=str, help='Path to CTDKD/GRLCTDKD worklog.txt (used as fallback or if no JSON log)')
    parser.add_argument('--save_dir', type=str, default='./output/temperature_plots', help='Directory to save plots')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic curve if no CTDKD/GRLCTDKD data found')
    parser.add_argument('--num_steps', type=int, default=50000, help='Approximate number of training steps/iterations (used for DKD array length and synthetic data)')
    parser.add_argument('--auto_detect_dir', type=str, default='./output', help='Directory to auto-detect log files in')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load CTDKD / GRLCTDKD Data --- 
    ctdkd_temps = []
    ctdkd_log_file = None
    task_id = "unknown_task"
    ctdkd_type = "CTDKD" # Default

    # Priority: specific log file > directory search > auto-detect > worklog > synthetic
    if args.ctdkd_log and os.path.exists(args.ctdkd_log):
        print(f"Using specific log file: {args.ctdkd_log}")
        ctdkd_temps = load_log(args.ctdkd_log)
        ctdkd_log_file = args.ctdkd_log
        task_id, ctdkd_type = extract_task_id_from_filename(args.ctdkd_log)
    elif args.ctdkd_dir and os.path.exists(args.ctdkd_dir):
        print(f"Searching for log files in directory: {args.ctdkd_dir}")
        log_files = find_temperature_logs(args.ctdkd_dir)
        if log_files:
            ctdkd_log_file = log_files[0] # Use the first found log file
            print(f"Using detected log file: {ctdkd_log_file}")
            ctdkd_temps = load_log(ctdkd_log_file)
            task_id, ctdkd_type = extract_task_id_from_filename(ctdkd_log_file)
        else:
            task_id = extract_task_id_from_path(args.ctdkd_dir)
            print(f"No JSON logs found in {args.ctdkd_dir}. Task ID set to {task_id}.")
    elif args.auto_detect_dir and os.path.exists(args.auto_detect_dir):
        print(f"Auto-detecting log files in: {args.auto_detect_dir}")
        log_files = find_temperature_logs(args.auto_detect_dir)
        if log_files:
            ctdkd_log_file = log_files[0]
            print(f"Using auto-detected log file: {ctdkd_log_file}")
            ctdkd_temps = load_log(ctdkd_log_file)
            task_id, ctdkd_type = extract_task_id_from_filename(ctdkd_log_file)
        else:
             print(f"Auto-detect found no JSON logs in {args.auto_detect_dir}.")
             
    # Try worklog if still no data
    if not ctdkd_temps and args.ctdkd_worklog and os.path.exists(args.ctdkd_worklog):
        print(f"No JSON log data found. Attempting fallback to worklog: {args.ctdkd_worklog}")
        ctdkd_temps = extract_temps_from_worklog(args.ctdkd_worklog)
        if ctdkd_temps: # Only update task ID if we got data from worklog
            worklog_dir = os.path.dirname(args.ctdkd_worklog)
            task_id = extract_task_id_from_path(worklog_dir)
            ctdkd_type = "CTDKD_from_worklog"
            
    # Use synthetic data if requested and still no data
    synthetic_used = False
    if not ctdkd_temps and args.synthetic:
        print("No real temperature data found. Creating synthetic data.")
        ctdkd_temps = create_synthetic_temperature_curve(length=args.num_steps)
        ctdkd_type = "Synthetic"
        synthetic_used = True
        # Use a generic task ID for synthetic data if none was determined
        if task_id == "unknown_task": task_id = "synthetic_run"

    # Determine length for DKD curve (use steps)
    dkd_length = len(ctdkd_temps) if ctdkd_temps else args.num_steps
    dkd_temps = get_constant_temperature(args.dkd_config, length=dkd_length)
    
    # Determine label for CTDKD/GRL plot
    ctdkd_label = "GRL-CTDKD" if ctdkd_type == "GRLCTDKD" else "CTDKD"
    if synthetic_used: ctdkd_label = "CTDKD (Synthetic)"
    if ctdkd_type == "CTDKD_from_worklog": ctdkd_label = "CTDKD (from Worklog)"

    # --- Plotting --- 
    # Plot individual curves
    plot_temperature_curve(
        ctdkd_temps,
        os.path.join(args.save_dir, f'temperature_curve_{ctdkd_type}_{task_id}.png'),
        f'Temperature Curve ({ctdkd_label} - {task_id})'
    )
    plot_temperature_curve(
        dkd_temps,
        os.path.join(args.save_dir, f'temperature_curve_dkd_{task_id}.png'),
        f'Temperature Curve (DKD Baseline - {task_id})'
    )
    
    # Plot comparison
    if dkd_temps or ctdkd_temps:
        plt.figure(figsize=(12, 8))
        max_length = max(len(dkd_temps), len(ctdkd_temps))
        if max_length == 0: return # Should not happen if DKD has data
        
        iterations = range(1, max_length + 1)
        
        # Plot DKD (fixed temperature)
        if dkd_temps:
            plt.plot(iterations[:len(dkd_temps)], dkd_temps, 'r--', linewidth=2, label='DKD (Fixed Temp)')
            plt.text(max_length * 0.8, dkd_temps[0] * 1.05, f'T={dkd_temps[0]}', color='r')
        
        # Plot CTDKD/GRLCTDKD (dynamic temperature)
        if ctdkd_temps:
            plt.plot(iterations[:len(ctdkd_temps)], ctdkd_temps, 'b-', linewidth=2, label=ctdkd_label)
        else:
            # Indicate missing data on plot
            plt.text(max_length * 0.1, dkd_temps[0]*0.8, f'{ctdkd_label}: No data available', color='b')
            
        plt.xlabel('Training Step/Iteration', fontsize=14)
        plt.ylabel('Temperature', fontsize=14)
        title_suffix = f" ({task_id})"
        title = f'Temperature Comparison: DKD vs {ctdkd_label}{title_suffix}'
        if synthetic_used:
            title += ' - Using synthetic data for CTDKD'
        plt.title(title, fontsize=16)
        
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        comparison_filename = f'temperature_comparison_{ctdkd_type}_{task_id}.png'
        save_path = os.path.join(args.save_dir, comparison_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Temperature comparison chart saved to {save_path}")
    else:
        print("Could not create comparison chart - no temperature data available for either method.")

if __name__ == '__main__':
    main() 