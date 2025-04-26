#!/usr/bin/env python3

import os
import sys
import time
import argparse
import subprocess
import torch

# Add the project path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from tools.train import main as train_main

def parse_args():
    parser = argparse.ArgumentParser(description="Run optimized knowledge distillation training")
    parser.add_argument("--method", type=str, choices=["GRLCTDKD", "CTDKD", "all"], default="all",
                      help="Which method to run (GRLCTDKD, CTDKD, or all)")
    parser.add_argument("--model", type=str, choices=["res110_res20", "res110_res32", "wrn40_2_wrn16_2"], 
                      default="res110_res20", help="Model configuration to use")
    parser.add_argument("--dataset", type=str, choices=["cifar100"], default="cifar100",
                      help="Dataset to use")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="optimized_output",
                      help="Directory to save outputs")
    parser.add_argument("--analyze", action="store_true", help="Run analysis after training")
    parser.add_argument("--compare", action="store_true", help="Compare with DKD after training")
    return parser.parse_args()

def prepare_config(method, model, dataset, args):
    """Prepare configuration file for training"""
    cfg_file = f"configs/{dataset}/{method.lower()}/{model}.yaml"
    cfg_instance = cfg.clone()
    cfg_instance.merge_from_file(cfg_file)
    
    # Override settings
    cfg_instance.SOLVER.EPOCHS = args.epochs
    cfg_instance.EXPERIMENT.NAME = f"{dataset}_{model}_{method}_optimized"
    
    # Set output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{method.lower()},{model}")
    cfg_instance.LOG.PREFIX = output_path
    os.makedirs(output_path, exist_ok=True)
    
    return cfg_instance

def run_training(method, model, dataset, args):
    """Run training for a specific method and model"""
    print(f"=" * 80)
    print(f"Running optimized training for {method} with {model} on {dataset}")
    print(f"=" * 80)
    
    # Set GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Prepare configuration
    cfg_instance = prepare_config(method, model, dataset, args)
    
    # Run training
    start_time = time.time()
    train_main(cfg_instance)
    end_time = time.time()
    
    # Print training time
    training_time = (end_time - start_time) / 3600  # in hours
    print(f"Training completed in {training_time:.2f} hours")
    
    return os.path.join(args.output_dir, f"{method.lower()},{model}")

def run_analysis(output_dir, method, model):
    """Run analysis on training results"""
    print(f"Running analysis for {method}...")
    
    # Create analysis output directory
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Command to run analysis
    cmd = [
        "python", "tools/monitor_training.py",
        "--log_dir", output_dir,
        "--method", method,
        "--output_dir", analysis_dir
    ]
    
    # Run analysis
    subprocess.run(cmd)
    
    return analysis_dir

def compare_results(methods, model, output_dir, dataset):
    """Compare results between different methods"""
    print(f"Comparing results for {methods}...")
    
    # Create comparison output directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Command to run comparison
    methods_str = " ".join(methods)
    cmd = [
        "python", "tools/analyze_results.py",
        "--logs_dir", output_dir,
        "--methods", *methods,
        "--model_name", f"{model}_comparison",
        "--model_pattern", model,
        "--output_dir", comparison_dir
    ]
    
    # Run comparison
    subprocess.run(cmd)
    
    return comparison_dir

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Determine which methods to run
    methods = []
    if args.method == "all":
        methods = ["CTDKD", "GRLCTDKD"]
    else:
        methods = [args.method]
    
    # Add DKD for comparison
    if args.compare:
        methods.append("DKD")
    
    # Run training for each method
    output_dirs = {}
    for method in methods:
        output_dir = run_training(method, args.model, args.dataset, args)
        output_dirs[method] = output_dir
        
        # Run analysis if requested
        if args.analyze:
            analysis_dir = run_analysis(output_dir, method, args.model)
            print(f"Analysis results saved to {analysis_dir}")
    
    # Compare results if requested and multiple methods were run
    if args.compare and len(methods) > 1:
        comparison_dir = compare_results(methods, args.model, args.output_dir, args.dataset)
        print(f"Comparison results saved to {comparison_dir}")
    
    # Print final message
    print(f"=" * 80)
    print(f"Optimized training complete!")
    print(f"Results saved to {args.output_dir}")
    print(f"=" * 80)

if __name__ == "__main__":
    main() 