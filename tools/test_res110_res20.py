import os
import torch
import argparse
from mdistiller.distillers import DKD, CTDKD, GRLCTDKD
from mdistiller.models import resnet110, resnet20
from mdistiller.engine.utils import load_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['dkd', 'ctdkd', 'grlctdkd'],
                       required=True, help='distillation method')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.cfg)
    
    # Setup models
    teacher = resnet110()
    student = resnet20()
    
    # Setup distiller based on method
    if args.method == 'dkd':
        distiller = DKD(student=student, teacher=teacher, cfg=cfg)
    elif args.method == 'ctdkd':
        # Modified temperature settings for CTDKD
        cfg.CTDKD.INIT_TEMPERATURE = 2.0
        cfg.CTDKD.MAX_TEMPERATURE = 8.0
        cfg.CTDKD.MIN_TEMPERATURE = 1.0
        cfg.CTDKD.LEARNING_RATE = 0.0001
        distiller = CTDKD(student=student, teacher=teacher, cfg=cfg)
    else:  # grlctdkd
        # Modified temperature settings for GRLCTDKD
        cfg.GRLCTDKD.INIT_TEMPERATURE = 2.0
        cfg.GRLCTDKD.MAX_TEMPERATURE = 8.0
        cfg.GRLCTDKD.MIN_TEMPERATURE = 1.0
        cfg.GRLCTDKD.LEARNING_RATE = 0.0001
        cfg.GRLCTDKD.GRL_LAMBDA = 0.1
        distiller = GRLCTDKD(student=student, teacher=teacher, cfg=cfg)
    
    # Print model configuration
    print(f"\nTesting {args.method.upper()} with ResNet110 → ResNet20")
    print(f"Temperature settings:")
    if args.method != 'dkd':
        print(f"- Initial temperature: {cfg[args.method.upper()].INIT_TEMPERATURE}")
        print(f"- Max temperature: {cfg[args.method.upper()].MAX_TEMPERATURE}")
        print(f"- Min temperature: {cfg[args.method.upper()].MIN_TEMPERATURE}")
        print(f"- Learning rate: {cfg[args.method.upper()].LEARNING_RATE}")
    
    # Start training
    try:
        from mdistiller.engine.trainer import train_distiller
        train_distiller(distiller, cfg)
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    main()