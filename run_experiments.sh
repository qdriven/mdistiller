#!/bin/bash

# Set PYTHONPATH to include mdistiller
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create directories for logs and checkpoints
mkdir -p logs
mkdir -p save

# Run DKD experiments
echo "Running DKD experiments..."
python tools/train.py \
    --cfg configs/cifar100/dkd/res110_res20.yaml \
    LOG.PREFIX logs/dkd_res110_res20

python tools/train.py \
    --cfg configs/cifar100/dkd/vgg13_vgg8.yaml \
    LOG.PREFIX logs/dkd_vgg13_vgg8

# Run CTDKD experiments
echo "Running CTDKD experiments..."
python tools/train.py \
    --cfg configs/cifar100/ctdkd/res110_res20.yaml \
    LOG.PREFIX logs/ctdkd_res110_res20

python tools/train.py \
    --cfg configs/cifar100/ctdkd/vgg13_vgg8.yaml \
    LOG.PREFIX logs/ctdkd_vgg13_vgg8

# Run GRLCTDKD experiments
echo "Running GRLCTDKD experiments..."
python tools/train.py \
    --cfg configs/cifar100/grlctdkd/res110_res32.yaml \
    LOG.PREFIX logs/grlctdkd_res110_res32

python tools/train.py \
    --cfg configs/cifar100/grlctdkd/res56_res20.yaml \
    LOG.PREFIX logs/grlctdkd_res56_res20

python tools/train.py \
    --cfg configs/cifar100/grlctdkd/wrn40_2_wrn16_2.yaml \
    LOG.PREFIX logs/grlctdkd_wrn40_2_wrn16_2

# After all experiments are complete, run the comparison script
echo "Generating comparison results..."
python tools/compare_results.py \
    --methods DKD CTDKD GRLCTDKD \
    --log_files logs/dkd_res110_res20/worklog.txt logs/ctdkd_res110_res20/worklog.txt logs/grlctdkd_res110_res32/worklog.txt \
    --output ablation_results.md 


## 结果比较

python tools/compare_results.py \
  --methods DKD CTKD GRLCTDKD \
  --logs "output/cifar100_baselines/dkd,res110,res20/worklog.txt" "output/cifar100_baselines/ctdkd,res110,res20/worklog.txt" "output/cifar100_baselines/grlctdkd,res110,res32/worklog.txt" \
  --metric accuracy \
  --output_dir ./comparison_results_3

python tools/analyze_results.py \
  --logs_dir output/cifar100_baselines \
  --methods DKD CTKD GRLCTDKD \
  --model_name res110_res20_comparison \
  --output_dir result_analysis


  python tools/plot_temperature.py --dkd_config path/to/dkd/config.yaml --ctdkd_dir path/to/ctdkd/logs



  python tools/analyze_results.py \
  --logs_dir output/cifar100_baselines \
  --methods DKD CTDKD GRLCTDKD \
  --model_name res110_res20_comparison \
  --model_pattern res110,res20 \
  --output_dir result_analysis