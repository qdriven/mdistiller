 
# 运行原始DKD: 基本比较：DKD vs CTDKD
python tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml

# 运行CTDKD
python tools/train.py --cfg configs/cifar100/ctdkd/res32x4_res8x4.yaml
## 固定温度的消融实验
# 运行温度T=4的DKD
python tools/train.py --cfg configs/cifar100/dkd_ablation/fixed_temp_4.yaml

# 运行温度T=8的DKD
python tools/train.py --cfg configs/cifar100/dkd_ablation/fixed_temp_8.yaml

# 运行初始温度T=2的CTDKD
python tools/train.py --cfg configs/cifar100/ctdkd_ablation/init_temp_2.yaml

 ## get loss picture
 python tools/compare_results.py \
     --methods "Vanilla KD" "CTKD" \
     --logs output/cifar100_baselines/dkd,res32x4,res8x4/worklog.txt output/cifar100_baselines/ctdkd,res32x4,res8x4/worklog.txt \
     --metric loss \
     --output_dir ./output/baseline_comparison
# 如果要生成温度变化曲线:
   python tools/plot_temperature.py --ctdkd_log output/cifar100_baselines/ctdkd,res32x4,res8x4/temperature_log.json

## 生成t-SNE可视化
# DKD的t-SNE
python tools/visualize_tsne.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml --ckpt output/cifar100_baselines/dkd,res32x4,res8x4/latest.pth

# CTDKD的t-SNE
python tools/visualize_tsne.py --cfg configs/cifar100/ctdkd/res32x4_res8x4.yaml --ckpt output/cifar100_baselines/ctdkd,res32x4,res8x4/latest

## 绘制温度学习曲线
python tools/plot_temperature.py --dkd_log output/dkd_res32x4_res8x4/log.txt --ctdkd_log output/ctdkd_res32x4_res8x4/log.txt

## 结果比较
python tools/compare_results.py \
  --methods "DKD" "CTDKD" "DKD-T4" "DKD-T8" "CTDKD-initT2" \
  --logs output/dkd_res32x4_res8x4/log.txt output/ctdkd_res32x4_res8x4/log.txt output/ablation/dkd_fixed_t4_res32x4_res8x4/log.txt output/ablation/dkd_fixed_t8_res32x4_res8x4/log.txt output/ablation/ctdkd_init_t2_res32x4_res8x4/log.txt \
  --configs configs/cifar100/dkd/res32x4_res8x4.yaml configs/cifar100/ctdkd/res32x4_res8x4.yaml configs/cifar100/dkd_ablation/fixed_temp_4.yaml configs/cifar100/dkd_ablation/fixed_temp_8.yaml configs/cifar100/ctdkd_ablation/init_temp_2.yaml

# 方式1：如果有worklog.txt，使用它作为备选数据源
python tools/plot_temperature.py --ctdkd_worklog output/ctdkd_res32x4_res8x4/worklog.txt --dkd_config configs/cifar100/dkd/res32x4_res8x4.yaml --save_dir ./output/comparison

# 方式2：使用合成数据
python tools/plot_temperature.py --synthetic --num_epochs 240 --dkd_config configs/cifar100/dkd/res32x4_res8x4.yaml --save_dir ./output/comparison



## 训练脚本

python tools/train.py --cfg configs/cifar100/grlctdkd/res110_res20.yaml