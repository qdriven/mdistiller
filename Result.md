#  实验结果

- 基准测试：

```sh
python tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml
```

- CTKD+DKD实验：

``sh
python tools/train.py --cfg configs/cifar100/ctdkd/res32x4_res8x4.yaml
```


## 需要做的

1. 准备训练日志
确保您已经完成了所有三种方法的训练，并且有相应的日志文件（通常在 save/ 或 output/ 目录下）。每个方法的日志文件应该包含训练和测试精度、损失等信息。
2. 使用 compare_results.py 生成比较

```sh
python tools/compare_results.py \
  --methods DKD CTKD GRLCTDKD \
  --log-files path/to/dkd/worklog.txt path/to/ctkd/worklog.txt path/to/grlctdkd/worklog.txt \
  --plot --metric accuracy --save-path comparison.png \
  --print-final
```


```
python tools/compare_results.py \
  --methods DKD CTKD GRLCTDKD \
  --logs "output/cifar100_baselines/dkd,res110,res20/worklog.txt" "output/cifar100_baselines/ctdkd,res110,res20/worklog.txt" "output/cifar100_baselines/grlctdkd,res110,res20/worklog.txt" \
  --plot --metric accuracy --save-path comparison.png \
  --print-final

  ```
