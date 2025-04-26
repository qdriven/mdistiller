#!/bin/bash

# Test DKD
python tools/test_res110_res20.py \
    --method dkd \
    --cfg configs/cifar100/dkd/res110_res20.yaml

# Test CTDKD with modified temperature
python tools/test_res110_res20.py \
    --method ctdkd \
    --cfg configs/cifar100/ctdkd/res110_res20.yaml

# Test GRLCTDKD with modified temperature
python tools/test_res110_res20.py \
    --method grlctdkd \
    --cfg configs/cifar100/grlctdkd/res110_res20.yaml