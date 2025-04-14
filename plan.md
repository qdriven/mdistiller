# CDKD实现环境部署指南

在实现CDKD（Curriculum Decoupled Knowledge Distillation）代码之前，我们需要先搭建合适的开发环境。基于您之前提到的两个GitHub仓库，我建议以DKD（Decoupled Knowledge Distillation）仓库为基础进行开发，因为它提供了更完整的知识蒸馏框架，然后将CTKD的自适应温度机制整合进去。

## 1. 环境准备

### 基础仓库选择

我们将使用DKD的官方实现仓库作为基础：
- 仓库地址：https://github.com/megvii-research/mdistiller

这个仓库提供了多种知识蒸馏方法的实现，包括DKD，并且有良好的框架结构，便于我们扩展新的方法。

### 系统要求

- Python 3.6+
- PyTorch 1.8+
- CUDA 10.1+ (如需GPU加速)

## 2. 环境搭建步骤

### 克隆仓库

```bash
git clone https://github.com/megvii-research/mdistiller.git
cd mdistiller
```

### 创建虚拟环境

使用Conda创建虚拟环境（推荐）：

```bash
# 创建名为cdkd的环境
conda create -n cdkd python=3.8
conda activate cdkd

# 安装PyTorch (根据您的CUDA版本选择合适的命令)
# 对于CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 对于CPU版本
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 安装依赖

```bash
# 安装mdistiller的依赖
pip install -r requirements.txt

# 安装额外依赖
pip install scikit-learn matplotlib tensorboard
```

## 3. 项目结构理解

在开始开发前，让我们了解mdistiller的项目结构：

```
mdistiller/
├── configs/                  # 配置文件
│   ├── cifar100/             # CIFAR-100数据集配置
│   └── cifar10/              # CIFAR-10数据集配置
├── dataset/                  # 数据集加载
├── distiller/                # 蒸馏方法实现
│   ├── base.py               # 基础蒸馏器类
│   ├── dkd.py                # DKD实现
│   └── ...
├── engine/                   # 训练引擎
├── models/                   # 模型定义
└── tools/                    # 训练和评估脚本
```

## 4. 整合CTKD的关键步骤

我们需要参考CTKD的实现，将其自适应温度机制整合到DKD中：

1. **查看CTKD仓库**：
   - 仓库地址：https://github.com/winycg/CTKD
   - 了解其温度模块和GRL层的实现

2. **创建新的蒸馏方法文件**：
   ```bash
   # 在mdistiller的distiller目录下创建cdkd.py
   touch distiller/cdkd.py
   ```

3. **创建配置文件**：
   ```bash
   # 在configs目录下创建CDKD的配置文件
   mkdir -p configs/cifar100/cdkd
   touch configs/cifar100/cdkd/resnet110_resnet20.yaml
   mkdir -p configs/cifar100/cdkd/vgg18_vgg8.yaml
   
   mkdir -p configs/cifar10/cdkd
   touch configs/cifar10/cdkd/resnet110_resnet20.yaml
   ```

## 5. 配置文件示例

以下是CDKD的配置文件示例（放在`configs/cifar100/cdkd/resnet110_resnet20.yaml`）：

```yaml
EXPERIMENT:
  NAME: ""
  TAG: "cdkd_r110_r20"
  PROJECT: "cifar100_baselines"

DATASET:
  TYPE: "cifar100"
  NUM_WORKERS: 8
  TEST_BATCH_SIZE: 64

TRAIN:
  BATCH_SIZE: 64
  EPOCHS: 240
  OPTIMIZER:
    TYPE: "sgd"
    LR: 0.05
    WEIGHT_DECAY: 5e-4
    MOMENTUM: 0.9
  LR_SCHEDULER:
    TYPE: "cosine"
    MIN_LR: 0.0
  WARMUP_EPOCHS: 0

MODEL:
  STUDENT: "resnet20"
  TEACHER: "resnet110"
  TEACHER_CHECKPOINT: "checkpoints/teachers/cifar100_resnet110_teacher.pth"

DISTILLER:
  TYPE: "CDKD"
  CDKD:
    CE_WEIGHT: 1.0
    ALPHA: 1.0
    BETA: 8.0
    WARMUP: 30
    TEMP_TYPE: "instance"  # "global" 或 "instance"
```

## 6. 开发流程

完成环境搭建后，我们可以按照以下流程开发CDKD方法：

1. **实现GRL层**：
   - 在`distiller/cdkd.py`中实现梯度反转层

2. **实现温度模块**：
   - 在`distiller/cdkd.py`中实现全局温度和实例温度模块

3. **实现CDKD蒸馏器**：
   - 在`distiller/cdkd.py`中实现CDKD类，继承自基础蒸馏器

4. **注册新方法**：
   - 在`distiller/__init__.py`中注册CDKD方法

5. **运行实验**：
   ```bash
   # 在CIFAR-10上运行小规模实验
   python tools/train.py --cfg configs/cifar10/cdkd/resnet110_resnet20.yaml
   
   # 在CIFAR-100上运行完整实验
   python tools/train.py --cfg configs/cifar100/cdkd/resnet110_resnet20.yaml
   ```

6. **可视化和分析**：
   - 使用TensorBoard或自定义脚本分析结果
   ```bash
   tensorboard --logdir runs/
   ```

## 7. 关键文件修改

### 在`distiller/__init__.py`中注册CDKD

```python
from .dkd import DKD
from .cdkd import CDKD  # 导入新实现的CDKD

distiller_dict = {
    "DKD": DKD,
    "CDKD": CDKD,  # 注册CDKD
    # ...其他蒸馏方法
}
```

### 创建实验脚本

您可以创建一个专门的脚本来运行消融实验：

```bash
touch tools/run_ablation.py
```

## 8. 数据和结果管理

- **模型检查点**：保存在`checkpoints/`目录
- **实验日志**：保存在`runs/`目录（使用TensorBoard）
- **可视化结果**：保存在`visualizations/`目录（需自行创建）

```bash
mkdir -p visualizations/tsne
mkdir -p visualizations/temp_curves
mkdir -p visualizations/loss_curves
```

通过以上步骤，您将拥有一个完整的开发环境，可以在DKD的基础上实现CDKD方法，并进行各种实验和分析。接下来，我们可以开始实现具体的代码。


根据您的要求，我来帮您规划运行DKD和CTKD+DKD的对比实验。我们可以使用res32x4_res8x4这对师生网络来做实验，步骤如下：

1. 首先运行原始DKD:
```bash
python tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml
```

2. 对于CTKD+DKD的方案，我们需要:
- 修改配置文件以添加CTKD
- 在DKD基础上集成CTKD的温度调节机制

让我先检查一下现有的配置文件结构：

Read file: mdistiller/configs/cifar100/dkd/res32x4_res8x4.yaml
基于您的需求，我建议以下实验计划：

1. DKD基准实验：
```bash
# 运行原始DKD
python tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml
```

2. CTKD+DKD实验：
需要创建新的配置文件 `configs/cifar100/ctdkd/res32x4_res8x4.yaml`，我们需要：

```yaml:mdistiller/configs/cifar100/ctdkd/res32x4_res8x4.yaml
EXPERIMENT:
  NAME: ""
  TAG: "ctdkd,res32x4,res8x4"
  PROJECT: "cifar100_baselines"

DATASET:
  TYPE: "cifar100"
  NUM_WORKERS: 8
  TEST_BATCH_SIZE: 64

DISTILLER:
  TYPE: "CTDKD"  # 修改为CTDKD
  TEACHER: "resnet32x4"
  STUDENT: "resnet8x4"

SOLVER:
  BATCH_SIZE: 64
  EPOCHS: 240
  LR: 0.05
  LR_DECAY_STAGES: [150, 180, 210]
  LR_DECAY_RATE: 0.1
  WEIGHT_DECAY: 0.0005
  MOMENTUM: 0.9
  TYPE: "SGD"
  TRAINER: "base"

LOG:
  PRINT_FREQ: 100
  SAVE_CHECKPOINT_FREQ: 40
  PREFIX: "./output"
  WANDB: false

DKD:
  CE_WEIGHT: 1.0
  ALPHA: 1.0
  BETA: 8.0
  T: 4.0
  WARMUP: 20

CTDKD:
  INIT_TEMPERATURE: 4.0
  MIN_TEMPERATURE: 1.0
  MAX_TEMPERATURE: 10.0
  LEARNING_RATE: 0.001  # 温度学习率
```
实验计划总结：

1. 运行基准DKD实验：
```bash
python tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml
```

2. 运行CTKD+DKD实验：
```bash
python tools/train.py --cfg configs/cifar100/ctdkd/res32x4_res8x4.yaml
```

3. 可视化分析：
- 对比两个模型的t-SNE图
- 绘制温度学习曲线对比图
- 分析性能指标（准确率、损失等）

需要注意的点：
1. 需要实现CTDKD的distiller类来处理温度学习
2. 需要修改训练循环以记录温度变化
3. 需要添加t-SNE可视化代码

您想先从哪一步开始？我建议先：
1. 实现CTDKD的distiller
2. 或者先运行基准DKD实验
请告诉我您想先进行哪个步骤？
