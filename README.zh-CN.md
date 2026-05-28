# PetData

[English](README.md) | [中文](README.zh-CN.md)

这是一个 Python + PyTorch 项目，用于在本地宠物图片分类数据集上微调 EfficientNet-B0。

期望的数据目录结构：

```text
PetData/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

## 环境准备

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

项目主要使用：

- `torch`
- `torchvision`
- `torchaudio`

## 微调 EfficientNet-B0

使用 ImageNet 预训练权重运行训练：

```powershell
python -m petdata --epochs 10 --batch-size 32 --learning-rate 3e-4 --eval-test
```

先只训练分类头：

```powershell
python -m petdata --epochs 5 --freeze-backbone
```

在不下载预训练权重的情况下，检查数据加载器和模型是否可以正常构建：

```powershell
python -m petdata --weights none --dry-run
```

输出会写入 `runs/efficientnet_b0/`：

- `best.pt`：验证集表现最好的 checkpoint
- `last.pt`：最新 checkpoint
- `labels.json`：数字标签到类别名称的映射

## 预测单张图片

微调完成后，可以用一张图片测试模型：

```powershell
python -m petdata --predict-image .\test\Abyssinian\Abyssinian_10.jpg
```

默认会加载 `runs/efficientnet_b0/best.pt`，并打印预测出的宠物类别和排名靠前的候选类别。也可以指定其他 checkpoint，或减少打印的候选数量：

```powershell
python -m petdata --predict-image .\my_pet.jpg --checkpoint .\runs\efficientnet_b0\last.pt --top-k 3
```

## 常用选项

```powershell
python -m petdata --help
```

常用参数：

- `--data-dir`：数据集根目录，默认为当前项目目录
- `--epochs`：训练轮数
- `--batch-size`：批大小
- `--num-workers`：DataLoader worker 数量
- `--weights imagenet|none`：使用 ImageNet 预训练权重或随机初始化
- `--freeze-backbone`：只训练最后的分类层
- `--eval-test`：训练结束后，用最佳 checkpoint 在 `test/` 上评估
- `--predict-image`：加载 checkpoint 并预测单张图片的宠物类别
- `--checkpoint`：供 `--predict-image` 使用的 checkpoint 路径
- `--top-k`：打印的预测候选数量
