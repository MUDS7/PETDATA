# PetData

[English](README.md) | [中文](README.zh-CN.md)

Python + PyTorch project for fine-tuning EfficientNet-B0 on the local pet image
classification dataset.

Expected folder layout:

```text
PetData/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

The project uses:

- `torch`
- `torchvision`
- `torchaudio`

## Fine-Tune EfficientNet-B0

Run with ImageNet pretrained weights:

```powershell
python -m petdata --epochs 10 --batch-size 32 --learning-rate 3e-4 --eval-test
```

Train only the classifier head first:

```powershell
python -m petdata --epochs 5 --freeze-backbone
```

Check that the data loaders and model can be built without downloading pretrained
weights:

```powershell
python -m petdata --weights none --dry-run
```

Outputs are written to `runs/efficientnet_b0/`:

- `best.pt`: best validation checkpoint
- `last.pt`: latest checkpoint
- `labels.json`: numeric label to class-name mapping

## Predict One Image

After fine-tuning, test the model with one image:

```powershell
python -m petdata --predict-image .\test\Abyssinian\Abyssinian_10.jpg
```

By default this loads `runs/efficientnet_b0/best.pt` and prints the predicted
pet class with the top candidates. You can choose another checkpoint or reduce
the number of printed candidates:

```powershell
python -m petdata --predict-image .\my_pet.jpg --checkpoint .\runs\efficientnet_b0\last.pt --top-k 3
```

## Useful Options

```powershell
python -m petdata --help
```

Common options:

- `--data-dir`: dataset root, defaults to the current project folder
- `--epochs`: number of training epochs
- `--batch-size`: batch size
- `--num-workers`: DataLoader worker count
- `--weights imagenet|none`: pretrained ImageNet weights or random init
- `--freeze-backbone`: train only the final classifier layer
- `--eval-test`: evaluate the best checkpoint on `test/` after training
- `--predict-image`: load a checkpoint and predict the pet class for one image
- `--checkpoint`: checkpoint path for `--predict-image`
- `--top-k`: number of prediction candidates to print
