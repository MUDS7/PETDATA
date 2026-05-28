from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNet-B0 on the PetData image folders."
    )
    parser.add_argument("--data-dir", type=Path, default=Path.cwd())
    parser.add_argument("--train-dir", type=str, default="train")
    parser.add_argument("--val-dir", type=str, default="val")
    parser.add_argument("--test-dir", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "efficientnet_b0")
    parser.add_argument(
        "--predict-image",
        type=Path,
        help="Run inference on one image and print the predicted pet class.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to use for --predict-image. Defaults to <output-dir>/best.pt.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of prediction candidates to print for --predict-image.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet pretrained weights or train from scratch.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the classifier head.",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate the best checkpoint on the test split after training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets and model, then exit before training.",
    )
    return parser.parse_args()


def load_torch_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_datasets(
    data_dir: Path,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    image_size: int,
) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder | None]:
    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(data_dir / train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / val_dir, transform=eval_transform)

    test_path = data_dir / test_dir
    test_dataset = (
        datasets.ImageFolder(test_path, transform=eval_transform)
        if test_path.exists()
        else None
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("train and val class folders do not match.")
    if test_dataset is not None and train_dataset.classes != test_dataset.classes:
        raise ValueError("train and test class folders do not match.")

    return train_dataset, val_dataset, test_dataset


def build_loaders(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder | None,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = (
        DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        if test_dataset is not None
        else None
    )
    return train_loader, val_loader, test_loader


def build_model(
    num_classes: int,
    weights_name: str,
    dropout: float,
    freeze_backbone: bool,
) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if weights_name == "imagenet" else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Metrics:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return Metrics(loss=running_loss / total, accuracy=correct / total)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Metrics:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return Metrics(loss=running_loss / total, accuracy=correct / total)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    class_to_idx: dict[str, int],
    metrics: Metrics,
    args: argparse.Namespace,
) -> None:
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
            "metrics": asdict(metrics),
            "args": args_dict,
        },
        path,
    )


def save_label_map(path: Path, class_to_idx: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    with path.open("w", encoding="utf-8") as file:
        json.dump(idx_to_class, file, indent=2, ensure_ascii=False)


def load_idx_to_class(
    checkpoint: dict,
    labels_path: Path,
) -> dict[int, str]:
    class_to_idx = checkpoint.get("class_to_idx")
    if class_to_idx:
        return {int(idx): class_name for class_name, idx in class_to_idx.items()}

    if not labels_path.exists():
        raise FileNotFoundError(
            "Could not find class labels in the checkpoint or at "
            f"{labels_path}."
        )

    with labels_path.open("r", encoding="utf-8") as file:
        labels = json.load(file)
    return {int(idx): class_name for idx, class_name in labels.items()}


def get_checkpoint_arg(checkpoint: dict, name: str, default: int | float) -> int | float:
    value = checkpoint.get("args", {}).get(name, default)
    return type(default)(value)


@torch.inference_mode()
def predict_image(args: argparse.Namespace) -> None:
    image_path = args.predict_image.resolve()
    checkpoint_path = (
        args.checkpoint if args.checkpoint is not None else args.output_dir / "best.pt"
    ).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    idx_to_class = load_idx_to_class(checkpoint, checkpoint_path.parent / "labels.json")
    image_size = int(get_checkpoint_arg(checkpoint, "image_size", args.image_size))
    dropout = float(get_checkpoint_arg(checkpoint, "dropout", args.dropout))

    model = build_model(
        num_classes=len(idx_to_class),
        weights_name="none",
        dropout=dropout,
        freeze_backbone=False,
    ).to(device)
    model_state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(model_state)
    model.eval()

    _, eval_transform = build_transforms(image_size)
    with Image.open(image_path) as image:
        image_tensor = eval_transform(image.convert("RGB")).unsqueeze(0).to(device)

    probabilities = torch.softmax(model(image_tensor), dim=1).squeeze(0)
    top_k = min(max(args.top_k, 1), len(idx_to_class))
    top_probabilities, top_indices = probabilities.topk(top_k)

    best_idx = int(top_indices[0].item())
    best_class = idx_to_class[best_idx]
    best_confidence = float(top_probabilities[0].item())

    print(f"Image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Predicted: {best_class} ({best_confidence:.2%})")
    if top_k > 1:
        print(f"Top {top_k}:")
        for rank, (probability, index) in enumerate(
            zip(top_probabilities.tolist(), top_indices.tolist()),
            start=1,
        ):
            print(f"  {rank}. {idx_to_class[int(index)]}: {probability:.2%}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.predict_image is not None:
        predict_image(args)
        return

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_dataset, val_dataset, test_dataset = build_datasets(
        data_dir,
        args.train_dir,
        args.val_dir,
        args.test_dir,
        args.image_size,
    )
    train_loader, val_loader, test_loader = build_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        args.batch_size,
        args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=len(train_dataset.classes),
        weights_name=args.weights,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Classes: {len(train_dataset.classes)}")
    print(
        "Images: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset) if test_dataset is not None else 0}"
    )
    print(f"Output: {output_dir}")

    save_label_map(output_dir / "labels.json", train_dataset.class_to_idx)

    if args.dry_run:
        print("Dry run complete. Model and dataloaders are ready.")
        return

    best_accuracy = 0.0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            train_dataset.class_to_idx,
            val_metrics,
            args,
        )
        if val_metrics.accuracy >= best_accuracy:
            best_accuracy = val_metrics.accuracy
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                train_dataset.class_to_idx,
                val_metrics,
                args,
            )

        elapsed = perf_counter() - start
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} "
            f"time={elapsed:.1f}s"
        )

    if args.eval_test and test_loader is not None:
        checkpoint = load_torch_checkpoint(best_path, device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(
            f"test_loss={test_metrics.loss:.4f} "
            f"test_acc={test_metrics.accuracy:.4f}"
        )

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
