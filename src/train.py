"""Fine-tuning SSDLite; pemilihan best model memakai validation saja."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from src.dataset import load_split, validate_dataset_splits
from src.evaluate import evaluate_model
from src.model import build_model
from src.utils import (
    capture_rng_state, collate_fn, configuration_snapshot, get_device,
    load_checkpoint, restore_rng_state, save_checkpoint, save_history,
    save_json, save_metrics_report, seed_everything, seed_worker,
)


@contextmanager
def singleton_batch_norm(model, batch_size):
    """Batch akhir berisi satu gambar: gunakan running stats BN sementara.

    SSDLite mempunyai feature map 1x1; BN training tidak dapat mengestimasi
    variansi dari satu nilai. Gambar terakhir tetap dilatih tanpa drop_last.
    Parameter affine BN tetap mendapat gradien.
    """
    changed = []
    if batch_size == 1:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
                module.eval()
                changed.append(module)
    try:
        yield
    finally:
        for module in changed:
            module.train()


def dataset_signature(datasets):
    digest = hashlib.sha256()
    for split, dataset in sorted(datasets.items()):
        digest.update(split.encode())
        digest.update(dataset.annotation_file.read_bytes())
        digest.update(json.dumps(dataset.category_mapping, sort_keys=True).encode())
        for path, content_hash in sorted(dataset.content_hashes().items()):
            digest.update(str(path.relative_to(dataset.images_dir)).encode())
            digest.update(content_hash.encode())
    return digest.hexdigest()


def train_model(train_dataset, val_dataset, num_epochs=None, checkpoint_dir=None,
                metrics_dir=None, pretrained=None, resume=None, device=None,
                batch_size=None):
    num_epochs = config.NUM_EPOCHS if num_epochs is None else num_epochs
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    pretrained = config.USE_PRETRAINED if pretrained is None else pretrained
    checkpoint_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR).resolve()
    metrics_dir = Path(metrics_dir or config.METRICS_DIR).resolve()
    device = get_device() if device is None else torch.device(device)
    if not isinstance(num_epochs, int) or num_epochs <= 0:
        raise ValueError("Jumlah epoch harus integer positif.")
    if not isinstance(batch_size, int) or batch_size < 2:
        raise ValueError("Batch size awal minimal 2 untuk BatchNorm SSDLite.")
    if config.OPTIMIZER != "SGD":
        raise ValueError("Baseline proyek ini menggunakan SGD; ubah optimizer melalui implementasi eksplisit.")
    if config.LEARNING_RATE <= 0 or config.WEIGHT_DECAY < 0:
        raise ValueError("Learning rate harus positif dan weight decay tidak boleh negatif.")
    if len(train_dataset) < 2:
        raise ValueError("Training membutuhkan minimal dua gambar.")

    datasets = {"train": train_dataset, "val": val_dataset}
    audit = validate_dataset_splits(datasets)
    if sum(train_dataset.class_counts.values()) == 0:
        raise ValueError("Split train tidak memiliki objek beranotasi.")
    if not any(not annotation["iscrowd"] for items in val_dataset.annotations_by_image.values() for annotation in items):
        raise ValueError("Validation membutuhkan objek non-crowd agar best mAP dapat dipilih.")
    if any(annotation["iscrowd"] for items in train_dataset.annotations_by_image.values() for annotation in items):
        raise ValueError("Loss SSD tidak menangani crowd secara khusus. Gunakan anotasi per biji iscrowd=0 untuk training.")
    signature = dataset_signature(datasets)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics_dir / "train_val_audit.json", audit)
    last_path = checkpoint_dir / "last_model.pth"
    best_path = checkpoint_dir / "best_model.pth"
    if resume is None and (last_path.exists() or best_path.exists()):
        raise FileExistsError(
            f"Checkpoint sudah ada pada {checkpoint_dir}. Gunakan --resume last_model.pth "
            "atau direktori checkpoint baru untuk eksperimen baru."
        )

    seed_everything()
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        worker_init_fn=seed_worker, generator=generator,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        worker_init_fn=seed_worker, pin_memory=device.type == "cuda",
    )
    model = build_model(pretrained=bool(pretrained and resume is None)).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY,
    )
    hyperparameters = {
        "batch_size": batch_size, "learning_rate": config.LEARNING_RATE,
        "momentum": config.MOMENTUM, "weight_decay": config.WEIGHT_DECAY,
        "seed": config.RANDOM_SEED, "optimizer": config.OPTIMIZER,
        "horizontal_flip_probability": config.HORIZONTAL_FLIP_PROBABILITY,
        "map_score_threshold": config.MAP_SCORE_THRESHOLD,
        "nms_iou_threshold": config.NMS_IOU_THRESHOLD,
        "max_detections_per_image": config.MAX_DETECTIONS_PER_IMAGE,
        "coco_max_detections": list(config.COCO_MAX_DETECTIONS),
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "iou_threshold": config.IOU_THRESHOLD,
        "deterministic": config.DETERMINISTIC,
        "num_workers": config.NUM_WORKERS,
    }
    start_epoch, best_map, best_epoch, history = 0, -1.0, 0, []
    if resume is not None:
        checkpoint = load_checkpoint(resume, device)
        for key in ("optimizer_state_dict", "epoch", "history", "rng_state", "loader_rng_state"):
            if key not in checkpoint:
                raise ValueError(f"Checkpoint resume tidak memiliki {key}.")
        if checkpoint.get("dataset_signature") != signature:
            raise ValueError("Data/anotasi train atau val berubah sejak checkpoint; mulailah eksperimen baru.")
        if checkpoint.get("hyperparameters") != hyperparameters:
            raise ValueError("Hyperparameter resume berbeda. Pertahankan konfigurasi dan batch size run awal.")
        start_epoch = int(checkpoint["epoch"])
        if num_epochs <= start_epoch:
            raise ValueError(f"--epochs adalah total epoch dan harus lebih besar dari epoch checkpoint ({start_epoch}).")
        source_best = Path(resume).resolve().parent / "best_model.pth"
        if not source_best.is_file():
            raise FileNotFoundError("Resume memerlukan best_model.pth di samping last_model.pth untuk menjaga best validation.")
        best_checkpoint = load_checkpoint(source_best, device)
        if (best_checkpoint.get("dataset_signature") != signature
                or best_checkpoint.get("epoch") != checkpoint.get("best_epoch")):
            raise ValueError("best_model.pth tidak sesuai dengan riwayat/dataset checkpoint resume.")
        if source_best != best_path:
            if best_path.exists() or last_path.exists():
                raise FileExistsError("Direktori tujuan resume sudah berisi checkpoint lain.")
            shutil.copy2(source_best, best_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_map = float(checkpoint["best_map"])
        best_epoch = int(checkpoint["best_epoch"])
        history = checkpoint["history"]
        generator.set_state(checkpoint["loader_rng_state"].cpu())
        restore_rng_state(checkpoint["rng_state"])

    print(f"Device: {device} | train={len(train_dataset)} | val={len(val_dataset)} | batch={batch_size}")
    for epoch_index in range(start_epoch, num_epochs):
        model.train()
        sums = {"training_loss": 0.0, "classification": 0.0, "bbox_regression": 0.0}
        seen_images = 0
        singleton_batches = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with singleton_batch_norm(model, len(images)):
                losses = model(images, targets)
                if set(losses) != {"classification", "bbox_regression"}:
                    raise RuntimeError(f"Loss SSD tidak sesuai: {list(losses)}")
                loss = sum(losses.values())
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Loss tidak finite pada epoch {epoch_index + 1}.")
                loss.backward()
                if any(not torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):
                    raise RuntimeError("Gradien tidak finite; optimizer tidak dijalankan.")
                optimizer.step()
            count = len(images)
            singleton_batches += int(count == 1)
            seen_images += count
            sums["training_loss"] += float(loss.detach()) * count
            for key in losses:
                sums[key] += float(losses[key].detach()) * count
        if seen_images != len(train_dataset):
            raise RuntimeError("Tidak semua gambar train diproses.")
        averages = {key: value / seen_images for key, value in sums.items()}
        validation = evaluate_model(model, val_loader, device)
        validation_map = validation["map"]
        if validation_map is None or not math.isfinite(validation_map):
            raise RuntimeError("mAP validation tidak terdefinisi; periksa anotasi validation.")
        epoch = epoch_index + 1
        row = {
            "epoch": epoch, **averages,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_map": validation_map, "val_map_50": validation["map_50"],
            "val_precision": validation["precision"], "val_recall": validation["recall"],
            "val_f1": validation["f1"], "train_images": seen_images,
            "singleton_batches": singleton_batches,
        }
        history.append(row)
        is_best = validation_map > best_map
        if is_best:
            best_map, best_epoch = validation_map, epoch
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "training_loss": averages["training_loss"],
            "best_map": best_map, "best_epoch": best_epoch,
            "validation_metrics": validation, "history": history,
            "model_name": config.MODEL_NAME, "num_classes": config.NUM_CLASSES,
            "class_names": config.CLASS_NAMES, "config": configuration_snapshot(),
            "hyperparameters": hyperparameters, "dataset_signature": signature,
            "rng_state": capture_rng_state(), "loader_rng_state": generator.get_state(),
            "versions": {"torch": str(torch.__version__)},
        }
        if is_best:
            save_checkpoint(best_path, checkpoint)
            save_json(metrics_dir / "best_validation_metrics.json", validation)
        save_checkpoint(last_path, checkpoint)
        save_metrics_report(metrics_dir, "validation", validation)
        save_history(metrics_dir, history)
        print(
            f"Epoch {epoch}/{num_epochs} | loss={averages['training_loss']:.4f} "
            f"(classification={averages['classification']:.4f}, bbox={averages['bbox_regression']:.4f}) "
            f"| lr={row['learning_rate']:.6f} | val mAP={validation_map:.4f}"
        )
    return {
        "best_checkpoint": str(best_path), "last_checkpoint": str(last_path),
        "best_epoch": best_epoch, "best_validation_map": best_map,
        "completed_epochs": num_epochs, "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=config.METRICS_DIR)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    result = train_model(
        load_split(args.dataset, "train", train=True), load_split(args.dataset, "val"),
        num_epochs=args.epochs, batch_size=args.batch_size, resume=args.resume,
        checkpoint_dir=args.checkpoint_dir, metrics_dir=args.metrics_dir,
        device=get_device(args.require_cuda),
    )
    print(f"Best checkpoint: {result['best_checkpoint']} (epoch {result['best_epoch']})")


if __name__ == "__main__":
    main()
