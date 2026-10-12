"""Seed, DataLoader collate, checkpoint, JSON/CSV, dan grafik eksperimen."""

import csv
import json
import os
from pathlib import Path
import random

import config

os.environ.setdefault("MPLCONFIGDIR", str(config.PROJECT_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(config.PROJECT_DIR / ".cache"))

import numpy as np
import torch


def collate_fn(batch):
    # Jumlah objek dan ukuran citra dapat berbeda untuk setiap gambar.
    return tuple(zip(*batch))


def get_device(require_cuda=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA tidak tersedia. Periksa driver, build CUDA PyTorch, dan interpreter .venv.")
    if device.type == "cpu":
        torch.set_num_threads(config.CPU_NUM_THREADS)
    return device


def seed_everything(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not config.DETERMINISTIC
    torch.backends.cudnn.deterministic = config.DETERMINISTIC
    if config.DETERMINISTIC:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(config.DETERMINISTIC, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def configuration_snapshot():
    def plain(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        return value
    return {key: plain(value) for key, value in vars(config).items() if key.isupper()}


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}. Jalankan training dahulu.")
    # weights_only menghindari kebutuhan memuat object Python arbitrer.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint harus berisi model_state_dict.")
    if checkpoint.get("model_name") != config.MODEL_NAME:
        raise ValueError("Arsitektur checkpoint tidak cocok dengan SSDLite proyek ini.")
    if checkpoint.get("num_classes") != config.NUM_CLASSES:
        raise ValueError("Jumlah kelas checkpoint tidak cocok.")
    classes = {int(key): value for key, value in checkpoint.get("class_names", {}).items()}
    if classes != config.CLASS_NAMES:
        raise ValueError("Nama/urutan kelas checkpoint berbeda dari config.py; hasil label bisa salah.")
    return checkpoint


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary_path.replace(path)


def save_history(metrics_dir, history):
    metrics_dir = Path(metrics_dir)
    save_json(metrics_dir / "training_history.json", history)
    if not history:
        return
    with (metrics_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for key in ("training_loss", "classification", "bbox_regression"):
        axes[0].plot(epochs, [row[key] for row in history], label=key)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training detection loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_map"] for row in history], label="mAP@0.5:0.95")
    axes[1].plot(epochs, [row["val_map_50"] for row in history], label="mAP@0.5")
    axes[1].set(xlabel="Epoch", ylabel="mAP", title="Validation")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(metrics_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def save_metrics_report(metrics_dir, split, metrics):
    metrics_dir = Path(metrics_dir)
    save_json(metrics_dir / f"{split}_metrics.json", metrics)
    with (metrics_dir / f"{split}_per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["label", "name", "support", "tp", "fp", "fn", "precision", "recall", "f1", "ap", "ap_50"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for label, values in metrics["per_class"].items():
            writer.writerow({"label": label, **values})
