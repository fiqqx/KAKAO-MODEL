"""Uji perangkat lunak menggunakan data sintetis, bukan evaluasi dataset kakao."""

import argparse
import json
import os
from pathlib import Path
import random
import sys

# Cache tetap di checkout, tanpa memerlukan penulisan ke home pengguna.
PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(PROJECT_DIR / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
sys.path.insert(0, str(PROJECT_DIR))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms

import config
from src.model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda", action="store_true",
        help="Gagal jika GPU CUDA tidak tersedia; gunakan pada Windows NVIDIA.",
    )
    args = parser.parse_args()
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA tidak tersedia. Periksa driver NVIDIA, build PyTorch CUDA, "
            "dan interpreter .venv yang dipilih pada VS Code."
        )

    # Operasi C++ NMS akan gagal bila binary torch/torchvision tidak kompatibel.
    boxes = torch.tensor([[10., 10., 80., 80.], [11., 11., 79., 79.]])
    scores = torch.tensor([0.9, 0.8])
    if nms(boxes, scores, 0.5).tolist() != [0]:
        raise RuntimeError("Operasi NMS menghasilkan keluaran yang tidak sesuai.")

    # Transfer learning COCO dengan head 3 kelas kakao + background.
    model = build_model(pretrained=True).to(device)
    if model.head.classification_head.num_columns != config.NUM_CLASSES:
        raise RuntimeError("Classification head tidak memiliki 4 kelas.")
    images = [torch.rand(3, 320, 320, device=device) for _ in range(2)]
    targets = [
        {"boxes": torch.tensor([[40., 50., 120., 160.]], device=device),
         "labels": torch.tensor([1], dtype=torch.int64, device=device)}
        for _ in images
    ]

    model.train()
    losses = model(images, targets)
    if set(losses) != {"bbox_regression", "classification"}:
        raise RuntimeError(f"Nama loss tidak sesuai: {list(losses)}")
    total_loss = sum(losses.values())
    if not torch.isfinite(total_loss).item():
        raise RuntimeError("Loss sintetis tidak finite.")
    total_loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError("Backward tidak menghasilkan gradien.")
    if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError("Backward menghasilkan gradien tidak finite.")
    model.zero_grad(set_to_none=True)

    model.eval()
    with torch.inference_mode():
        predictions = model(images[:1])
    prediction = predictions[0]
    if set(prediction) != {"boxes", "labels", "scores"}:
        raise RuntimeError("Keluaran deteksi tidak lengkap.")
    if prediction["boxes"].ndim != 2 or prediction["boxes"].shape[1] != 4:
        raise RuntimeError("Bentuk bounding box tidak sesuai.")
    if not all(torch.isfinite(v).all().item() for v in prediction.values()):
        raise RuntimeError("Prediksi berisi nilai tidak finite.")
    if set(prediction["labels"].tolist()) - {1, 2, 3}:
        raise RuntimeError("Prediksi tidak memakai label internal kakao 1..3.")

    # Prediksi identik dengan anotasi harus menghasilkan mAP = 1.
    # Ini self-test backend metrik, bukan angka performa penelitian.
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    truth = {"boxes": boxes[:1], "labels": torch.tensor([1])}
    exact_prediction = {**truth, "scores": torch.tensor([0.99])}
    metric.update([exact_prediction], [truth])
    result = metric.compute()
    if not torch.isclose(result["map"], torch.tensor(1.)).item():
        raise RuntimeError("Self-test COCO mAP gagal.")

    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    Image.fromarray(rgb).convert("RGB")
    success, encoded = cv2.imencode(".png", rgb)
    if not success or cv2.imdecode(encoded, cv2.IMREAD_COLOR).shape != rgb.shape:
        raise RuntimeError("OpenCV tidak dapat melakukan encode/decode gambar.")
    figure, axis = plt.subplots()
    axis.imshow(rgb)
    figure.canvas.draw()
    plt.close(figure)

    print(json.dumps({
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "pretrained_model": config.MODEL_NAME,
        "num_classes_including_background": config.NUM_CLASSES,
        "nms_check": "passed",
        "synthetic_detection_forward_backward": "passed",
        "loss_keys": sorted(losses),
        "prediction_keys": sorted(prediction),
        "synthetic_coco_map_check": "passed",
        "image_and_plot_check": "passed",
        "research_evaluation": "belum dijalankan; dataset kakao lokal belum diperiksa",
    }, indent=2))


if __name__ == "__main__":
    main()
