"""Deteksi satu foto dan simpan seluruh bounding box beserta jumlah per kelas.

TorchVision SSDLite melakukan resize dan normalisasi sendiri. Modul ini hanya
mengubah RGB menjadi tensor float [0, 1] dengan ukuran foto aslinya.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

import config


# Warna RGB: setiap kelas selalu menggunakan warna yang sama di semua foto.
CLASS_COLORS = {1: (230, 70, 70), 2: (245, 185, 40), 3: (50, 180, 85)}


def _output_stem(image_path: Path) -> str:
    """Batasi nama file; bagian path input tidak menjadi path output."""
    return re.sub(r"[^\w-]+", "_", image_path.stem).strip("_")[:100] or "image"


def _draw_predictions(rgb_image: np.ndarray, boxes, labels, scores) -> np.ndarray:
    annotated = rgb_image.copy()
    height, width = annotated.shape[:2]
    thickness = max(1, round(min(width, height) / 400))
    font_scale = max(0.4, min(1.0, min(width, height) / 900))
    for box, label, score in zip(boxes, labels, scores):
        # Koordinat JSON tetap float. Pembulatan/clipping hanya untuk rendering.
        x1 = min(width - 1, max(0, round(box[0])))
        y1 = min(height - 1, max(0, round(box[1])))
        x2 = min(width - 1, max(0, round(box[2])))
        y2 = min(height - 1, max(0, round(box[3])))
        color = CLASS_COLORS.get(label, (80, 140, 230))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        text = f"{config.CLASS_NAMES[label]} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_x = min(x1, max(0, width - text_width - 4))
        text_y = max(text_height + 3, y1 - 3)
        text_y = min(height - 1, text_y)
        cv2.rectangle(
            annotated,
            (text_x, max(0, text_y - text_height - 3)),
            (min(width - 1, text_x + text_width + 4), min(height - 1, text_y + baseline)),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            text,
            (text_x + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 15, 15),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


@torch.inference_mode()
def predict_image(model, image_path, device, confidence_threshold=None, output_dir=None) -> dict:
    """Inferensi RGB tanpa batas buatan pada jumlah objek yang lolos threshold.

    Hasil berisi box xyxy dalam piksel foto asli, label internal 1..3, confidence,
    nama kelas per deteksi, dan jumlah per kelas. PNG serta JSON disimpan tanpa
    membuka GUI. Menjalankan foto yang sama ke direktori yang sama memperbarui
    file hasil dengan nama deterministik.
    """
    from src.utils import save_json

    threshold = config.CONFIDENCE_THRESHOLD if confidence_threshold is None else float(confidence_threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold harus berupa angka antara 0 dan 1.")
    input_path = Path(image_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {input_path}")

    with Image.open(input_path) as opened:
        rgb_image = opened.convert("RGB")
        image_tensor = pil_to_tensor(rgb_image).to(dtype=torch.float32).div(255.0)
        rgb_array = np.asarray(rgb_image).copy()

    model.eval()
    outputs = model([image_tensor.to(device)])
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise ValueError("Model harus mengembalikan tepat satu hasil untuk satu foto.")
    prediction = outputs[0]
    boxes = prediction["boxes"].detach().to("cpu", dtype=torch.float32).reshape(-1, 4)
    labels = prediction["labels"].detach().to("cpu", dtype=torch.int64).reshape(-1)
    scores = prediction["scores"].detach().to("cpu", dtype=torch.float32).reshape(-1)
    if not len(boxes) == len(labels) == len(scores):
        raise ValueError("Jumlah boxes, labels, dan scores prediksi harus sama.")
    if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all()):
        raise ValueError("Prediksi mengandung koordinat atau confidence yang tidak finite.")
    if bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError("Confidence model harus berada antara 0 dan 1.")
    class_ids = set(config.CLASS_NAMES) - {0}
    unknown = set(labels.tolist()) - class_ids
    if unknown:
        raise ValueError(f"Label prediksi tidak dikenal: {sorted(unknown)}.")

    keep = scores >= threshold
    boxes_list = boxes[keep].tolist()
    labels_list = labels[keep].tolist()
    scores_list = scores[keep].tolist()
    names = [config.CLASS_NAMES[label] for label in labels_list]
    counts = {config.CLASS_NAMES[label]: labels_list.count(label) for label in sorted(class_ids)}

    destination = Path(config.PREDICTION_DIR if output_dir is None else output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(input_path)
    annotated_path = destination / f"{stem}_annotated.png"
    json_path = destination / f"{stem}_predictions.json"
    annotated_rgb = _draw_predictions(rgb_array, boxes_list, labels_list, scores_list)
    success, encoded = cv2.imencode(".png", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError("OpenCV gagal mengenkode gambar anotasi PNG.")
    # imencode + tofile mendukung nama/path Unicode pada Windows.
    encoded.tofile(str(annotated_path))

    result = {
        "image": str(input_path),
        "input_path": str(input_path),
        "annotated_image": str(annotated_path),
        "prediction_json": str(json_path),
        "confidence_threshold": threshold,
        "boxes": boxes_list,
        "labels": labels_list,
        "scores": scores_list,
        "class_names": names,
        "counts": counts,
        "total": len(labels_list),
    }
    save_json(json_path, result)
    return result


def main(argv=None) -> None:
    from src.model import build_model
    from src.utils import get_device, load_checkpoint, seed_everything

    parser = argparse.ArgumentParser(description="Deteksi seluruh biji kakao dalam satu foto.")
    parser.add_argument("--image", required=True, type=Path, help="Foto JPG/PNG yang akan dideteksi.")
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_DIR / "best_model.pth")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=config.PREDICTION_DIR)
    args = parser.parse_args(argv)

    seed_everything(config.RANDOM_SEED)
    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint tidak memiliki model_state_dict; gunakan checkpoint hasil training proyek ini.")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    result = predict_image(model, args.image, device, args.confidence, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
