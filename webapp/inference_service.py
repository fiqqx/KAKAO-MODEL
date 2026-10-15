"""Satu model bersama, inferensi terkunci, dan hasil gambar di memori."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import BytesIO
import math
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

import config
from src.model import build_model
from src.predict import _draw_predictions
from src.utils import load_checkpoint


class InferenceUnavailable(RuntimeError):
    """Model belum siap; kondisi ini berbeda dari hasil nol objek."""


class InferenceService:
    """Muat checkpoint sekali ketika aplikasi dibuat, bukan setiap request.

    ``model`` hanya untuk dependency injection/pengujian tanpa checkpoint.
    Aplikasi biasa selalu memuat checkpoint melalui konfigurasi eksplisit.
    """

    def __init__(self, options, model=None):
        self._lock = Lock()
        self.model = None
        self.error = None
        self.device = "belum tersedia"
        self.threshold = float(options["WEB_CONFIDENCE_THRESHOLD"])
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("WEB_CONFIDENCE_THRESHOLD harus antara 0 dan 1.")
        try:
            requested = str(options.get("DEVICE", "auto")).lower()
            if requested not in {"auto", "cuda", "cpu"}:
                raise ValueError("WEB_DEVICE harus auto, cuda, atau cpu.")
            if requested == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("GPU CUDA diminta, tetapi belum tersedia pada Python yang dipilih.")
            self.device = "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                torch.set_num_threads(config.CPU_NUM_THREADS)
            if model is None:
                checkpoint_path = Path(options["CHECKPOINT_PATH"]).expanduser()
                if not checkpoint_path.is_absolute():
                    checkpoint_path = config.PROJECT_DIR / checkpoint_path
                checkpoint = load_checkpoint(checkpoint_path, torch.device(self.device))
                model = build_model(pretrained=False)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.model = model.to(self.device).eval()
        except Exception as exc:
            # Home dan petunjuk tetap dapat dibuka; tidak membuat model contoh.
            self.error = f"Model belum siap: {exc}"

    @property
    def ready(self):
        return self.model is not None and self.error is None

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        if not self.ready:
            raise InferenceUnavailable(self.error or "Model belum siap.")
        started = perf_counter()
        # SSDLite melakukan resize/normalisasi internal. Cukup tensor RGB [0,1].
        image = image.convert("RGB")
        tensor = pil_to_tensor(image).to(dtype=torch.float32).div(255.0)
        with self._lock:
            outputs = self.model([tensor.to(self.device)])
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
                raise ValueError("Model harus menghasilkan satu prediksi untuk satu gambar.")
            prediction = outputs[0]
            boxes = prediction["boxes"].detach().to("cpu", dtype=torch.float32).reshape(-1, 4)
            labels = prediction["labels"].detach().to("cpu", dtype=torch.int64).reshape(-1)
            scores = prediction["scores"].detach().to("cpu", dtype=torch.float32).reshape(-1)
        if not len(boxes) == len(labels) == len(scores):
            raise ValueError("Jumlah boxes, labels, dan scores model tidak sama.")
        if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all()):
            raise ValueError("Prediksi model berisi nilai tidak valid.")
        if bool(((scores < 0) | (scores > 1)).any()):
            raise ValueError("Confidence model harus antara 0 dan 1.")
        class_ids = sorted(set(config.CLASS_NAMES) - {0})
        unknown = set(labels.tolist()) - set(class_ids)
        if unknown:
            raise ValueError(f"Label model tidak dikenal: {sorted(unknown)}.")
        keep = scores >= self.threshold
        box_list = boxes[keep].tolist()
        label_list = labels[keep].tolist()
        score_list = scores[keep].tolist()
        counts = {config.CLASS_NAMES[label]: label_list.count(label) for label in class_ids}
        annotated = _draw_predictions(np.asarray(image), box_list, label_list, score_list)
        buffer = BytesIO()
        Image.fromarray(annotated).save(buffer, format="JPEG", quality=90)
        return {
            "result_id": uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "image_mime": "image/jpeg",
            "width": image.width,
            "height": image.height,
            "boxes": box_list,
            "labels": label_list,
            "scores": score_list,
            "counts": counts,
            "total": len(label_list),
            "confidence_threshold": self.threshold,
            "inference_ms": round((perf_counter() - started) * 1000, 1),
        }
