"""Pengaturan website, terpisah dari pengaturan eksperimen training."""

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


class WebConfig:
    """Baca ulang environment saat aplikasi dibuat, termasuk dari notebook."""

    @staticmethod
    def from_env():
        return {
            # Pilih checkpoint secara eksplisit; tidak mencari file terbaru.
            "CHECKPOINT_PATH": os.environ.get(
                "CHECKPOINT_PATH", str(PROJECT_DIR / "checkpoints" / "best_model.pth")
            ),
            "WEB_CONFIDENCE_THRESHOLD": float(os.environ.get("WEB_CONFIDENCE_THRESHOLD", "0.20")),
            "DEVICE": os.environ.get("WEB_DEVICE", "auto"),
            "MAX_CONTENT_LENGTH": 15 * 1024 * 1024,
            "MAX_IMAGE_PIXELS": 50_000_000,
            "JSON_AS_ASCII": False,
        }
