"""Uji integrasi CPU sementara, bukan training/evaluasi dataset penelitian.

Fixture COCO dibuat di folder temporary lalu dihapus. Satu laporan pemeriksaan
tersisa di outputs/metrics; angka AP sintetis tidak menggambarkan mutu model kakao.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(PROJECT_DIR / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from PIL import Image, ImageDraw
import torch

import config
from run_pipeline import run_pipeline
from src.dataset import load_split
from src.model import build_model
from src.train import train_model
from src.utils import load_checkpoint, save_json


def require(condition, message):
    if not condition:
        raise RuntimeError(f"Smoke test gagal: {message}")


def create_synthetic_dataset(dataset_dir):
    """20 box per gambar; category_id 0 tetap kelas objek, bukan background."""
    categories = [
        {"id": raw_id, "name": config.CLASS_NAMES[label]}
        for raw_id, label in zip((0, 10, 20), (1, 2, 3))
    ]
    colors = [(145, 95, 65), (190, 115, 70), (100, 65, 45)]
    image_sequence = 0
    for split, count in (("train", 3), ("valid", 1), ("test", 1)):
        directory = dataset_dir / split
        directory.mkdir(parents=True)
        coco = {"images": [], "annotations": [], "categories": categories}
        for image_index in range(count):
            image_sequence += 1
            # Independent pixel contents prevent cross-split duplicate detection.
            rng = np.random.default_rng(1000 + image_sequence)
            pixels = rng.integers(200, 240, (320, 320, 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image)
            filename = f"synthetic_{split}_{image_index}.png"
            coco["images"].append({
                "id": image_sequence, "file_name": filename,
                "width": 320, "height": 320,
            })
            for object_index in range(20):
                column, row = object_index % 5, object_index // 5
                x = 15 + column * 60 + image_sequence
                y = 20 + row * 70 + image_sequence
                width, height = 32, 46
                class_index = object_index % 3
                draw.rectangle((x, y, x + width, y + height), fill=colors[class_index])
                coco["annotations"].append({
                    "id": image_sequence * 100 + object_index,
                    "image_id": image_sequence,
                    "category_id": (0, 10, 20)[class_index],
                    "bbox": [x, y, width, height], "area": width * height,
                    "iscrowd": 0,
                })
            image.save(directory / filename)
        save_json(directory / "_annotations.coco.json", coco)


def check_checkpoint(checkpoint, epoch):
    require(checkpoint["epoch"] == epoch, "epoch checkpoint tidak sesuai")
    require(checkpoint["num_classes"] == 4, "head harus memiliki empat kelas")
    require(checkpoint["class_names"] == config.CLASS_NAMES, "nama/urutan kelas berubah")
    require(bool(checkpoint["optimizer_state_dict"]["state"]), "state momentum SGD tidak disimpan")
    require(len(checkpoint["history"]) == epoch, "history epoch tidak lengkap")
    require(set(checkpoint["rng_state"]) == {"python", "numpy", "torch", "cuda"}, "state RNG tidak lengkap")
    require(checkpoint["loader_rng_state"].dtype == torch.uint8, "state generator loader tidak disimpan")
    for expected_epoch, row in enumerate(checkpoint["history"], start=1):
        require(row["epoch"] == expected_epoch, "urutan history epoch salah")
        require(row["train_images"] == 3, "batch akhir satu gambar dibuang")
        require(row["singleton_batches"] == 1, "batch singleton tidak diproses")
        for name in ("training_loss", "classification", "bbox_regression", "val_map", "val_map_50"):
            require(math.isfinite(row[name]), f"{name} tidak finite")


def run_smoke_test(pretrained=True):
    started = time.monotonic()
    # Satu thread membuat pembandingan resume tidak dipengaruhi urutan reduksi
    # floating point paralel CPU; training penelitian tetap memakai config.py.
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="cocoa-smoke-") as temporary:
        root = Path(temporary)
        dataset_dir = root / "synthetic_dataset"
        create_synthetic_dataset(dataset_dir)
        checkpoints, metrics, predictions = root / "checkpoints", root / "metrics", root / "predictions"
        # Smoke test selalu CPU sehingga dapat dijalankan pada laptop tanpa GPU.
        with patch("run_pipeline.get_device", return_value=torch.device("cpu")):
            result = run_pipeline(
                dataset_dir=dataset_dir, epochs=1, batch_size=2,
                checkpoint_dir=checkpoints, metrics_dir=metrics,
                prediction_dir=predictions, pretrained=pretrained,
            )
        best = load_checkpoint(checkpoints / "best_model.pth", "cpu")
        last = load_checkpoint(checkpoints / "last_model.pth", "cpu")
        check_checkpoint(best, 1)
        check_checkpoint(last, 1)
        require(result["best_epoch"] == 1, "best validation checkpoint belum dipilih")
        test_metrics = result["test_metrics"]
        require(test_metrics["evaluated_images"] == 1, "test tidak mengevaluasi tepat satu gambar")
        require(test_metrics["support"] == 20, "target multi-objek test tidak lengkap")
        for name in ("map", "map_50", "precision", "recall", "f1"):
            require(math.isfinite(test_metrics[name]), f"metrik {name} tidak finite")
        require(set(test_metrics["per_class"]) == {"1", "2", "3"}, "laporan per kelas tidak lengkap")
        for filename in (
            "dataset_audit.json", "train_val_audit.json", "training_history.json",
            "training_history.csv", "training_curves.png", "validation_metrics.json",
            "validation_per_class.csv", "test_metrics.json", "test_per_class.csv",
            "best_validation_metrics.json", "pipeline_summary.json",
        ):
            require((metrics / filename).is_file(), f"output {filename} tidak ditemukan")
        with (metrics / "test_per_class.csv").open(encoding="utf-8", newline="") as handle:
            per_class_rows = list(csv.DictReader(handle))
        require(len(per_class_rows) == 3, "CSV tidak mempunyai tiga kelas objek")
        require(sum(int(row["support"]) for row in per_class_rows) == 20, "support CSV salah")
        require(Path(result["prediction"]["annotated_image"]).is_file(), "gambar prediksi tidak tersimpan")
        require(Path(result["prediction"]["prediction_json"]).is_file(), "JSON prediksi tidak tersimpan")
        require(sum(result["prediction"]["counts"].values()) == result["prediction"]["total"], "jumlah prediksi tidak konsisten")

        # Checkpoint harus dapat direstore tanpa bobot dari internet, lalu inferensi.
        with patch("torch.hub.download_url_to_file", side_effect=RuntimeError("Unduhan tidak boleh terjadi saat restore")):
            restored = build_model(pretrained=False)
            restored.load_state_dict(best["model_state_dict"], strict=True)
            restored.eval()
            image, target = load_split(dataset_dir, "test")[0]
            with torch.inference_mode():
                output = restored([image])[0]
        require(restored.head.classification_head.num_columns == 4, "jumlah kolom classifier salah")
        require(set(target["labels"].tolist()) == {1, 2, 3}, "pemetaan label raw COCO salah")
        require(set(output["labels"].tolist()) <= {1, 2, 3}, "inferensi menghasilkan label background/asing")
        require(all(torch.equal(value, restored.state_dict()[key]) for key, value in best["model_state_dict"].items()), "bobot restore berbeda dari checkpoint")

        train_dataset = load_split(dataset_dir, "train", train=True)
        val_dataset = load_split(dataset_dir, "val")
        resumed = train_model(
            train_dataset, val_dataset, num_epochs=2, batch_size=2,
            checkpoint_dir=checkpoints, metrics_dir=metrics,
            pretrained=pretrained, resume=checkpoints / "last_model.pth", device="cpu",
        )
        resumed_checkpoint = load_checkpoint(checkpoints / "last_model.pth", "cpu")
        check_checkpoint(resumed_checkpoint, 2)
        require(resumed_checkpoint["history"][0] == last["history"][0], "resume mengubah history epoch pertama")
        require(resumed["best_epoch"] in (1, 2), "resume kehilangan best epoch")
        require(resumed["best_validation_map"] >= result["best_validation_map"], "resume kehilangan best validation lama")

        # Dua epoch tanpa interupsi harus identik dengan satu epoch + resume CPU.
        train_model(
            train_dataset, val_dataset, num_epochs=2, batch_size=2,
            checkpoint_dir=root / "uninterrupted_checkpoints",
            metrics_dir=root / "uninterrupted_metrics", pretrained=pretrained, device="cpu",
        )
        uninterrupted = load_checkpoint(root / "uninterrupted_checkpoints" / "last_model.pth", "cpu")
        history_differences = [
            (index + 1, key, left[key], right[key])
            for index, (left, right) in enumerate(zip(uninterrupted["history"], resumed_checkpoint["history"]))
            for key in left if left[key] != right[key]
        ]
        parameter_differences = [
            (key, float((value - uninterrupted["model_state_dict"][key]).abs().max()))
            for key, value in resumed_checkpoint["model_state_dict"].items()
            if not torch.equal(value, uninterrupted["model_state_dict"][key])
        ]
        require(not history_differences, f"history resume berbeda dari run tanpa interupsi: {history_differences[:10]}; perbedaan tensor terbesar: {sorted(parameter_differences, key=lambda item: item[1], reverse=True)[:5]}")
        require(not parameter_differences, f"bobot resume berbeda dari run tanpa interupsi: {parameter_differences[:10]}")

        summary = {
            "status": "passed", "device": "cpu", "synthetic_data": True,
            "pretrained_coco": pretrained, "temporary_dataset_removed": True,
            "fixture_images": {"train": 3, "valid": 1, "test": 1},
            "objects_per_image": 20, "num_classes_including_background": 4,
            "all_train_images_processed": True, "singleton_batch_processed": True,
            "checkpoint_and_optimizer_restore": "passed",
            "restore_without_download": "passed",
            "resume_matches_uninterrupted_training": "passed",
            "test_evaluation_and_per_class_reports": "passed",
            "prediction_png_json_and_counts": "passed",
            "completed_epochs_after_resume": resumed_checkpoint["epoch"],
            "research_results": "Belum tersedia. Fixture sintetis hanya menguji fungsi program; gunakan dataset kakao asli untuk hasil skripsi.",
        }
    summary["duration_seconds"] = round(time.monotonic() - started, 3)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Inisialisasi acak; tidak mengunduh bobot pretrained. Ini hanya uji software.")
    parser.add_argument("--output", type=Path, default=config.METRICS_DIR / "smoke_test.json")
    args = parser.parse_args()
    summary = run_smoke_test(pretrained=not args.offline)
    save_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Laporan smoke test: {args.output.resolve()}")


if __name__ == "__main__":
    main()
