# Penjelasan kode lengkap: Tahap 1 sampai 8

Seluruh implementasi berada dalam satu root proyek. Baca README untuk perintah lokal dan VALIDASI.md untuk bukti pengujian. Sumber lengkap di bawah sesuai file pada paket ZIP; tidak ada deployment web.

## config.py

Fungsi: Path, kelas dan seluruh hyperparameter awal.

Input: Pengaturan peneliti dan nama kategori COCO.

Output: Konstanta Python.

Bagian penting: NUM_CLASSES=4 mencakup background. CATEGORY_NAME_TO_LABEL memetakan nama aktual, bukan menganggap urutan ID COCO. Nilai baseline dijelaskan pada README.

Kode lengkap:

```python
"""Konfigurasi awal eksperimen; nilai ini belum merupakan hasil tuning."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# Ekspor COCO Roboflow biasanya memakai nama ini. Verifikasi pada Tahap 2.
TRAIN_ANNOTATIONS = TRAIN_DIR / "_annotations.coco.json"
VAL_ANNOTATIONS = VAL_DIR / "_annotations.coco.json"
TEST_ANNOTATIONS = TEST_DIR / "_annotations.coco.json"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
PREDICTION_DIR = PROJECT_DIR / "outputs" / "predictions"
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"

CLASS_NAMES = {
    0: "background",
    1: "Tidak Fermentasi",
    2: "Setengah Fermentasi",
    3: "Full Fermentasi",
}
NUM_CLASSES = len(CLASS_NAMES)  # Tiga kelas objek + background = 4.

# Inilah label internal model, bukan asumsi category_id dalam JSON COCO.
# Pemetaan category_id COCO dibuat loader berdasarkan nama, bukan urutan ID.
CATEGORY_NAME_TO_LABEL = {
    "Tidak Fermentasi": 1,
    "Setengah Fermentasi": 2,
    "Full Fermentasi": 3,
}
# Tambahkan alias sesuai nama asli categories dalam JSON Anda, misalnya
# "unfermented": 1, jika dan hanya jika maknanya memang Tidak Fermentasi.
IGNORED_CATEGORY_NAMES = ()  # Jangan abaikan kelas penelitian.
MODEL_NAME = "ssdlite320_mobilenet_v3_large"
USE_PRETRAINED = True

# Titik awal konservatif untuk fine-tuning SGD; tuning hanya memakai val.
OPTIMIZER = "SGD"
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 4  # Batch > 1 membantu BatchNorm; sesuaikan dengan memori.
NUM_EPOCHS = 30  # Anggaran awal, bukan jaminan konvergensi.
NUM_WORKERS = 0  # Mudah ditelusuri dan aman pada CPU/cloud/notebook.
CPU_NUM_THREADS = 4  # Hindari thread berlebihan pada mesin CPU.
DETERMINISTIC = True  # Mengurangi sumber variasi; lintas perangkat tetap bisa berbeda.

# Threshold operasional Precision/Recall/F1 dan visualisasi.
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.55  # Default SSDLite TorchVision; berbeda dari IoU evaluasi.

# mAP harus memakai prediksi dengan ambang rendah agar kurva PR tetap utuh.
MAP_SCORE_THRESHOLD = 0.001  # Default postprocessing TorchVision SSD.
MAX_DETECTIONS_PER_IMAGE = 300  # Default SSD; dapat menangani banyak biji.
COCO_MAX_DETECTIONS = (1, 10, 100)  # Standar pelaporan COCO mAP.

HORIZONTAL_FLIP_PROBABILITY = 0.5  # Posisi berubah; warna tetap dipertahankan.
RANDOM_SEED = 42  # Nilai tetap untuk membandingkan eksperimen, bukan tuning.
```

## requirements.txt

Fungsi: Versi library yang dipakai.

Input: Dibaca pip atau uv.

Output: Dependensi pada virtual environment.

Bagian penting: Pasangan torch 2.9.1 / torchvision 0.24.1 wajib kompatibel. Build CPU/CUDA dipilih launcher; pycocotools adalah backend COCO mAP.

Kode lengkap:

```text
# Pasangan versi resmi: torchvision 0.24.1 memerlukan torch 2.9.1.
# Untuk cloud CPU, pasang kedua paket dari indeks CPU resmi terlebih dahulu.
torch==2.9.1
torchvision==0.24.1
numpy==2.2.6
Pillow==12.3.0
opencv-python-headless==4.12.0.88
matplotlib==3.10.8
torchmetrics==1.9.0
pycocotools==2.0.11
```

## src/__init__.py

Fungsi: Inisialisasi package dan cache lokal.

Input: Folder root proyek.

Output: Variabel lokasi cache bila belum ditetapkan pengguna.

Bagian penting: Cache tidak bergantung pada home pengguna; lokasi dapat mengikuti environment yang telah ditetapkan.

Kode lengkap:

```python
"""Modul penelitian deteksi multi-objek; cache tetap di folder proyek."""

import os
import config

os.environ.setdefault("TORCH_HOME", str(config.PROJECT_DIR / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(config.PROJECT_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(config.PROJECT_DIR / ".cache"))
```

## src/dataset.py

Fungsi: Custom COCO Dataset dan audit split.

Input: Gambar dan JSON images/annotations/categories.

Output: Image float RGB CHW dan target multi-objek; laporan audit.

Bagian penting: Box xywh diubah menjadi xyxy, labels int64, boxes float32. Semua objek pada gambar dikumpulkan. Kategori mentah0 bisa menjadi label objek1. Dimensi, referensi, box dan split diperiksa; SHA256 hanya mendeteksi salinan byte identik.

Kode lengkap:

```python
"""Membaca ekspor COCO Roboflow dan memeriksa integritas split penelitian."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image
import torch
from torch.utils.data import Dataset

import config
from src.transforms import make_transforms


def normalize_category_name(name):
    """Spasi, underscore, dan tanda hubung tidak mengubah nama kelas."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Nama kategori harus berupa teks yang tidak kosong.")
    return re.sub(r"[\s_-]+", " ", name.strip().casefold())


def _integer(value, description):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} harus berupa integer; ditemukan {value!r}.")
    return value


def _records_by_id(records, description):
    if not isinstance(records, list):
        raise ValueError(f"COCO '{description}' harus berupa list.")
    by_id = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Setiap entri '{description}' harus berupa object JSON.")
        record_id = _integer(record.get("id"), f"id pada {description}")
        if record_id in by_id:
            raise ValueError(f"ID duplikat pada {description}: {record_id}.")
        by_id[record_id] = record
    return by_id


def _configured_categories():
    configured = {}
    for name, label in config.CATEGORY_NAME_TO_LABEL.items():
        normalized = normalize_category_name(name)
        label = _integer(label, f"Label konfigurasi {name!r}")
        if label <= 0 or label >= config.NUM_CLASSES:
            raise ValueError(f"Label {name!r} harus 1..{config.NUM_CLASSES - 1}.")
        if normalized in configured and configured[normalized] != label:
            raise ValueError(f"Pemetaan nama kategori bertentangan: {name!r}.")
        configured[normalized] = label
    ignored = {normalize_category_name(name) for name in config.IGNORED_CATEGORY_NAMES}
    overlap = set(configured) & ignored
    if overlap:
        raise ValueError(f"Nama kelas sekaligus dipetakan dan diabaikan: {sorted(overlap)}.")
    return configured, ignored


class CocoaCocoDataset(Dataset):
    """Satu gambar dapat menghasilkan nol, satu, atau banyak target objek.

    category_id COCO tidak diasumsikan sama dengan label model. Nama kategori
    dipetakan melalui CATEGORY_NAME_TO_LABEL dalam config.py.
    """

    def __init__(self, images_dir, annotation_file=None, transforms=None):
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.annotation_file = (
            Path(annotation_file).expanduser().resolve()
            if annotation_file is not None
            else self.images_dir / "_annotations.coco.json"
        )
        if not self.annotation_file.is_file():
            raise ValueError(
                f"Anotasi COCO tidak ditemukan: {self.annotation_file}. "
                "Salin gambar dan _annotations.coco.json ekspor Roboflow ke split ini."
            )
        try:
            with self.annotation_file.open("r", encoding="utf-8-sig") as handle:
                coco = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Tidak dapat membaca COCO {self.annotation_file}: {error}") from error
        if not isinstance(coco, dict):
            raise ValueError("Dokumen COCO harus berupa object JSON.")

        self.images = _records_by_id(coco.get("images"), "images")
        categories = _records_by_id(coco.get("categories"), "categories")
        annotations = _records_by_id(coco.get("annotations"), "annotations")
        self.image_ids = list(self.images)
        self.category_names = {
            category_id: category.get("name") for category_id, category in categories.items()
        }
        configured, ignored = _configured_categories()
        normalized_names = {
            category_id: normalize_category_name(name)
            for category_id, name in self.category_names.items()
        }
        self.category_mapping = {
            category_id: configured[name]
            for category_id, name in normalized_names.items()
            if name in configured
        }
        self.ignored_category_ids = {
            category_id for category_id, name in normalized_names.items() if name in ignored
        }
        self.transforms = transforms if transforms is not None else make_transforms(train=False)
        self.image_paths = []
        self.annotations_by_image = {image_id: [] for image_id in self.image_ids}
        self.class_counts = Counter({label: 0 for label in range(1, config.NUM_CLASSES)})
        self.ignored_object_count = 0
        self._content_hashes = None
        self._validate_images()
        self._validate_annotations(annotations, categories)
        annotated_category_ids = {annotation["category_id"] for annotation in annotations.values()}
        self.unused_category_ids = set(categories) - annotated_category_ids
        self.metadata = self.audit_summary()

    def _validate_images(self):
        seen_paths = set()
        for image_id, record in self.images.items():
            width = _integer(record.get("width"), f"width gambar {image_id}")
            height = _integer(record.get("height"), f"height gambar {image_id}")
            if width <= 0 or height <= 0:
                raise ValueError(f"Dimensi gambar {image_id} harus positif.")
            filename = record.get("file_name")
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError(f"file_name gambar {image_id} harus berupa nama file.")
            # Backslash juga dibaca sebagai separator agar ekspor Windows portabel.
            image_path = (self.images_dir / filename.replace("\\", "/")).resolve()
            if not image_path.is_relative_to(self.images_dir):
                raise ValueError(f"file_name gambar {image_id} keluar dari folder split: {filename!r}.")
            if image_path in seen_paths:
                raise ValueError(f"File gambar dirujuk lebih dari sekali: {image_path}.")
            if not image_path.is_file():
                raise ValueError(f"File gambar tidak ditemukan untuk image_id={image_id}: {image_path}.")
            try:
                with Image.open(image_path) as image:
                    actual_size = image.size
                    image.verify()
            except (OSError, ValueError) as error:
                raise ValueError(f"File gambar rusak/tidak dapat dibaca: {image_path}: {error}") from error
            if actual_size != (width, height):
                raise ValueError(
                    f"Dimensi COCO berbeda dari file {filename!r}: "
                    f"COCO={(width, height)}, file={actual_size}. Perbaiki ekspor/anotasinya."
                )
            seen_paths.add(image_path)
            self.image_paths.append(image_path)

    def _validate_annotations(self, annotations, categories):
        for annotation_id, annotation in annotations.items():
            image_id = _integer(annotation.get("image_id"), f"image_id anotasi {annotation_id}")
            category_id = _integer(annotation.get("category_id"), f"category_id anotasi {annotation_id}")
            if image_id not in self.images:
                raise ValueError(f"Anotasi {annotation_id} merujuk image_id yang tidak ada: {image_id}.")
            if category_id not in categories:
                raise ValueError(f"Anotasi {annotation_id} merujuk category_id yang tidak ada: {category_id}.")
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"bbox anotasi {annotation_id} harus [x, y, width, height].")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
                raise ValueError(f"bbox anotasi {annotation_id} harus berisi empat angka.")
            x, y, width, height = map(float, bbox)
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise ValueError(f"bbox anotasi {annotation_id} mengandung NaN/Infinity.")
            image_record = self.images[image_id]
            if (
                x < 0 or y < 0 or width <= 0 or height <= 0
                or x + width > image_record["width"]
                or y + height > image_record["height"]
            ):
                raise ValueError(
                    f"bbox anotasi {annotation_id} tidak valid/keluar gambar: {bbox}, "
                    f"ukuran gambar={(image_record['width'], image_record['height'])}. "
                    "Bounding box tidak dipotong diam-diam; perbaiki anotasi sumber."
                )
            iscrowd = _integer(annotation.get("iscrowd", 0), f"iscrowd anotasi {annotation_id}")
            if iscrowd not in (0, 1):
                raise ValueError(f"iscrowd anotasi {annotation_id} harus 0 atau 1.")
            if category_id in self.ignored_category_ids:
                self.ignored_object_count += 1
                continue
            if category_id not in self.category_mapping:
                raise ValueError(
                    f"Kategori beranotasi belum dipetakan: id={category_id}, "
                    f"name={self.category_names[category_id]!r}. Tambahkan nama persis/alias "
                    "ke config.CATEGORY_NAME_TO_LABEL dengan label 1, 2, atau 3. "
                    "Gunakan IGNORED_CATEGORY_NAMES hanya jika kategori memang bukan objek penelitian."
                )
            label = self.category_mapping[category_id]
            self.annotations_by_image[image_id].append({
                "boxes": [x, y, x + width, y + height],
                "labels": label,
                "area": width * height,
                "iscrowd": iscrowd,
            })
            self.class_counts[label] += 1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
        annotations = self.annotations_by_image[image_id]
        target = {
            "boxes": torch.tensor([item["boxes"] for item in annotations], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([item["labels"] for item in annotations], dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor([item["area"] for item in annotations], dtype=torch.float32),
            "iscrowd": torch.tensor([item["iscrowd"] for item in annotations], dtype=torch.int64),
        }
        return self.transforms(image, target)

    def content_hashes(self):
        """SHA-256 isi file untuk mendeteksi salinan gambar lintas split."""
        if self._content_hashes is None:
            hashes = {}
            for path in self.image_paths:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[path] = digest.hexdigest()
            self._content_hashes = hashes
        return self._content_hashes

    def audit_summary(self):
        empty_images = sum(not annotations for annotations in self.annotations_by_image.values())
        return {
            "images_dir": str(self.images_dir),
            "annotation_file": str(self.annotation_file),
            "images": len(self),
            "objects": sum(self.class_counts.values()),
            "images_without_objects": empty_images,
            "ignored_objects": self.ignored_object_count,
            "class_counts": {
                config.CLASS_NAMES[label]: self.class_counts[label]
                for label in range(1, config.NUM_CLASSES)
            },
            "category_mapping": {
                str(category_id): {
                    "coco_name": self.category_names[category_id],
                    "model_label": label,
                    "model_name": config.CLASS_NAMES[label],
                }
                for category_id, label in self.category_mapping.items()
            },
            "unmapped_unannotated_categories": [
                {"id": category_id, "name": self.category_names[category_id]}
                for category_id in sorted(getattr(self, "unused_category_ids", set()))
                if category_id not in self.category_mapping
            ],
            "ignored_categories": [self.category_names[category_id] for category_id in sorted(self.ignored_category_ids)],
        }


def load_split(dataset_dir, split, train=False):
    """Mendukung 'val' lokal dan 'valid' hasil ekspor Roboflow."""
    root = Path(dataset_dir).expanduser().resolve()
    split = "val" if split in ("valid", "validation") else split
    if split not in ("train", "val", "test"):
        raise ValueError(f"Nama split tidak didukung: {split!r}.")
    names = ("val", "valid") if split == "val" else (split,)
    candidates = [root / name for name in names if (root / name / "_annotations.coco.json").is_file()]
    if len(candidates) > 1:
        raise ValueError(f"Ada dua split validation: {candidates}. Gunakan satu folder val atau valid.")
    split_dir = candidates[0] if candidates else root / names[0]
    dataset = CocoaCocoDataset(split_dir, transforms=make_transforms(train=train))
    if not len(dataset):
        raise ValueError(f"Split {split!r} kosong: {dataset.annotation_file}.")
    return dataset


def validate_dataset_splits(datasets):
    """Memeriksa split yang diberikan tanpa memerlukan/membaca split lainnya.

    Training memberikan train+val saja. Audit awal/pelaporan akhir memberikan
    train+val+test. Tes ini mendeteksi salinan identik, bukan kemiripan citra.
    """
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Berikan dictionary split -> CocoaCocoDataset yang tidak kosong.")
    seen_annotations, seen_paths, seen_hashes, seen_category_labels = {}, {}, {}, {}
    leakage = []
    summaries = {}
    for split, dataset in datasets.items():
        if not isinstance(dataset, CocoaCocoDataset) or not len(dataset):
            raise ValueError(f"Split {split!r} hilang atau tidak memiliki gambar.")
        for category_id, label in dataset.category_mapping.items():
            normalized = normalize_category_name(dataset.category_names[category_id])
            if normalized in seen_category_labels and seen_category_labels[normalized] != label:
                raise ValueError(f"Pemetaan label berbeda antar split untuk kategori {normalized!r}.")
            seen_category_labels[normalized] = label
        annotation_path = dataset.annotation_file.resolve()
        if annotation_path in seen_annotations:
            leakage.append(f"Anotasi sama pada {seen_annotations[annotation_path]} dan {split}: {annotation_path}")
        seen_annotations[annotation_path] = split
        for path, digest in dataset.content_hashes().items():
            if path in seen_paths and seen_paths[path] != split:
                leakage.append(f"Path gambar sama pada {seen_paths[path]} dan {split}: {path}")
            if digest in seen_hashes and seen_hashes[digest][0] != split:
                previous_split, previous_path = seen_hashes[digest]
                leakage.append(f"Isi gambar identik pada {previous_split} dan {split}: {previous_path.name} / {path.name}")
            seen_paths[path] = split
            seen_hashes[digest] = (split, path)
        summaries[split] = dataset.audit_summary()
    if leakage:
        examples = "\n- ".join(leakage[:10])
        raise ValueError(f"Data leakage lintas split ditemukan:\n- {examples}\nPisahkan ulang dataset sebelum training.")
    return {
        "status": "passed",
        "split_checks": "unique annotation paths, unique image paths, SHA-256 image file contents",
        "class_names": config.CLASS_NAMES,
        "splits": summaries,
        "limitations": "Tidak mendeteksi gambar yang mirip, hasil crop, atau duplikat dengan encoding berbeda; periksa sumber gambar secara manual.",
    }


def build_datasets(dataset_dir=None):
    """Memuat tiga split untuk audit; bukan fungsi loader training."""
    root = config.DATASET_DIR if dataset_dir is None else dataset_dir
    datasets = {split: load_split(root, split, train=False) for split in ("train", "val", "test")}
    validate_dataset_splits(datasets)
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Audit COCO dan cek kebocoran train/val/test.")
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--output", type=Path, default=config.METRICS_DIR / "dataset_audit.json")
    args = parser.parse_args()
    try:
        datasets = build_datasets(args.dataset)
        audit = validate_dataset_splits(datasets)
    except ValueError as error:
        parser.exit(1, f"Audit dataset gagal: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"\nAudit disimpan: {args.output.resolve()}")


if __name__ == "__main__":
    main()
```

## src/transforms.py

Fungsi: Preprocessing dan augmentasi aman.

Input: Pasangan image/target.

Output: Tensor gambar dan target sinkron.

Bagian penting: Flip memakai xmin baru=W - xmax dan xmax baru=W - xmin; tidak mengubah warna atau luas box. Validation/test tanpa augmentasi acak.

Kode lengkap:

```python
"""Transformasi deteksi: tensor RGB dan flip dengan bounding box yang sinkron."""

from __future__ import annotations

import math

import torch
from torchvision.transforms import functional as F

import config


class Compose:
    """Menerapkan transformasi berurutan pada pasangan image dan target."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToTensor:
    """PIL RGB uint8 -> tensor float32 [C, H, W] pada rentang [0, 1]."""

    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    """Flip kiri-kanan; warna dan luas objek tetap sama."""

    def __init__(self, probability=0.5):
        self.probability = float(probability)
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("Probabilitas horizontal flip harus berada pada [0, 1].")

    def __call__(self, image, target):
        if torch.rand(1).item() >= self.probability:
            return image, target
        width = F.get_dimensions(image)[2]
        image = F.hflip(image)
        target = dict(target)
        boxes = target["boxes"].clone()
        # xyxy: xmin baru = W - xmax lama, xmax baru = W - xmin lama.
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target["boxes"] = boxes
        return image, target


def make_transforms(train=False):
    """Validation/test tidak memakai augmentasi acak."""
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(config.HORIZONTAL_FLIP_PROBABILITY))
    return Compose(transforms)
```

## src/model.py

Fungsi: Membuat SSDLite-MobileNetV3-Large resmi.

Input: Flag pretrained dan konfigurasi 4 kelas.

Output: Model TorchVision.

Bagian penting: Muat weights detection COCO tanpa num_classes=4 terlebih dahulu, lalu ganti classification_head memakai SSDLiteClassificationHead. Backbone/features dan regresi box tetap pretrained. Restore menggunakan pretrained=False agar tidak mengunduh bobot lagi.

Kode lengkap:

```python
"""SSDLite-MobileNetV3-Large resmi, dengan head klasifikasi empat kelas."""

from functools import partial
import os

import torch
from torch import nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

import config

os.environ.setdefault("TORCH_HOME", str(config.PROJECT_DIR / ".cache" / "torch"))


def build_model(pretrained=True):
    """COCO transfer learning; pretrained=False hanya untuk restore/offline test.

    Jangan memasukkan num_classes=4 saat memuat bobot detection COCO karena
    builder pretrained mengharuskan jumlah kelas asli COCO (91).
    """
    if config.NUM_CLASSES != 4 or sorted(config.CLASS_NAMES) != [0, 1, 2, 3]:
        raise ValueError("Penelitian memerlukan 3 kelas objek + background (4 total).")
    if config.MODEL_NAME != "ssdlite320_mobilenet_v3_large":
        raise ValueError("Model utama harus ssdlite320_mobilenet_v3_large.")
    options = {
        "weights_backbone": None,
        "score_thresh": config.MAP_SCORE_THRESHOLD,
        "nms_thresh": config.NMS_IOU_THRESHOLD,
        "detections_per_img": config.MAX_DETECTIONS_PER_IMAGE,
        "progress": False,
    }
    if not pretrained:
        # Checkpoint akan mengisi seluruh bobot; jangan mengunduh backbone lain.
        return ssdlite320_mobilenet_v3_large(
            weights=None, num_classes=config.NUM_CLASSES, **options,
        )

    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT, **options,
    )
    old_head = model.head.classification_head
    in_channels = [block[0][0].in_channels for block in old_head.module_list]
    anchors = model.anchor_generator.num_anchors_per_location()
    # Hanya head kelas yang diinisialisasi ulang. Backbone, extra feature
    # blocks, dan head regresi box tetap memakai bobot pretrained COCO.
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels,
        anchors,
        config.NUM_CLASSES,
        partial(nn.BatchNorm2d, eps=0.001, momentum=0.03),
    )
    return model
```

## src/utils.py

Fungsi: Seed, collate, device, checkpoint dan pelaporan.

Input: Batch bervariasi, data metrik, checkpoint, parameter helper.

Output: List batch, perangkat, JSON/CSV/grafik, checkpoint tervalidasi.

Bagian penting: Seed Python/NumPy/PyTorch/CUDA disimpan dan direstore. torch.load(weights_only=True) membaca checkpoint sederhana dan memeriksa arsitektur/nama kelas. File disimpan melalui file sementara. Tidak ada accuracy klasifikasi gambar.

Kode lengkap:

```python
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
```

## src/train.py

Fungsi: Fine-tuning dengan SGD dan validation.

Input: Train/val Dataset, hyperparameter, optional resume.

Output: Best/last checkpoint, history loss/lr/mAP, grafik.

Bagian penting: Loss bawaan classification+bbox_regression dijumlahkan. Best memakai validation mAP; test tidak dibaca trainer. BN memakai running stats sementara untuk batch terakhir satu gambar, tanpa membuang gambar. Resume memeriksa data, mapping dan protokol evaluasi; --epochs adalah total akhir.

Kode lengkap:

```python
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
```

## src/evaluate.py

Fungsi: Evaluasi object detection keseluruhan dan per kelas.

Input: Deteksi model dan target pada val/test.

Output: IoU TP cocok, PR/F1, AP, mAP50 dan mAP50:95.

Bagian penting: TP memerlukan kelas sama serta IoU cukup dan ground truth yang belum terpakai. Confidence diurutkan; duplikat FP, salah kelas FP+FN. AP memakai skor rendah, bukan hasil yang sudah difilter pada 0.5. AP tak terdefinisi menjadi null; crowd ditangani eksplisit; maxDet COCO 100 dijelaskan.

Kode lengkap:

```python
"""Evaluasi deteksi: COCO AP dan pencocokan box per kelas untuk PR/F1.

AP menggunakan prediksi ber-confidence rendah untuk membentuk kurva PR.
Precision/recall/F1 memakai confidence operasional yang terpisah. Prediksi
diurutkan menurut confidence lalu dipasangkan dengan satu ground truth
berkelas sama dan IoU yang cukup; satu anotasi tidak boleh dipakai dua kali.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

import config


def _number_or_none(value):
    """COCO memakai -1 untuk AP yang tidak terdefinisi, bukan AP nol."""
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _rates(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "predicted_count": tp + fp,
    }


def _intersection_over_prediction(prediction_box, crowd_boxes):
    """COCO memperlakukan satu box crowd sebagai area yang boleh diabaikan."""
    top_left = torch.maximum(prediction_box[:2], crowd_boxes[:, :2])
    bottom_right = torch.minimum(prediction_box[2:], crowd_boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    prediction_area = (prediction_box[2:] - prediction_box[:2]).clamp(min=0).prod()
    return intersection / prediction_area.clamp(min=torch.finfo(torch.float32).eps)


def _match_image(prediction, target, confidence_threshold, iou_threshold, class_ids):
    """Hitung TP/FP/FN pada satu gambar; box salah kelas menjadi FP dan FN."""
    counts = {label: {"tp": 0, "fp": 0, "fn": 0, "ignored": 0} for label in class_ids}
    matched_ious = []
    crowds = target.get("iscrowd", torch.zeros_like(target["labels"])).bool()

    for label in class_ids:
        ordinary_boxes = target["boxes"][(target["labels"] == label) & ~crowds]
        crowd_boxes = target["boxes"][(target["labels"] == label) & crowds]
        keep = (prediction["labels"] == label) & (prediction["scores"] >= confidence_threshold)
        boxes = prediction["boxes"][keep]
        scores = prediction["scores"][keep]
        # Stable sorting makes tied confidences reproducible in input order.
        boxes = boxes[torch.argsort(scores, descending=True, stable=True)]
        used = torch.zeros(len(ordinary_boxes), dtype=torch.bool)
        overlaps = box_iou(boxes, ordinary_boxes)

        for prediction_index, prediction_box in enumerate(boxes):
            if len(ordinary_boxes):
                candidates = overlaps[prediction_index].clone()
                candidates[used] = -1
                best_iou, ground_truth_index = candidates.max(dim=0)
                if float(best_iou) >= iou_threshold:
                    used[ground_truth_index] = True
                    counts[label]["tp"] += 1
                    matched_ious.append(float(best_iou))
                    continue

            # Crowd is excluded from support/FN. An unmatched same-class
            # prediction inside crowd is ignored, not counted as a false positive.
            if len(crowd_boxes) and bool(
                (_intersection_over_prediction(prediction_box, crowd_boxes) >= iou_threshold).any()
            ):
                counts[label]["ignored"] += 1
            else:
                counts[label]["fp"] += 1
        counts[label]["fn"] = int((~used).sum())
    return counts, matched_ious


def _cpu_prediction(prediction, class_ids):
    result = {
        "boxes": prediction["boxes"].detach().to("cpu", dtype=torch.float32).reshape(-1, 4),
        "labels": prediction["labels"].detach().to("cpu", dtype=torch.int64).reshape(-1),
        "scores": prediction["scores"].detach().to("cpu", dtype=torch.float32).reshape(-1),
    }
    if len(result["boxes"]) != len(result["labels"]) or len(result["boxes"]) != len(result["scores"]):
        raise ValueError("Jumlah boxes, labels, dan scores prediksi harus sama.")
    unknown = set(result["labels"].tolist()) - set(class_ids)
    if unknown:
        raise ValueError(f"Label prediksi tidak dikenal: {sorted(unknown)}.")
    return result


def _cpu_target(target, class_ids):
    boxes = target["boxes"].detach().to("cpu", dtype=torch.float32).reshape(-1, 4)
    labels = target["labels"].detach().to("cpu", dtype=torch.int64).reshape(-1)
    result = {"boxes": boxes, "labels": labels}
    result["area"] = (
        target["area"].detach().to("cpu", dtype=torch.float32).reshape(-1)
        if "area" in target
        else (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    )
    result["iscrowd"] = (
        target["iscrowd"].detach().to("cpu", dtype=torch.int64).reshape(-1)
        if "iscrowd" in target
        else torch.zeros_like(labels)
    )
    if any(len(result[key]) != len(boxes) for key in ("labels", "area", "iscrowd")):
        raise ValueError("Jumlah boxes, labels, area, dan iscrowd anotasi harus sama.")
    unknown = set(labels.tolist()) - set(class_ids)
    if unknown:
        raise ValueError(f"Label anotasi tidak dikenal: {sorted(unknown)}.")
    return result


@torch.inference_mode()
def evaluate_model(model, data_loader, device, confidence_threshold=None, iou_threshold=None):
    """Kembalikan metrik Python biasa yang aman disimpan sebagai JSON.

    Tidak ada ground truth kelas tertentu: support=0, recall/F1=0 dan AP=None.
    AP=None juga berlaku untuk kelas yang hanya mempunyai anotasi crowd.
    mean_matched_iou hanya mencakup TP non-crowd pada threshold operasional.
    """
    confidence_threshold = (
        config.CONFIDENCE_THRESHOLD if confidence_threshold is None else float(confidence_threshold)
    )
    iou_threshold = config.IOU_THRESHOLD if iou_threshold is None else float(iou_threshold)
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold harus berada pada rentang 0 sampai 1.")
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold harus lebih dari 0 dan paling besar 1.")

    class_ids = sorted(label for label in config.CLASS_NAMES if label != 0)
    counts = {label: {"tp": 0, "fp": 0, "fn": 0, "ignored": 0} for label in class_ids}
    matched_ious = []
    evaluated_images = 0
    metric = MeanAveragePrecision(
        box_format="xyxy",
        class_metrics=True,
        extended_summary=True,
        max_detection_thresholds=list(config.COCO_MAX_DETECTIONS),
    )

    was_training = model.training
    previous_score_threshold = getattr(model, "score_thresh", None)
    # TorchVision SSD filters scores inside forward(). Keep low scores for AP;
    # the operational PR filter happens later and never truncates the AP input.
    if previous_score_threshold is not None:
        model.score_thresh = min(config.MAP_SCORE_THRESHOLD, confidence_threshold)
    model.eval()
    try:
        for images, targets in data_loader:
            predictions = model([image.to(device) for image in images])
            if len(predictions) != len(targets):
                raise ValueError("Model harus mengembalikan satu prediksi per gambar.")
            cpu_predictions = [_cpu_prediction(prediction, class_ids) for prediction in predictions]
            cpu_targets = [_cpu_target(target, class_ids) for target in targets]
            map_predictions = []
            for prediction, target in zip(cpu_predictions, cpu_targets):
                keep = prediction["scores"] >= config.MAP_SCORE_THRESHOLD
                map_predictions.append({key: value[keep] for key, value in prediction.items()})
                image_counts, image_ious = _match_image(
                    prediction, target, confidence_threshold, iou_threshold, class_ids
                )
                for label in class_ids:
                    for key in counts[label]:
                        counts[label][key] += image_counts[label][key]
                matched_ious.extend(image_ious)
            if cpu_targets:
                metric.update(map_predictions, cpu_targets)
                evaluated_images += len(cpu_targets)
        raw_map = metric.compute() if evaluated_images else None
    finally:
        if previous_score_threshold is not None:
            model.score_thresh = previous_score_threshold
        model.train(was_training)

    per_class_ap = {}
    if raw_map is not None:
        observed_classes = raw_map["classes"].reshape(-1).tolist()
        average_precisions = raw_map["map_per_class"].reshape(-1)
        iou_50_index = next(
            index for index, threshold in enumerate(metric.iou_thresholds) if abs(threshold - 0.5) < 1e-6
        )
        for class_index, label in enumerate(observed_classes):
            # Public extended summary: [IoU, recall, class, area, maxDet].
            # Area index 0 means all sizes; last maxDet is the COCO limit 100.
            precision_50 = raw_map["precision"][iou_50_index, :, class_index, 0, -1]
            valid_precision = precision_50[precision_50 >= 0]
            per_class_ap[int(label)] = {
                "ap": _number_or_none(average_precisions[class_index]),
                "ap_50": _number_or_none(valid_precision.mean()) if len(valid_precision) else None,
            }

    per_class = {}
    for label in class_ids:
        values = counts[label]
        per_class[str(label)] = {
            "name": config.CLASS_NAMES[label],
            **_rates(values["tp"], values["fp"], values["fn"]),
            "ignored": values["ignored"],
            **per_class_ap.get(label, {"ap": None, "ap_50": None}),
        }
    total = {key: sum(values[key] for values in counts.values()) for key in ("tp", "fp", "fn", "ignored")}
    return {
        "map": _number_or_none(raw_map["map"]) if raw_map is not None else None,
        "map_50": _number_or_none(raw_map["map_50"]) if raw_map is not None else None,
        "map_75": _number_or_none(raw_map["map_75"]) if raw_map is not None else None,
        **_rates(total["tp"], total["fp"], total["fn"]),
        "ignored": total["ignored"],
        "per_class": per_class,
        "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
        "evaluated_images": evaluated_images,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "map_score_threshold": config.MAP_SCORE_THRESHOLD,
        "coco_max_detections": list(config.COCO_MAX_DETECTIONS),
        "crowd_policy": "Exclude crowd from PR support; ignore unmatched same-class predictions with intersection/prediction-area >= IoU threshold. COCO AP receives original crowd annotations.",
        "zero_support_policy": "PR denominators of zero return 0; undefined COCO AP returns null.",
    }


def main():
    from src.dataset import load_split
    from src.model import build_model
    from src.utils import collate_fn, get_device, load_checkpoint, save_json, seed_everything

    parser = argparse.ArgumentParser(description="Evaluasi checkpoint SSDLite pada split validation atau test.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confidence", type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    args = parser.parse_args()

    seed_everything(config.RANDOM_SEED)
    device = get_device()
    dataset = load_split(args.dataset, args.split, train=False)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(model, loader, device, args.confidence, args.iou)
    output = args.output or config.METRICS_DIR / f"{args.split}_metrics.json"
    save_json(output, metrics)
    print(f"Evaluasi {args.split}: {metrics['evaluated_images']} gambar, device={device}")
    for name in ("map", "map_50", "map_75", "precision", "recall", "f1", "mean_matched_iou"):
        value = metrics[name]
        print(f"{name}: {value:.4f}" if value is not None else f"{name}: tidak terdefinisi")
    for values in metrics["per_class"].values():
        ap = f"{values['ap']:.4f}" if values["ap"] is not None else "tidak terdefinisi"
        ap_50 = f"{values['ap_50']:.4f}" if values["ap_50"] is not None else "tidak terdefinisi"
        print(
            f"{values['name']}: precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, F1={values['f1']:.4f}, "
            f"AP={ap}, AP50={ap_50}, support={values['support']}"
        )
    print(f"Metrik tersimpan: {output}")


if __name__ == "__main__":
    main()
```

## src/predict.py

Fungsi: Inferensi satu foto dengan banyak biji.

Input: Foto, model/checkpoint, confidence dan outputdir.

Output: PNG beranotasi dan JSON boxes/labels/scores/counts.

Bagian penting: Gambar RGB menjadi float [0,1]; resize/normalisasi dilakukan model. Semua deteksi yang lolos threshold dihitung, bukan satu label untuk seluruh foto. cv2 RGB/BGR disesuaikan saat menyimpan; nama/path Unicode didukung.

Kode lengkap:

```python
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
```

## run_pipeline.py

Fungsi: Mengurutkan seluruh tahap dalam satu perintah.

Input: Lokasi dataset dan opsi epoch,batch,image,resume.

Output: Audit, checkpoint, metrik test dan prediksi.

Bagian penting: Audit tiga split mendahului training, augmentasi diaktifkan hanya train. Best checkpoint dipilih validation sebelum test. Foto baru diverifikasi sebelum training; jika takdiberikan, gambar pertama test dipakai contohvisualisasi.

Kode lengkap:

```python
"""Satu perintah: audit dataset -> training -> evaluasi test -> prediksi."""

import argparse
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

import config
from src.dataset import build_datasets, validate_dataset_splits
from src.evaluate import evaluate_model
from src.model import build_model
from src.predict import predict_image
from src.train import train_model
from src.transforms import make_transforms
from src.utils import collate_fn, get_device, load_checkpoint, save_json, save_metrics_report, seed_everything


def run_pipeline(dataset_dir=None, epochs=None, batch_size=None, image_path=None,
                 resume=None, require_cuda=False, checkpoint_dir=None,
                 metrics_dir=None, prediction_dir=None, pretrained=None):
    dataset_dir = Path(dataset_dir or config.DATASET_DIR)
    checkpoint_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR)
    metrics_dir = Path(metrics_dir or config.METRICS_DIR)
    prediction_dir = Path(prediction_dir or config.PREDICTION_DIR)
    if image_path is not None:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Gambar prediksi tidak ditemukan: {image_path}")
        with Image.open(image_path) as image:
            image.verify()  # Tolak input rusak sebelum training yang panjang.
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    device = get_device(require_cuda)
    seed_everything()
    print("Tahap 1–2: periksa environment, struktur, dan anotasi semua split.")
    datasets = build_datasets(dataset_dir)
    audit = validate_dataset_splits(datasets)
    save_json(metrics_dir / "dataset_audit.json", audit)
    # Audit memakai transform deterministik; augmentasi hanya pada split train.
    datasets["train"].transforms = make_transforms(train=True)
    print("Tahap 3–6: loader, augmentasi, model pretrained, dan training train/val.")
    training = train_model(
        datasets["train"], datasets["val"], num_epochs=epochs,
        batch_size=batch_size, resume=resume, device=device,
        checkpoint_dir=checkpoint_dir, metrics_dir=metrics_dir, pretrained=pretrained,
    )
    print("Tahap 7: evaluasi best validation checkpoint pada test.")
    checkpoint = load_checkpoint(training["best_checkpoint"], device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_loader = DataLoader(
        datasets["test"], batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
    )
    test_metrics = evaluate_model(model, test_loader, device)
    save_metrics_report(metrics_dir, "test", test_metrics)
    print(f"Test mAP@0.5={test_metrics['map_50']} | mAP@0.5:0.95={test_metrics['map']}")
    print("Tahap 8: simpan visualisasi, confidence, dan jumlah tiap kelas.")
    if image_path is None:
        image_path = datasets["test"].image_paths[0]
    prediction = predict_image(model, image_path, device, output_dir=prediction_dir)
    for name, count in prediction["counts"].items():
        print(f"{name}: {count}")
    print(f"Total: {prediction['total']} biji")
    result = {
        "dataset": str(dataset_dir.resolve()), "device": str(device),
        "best_checkpoint": training["best_checkpoint"],
        "last_checkpoint": training["last_checkpoint"],
        "completed_epochs": training["completed_epochs"],
        "best_epoch": training["best_epoch"],
        "best_validation_map": training["best_validation_map"],
        "test_metrics": test_metrics, "prediction": prediction,
    }
    save_json(metrics_dir / "pipeline_summary.json", result)
    print(f"Selesai. Checkpoint: {checkpoint_dir} | metrik: {metrics_dir} | prediksi: {prediction_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=config.METRICS_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=config.PREDICTION_DIR)
    args = parser.parse_args()
    run_pipeline(
        dataset_dir=args.dataset, epochs=args.epochs, batch_size=args.batch_size,
        image_path=args.image, resume=args.resume, require_cuda=args.require_cuda,
        checkpoint_dir=args.checkpoint_dir, metrics_dir=args.metrics_dir,
        prediction_dir=args.prediction_dir,
    )


if __name__ == "__main__":
    main()
```

## scripts/check_environment.py

Fungsi: Memeriksa kemampuan software/perangkat.

Input: Bobot pretrained resmi, data sintetis, optional --require-cuda.

Output: JSON status dan exit code.

Bagian penting: Memeriksa NMS, head4kelas, loss/backward/gradienfinite, outputmodel, COCO mAP, OpenCV dan Matplotlib. --require-cuda gagal bila GPU taktersedia; ini bukan nilai performa kakao.

Kode lengkap:

```python
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
```

## scripts/smoke_test.py

Fungsi: Menguji pipeline lengkap sementara.

Input: Fixture 3 train / 1 valid / 1 test, masing-masing20 objek; optional --offline.

Output: Laporan status; semua fixture/checkpoint sementara dibersihkan.

Bagian penting: Memeriksa training, batch singleton, optimizer/RNG, evaluasi, prediksi dan restore tanpa unduhan. Membandingkan 1 epoch + resume dengan2 epoch langsung pada CPU satu thread. Mode offline dengan inisialisasi acak hanya untuk self-test, bukan default penelitian.

Kode lengkap:

```python
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
```

## scripts/run_windows.ps1

Fungsi: Instalasi dan orkestrasi lokal Windows.

Input: DatasetPath,TorchIndex,Epochs,BatchSize,PredictionImage,Resume.

Output: Venv/dependensi, checks dan pipeline.

Bagian penting: Mendukung path berspasi, Python 3.12 64-bit, wheel suffix versi yang tepat CPU/CUDA. Seluruh exit code native diperiksa. Default cu126, RTX 50 memakai cu128, CPU harus dipilih eksplisit. Environment lama tidak dihapus otomatis.

Kode lengkap:

```powershell
[CmdletBinding()]
param(
    [string]$DatasetPath,
    [ValidateRange(1, 100000)]
    [int]$Epochs,
    [ValidateRange(2, 100000)]
    [int]$BatchSize,
    [ValidateSet('cu126', 'cu128', 'cpu')]
    [string]$TorchIndex = 'cu126',
    [string]$PredictionImage,
    [string]$Resume
)

# Compatible with Windows PowerShell 5.1; no environment activation is needed.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:FailureExitCode = 1
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$StartingDirectory = (Get-Location).Path

function Get-AbsoluteInputPath {
    param([string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $StartingDirectory $Value))
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    # Inspect the native exit code explicitly, including on PowerShell 5.1.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = $null
        & $Executable @Arguments
        $nativeExitCode = $LASTEXITCODE
        if ($null -eq $nativeExitCode) {
            $nativeExitCode = 1
        }
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($nativeExitCode -ne 0) {
        $script:FailureExitCode = $nativeExitCode
        throw "Perintah gagal (exit $nativeExitCode): $Executable $($Arguments -join ' ')"
    }
}

function Test-Python312 {
    param([string]$Executable, [string[]]$PrefixArguments = @())
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = $null
        & $Executable @PrefixArguments -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)' 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

try {
    if ([string]::IsNullOrWhiteSpace($DatasetPath)) {
        $DatasetPath = Join-Path $ProjectRoot 'dataset'
    }
    else {
        $DatasetPath = Get-AbsoluteInputPath $DatasetPath
    }
    if (-not (Test-Path -LiteralPath $DatasetPath -PathType Container)) {
        throw "Folder dataset tidak ditemukan: $DatasetPath. Ekstrak dataset COCO dan gunakan -DatasetPath untuk menentukan foldernya."
    }
    if (-not [string]::IsNullOrWhiteSpace($PredictionImage)) {
        $PredictionImage = Get-AbsoluteInputPath $PredictionImage
        if (-not (Test-Path -LiteralPath $PredictionImage -PathType Leaf)) {
            throw "Gambar prediksi tidak ditemukan: $PredictionImage"
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($Resume)) {
        $Resume = Get-AbsoluteInputPath $Resume
        if (-not (Test-Path -LiteralPath $Resume -PathType Leaf)) {
            throw "Checkpoint untuk resume tidak ditemukan: $Resume"
        }
    }

    Push-Location -LiteralPath $ProjectRoot
    try {
        Write-Host '[1/5] Menyiapkan Python 3.12 64-bit dan virtual environment.'
        if (Test-Path -LiteralPath $PythonPath -PathType Leaf) {
            if (-not (Test-Python312 -Executable $PythonPath)) {
                throw '.venv yang ada bukan Python 3.12 64-bit. Simpan environment lama dengan mengganti nama folder .venv, lalu jalankan lagi. Launcher tidak menghapus environment Anda.'
            }
        }
        else {
            # Preserve an existing, incomplete environment rather than replace it.
            if (Test-Path -LiteralPath (Join-Path $ProjectRoot '.venv')) {
                throw 'Folder .venv sudah ada tetapi python.exe tidak ditemukan. Ganti nama folder tersebut setelah memeriksanya, lalu jalankan lagi.'
            }
            $launcher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
            $systemPython = Get-Command 'python.exe' -ErrorAction SilentlyContinue
            if ($null -ne $launcher -and (Test-Python312 -Executable $launcher.Source -PrefixArguments @('-3.12'))) {
                Invoke-Checked -Executable $launcher.Source -Arguments @('-3.12', '-m', 'venv', (Join-Path $ProjectRoot '.venv'))
            }
            elseif ($null -ne $systemPython -and (Test-Python312 -Executable $systemPython.Source)) {
                Invoke-Checked -Executable $systemPython.Source -Arguments @('-m', 'venv', (Join-Path $ProjectRoot '.venv'))
            }
            else {
                throw 'Python 3.12 64-bit belum ditemukan. Instal Python 3.12 dari python.org (termasuk Python Launcher), buka ulang terminal VS Code, lalu jalankan kembali.'
            }
        }

        if ($TorchIndex -ne 'cpu') {
            Write-Host "Build GPU dipilih: $TorchIndex. Gunakan cu128 untuk RTX seri 50 / Blackwell; pastikan driver NVIDIA mendukung runtime tersebut."
            $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
            if ($null -ne $nvidiaSmi) {
                Invoke-Checked -Executable $nvidiaSmi.Source -Arguments @('--query-gpu=name,driver_version', '--format=csv,noheader')
            }
            else {
                Write-Warning 'nvidia-smi tidak ditemukan pada PATH. Pemeriksaan CUDA berikutnya tetap harus lulus; periksa driver NVIDIA jika gagal.'
            }
        }

        Write-Host '[2/5] Memasang dependensi dengan versi yang dipatok.'
        Invoke-Checked -Executable $PythonPath -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
        # The explicit local-version suffix also switches an old CPU installation
        # to CUDA (or the reverse); a bare ==2.9.1 would accept either build.
        Invoke-Checked -Executable $PythonPath -Arguments @(
            '-m', 'pip', 'install',
            "torch==2.9.1+$TorchIndex", "torchvision==0.24.1+$TorchIndex",
            '--index-url', "https://download.pytorch.org/whl/$TorchIndex"
        )
        Invoke-Checked -Executable $PythonPath -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $ProjectRoot 'requirements.txt'))

        Write-Host '[3/5] Memeriksa kompatibilitas dependensi.'
        Invoke-Checked -Executable $PythonPath -Arguments @('-m', 'pip', 'check')

        Write-Host '[4/5] Memeriksa model, backward, backend metrik, dan perangkat.'
        $checkArguments = @((Join-Path $ProjectRoot 'scripts\check_environment.py'))
        if ($TorchIndex -ne 'cpu') {
            $checkArguments += '--require-cuda'
        }
        Invoke-Checked -Executable $PythonPath -Arguments $checkArguments

        Write-Host '[5/5] Menjalankan validasi dataset, training, evaluasi, dan prediksi.'
        $pipelineArguments = @((Join-Path $ProjectRoot 'run_pipeline.py'), '--dataset', $DatasetPath)
        if ($PSBoundParameters.ContainsKey('Epochs')) {
            $pipelineArguments += @('--epochs', [string]$Epochs)
        }
        if ($PSBoundParameters.ContainsKey('BatchSize')) {
            $pipelineArguments += @('--batch-size', [string]$BatchSize)
        }
        if ($TorchIndex -ne 'cpu') {
            $pipelineArguments += '--require-cuda'
        }
        if (-not [string]::IsNullOrWhiteSpace($PredictionImage)) {
            $pipelineArguments += @('--image', $PredictionImage)
        }
        if (-not [string]::IsNullOrWhiteSpace($Resume)) {
            $pipelineArguments += @('--resume', $Resume)
        }
        Invoke-Checked -Executable $PythonPath -Arguments $pipelineArguments
        Write-Host 'Selesai. Periksa checkpoint dan hasil pada direktori yang dilaporkan pipeline.'
    }
    finally {
        Pop-Location
    }
}
catch {
    [Console]::Error.WriteLine("GAGAL: $($_.Exception.Message)")
    [Console]::Error.WriteLine('Perbaiki penyebab yang ditampilkan, lalu jalankan kembali. Tidak ada fallback otomatis dari GPU ke CPU.')
    exit $script:FailureExitCode
}

exit 0
```

## RUN_WINDOWS.bat

Fungsi: Entry point satu perintah Windows.

Input: Argumen pengguna.

Output: Memanggil PowerShell dan meneruskan exit code.

Bagian penting: ExecutionPolicy Bypass berlaku hanya pada proses ini, tidak mengubah policy mesin secara permanen.

Kode lengkap:

```bat
@echo off
rem Execution policy override applies only to this PowerShell process.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_windows.ps1" %*
exit /b %errorlevel%
```

## tests/test_dataset.py

Fungsi: Pengujian COCO dan flip.

Input: Fixture sintetis kecil.

Output: Status unittest.

Bagian penting: Menguji banyak objek, kategori ID 0, target kosong, flip box, anotasi salah, files/dimensi dan duplikat lintas split.

Kode lengkap:

```python
"""Kasus penting COCO: multi-objek, flip, anotasi rusak, dan data leakage."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from src.dataset import CocoaCocoDataset, load_split, validate_dataset_splits
from src.transforms import Compose, RandomHorizontalFlip, ToTensor


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mapping = patch("config.CATEGORY_NAME_TO_LABEL", {
            "Tidak Fermentasi": 1, "Setengah Fermentasi": 2, "Full Fermentasi": 3,
        }, create=True)
        self.ignored = patch("config.IGNORED_CATEGORY_NAMES", (), create=True)
        self.mapping.start()
        self.ignored.start()
        self.addCleanup(self.mapping.stop)
        self.addCleanup(self.ignored.stop)
        self.addCleanup(self.temporary.cleanup)

    def write_split(self, name="train", coco=None, color=(60, 90, 120)):
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), color).save(directory / "sample.png")
        if coco is None:
            coco = {
                "images": [{"id": 42, "file_name": "sample.png", "width": 16, "height": 12}],
                "categories": [{"id": 0, "name": " Full_Fermentasi "}, {"id": 17, "name": "tidak-fermentasi"}],
                "annotations": [
                    {"id": 90, "image_id": 42, "category_id": 0, "bbox": [1, 2, 4, 5]},
                    {"id": 91, "image_id": 42, "category_id": 17, "bbox": [8, 1, 3, 2]},
                ],
            }
        path = directory / "_annotations.coco.json"
        path.write_text(json.dumps(coco), encoding="utf-8")
        return directory, coco

    def rewrite(self, directory, coco):
        (directory / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    def test_multi_object_and_zero_based_raw_category_id(self):
        directory, _ = self.write_split()
        dataset = CocoaCocoDataset(directory)
        image, target = dataset[0]
        self.assertEqual(image.shape, (3, 12, 16))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(0 <= image.min() <= image.max() <= 1)
        self.assertEqual(target["boxes"].tolist(), [[1, 2, 5, 7], [8, 1, 11, 3]])
        self.assertEqual(target["boxes"].dtype, torch.float32)
        self.assertEqual(target["labels"].dtype, torch.int64)
        self.assertEqual(target["labels"].tolist(), [3, 1])
        self.assertEqual(target["area"].tolist(), [20, 6])
        self.assertEqual(dataset.category_mapping, {0: 3, 17: 1})

    def test_empty_target_is_valid(self):
        directory, coco = self.write_split()
        coco["annotations"] = []
        self.rewrite(directory, coco)
        _, target = CocoaCocoDataset(directory)[0]
        self.assertEqual(target["boxes"].shape, (0, 4))
        self.assertEqual(target["labels"].shape, (0,))

    def test_flip_updates_each_box_and_preserves_area(self):
        directory, _ = self.write_split()
        dataset = CocoaCocoDataset(directory, transforms=Compose([ToTensor(), RandomHorizontalFlip(1)]))
        _, target = dataset[0]
        self.assertEqual(target["boxes"].tolist(), [[11, 2, 15, 7], [5, 1, 8, 3]])
        self.assertEqual(target["area"].tolist(), [20, 6])
        self.assertEqual(target["labels"].tolist(), [3, 1])

    def test_unannotated_parent_is_allowed_but_unknown_objects_fail(self):
        directory, coco = self.write_split()
        coco["categories"].append({"id": 99, "name": "cocoa parent"})
        self.rewrite(directory, coco)
        dataset = CocoaCocoDataset(directory)
        self.assertNotIn(99, dataset.category_mapping)
        coco["annotations"][0]["category_id"] = 99
        self.rewrite(directory, coco)
        with self.assertRaisesRegex(ValueError, "belum dipetakan"):
            CocoaCocoDataset(directory)

    def test_explicit_ignored_category(self):
        directory, coco = self.write_split()
        coco["categories"].append({"id": 99, "name": "label papan"})
        coco["annotations"][0]["category_id"] = 99
        self.rewrite(directory, coco)
        with patch("config.IGNORED_CATEGORY_NAMES", ("label papan",)):
            dataset = CocoaCocoDataset(directory)
        self.assertEqual(dataset.ignored_object_count, 1)
        self.assertEqual(dataset[0][1]["labels"].tolist(), [1])

    def test_invalid_boxes_fail_without_clipping(self):
        directory, coco = self.write_split()
        for bbox in ([-1, 2, 4, 5], [1, 2, 0, 5], [15, 2, 4, 5], [1, 2, float("nan"), 5]):
            with self.subTest(bbox=bbox):
                coco["annotations"][0]["bbox"] = bbox
                self.rewrite(directory, coco)
                with self.assertRaisesRegex(ValueError, "bbox"):
                    CocoaCocoDataset(directory)

    def test_invalid_references_and_duplicate_ids_fail(self):
        directory, coco = self.write_split()
        coco["annotations"][0]["image_id"] = 999
        self.rewrite(directory, coco)
        with self.assertRaisesRegex(ValueError, "image_id yang tidak ada"):
            CocoaCocoDataset(directory)
        coco["annotations"][0]["image_id"] = 42
        coco["annotations"].append(dict(coco["annotations"][0]))
        self.rewrite(directory, coco)
        with self.assertRaisesRegex(ValueError, "ID duplikat"):
            CocoaCocoDataset(directory)

    def test_dimensions_and_missing_file_fail(self):
        directory, coco = self.write_split()
        coco["images"][0]["width"] = 15
        self.rewrite(directory, coco)
        with self.assertRaisesRegex(ValueError, "Dimensi COCO berbeda"):
            CocoaCocoDataset(directory)
        coco["images"][0]["width"] = 16
        coco["images"][0]["file_name"] = "missing.png"
        self.rewrite(directory, coco)
        with self.assertRaisesRegex(ValueError, "File gambar tidak ditemukan"):
            CocoaCocoDataset(directory)

    def test_cross_split_copy_is_rejected(self):
        train_dir, _ = self.write_split("train")
        val_dir, _ = self.write_split("val", color=(10, 20, 30))
        train, val = CocoaCocoDataset(train_dir), CocoaCocoDataset(val_dir)
        self.assertEqual(validate_dataset_splits({"train": train, "val": val})["status"], "passed")
        shutil.copyfile(train_dir / "sample.png", val_dir / "sample.png")
        with self.assertRaisesRegex(ValueError, "Data leakage"):
            validate_dataset_splits({"train": train, "val": CocoaCocoDataset(val_dir)})

    def test_roboflow_valid_folder_with_empty_val_placeholder(self):
        self.write_split("valid")
        (self.root / "val").mkdir()
        dataset = load_split(self.root, "val")
        self.assertEqual(dataset.images_dir.name, "valid")


if __name__ == "__main__":
    unittest.main()
```

## tests/test_metrics.py

Fungsi: Pengujian kebenaran metrik.

Input: Prediksi terkontrol dan target.

Output: Status unittest.

Bagian penting: Memeriksa exactAP, urutan kelas, salah kelas, duplikat, threshold IoU, confidence rendah untukAP, target kosong dan crowd.

Kode lengkap:

```python
"""Kasus kecil yang memverifikasi deteksi, bukan accuracy klasifikasi."""

import json
import unittest

import torch

from src.evaluate import evaluate_model


def prediction(boxes=(), labels=(), scores=()):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def target(boxes=(), labels=(), crowds=None):
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return {
        "boxes": boxes_tensor,
        "labels": torch.tensor(labels, dtype=torch.int64),
        "area": (boxes_tensor[:, 2:] - boxes_tensor[:, :2]).prod(dim=1),
        "iscrowd": torch.tensor(crowds if crowds is not None else [0] * len(labels), dtype=torch.int64),
    }


class FixedDetector(torch.nn.Module):
    """Model sederhana agar tes metrik tidak mengunduh bobot atau training."""

    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions
        self.score_thresh = 0.9

    def forward(self, images):
        outputs = []
        for image in images:
            output = self.predictions[int(image[0, 0, 0])]
            keep = output["scores"] >= self.score_thresh
            outputs.append({key: value[keep].clone() for key, value in output.items()})
        return outputs


def run_evaluation(predictions, targets, confidence=0.5, iou=0.5):
    images = [torch.full((3, 16, 16), float(index)) for index in range(len(targets))]
    detector = FixedDetector(predictions)
    loader = [(images, targets)] if targets else []
    result = evaluate_model(detector, loader, torch.device("cpu"), confidence, iou)
    # No tensors, Path, NaN, Infinity or invalid COCO sentinel in saved JSON.
    json.dumps(result, allow_nan=False)
    return result, detector


class DetectionMetricsTests(unittest.TestCase):
    def test_exact_boxes_and_per_class_ap_use_observed_class_order(self):
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
        result, detector = run_evaluation(
            [prediction(boxes, [1, 3], [0.95, 0.8])], [target(boxes, [1, 3])]
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 0, 0))
        for key in ("map", "map_50", "map_75", "precision", "recall", "f1", "mean_matched_iou"):
            self.assertAlmostEqual(result[key], 1.0)
        self.assertAlmostEqual(result["per_class"]["3"]["ap_50"], 1.0)
        self.assertEqual(result["per_class"]["2"]["support"], 0)
        self.assertIsNone(result["per_class"]["2"]["ap"])
        self.assertTrue(detector.training)
        self.assertEqual(detector.score_thresh, 0.9)

    def test_wrong_class_is_false_positive_and_false_negative(self):
        box = [[0, 0, 10, 10]]
        result, _ = run_evaluation([prediction(box, [2], [0.9])], [target(box, [1])])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        self.assertEqual(result["per_class"]["1"]["fn"], 1)
        self.assertEqual(result["per_class"]["2"]["fp"], 1)
        self.assertEqual(result["per_class"]["2"]["support"], 0)
        self.assertIsNone(result["per_class"]["2"]["ap"])
        self.assertEqual(result["map_50"], 0.0)
        self.assertIsNone(result["mean_matched_iou"])

    def test_duplicate_detection_matches_one_ground_truth_only(self):
        box = [0, 0, 10, 10]
        result, _ = run_evaluation(
            [prediction([box, box], [1, 1], [0.95, 0.8])], [target([box], [1])]
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_matching_sorts_confidences_before_using_ground_truth(self):
        result, _ = run_evaluation(
            [prediction([[2, 0, 12, 10], [0, 0, 10, 10]], [1, 1], [0.8, 0.95])],
            [target([[0, 0, 10, 10], [4, 0, 14, 10]], [1, 1])],
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 0, 0))
        self.assertAlmostEqual(result["mean_matched_iou"], (1 + 2 / 3) / 2, places=6)

    def test_low_confidence_still_contributes_to_ap(self):
        box = [[0, 0, 10, 10]]
        result, detector = run_evaluation([prediction(box, [1], [0.2])], [target(box, [1])])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))
        self.assertAlmostEqual(result["map"], 1.0)
        self.assertAlmostEqual(result["map_50"], 1.0)
        self.assertEqual(detector.score_thresh, 0.9)

    def test_iou_threshold_applies_to_bounding_boxes(self):
        result, _ = run_evaluation(
            [prediction([[6, 0, 16, 10]], [1], [0.9])], [target([[0, 0, 10, 10]], [1])]
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        self.assertEqual(result["map_50"], 0.0)

    def test_empty_image_with_prediction_counts_false_positive(self):
        box = [[0, 0, 10, 10]]
        result, _ = run_evaluation(
            [prediction(box, [1], [0.95]), prediction(box, [1], [0.7]), prediction()],
            [target(box, [1]), target(), target()],
        )
        self.assertEqual(result["evaluated_images"], 3)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))

    def test_no_ground_truth_has_undefined_ap_and_zero_support(self):
        result, _ = run_evaluation([prediction([[0, 0, 10, 10]], [1], [0.9])], [target()])
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["support"]), (0, 1, 0, 0))
        self.assertEqual(result["recall"], 0.0)
        self.assertIsNone(result["map"])
        self.assertIsNone(result["per_class"]["1"]["ap_50"])

    def test_crowd_prediction_is_ignored_only_for_same_class(self):
        result, _ = run_evaluation(
            [prediction([[0, 0, 10, 10], [20, 20, 30, 30]], [1, 2], [0.9, 0.8])],
            [target([[0, 0, 100, 100]], [1], [1])],
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 0))
        self.assertEqual(result["ignored"], 1)
        self.assertEqual(result["per_class"]["1"]["support"], 0)
        self.assertIsNone(result["map"])

    def test_non_crowd_match_is_preferred_to_overlapping_crowd(self):
        result, _ = run_evaluation(
            [prediction([[0, 0, 10, 10]], [1], [0.9])],
            [target([[0, 0, 10, 10], [0, 0, 100, 100]], [1, 1], [0, 1])],
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["ignored"]), (1, 0, 0, 0))
        self.assertAlmostEqual(result["map"], 1.0)

    def test_empty_loader_serializes_undefined_ap_as_null(self):
        result, _ = run_evaluation([], [])
        self.assertEqual(result["evaluated_images"], 0)
        self.assertIsNone(result["map"])
        self.assertIsNone(result["mean_matched_iou"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 0))

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            run_evaluation([], [], confidence=float("nan"))
        with self.assertRaises(ValueError):
            run_evaluation([], [], iou=0)


if __name__ == "__main__":
    unittest.main()
```

## tests/test_model_training.py

Fungsi: Pengujian model dan training.

Input: Model4kelas/data sintetis, bobot COCO yang dicache.

Output: Status unittest.

Bagian penting: Memeriksa pembaruan SGD, mempertahankan backbone/regresi COCO, mode BatchNorm singleton, dan menolak checkpoint dengan label tertukar. Tes pretrained dapat skipped bila cache belum ada; jalankan pemeriksaan environment terlebih dahulu.

Kode lengkap:

```python
"""Pemeriksaan semantik transfer learning, SGD, BatchNorm, dan checkpoint."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

import config
from src.model import build_model
from src.train import singleton_batch_norm
from src.utils import load_checkpoint, save_checkpoint, seed_everything


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(config.CPU_NUM_THREADS)
        seed_everything()
        with patch("torch.hub.download_url_to_file", side_effect=RuntimeError("Offline model mengunduh bobot")):
            cls.model = build_model(pretrained=False)

    def test_four_class_detection_head_and_sgd_update(self):
        model = self.model
        self.assertEqual(model.head.classification_head.num_columns, 4)
        model.train()
        images = [torch.rand(3, 320, 320) for _ in range(2)]
        targets = [
            {"boxes": torch.tensor([[25., 30., 65., 80.], [100., 110., 140., 160.], [190., 200., 230., 250.]]),
             "labels": torch.tensor([1, 2, 3], dtype=torch.int64)}
            for _ in images
        ]
        head_parameters = list(model.head.classification_head.parameters())
        before = [parameter.detach().clone() for parameter in head_parameters]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        self.assertEqual(set(losses), {"classification", "bbox_regression"})
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total).item())
        total.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in head_parameters))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, head_parameters)), "SGD tidak memperbarui classifier")
        model.eval()
        with torch.inference_mode():
            prediction = model(images[:1])[0]
        self.assertEqual(set(prediction), {"boxes", "labels", "scores"})
        self.assertTrue(set(prediction["labels"].tolist()) <= {1, 2, 3})

    def test_singleton_batch_keeps_affine_gradients_and_restores_modes(self):
        model = self.model
        model.train()
        modules = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        # Pengguna dapat membekukan sebagian BN; konteks jangan mengaktifkannya.
        modules[0].eval()
        before_modes = [module.training for module in modules]
        parameter_grad_states = [parameter.requires_grad for module in modules for parameter in module.parameters()]
        with singleton_batch_norm(model, 1):
            self.assertTrue(all(not module.training for module in modules))
            self.assertEqual(parameter_grad_states, [parameter.requires_grad for module in modules for parameter in module.parameters()])
        self.assertEqual(before_modes, [module.training for module in modules])
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            with singleton_batch_norm(model, 1):
                raise RuntimeError("simulated forward failure")
        self.assertEqual(before_modes, [module.training for module in modules])

    def test_pretrained_backbone_and_box_regression_are_preserved(self):
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        cache = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        if not cache.is_file():
            self.skipTest("Bobot COCO belum ada di cache; jalankan smoke_test.py untuk validasi pretrained.")
        # Cache digunakan agar unit test tidak memerlukan koneksi internet.
        with patch("torch.hub.download_url_to_file", side_effect=RuntimeError("Test cache mengunduh bobot")):
            original = weights.get_state_dict(progress=False, check_hash=True)
            transferred = build_model(pretrained=True)
        state = transferred.state_dict()
        preserved_keys = [key for key in original if not key.startswith("head.classification_head.")]
        self.assertTrue(any(key.startswith("backbone.") for key in preserved_keys))
        self.assertTrue(any(key.startswith("head.regression_head.") for key in preserved_keys))
        for key in preserved_keys:
            self.assertTrue(torch.equal(original[key], state[key]), f"Bobot pretrained berubah: {key}")
        self.assertEqual(transferred.head.classification_head.num_columns, 4)
        self.assertTrue(any(original[key].shape != state[key].shape for key in original if key.startswith("head.classification_head.")))

    def test_checkpoint_rejects_reordered_class_names(self):
        with tempfile.TemporaryDirectory(prefix="cocoa-checkpoint-test-") as temporary:
            path = Path(temporary) / "mismatched.pth"
            names = dict(config.CLASS_NAMES)
            names[1], names[2] = names[2], names[1]
            save_checkpoint(path, {
                "model_state_dict": {}, "model_name": config.MODEL_NAME,
                "num_classes": config.NUM_CLASSES, "class_names": names,
            })
            with self.assertRaisesRegex(ValueError, "Nama/urutan kelas"):
                load_checkpoint(path, "cpu")


if __name__ == "__main__":
    unittest.main()
```

## tests/test_predict.py

Fungsi: Pengujian inferensi multibiji.

Input: Detector terkontrol berisi 20 hasil.

Output: Status unittest.

Bagian penting: Seluruh 20 deteksi dipertahankan/dihitung; RGB, grayscale, threshold, target kosong, warna dan path Unicode diperiksa.

Kode lengkap:

```python
"""Validasi inferensi multiobjek tanpa download bobot atau GPU."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from src.predict import CLASS_COLORS, predict_image


class FakeDetector(torch.nn.Module):
    def __init__(self, object_count=20):
        super().__init__()
        self.object_count = object_count
        self.last_input = None

    def forward(self, images):
        self.last_input = images[0].detach().cpu()
        boxes = torch.tensor([[2 + i, 20, 12 + i, 32] for i in range(self.object_count)], dtype=torch.float32)
        boxes = boxes.reshape(-1, 4)
        labels = torch.tensor([i % 3 + 1 for i in range(self.object_count)], dtype=torch.int64)
        scores = torch.full((self.object_count,), 0.9)
        return [{"boxes": boxes, "labels": labels, "scores": scores}]


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "biji kakao_日本.png"
        self.rgb = np.zeros((48, 96, 3), dtype=np.uint8)
        self.rgb[:] = (255, 128, 0)
        Image.fromarray(self.rgb).save(self.image_path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_twenty_objects_coordinates_counts_and_rgb_input(self):
        model = FakeDetector()
        result = predict_image(model, self.image_path, "cpu", output_dir=self.root / "hasil_日本")
        self.assertEqual(result["total"], 20)
        self.assertEqual(result["counts"], {
            config.CLASS_NAMES[1]: 7,
            config.CLASS_NAMES[2]: 7,
            config.CLASS_NAMES[3]: 6,
        })
        self.assertEqual(result["boxes"][0], [2.0, 20.0, 12.0, 32.0])
        self.assertEqual(tuple(model.last_input.shape), (3, 48, 96))
        self.assertEqual(model.last_input.dtype, torch.float32)
        self.assertAlmostEqual(float(model.last_input[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(model.last_input[1, 0, 0]), 128 / 255, places=6)
        self.assertEqual(float(model.last_input[2, 0, 0]), 0.0)
        self.assertFalse(model.training)
        saved = json.loads(Path(result["prediction_json"]).read_text(encoding="utf-8"))
        self.assertEqual(saved, result)
        with Image.open(result["annotated_image"]) as annotated:
            self.assertEqual(annotated.mode, "RGB")
            self.assertEqual(annotated.size, (96, 48))
            self.assertEqual(annotated.getpixel((95, 47)), (255, 128, 0))

    def test_confidence_threshold_filters_without_extra_object_cap(self):
        model = FakeDetector()
        result = predict_image(model, self.image_path, "cpu", 0.95, self.root / "hasil")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["boxes"], [])
        self.assertEqual(result["labels"], [])
        self.assertEqual(result["scores"], [])
        self.assertEqual(result["class_names"], [])
        self.assertTrue(all(count == 0 for count in result["counts"].values()))
        with Image.open(result["annotated_image"]) as annotated:
            np.testing.assert_array_equal(np.asarray(annotated), self.rgb)

    def test_empty_detections(self):
        result = predict_image(FakeDetector(0), self.image_path, "cpu", output_dir=self.root / "hasil")
        self.assertEqual(result["total"], 0)
        self.assertTrue(Path(result["annotated_image"]).is_file())

    def test_grayscale_is_converted_to_rgb(self):
        grayscale = self.root / "grayscale.png"
        Image.new("L", (40, 40), 200).save(grayscale)
        model = FakeDetector(0)
        predict_image(model, grayscale, "cpu", output_dir=self.root / "hasil")
        self.assertEqual(tuple(model.last_input.shape), (3, 40, 40))
        torch.testing.assert_close(model.last_input[0], model.last_input[1])

    def test_rgb_box_color_is_preserved_when_saved(self):
        result = predict_image(FakeDetector(1), self.image_path, "cpu", output_dir=self.root / "hasil")
        with Image.open(result["annotated_image"]) as annotated:
            self.assertEqual(annotated.getpixel((2, 30)), CLASS_COLORS[1])

    def test_invalid_threshold_rejected(self):
        for threshold in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                predict_image(FakeDetector(), self.image_path, "cpu", threshold, self.root / "hasil")

    def test_output_file_stays_inside_requested_directory(self):
        unusual_name = self.root / "..\\foto; nama.png"
        Image.new("RGB", (40, 40)).save(unusual_name)
        output = self.root / "outputs"
        result = predict_image(FakeDetector(0), unusual_name, "cpu", output_dir=output)
        self.assertEqual(Path(result["annotated_image"]).parent, output.resolve())
        self.assertEqual(Path(result["prediction_json"]).parent, output.resolve())


if __name__ == "__main__":
    unittest.main()
```

## .gitignore

Fungsi: Pola pelacakanGit.

Input: Path proyek.

Output: Environment/cache/data/checkpoint/hasil tidak otomatis dilacak.

Bagian penting: File .gitkeep mempertahankan direktori kosong; aturanini tidak menghapus file penelitian.

Kode lengkap:

```gitignore
.venv/
.cache/
__pycache__/
*.py[cod]
dataset/**
!dataset/**/
!dataset/**/.gitkeep
checkpoints/**
!checkpoints/.gitkeep
outputs/**
!outputs/**/
!outputs/**/.gitkeep
```

## File dokumentasi dan placeholder

README.md memuat konsep, konfigurasi dan instruksi menjalankan proyek; inputnya pembaca, outputnya panduan. VALIDASI.md memuat hasil checks yang sudah dilaksanakan, dengan batas Linux CPU/dataset sintetis. Dokumen ini memuat sumber serta penjelasannya. Isi lengkap dokumen Markdown dapat dibaca langsung pada masing-masing file.

File .gitkeep di dataset/train, dataset/val, dataset/test, checkpoints, outputs/predictions dan outputs/metrics kosong tanpa kode, input atau output runtime. Fungsinya mempertahankan struktur folder kosong.
