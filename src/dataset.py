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
