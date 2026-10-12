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
