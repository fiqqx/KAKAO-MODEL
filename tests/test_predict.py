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
