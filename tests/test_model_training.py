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
