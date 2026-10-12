"""Regresi DataLoader: pinning opsional tanpa memerlukan perangkat CUDA."""

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import config
from src.train import _make_training_loaders


class MemoryTransferTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            (
                torch.full((3, 8, 8), float(index), dtype=torch.float32),
                {
                    "boxes": torch.tensor([[1., 1., 5., 5.]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64),
                    "image_id": torch.tensor([index], dtype=torch.int64),
                },
            )
            for index in range(5)
        ]

    def _loaders(self, device):
        return _make_training_loaders(
            self.samples, self.samples, 4, torch.device(device),
            torch.Generator().manual_seed(42),
        )

    def _assert_all_samples_preserved(self, loaders):
        for loader_index, loader in enumerate(loaders):
            self.assertIsInstance(loader, DataLoader)
            batches = list(loader)
            self.assertEqual([len(images) for images, _ in batches], [4, 1])
            image_ids = []
            for images, targets in batches:
                self.assertEqual(len(images), len(targets))
                for image, target in zip(images, targets):
                    image_id = int(target["image_id"].item())
                    image_ids.append(image_id)
                    self.assertEqual(image.dtype, torch.float32)
                    self.assertEqual(image.device.type, "cpu")
                    self.assertTrue(torch.equal(image, self.samples[image_id][0]))
                    self.assertEqual(target["labels"].dtype, torch.int64)
                    self.assertEqual(target["image_id"].dtype, torch.int64)
                    self.assertEqual(target["boxes"].dtype, torch.float32)
                    self.assertTrue(torch.equal(target["boxes"], self.samples[image_id][1]["boxes"]))
                    self.assertEqual(target["labels"].tolist(), [1])
            self.assertEqual(sorted(image_ids), list(range(5)))
            if loader_index == 1:
                self.assertEqual(image_ids, list(range(5)))

    def test_cuda_loader_without_pinning_keeps_all_samples(self):
        # Simulate an available accelerator so CPU fallback cannot hide an
        # accidental request for pinned memory. No tensor is moved to CUDA.
        with patch.object(config, "PIN_MEMORY", False), \
                patch.object(config, "NUM_WORKERS", 0), \
                patch("torch.accelerator.is_available", return_value=True), \
                patch("torch.accelerator.current_accelerator", return_value=torch.device("cuda")), \
                patch.object(torch.Tensor, "pin_memory", side_effect=RuntimeError("CUDA error: resource already mapped")) as allocator:
            self._assert_all_samples_preserved(self._loaders("cuda"))
            allocator.assert_not_called()

    def test_cuda_pinning_reproduces_injected_allocator_failure(self):
        # Positive control proves the real DataLoader reaches the failing
        # allocator when pinning is enabled; training itself is not mocked.
        with patch.object(config, "PIN_MEMORY", True), \
                patch.object(config, "NUM_WORKERS", 0), \
                patch("torch.accelerator.is_available", return_value=True), \
                patch("torch.accelerator.current_accelerator", return_value=torch.device("cuda")), \
                patch.object(torch.Tensor, "pin_memory", side_effect=RuntimeError("CUDA error: resource already mapped")) as allocator:
            for loader in self._loaders("cuda"):
                before = allocator.call_count
                with self.assertRaisesRegex(RuntimeError, "resource already mapped"):
                    next(iter(loader))
                self.assertGreater(allocator.call_count, before)

    def test_cpu_loader_skips_pinning_even_when_setting_enabled(self):
        with patch.object(config, "PIN_MEMORY", True), \
                patch.object(config, "NUM_WORKERS", 0), \
                patch("torch.accelerator.is_available", return_value=True), \
                patch("torch.accelerator.current_accelerator", return_value=torch.device("cuda")), \
                patch.object(torch.Tensor, "pin_memory", side_effect=RuntimeError("CUDA error: resource already mapped")) as allocator:
            self._assert_all_samples_preserved(self._loaders("cpu"))
            allocator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
