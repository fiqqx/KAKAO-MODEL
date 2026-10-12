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
