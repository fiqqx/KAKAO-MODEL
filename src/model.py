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
