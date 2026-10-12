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
