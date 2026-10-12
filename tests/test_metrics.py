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
