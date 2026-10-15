"""Perilaku API, inferensi, dan server notebook tanpa checkpoint nyata."""

import base64
from io import BytesIO
import json
from pathlib import Path
import unittest
from urllib.request import urlopen

import torch
from PIL import Image

from webapp.app import create_app
from webapp.inference_service import InferenceService
from webapp.notebook_server import NotebookServerManager


class SyntheticDetector(torch.nn.Module):
    def __init__(self, empty=False, fail=False):
        super().__init__()
        self.empty = empty
        self.fail = fail
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Kegagalan perangkat simulasi")
        if self.empty:
            return [{"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64),
                     "scores": torch.empty(0)}]
        return [{
            "boxes": torch.tensor([[1, 1, 10, 10], [11, 1, 20, 10], [21, 1, 30, 10], [1, 11, 10, 20]]),
            "labels": torch.tensor([1, 2, 3, 1]),
            "scores": torch.tensor([0.90, 0.50, 0.20, 0.19]),
        }]


def image_bytes(image_format="PNG", exif=None):
    buffer = BytesIO()
    kwargs = {"exif": exif} if exif is not None else {}
    Image.new("RGB", (40, 30), (230, 220, 200)).save(buffer, image_format, **kwargs)
    buffer.seek(0)
    return buffer


class WebBackendTests(unittest.TestCase):
    def setUp(self):
        self.options = {"WEB_CONFIDENCE_THRESHOLD": 0.20, "DEVICE": "cpu", "TESTING": True}

    def make_app(self, model=None, extra=None):
        service = InferenceService(self.options, model=model or SyntheticDetector())
        return create_app({**self.options, **(extra or {})}, inference_service=service)

    def upload(self, app, **extra):
        return app.test_client().post("/api/predict", data={"image": (image_bytes(), "foto.png"), **extra})

    def test_counts_match_filtered_boxes_and_output_image(self):
        model = SyntheticDetector()
        app = self.make_app(model)
        result = self.upload(app).get_json()
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["total"], sum(result["counts"].values()))
        self.assertEqual(result["labels"], [1, 2, 3])
        self.assertEqual(len(result["boxes"]), result["total"])
        with Image.open(BytesIO(base64.b64decode(result["image_base64"]))) as decoded:
            self.assertEqual(decoded.size, (result["width"], result["height"]))
        self.upload(app)
        self.assertEqual(model.calls, 2)
        self.assertIs(app.extensions["inference_service"].model, model)

    def test_valid_empty_is_success_with_three_zero_counts(self):
        result = self.upload(self.make_app(SyntheticDetector(empty=True)))
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json["total"], 0)
        self.assertEqual(len(result.json["counts"]), 3)
        self.assertEqual(sum(result.json["counts"].values()), 0)

    def test_failure_does_not_become_zero_result(self):
        app = self.make_app(SyntheticDetector(fail=True))
        with self.assertLogs(app.logger, level="ERROR"):
            response = self.upload(app)
        self.assertEqual(response.status_code, 503)
        self.assertIn("error", response.json)
        self.assertNotIn("total", response.json)

    def test_missing_checkpoint_has_health_diagnostic_and_no_fake_prediction(self):
        with self.assertLogs(level="WARNING"):
            app = create_app({**self.options, "CHECKPOINT_PATH": Path("/tmp/kakao-does-not-exist.pth")})
        health = app.test_client().get("/api/health")
        self.assertFalse(health.json["ready"])
        self.assertIn("Checkpoint", health.json["error"])
        self.assertEqual(self.upload(app).status_code, 503)

    def test_rotated_iphone_jpeg_applies_exif_before_inference(self):
        exif = Image.Exif()
        exif[274] = 6  # Rotasi 90 derajat: 40x30 menjadi 30x40.
        response = self.make_app().test_client().post("/api/predict", data={
            "image": (image_bytes("JPEG", exif), "iphone.jpg"),
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual((response.json["width"], response.json["height"]), (30, 40))

    def test_invalid_and_excessive_input(self):
        app = self.make_app()
        client = app.test_client()
        self.assertEqual(client.post("/api/predict").status_code, 400)
        corrupt = client.post("/api/predict", data={"image": (BytesIO(b"bad image"), "fake.png")})
        self.assertEqual(corrupt.status_code, 415)
        self.assertEqual(self.upload(app, mode="unknown").status_code, 400)
        self.assertEqual(self.upload(self.make_app(extra={"MAX_IMAGE_PIXELS": 100})).status_code, 413)
        self.assertEqual(self.upload(self.make_app(extra={"MAX_CONTENT_LENGTH": 30})).status_code, 413)

    def test_notebook_server_start_is_idempotent_and_stops(self):
        manager = NotebookServerManager()
        try:
            app = self.make_app()
            url = manager.start(app, port=0)
            self.assertTrue(manager.is_running)
            self.assertEqual(manager.start(app, port=0), url)
            with urlopen(url + "/api/health", timeout=5) as response:
                self.assertTrue(json.load(response)["ready"])
            thread = manager._thread
            dispatcher = manager._server.task_dispatcher
            port = manager._server.effective_port
            manager.stop()
            self.assertFalse(manager.is_running)
            self.assertFalse(thread.is_alive())
            self.assertFalse(dispatcher.threads)
            # Port yang sama dapat digunakan lagi setelah tombol Hentikan.
            restarted = manager.start(app, port=port)
            self.assertEqual(restarted, url)
            with urlopen(restarted + "/api/health", timeout=5) as response:
                self.assertTrue(json.load(response)["ready"])
        finally:
            manager.stop()


if __name__ == "__main__":
    unittest.main()
