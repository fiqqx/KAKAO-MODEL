"""Flask inferensi tanpa database dan tanpa penyimpanan gambar permanen."""

from __future__ import annotations

from io import BytesIO
import warnings

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from webapp.inference_service import InferenceService, InferenceUnavailable
from webapp.web_config import WebConfig

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    # JPEG/PNG/WebP tetap berfungsi; HEIC meminta decoder jika belum terpasang.
    pillow_heif = None


def create_app(config_override=None, inference_service=None):
    app = Flask(__name__)
    app.config.from_mapping(WebConfig.from_env())
    if config_override:
        app.config.update(config_override)
    app.json.ensure_ascii = False
    service = inference_service if inference_service is not None else InferenceService(app.config)
    app.extensions["inference_service"] = service
    if service.ready:
        app.logger.info("Model website siap. Perangkat: %s", service.device)
    else:
        app.logger.warning("%s", service.error)

    @app.get("/")
    def home():
        return render_template("home.html")

    @app.get("/upload")
    def upload():
        return render_template("upload.html")

    @app.get("/camera")
    def camera():
        return render_template("camera.html")

    @app.get("/history")
    def history():
        return render_template("history.html")

    @app.get("/api/health")
    def health():
        return jsonify(ready=bool(service.ready), device=str(service.device), error=service.error)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        limit = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
        return jsonify(error=f"Gambar terlalu besar. Ukuran maksimal {limit:g} MB."), 413

    @app.post("/api/predict")
    def predict():
        if not service.ready:
            return jsonify(error=service.error or "Model belum siap. Periksa lokasi checkpoint."), 503
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify(error="Pilih atau kirim gambar terlebih dahulu."), 400
        if request.form.get("mode", "upload") not in {"upload", "camera"}:
            return jsonify(error="Mode deteksi harus upload atau camera."), 400
        try:
            raw = uploaded.read()
            if not raw:
                return jsonify(error="Berkas gambar kosong."), 400
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(raw)) as opened:
                    if opened.format not in {"JPEG", "PNG", "WEBP", "HEIF", "HEIC"}:
                        return jsonify(error="Gunakan gambar JPEG, PNG, WebP, atau HEIC/HEIF."), 415
                    if opened.width * opened.height > app.config["MAX_IMAGE_PIXELS"]:
                        return jsonify(error="Resolusi gambar terlalu besar. Kecilkan gambar terlebih dahulu."), 413
                    opened.load()
                    image = ImageOps.exif_transpose(opened).convert("RGB")
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            return jsonify(error="Resolusi gambar terlalu besar. Kecilkan gambar terlebih dahulu."), 413
        except UnidentifiedImageError:
            return jsonify(error="Berkas bukan gambar yang didukung atau rusak. Untuk HEIC, pasang pillow-heif."), 415
        except (OSError, ValueError, SyntaxError):
            return jsonify(error="Gambar tidak dapat dibaca. Pilih berkas gambar yang valid."), 400
        try:
            return jsonify(service.predict(image))
        except InferenceUnavailable as exc:
            return jsonify(error=str(exc)), 503
        except Exception:
            app.logger.exception("Inferensi gambar gagal")
            return jsonify(error="Deteksi gagal diproses. Periksa log notebook/server, lalu coba kembali."), 503

    return app


if __name__ == "__main__":
    from waitress import serve
    serve(create_app(), host="127.0.0.1", port=5000, threads=4)
