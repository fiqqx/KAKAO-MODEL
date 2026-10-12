"""Konfigurasi awal eksperimen; nilai ini belum merupakan hasil tuning."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# Ekspor COCO Roboflow biasanya memakai nama ini. Verifikasi pada Tahap 2.
TRAIN_ANNOTATIONS = TRAIN_DIR / "_annotations.coco.json"
VAL_ANNOTATIONS = VAL_DIR / "_annotations.coco.json"
TEST_ANNOTATIONS = TEST_DIR / "_annotations.coco.json"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
PREDICTION_DIR = PROJECT_DIR / "outputs" / "predictions"
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"

CLASS_NAMES = {
    0: "background",
    1: "Tidak Fermentasi",
    2: "Setengah Fermentasi",
    3: "Full Fermentasi",
}
NUM_CLASSES = len(CLASS_NAMES)  # Tiga kelas objek + background = 4.

# Inilah label internal model, bukan asumsi category_id dalam JSON COCO.
# Pemetaan category_id COCO dibuat loader berdasarkan nama, bukan urutan ID.
CATEGORY_NAME_TO_LABEL = {
    "Tidak Fermentasi": 1,
    "Setengah Fermentasi": 2,
    "Full Fermentasi": 3,
}
# Tambahkan alias sesuai nama asli categories dalam JSON Anda, misalnya
# "unfermented": 1, jika dan hanya jika maknanya memang Tidak Fermentasi.
IGNORED_CATEGORY_NAMES = ()  # Jangan abaikan kelas penelitian.
MODEL_NAME = "ssdlite320_mobilenet_v3_large"
USE_PRETRAINED = True

# Titik awal konservatif untuk fine-tuning SGD; tuning hanya memakai val.
OPTIMIZER = "SGD"
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 4  # Batch > 1 membantu BatchNorm; sesuaikan dengan memori.
NUM_EPOCHS = 30  # Anggaran awal, bukan jaminan konvergensi.
NUM_WORKERS = 0  # Mudah ditelusuri dan aman pada CPU/cloud/notebook.
CPU_NUM_THREADS = 4  # Hindari thread berlebihan pada mesin CPU.
DETERMINISTIC = True  # Mengurangi sumber variasi; lintas perangkat tetap bisa berbeda.

# Threshold operasional Precision/Recall/F1 dan visualisasi.
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.55  # Default SSDLite TorchVision; berbeda dari IoU evaluasi.

# mAP harus memakai prediksi dengan ambang rendah agar kurva PR tetap utuh.
MAP_SCORE_THRESHOLD = 0.001  # Default postprocessing TorchVision SSD.
MAX_DETECTIONS_PER_IMAGE = 300  # Default SSD; dapat menangani banyak biji.
COCO_MAX_DETECTIONS = (1, 10, 100)  # Standar pelaporan COCO mAP.

HORIZONTAL_FLIP_PROBABILITY = 0.5  # Posisi berubah; warna tetap dipertahankan.
RANDOM_SEED = 42  # Nilai tetap untuk membandingkan eksperimen, bukan tuning.
