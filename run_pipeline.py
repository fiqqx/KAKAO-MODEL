"""Satu perintah: audit dataset -> training -> evaluasi test -> prediksi."""

import argparse
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

import config
from src.dataset import build_datasets, validate_dataset_splits
from src.evaluate import evaluate_model
from src.model import build_model
from src.predict import predict_image
from src.train import train_model
from src.transforms import make_transforms
from src.utils import collate_fn, get_device, load_checkpoint, save_json, save_metrics_report, seed_everything


def run_pipeline(dataset_dir=None, epochs=None, batch_size=None, image_path=None,
                 resume=None, require_cuda=False, checkpoint_dir=None,
                 metrics_dir=None, prediction_dir=None, pretrained=None):
    dataset_dir = Path(dataset_dir or config.DATASET_DIR)
    checkpoint_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR)
    metrics_dir = Path(metrics_dir or config.METRICS_DIR)
    prediction_dir = Path(prediction_dir or config.PREDICTION_DIR)
    if image_path is not None:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Gambar prediksi tidak ditemukan: {image_path}")
        with Image.open(image_path) as image:
            image.verify()  # Tolak input rusak sebelum training yang panjang.
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    device = get_device(require_cuda)
    seed_everything()
    print("Tahap 1–2: periksa environment, struktur, dan anotasi semua split.")
    datasets = build_datasets(dataset_dir)
    audit = validate_dataset_splits(datasets)
    save_json(metrics_dir / "dataset_audit.json", audit)
    # Audit memakai transform deterministik; augmentasi hanya pada split train.
    datasets["train"].transforms = make_transforms(train=True)
    print("Tahap 3–6: loader, augmentasi, model pretrained, dan training train/val.")
    training = train_model(
        datasets["train"], datasets["val"], num_epochs=epochs,
        batch_size=batch_size, resume=resume, device=device,
        checkpoint_dir=checkpoint_dir, metrics_dir=metrics_dir, pretrained=pretrained,
    )
    print("Tahap 7: evaluasi best validation checkpoint pada test.")
    checkpoint = load_checkpoint(training["best_checkpoint"], device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_loader = DataLoader(
        datasets["test"], batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
    )
    test_metrics = evaluate_model(model, test_loader, device)
    save_metrics_report(metrics_dir, "test", test_metrics)
    print(f"Test mAP@0.5={test_metrics['map_50']} | mAP@0.5:0.95={test_metrics['map']}")
    print("Tahap 8: simpan visualisasi, confidence, dan jumlah tiap kelas.")
    if image_path is None:
        image_path = datasets["test"].image_paths[0]
    prediction = predict_image(model, image_path, device, output_dir=prediction_dir)
    for name, count in prediction["counts"].items():
        print(f"{name}: {count}")
    print(f"Total: {prediction['total']} biji")
    result = {
        "dataset": str(dataset_dir.resolve()), "device": str(device),
        "best_checkpoint": training["best_checkpoint"],
        "last_checkpoint": training["last_checkpoint"],
        "completed_epochs": training["completed_epochs"],
        "best_epoch": training["best_epoch"],
        "best_validation_map": training["best_validation_map"],
        "test_metrics": test_metrics, "prediction": prediction,
    }
    save_json(metrics_dir / "pipeline_summary.json", result)
    print(f"Selesai. Checkpoint: {checkpoint_dir} | metrik: {metrics_dir} | prediksi: {prediction_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=config.METRICS_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=config.PREDICTION_DIR)
    args = parser.parse_args()
    run_pipeline(
        dataset_dir=args.dataset, epochs=args.epochs, batch_size=args.batch_size,
        image_path=args.image, resume=args.resume, require_cuda=args.require_cuda,
        checkpoint_dir=args.checkpoint_dir, metrics_dir=args.metrics_dir,
        prediction_dir=args.prediction_dir,
    )


if __name__ == "__main__":
    main()
