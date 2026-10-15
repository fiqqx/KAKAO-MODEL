# Skripsi Deteksi Biji Kakao

**Deteksi dan Klasifikasi Tingkat Fermentasi Biji Kakao Menggunakan
SSDLite-MobileNetV3-Large Berbasis Analisis Citra Bentuk dan Warna.**

Paket training dan website disediakan dalam **dua ZIP terpisah**.

| Paket | Unduh langsung |
| --- | --- |
| Training dari dataset COCO Roboflow, seluruh kode utama dalam notebook | [KAKAO_TRAINING_LENGKAP.zip](https://github.com/fiqqx/KAKAO-MODEL/raw/refs/heads/main/training/KAKAO_TRAINING_LENGKAP.zip) |
| Website Flask mandiri, tanpa ketergantungan ke folder training | [KAKAO_WEBSITE_FLASK.zip](https://github.com/fiqqx/KAKAO-MODEL/raw/refs/heads/main/website/KAKAO_WEBSITE_FLASK.zip) |

## Paket training

Ekstrak ZIP, lalu baca `BACA_DULU.txt`. Struktur utamanya:

```text
TRAINING_KAKAO/
├── KAKAO_TRAINING_DARI_NOL.ipynb
├── BACA_DULU.txt
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
├── hasil_penelitian/
├── foto_baru/
└── .cache/torch/
```

Isi masing-masing split dengan gambar dan `_annotations.coco.json` asli hasil
ekspor Roboflow. Jika ekspor memakai nama `val`, salin isinya ke `valid`;
jangan membuat `val` dan `valid` secara bersamaan.

Buka folder `TRAINING_KAKAO` di VS Code, pilih kernel Python 3.12, lalu jalankan
sel notebook berurutan melalui tombol ▶. Training baru memakai pretrained
COCO resmi dengan konfigurasi awal 100 epoch. Seluruh penjelasan tersedia
di dalam notebook; tidak membutuhkan file program `.py` dari proyek lain.

## Paket website

Ekstrak ZIP, lalu salin `best_model.pth` hasil training ke folder yang sama
dengan `app.py`. Baca `BACA_DULU.md`. Website menyediakan kamera, unggah gambar,
bounding box, hitungan tiga kelas, unduh hasil, dan riwayat browser tanpa database.

Klik dua kali `BUKA_WEBSITE_WINDOWS.bat` untuk environment CPU tersendiri.
Untuk memakai kernel GPU yang sudah tersedia, buka `JALANKAN_WEBSITE.ipynb`
dan gunakan tombol Mulai Website. Website tidak memerlukan dataset atau kode
training. Model PyTorch memakai format `.pth`; format Keras `.h5` tidak didukung.
Kamera Android/iPhone memerlukan akses HTTPS.

Dataset asli, bobot model, environment Python, dan hasil penelitian tidak
disertakan dalam kedua ZIP. Folder checkpoint, metrik, dan prediksi dibuat
otomatis setelah training dijalankan pada dataset Anda.
