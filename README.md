# Deteksi dan klasifikasi tingkat fermentasi biji kakao

**Mulai penelitian dari dataset Roboflow melalui satu notebook:** buka
[KAKAO_TRAINING_DARI_NOL.ipynb](KAKAO_TRAINING_DARI_NOL.ipynb).
Semua kode utama berada langsung di dalam sel `.ipynb`: instalasi, audit COCO,
preprocessing, augmentasi, transfer learning, training, grafik, evaluasi, dan
prediksi foto baru. Notebook ini mandiri: cukup salin file tersebut ke folder
baru bersama dataset, pilih kernel Python 3.12 di VS Code, lalu jalankan sel
berurutan dengan tombol ▶. Tidak memerlukan `src/`, `config.py`, `.bat`, atau
perintah terminal. Petunjuk lengkap terdapat di awal notebook.

Training membuat eksperimen baru menggunakan pretrained COCO resmi, tanpa
checkpoint kakao lama. Konfigurasi awal 100 epoch; setiap run memiliki folder
hasil sendiri. Test hanya dijalankan setelah pemilihan model/threshold final
dari validation. Dataset dan hasil training tidak disertakan dalam GitHub.

Untuk **website dengan model yang sudah selesai dilatih**, tersedia notebook
terpisah [KAKAO_LENGKAP.ipynb](KAKAO_LENGKAP.ipynb), dengan panduan
[README_NOTEBOOK.md](README_NOTEBOOK.md). Notebook website memerlukan folder
aplikasi proyek; notebook training mandiri di atas tidak memerlukannya.

Proyek skripsi menggunakan **SSDLite-MobileNetV3-Large** resmi TorchVision,
`torchvision.models.detection.ssdlite320_mobilenet_v3_large`.
Satu gambar dapat berisi banyak biji; setiap objek menghasilkan bounding box,
kelas fermentasi, dan confidence score. Implementasi Tahap 1 sampai 8 berada
**dalam satu folder** dan dapat dijalankan melalui satu perintah di Windows.

Model belajar pola bentuk dan warna dari citra RGB melalui jaringan saraf.
Tidak ada ekstraksi fitur warna/bentuk manual dalam baseline ini. Tiga kelas
objek ditambah background:

| Label internal | Nama |
| --- | --- |
| 0 | background, digunakan model dan tidak dihitung sebagai biji |
| 1 | Tidak Fermentasi |
| 2 | Setengah Fermentasi |
| 3 | Full Fermentasi |

Paket ini menyediakan implementasi lengkap; bobot terlatih dan hasil penelitian
akan dihasilkan setelah Anda menjalankan dataset kakao pada komputer lokal.
Dataset lokal Anda belum tersedia di cloud, sehingga angka performa penelitian
belum dapat dihitung. Pengujian dengan data sintetis hanya memeriksa program.

## Mulai di Windows NVIDIA dan VS Code

1. Ekstrak ZIP proyek ke satu folder, misalnya `D:\skripsi\cocoa_detection`.
2. Pasang **Python 3.12 64-bit**, VS Code, extension Python Microsoft, dan driver
   NVIDIA yang mendukung build CUDA yang dipilih. Pastikan `py -3.12 --version`
   dan `nvidia-smi` berjalan pada terminal PowerShell.
3. Ekstrak dataset **COCO JSON Roboflow**. Folder yang Anda berikan harus
   langsung berisi `train`, `valid` atau `val`, dan `test`.
4. Buka folder proyek di VS Code, pilih **Terminal → New Terminal**, dan jalankan
   perintah di bawah dari root proyek.

```powershell
.\RUN_WINDOWS.bat -DatasetPath "D:\dataset\kakao-coco" -PredictionImage "D:\foto\biji-baru.jpg"
```

Jika dataset disalin ke folder `dataset` proyek dan Anda ingin memakai gambar
pertama pada split test untuk contoh visualisasi, cukup:

```powershell
.\RUN_WINDOWS.bat
```

Launcher membuat `.venv`, memasang dependensi, memeriksa environment GPU,
mengaudit tiga split, melatih model, memilih checkpoint terbaik dari validation,
mengevaluasi checkpoint tersebut pada test, lalu menyimpan prediksi.
Run berhenti dengan pesan kesalahan jika prasyarat gagal; training tidak
berjalan diam-diam dengan kategori atau anotasi yang tidak valid.

Default memakai wheel PyTorch **CUDA 12.6**. Pilih build yang cocok dengan
GPU dan kemampuan driver; pada driver yang terlalu lama, perbarui driver
sebelum menjalankan build tersebut. Untuk **RTX seri 50/Blackwell**, gunakan
build CUDA 12.8 dan driver yang mendukungnya:

```powershell
.\RUN_WINDOWS.bat -TorchIndex cu128 -DatasetPath "D:\dataset\kakao-coco"
```

Untuk mencoba dengan CPU, gunakan `-TorchIndex cpu`; training biasanya lebih
lambat. Kolom `CUDA Version` pada `nvidia-smi` menunjukkan kemampuan driver.
Wheel PyTorch membawa runtime CUDA, sehingga CUDA Toolkit terpisah tidak
harus dipasang untuk menjalankan proyek ini. Pemeriksaan environment harus
menampilkan `"device": "cuda"` pada run NVIDIA.

Setelah `.venv` terbentuk, tekan **Ctrl+Shift+P → Python: Select Interpreter**
dan pilih `.venv\Scripts\python.exe`. Launcher memakai executable ini langsung,
sehingga Anda tidak perlu menjalankan aktivasi environment secara manual.
Internet diperlukan pada instalasi pertama dan unduhan pretrained weights;
cache bobot berada di `.cache/torch`.

### Mengubah lama training dan batch size

```powershell
.\RUN_WINDOWS.bat -DatasetPath "D:\dataset\kakao-coco" -Epochs 30 -BatchSize 4
```

Jika GPU kehabisan memori, coba `-BatchSize 2`. Gunakan batch minimal 2 untuk
training SSDLite. Jika batch terakhir hanya berisi satu gambar, kode menahan
statistik BatchNorm pada batch itu; gambar tetap dipakai dan gradien parameter
lain tetap dihitung. Batch size yang berubah dapat mengubah hasil eksperimen.

Untuk melanjutkan run yang sama hingga **total 50 epoch**, gunakan:

```powershell
.\RUN_WINDOWS.bat -DatasetPath "D:\dataset\kakao-coco" -Resume ".\checkpoints\last_model.pth" -Epochs 50 -BatchSize 4
```

`-Epochs 50` berarti berhenti pada epoch 50, bukan menambah 50 epoch. Gunakan
dataset, mapping kelas, dan konfigurasi yang sama saat resume. Checkpoint
menyimpan optimizer dan keadaan random untuk melanjutkan eksperimen.
Pertahankan `best_model.pth` di samping `last_model.pth` karena resume juga
memerlukan model terbaik run sebelumnya. Run baru menolak folder checkpoint
yang sudah berisi bobot, agar eksperimen lama terlindungi. Untuk eksperimen
baru, gunakan direktori hasil baru melalui opsi pipeline di bawah.

## Struktur satu folder

Nama root boleh `cocoa_detection` atau `KAKAO-MODEL`; tidak perlu membuat root
kedua di dalamnya.

```text
cocoa_detection/
├── dataset/
│   ├── train/               # gambar dan _annotations.coco.json
│   ├── val/                 # folder valid Roboflow juga didukung
│   └── test/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── transforms.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── scripts/
│   ├── check_environment.py
│   ├── smoke_test.py
│   └── run_windows.ps1
├── tests/
├── checkpoints/
├── outputs/
│   ├── predictions/
│   └── metrics/
├── config.py
├── run_pipeline.py
├── RUN_WINDOWS.bat
├── requirements.txt
├── PENJELASAN_KODE.md
└── README.md
```

Dataset, environment, cache, checkpoint, dan hasil run dikecualikan oleh
`.gitignore`. `.gitkeep` mempertahankan struktur folder kosong. File gambar
training dan bobot besar tidak perlu tersebar ke luar root proyek; dataset
juga boleh tetap berada di folder eksternal melalui `-DatasetPath`.

## Tahap 1–8 dan cara kerja

| Tahap | Implementasi | Hasil |
| --- | --- | --- |
| 1 | Instalasi dan pemeriksaan environment | Library deteksi, perangkat, forward/backward, backend mAP, dan visualisasi diperiksa |
| 2 | Audit COCO dan split | Kategori, file gambar, dimensi, box, dan salinan lintas split diperiksa |
| 3 | Custom dataset dan DataLoader | Tensor RGB serta target multi-object per gambar |
| 4 | Preprocessing dan augmentasi | Tensor dan horizontal flip yang mengubah box secara bersamaan |
| 5 | Transfer learning dan head empat kelas | Backbone serta regresi pretrained dipertahankan; klasifikasi disesuaikan |
| 6 | Training dan checkpoint | Loss, riwayat epoch, last model, dan best model dari validation |
| 7 | Evaluasi deteksi | IoU, Precision, Recall, F1, AP per kelas, mAP50, dan mAP50:95 |
| 8 | Prediksi gambar | Box, nama kelas, confidence, gambar anotasi, dan jumlah biji per kelas |

Semua tahap sudah tersedia dalam paket. Dataset train dipakai untuk update
bobot; validation dipakai untuk memilih model terbaik; test dipakai untuk
pelaporan akhir. Audit awal membaca metadata/gambar test untuk memeriksa
integritas split, tanpa memakai test untuk gradient, pemilihan epoch, atau
tuning hyperparameter.

### Tahap 2–3: COCO dan label yang benar

Struktur dataset yang diterima:

```text
kakao-coco/
├── train/
│   ├── _annotations.coco.json
│   └── ... gambar train ...
├── valid/                   # boleh bernama val; gunakan salah satu
│   ├── _annotations.coco.json
│   └── ... gambar validation ...
└── test/
    ├── _annotations.coco.json
    └── ... gambar test ...
```

COCO menyimpan box sebagai `[x, y, width, height]`. Loader mengubahnya menjadi
`[x, y, x + width, y + height]`, yakni format `xyxy` yang dibutuhkan PyTorch.
Setiap `__getitem__` mengembalikan `(image, target)`, dengan target:

```python
target = {
    "boxes": ...,     # float32, bentuk [N, 4], koordinat xyxy
    "labels": ...,    # int64, bentuk [N], label 1/2/3
    "image_id": ...,  # int64, ID gambar COCO
    "area": ...,      # float32, luas setiap box
    "iscrowd": ...,   # int64, status anotasi crowd
}
```

`N` boleh nol, satu, atau banyak. `collate_fn` membuat daftar gambar dan target
karena ukuran gambar serta jumlah biji pada setiap gambar dapat berbeda.

**ID kategori COCO tidak diasumsikan sama dengan label model.** Pemetaan dibuat
berdasarkan nama di `config.CATEGORY_NAME_TO_LABEL`. Huruf besar/kecil, spasi,
underscore, dan tanda hubung dinormalisasi; arti kelas tidak ditebak.
Default nama kelas adalah tiga nama Indonesia pada tabel awal. Jika JSON memakai
nama berbeda, periksa `categories` pada JSON dan tambahkan alias yang memang
sesuai dengan definisi kelas penelitian, misalnya:

```python
CATEGORY_NAME_TO_LABEL = {
    "Tidak Fermentasi": 1,
    "Setengah Fermentasi": 2,
    "Full Fermentasi": 3,
    "Unfermented": 1,  # hanya jika ini nama kelas yang sesuai pada ekspor Anda
}
```

Kategori beranotasi yang belum dipetakan menyebabkan audit gagal dengan nama
kategori yang harus diperiksa. Kategori induk tanpa anotasi hanya dilaporkan.
`IGNORED_CATEGORY_NAMES` tersedia untuk kategori yang sengaja berada di luar
objek penelitian; jangan menggunakannya untuk menyembunyikan kesalahan mapping.

Audit memeriksa referensi ID, file gambar, dimensi COCO terhadap gambar asli,
box positif dalam batas gambar, serta duplikasi path dan hash SHA-256 isi file
antar split. Box rusak tidak dipotong atau dibuang diam-diam. Pemeriksaan hash
hanya mendeteksi file identik: foto hampir sama, crop, atau JPEG yang dikodekan
ulang mungkin tidak terdeteksi. Pisahkan juga foto berdasarkan sumber/batch
biji agar citra yang sangat terkait tidak bocor ke split lain.
Periksa juga jumlah objek setiap kelas pada laporan audit. Jika suatu kelas
tidak muncul di validation atau test, AP kelas itu tidak dapat dinilai;
laporkan keterbatasan split tersebut ketika menyajikan hasil penelitian.
Anotasi training harus per biji dengan `iscrowd=0`; training menolak anotasi
crowd karena loss SSD tidak mempunyai penanganan crowd khusus. Validation/test
tetap mendukung crowd dengan kebijakan evaluasi yang dijelaskan di bawah.

### Tahap 4–5: preprocessing dan transfer learning

Gambar dibaca sebagai RGB dan diubah menjadi tensor float32 `[C, H, W]` dalam
rentang `[0, 1]`. TorchVision SSDLite melakukan resize ke **320 × 320**,
normalisasi, dan penyesuaian box di dalam model. Prediksi dikembalikan ke
koordinat gambar asli. Tidak perlu resize atau normalisasi kedua kali di loader.

Training memakai horizontal flip dengan peluang 0.5. Jika lebar gambar `W`,
koordinat baru adalah `xmin = W - xmax_lama` dan `xmax = W - xmin_lama`.
Validation/test hanya memakai konversi tensor. Warna biji dipertahankan agar
augmentasi tidak mengubah ciri yang dipakai untuk menilai fermentasi.

Di `src/model.py`, model pertama dimuat dengan
`SSDLite320_MobileNet_V3_Large_Weights.DEFAULT`. Detector pretrained COCO
memiliki head klasifikasi untuk 91 label TorchVision. Head tersebut kemudian
diganti dengan `SSDLiteClassificationHead` yang memakai jumlah kanal fitur
dan anchor model yang sama, tetapi **`num_classes = 4`**.
Backbone MobileNetV3-Large dan head regresi box tetap memakai bobot pretrained.
Head klasifikasi baru diinisialisasi untuk empat label dan dilatih bersama model.

Ini diperlukan karena memaksakan `num_classes=4` saat langsung memuat weights
COCO tidak cocok dengan dimensi classification head pretrained. Saat memuat
checkpoint kakao untuk evaluasi/prediksi, arsitektur empat kelas dibuat tanpa
unduhan pretrained baru, lalu `model_state_dict` dari checkpoint dimuat.

### Tahap 6: training, loss, dan checkpoint

Perangkat dipilih dengan:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Pada mode training, `model(images, targets)` mengembalikan loss deteksi bawaan
TorchVision: **`classification`** untuk prediksi label objek/background dan
**`bbox_regression`** untuk posisi box. Training menjumlahkan keduanya,
menjalankan backward, lalu memperbarui parameter melalui SGD. Loss klasifikasi
gambar tunggal tidak digunakan.

Checkpoint berikut disimpan:

| File | Pemakaian |
| --- | --- |
| `checkpoints/last_model.pth` | Model pada epoch terakhir; dipakai untuk resume |
| `checkpoints/best_model.pth` | Model dengan validation mAP@0.5:0.95 terbaik; dipakai untuk test/prediksi |

Checkpoint berisi `model_state_dict`, `optimizer_state_dict`, epoch,
training loss, metrik validation, konfigurasi eksperimen, riwayat, dan keadaan
random. Grafik serta tabel loss/validation disimpan pada `outputs/metrics`.
Test tidak dipakai untuk menentukan best checkpoint.

### Konfigurasi baseline dan alasan

Nilai utama ada di `config.py`; opsi CLI mengubah nilai run tertentu.
Angka berikut merupakan baseline awal yang perlu diuji memakai validation.

| Parameter | Default | Alasan |
| --- | --- | --- |
| Learning rate | 0.001 | Langkah awal kecil untuk fine-tuning pretrained detector |
| Optimizer / momentum | SGD / 0.9 | Optimizer sederhana; momentum meredam variasi pembaruan |
| Weight decay | 0.0005 | Regularisasi awal terhadap bobot besar |
| Batch size | 4 | Awal moderat untuk memori GPU dan BatchNorm |
| Epoch | 30 | Anggaran awal; model terbaik dipilih dari validation |
| Confidence operasional | 0.5 | Titik awal untuk visualisasi dan Precision/Recall/F1 |
| IoU pencocokan | 0.5 | Minimal overlap agar prediksi dapat dihitung benar |
| IoU NMS | 0.55 | Default SSDLite untuk menekan prediksi tumpang tindih |
| Skor minimum mAP | 0.001 | Mempertahankan prediksi skor rendah untuk kurva precision–recall |
| Maksimum deteksi model | 300 | Default SSD untuk gambar dengan banyak objek |
| Batas evaluasi COCO | 1, 10, 100 | Protokol standar pelaporan COCO mAP |
| Horizontal flip | 0.5 | Variasi posisi tanpa perubahan warna |
| Random seed | 42 | Nilai tetap untuk membandingkan eksperimen |
| DataLoader workers | 0 | Awal sederhana dan mudah ditelusuri pada Windows |
| Pinned memory | False | Menghindari kegagalan pemetaan memori pada Windows; model tetap menggunakan CUDA |

Parameter tersebut bukan hasil tuning dataset Anda. Tuning learning rate,
epoch, dan threshold memakai validation, kemudian pengaturan dikunci sebelum
pelaporan test. Seed diterapkan pada Python, NumPy, PyTorch, dan CUDA.
Catat versi library, GPU, batch size, dan konfigurasi; seed yang sama tidak
menjamin hasil bit-identik lintas perangkat atau versi library.

### Tahap 7: evaluasi object detection

IoU adalah luas irisan dua box dibagi luas gabungannya. Untuk Precision,
Recall, dan F1, prediksi difilter dengan confidence operasional, lalu diurutkan
menurut confidence pada setiap gambar dan kelas.

- **TP:** kelas benar dan IoU minimal threshold terhadap satu anotasi yang
  belum dipasangkan. Satu anotasi hanya dapat menjadi pasangan satu prediksi.
- **FP:** prediksi tidak mendapat pasangan; termasuk box duplikat, lokasi
  meleset, atau label salah.
- **FN:** anotasi yang tidak mendapat pasangan prediksi benar.

Contoh: satu biji `Full Fermentasi` diprediksi `Tidak Fermentasi`, walaupun
box tepat, menghasilkan FP untuk `Tidak Fermentasi` dan FN untuk
`Full Fermentasi`. Dua prediksi untuk satu biji hanya dapat menghasilkan
satu TP; prediksi tambahan menjadi FP.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

Metrik keseluruhan menggunakan jumlah TP/FP/FN seluruh kelas (micro), dan
laporan juga memberikan hasil setiap kelas. Mean matched IoU hanya dihitung
dari pasangan TP pada threshold operasional; metrik ini bukan pengganti mAP.

`torchmetrics.detection.MeanAveragePrecision` memakai backend `pycocotools`
untuk AP COCO dan metrik per kelas. AP merangkum kurva precision–recall pada
beragam confidence. `mAP@0.5` menggunakan IoU 0.5; `mAP@0.5:0.95`
merata-ratakan IoU 0.50, 0.55, ..., 0.95 dan kelas yang terdefinisi.
AP input memakai threshold skor rendah 0.001, terpisah dari confidence 0.5
untuk PR/F1 dan visualisasi. Memfilter semua prediksi pada 0.5 sebelum
menghitung mAP akan memangkas kurva dan mengubah hasil.

`iscrowd` diteruskan ke evaluasi COCO. Pada PR/F1, anotasi crowd dikeluarkan
dari support/FN; prediksi kelas sama yang cukup tertutup area crowd dan tidak
mendapat pasangan biasa diabaikan. Overlap crowd memakai luas irisan dibagi
luas prediksi. Kelas tanpa anotasi non-crowd memiliki support nol; PR/F1
menggunakan nol jika pembagi nol, sedangkan AP tak terdefinisi disimpan sebagai
`null`, bukan dianggap AP nol. Laporan mencantumkan kebijakan tersebut.
Nilai metrik ditulis dalam rentang 0–1; kalikan 100 jika menyajikannya sebagai
persentase pada skripsi.

### Tahap 8: prediksi banyak biji

`src/predict.py` membaca satu gambar RGB, memuat best checkpoint,
menjalankan model dalam mode evaluasi, memfilter confidence, kemudian
menggambar box, label kelas, dan confidence. Gambar hasil beserta data prediksi
dan jumlah biji setiap kelas disimpan pada `outputs/predictions`.
Output model tetap berupa `boxes`, `labels`, dan `scores` per objek.
Jumlah biji yang dilaporkan adalah jumlah **deteksi yang lolos threshold**;
hasil ini bergantung pada kualitas model dan ambang confidence.

Jika `-PredictionImage` tidak diberikan, pipeline memakai gambar pertama
test sebagai contoh visualisasi setelah evaluasi. Untuk menguji foto baru,
berikan path gambar Anda. Hasil berupa file gambar dan data; tidak memerlukan
jendela desktop atau server.

## Menjalankan komponen secara terpisah

Setelah dependensi terpasang, jalankan dari root proyek menggunakan interpreter
`.venv`. Langkah terpisah berguna untuk memeriksa dataset atau memprediksi
lagi tanpa mengulang training.

```powershell
# Periksa environment GPU
.\.venv\Scripts\python.exe scripts/check_environment.py --require-cuda

# Audit tiga split dan mapping kategori
.\.venv\Scripts\python.exe -m src.dataset --dataset "D:\dataset\kakao-coco"

# Training memakai train dan validation
.\.venv\Scripts\python.exe -m src.train --dataset "D:\dataset\kakao-coco" --epochs 30 --batch-size 4

# Evaluasi best checkpoint pada test
.\.venv\Scripts\python.exe -m src.evaluate --dataset "D:\dataset\kakao-coco" --checkpoint checkpoints/best_model.pth --split test

# Prediksi gambar baru, tanpa training ulang
.\.venv\Scripts\python.exe -m src.predict --image "D:\foto\biji-baru.jpg" --checkpoint checkpoints/best_model.pth
```

Resume training saja:

```powershell
.\.venv\Scripts\python.exe -m src.train --dataset "D:\dataset\kakao-coco" --epochs 50 --batch-size 4 --resume checkpoints/last_model.pth
```

Untuk pipeline dengan environment yang sudah terpasang:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --dataset "D:\dataset\kakao-coco" --epochs 30 --batch-size 4 --image "D:\foto\biji-baru.jpg" --require-cuda
```

Untuk eksperimen baru tanpa menimpa checkpoint sebelumnya, tentukan folder
hasil baru. Ini memakai environment yang sudah terpasang:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --dataset "D:\dataset\kakao-coco" --checkpoint-dir checkpoints/eksperimen_2 --metrics-dir outputs/metrics/eksperimen_2 --prediction-dir outputs/predictions/eksperimen_2 --require-cuda
```

### Instalasi manual jika diperlukan

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Ganti `cu126` dengan `cu128` untuk RTX seri 50 atau `cpu` untuk CPU.
PyTorch 2.9.1 dan TorchVision 0.24.1 dipasangkan agar operasi deteksi C++
kompatibel. Dependensi lain: NumPy untuk array/seed, Pillow untuk gambar RGB,
OpenCV headless untuk menyimpan visualisasi, Matplotlib untuk grafik,
TorchMetrics untuk mAP, dan pycocotools untuk backend COCO.

### Linux CPU

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/check_environment.py
.venv/bin/python run_pipeline.py --dataset /path/to/kakao-coco --epochs 30 --batch-size 4
```

## Fungsi setiap file, input, dan output

Kode lengkap dan penjelasan bagian penting setiap file tersedia di
[**PENJELASAN_KODE.md**](PENJELASAN_KODE.md). Baca berurutan sesuai tabel agar
mudah menjelaskan implementasi saat sidang.

| File | Fungsi dan bagian penting | Input | Output |
| --- | --- | --- | --- |
| `config.py` | Path, nama/mapping kelas, dan hyperparameter dalam satu tempat | Pengaturan peneliti | Konstanta untuk seluruh pipeline |
| `src/dataset.py` | Custom Dataset COCO, mapping nama kelas, box xywh→xyxy, audit split | Folder gambar dan JSON COCO | `(image, target)` serta laporan audit |
| `src/transforms.py` | Tensor RGB dan horizontal flip dengan perubahan box | Gambar dan target | Gambar/target yang tetap sinkron |
| `src/model.py` | Pretrained SSDLite dan penggantian classification head menjadi empat kelas | Konfigurasi model | Detector TorchVision |
| `src/train.py` | DataLoader train/val, loss bawaan, SGD, validation dan resume | Dataset, konfigurasi, opsional checkpoint | Best/last checkpoint dan riwayat training |
| `src/evaluate.py` | IoU matching per kelas dan backend COCO AP | Model/checkpoint dan split val/test | Metrik keseluruhan/per kelas dalam JSON |
| `src/predict.py` | Inferensi, filter skor, visualisasi box, dan jumlah biji | Gambar, checkpoint, confidence | Gambar anotasi dan data prediksi |
| `src/utils.py` | Seed, perangkat, collate, JSON, checkpoint, dan grafik | Data/helper parameter | Utilitas bersama dan file hasil |
| `run_pipeline.py` | Mengurutkan audit→train→test→predict | Path dataset dan opsi run | Hasil seluruh workflow pada folder proyek |
| `scripts/check_environment.py` | Uji NMS, pretrained model, loss/backward, mAP, OpenCV/Matplotlib | Data sintetis, opsional `--require-cuda` | Laporan readiness; exit nonzero jika gagal |
| `scripts/smoke_test.py` | Training/evaluasi/prediksi sementara, restore offline dan pembandingan resume | Fixture COCO sintetis; opsional `--offline` | Status pipeline; fixture dan checkpoint sementara dihapus |
| `scripts/run_windows.ps1` | Instalasi `.venv` dan orkestrasi run Windows | Parameter launcher | Environment terpasang dan pipeline dijalankan |
| `RUN_WINDOWS.bat` | Memanggil script PowerShell dari root proyek | Argumen pengguna | Satu perintah untuk menjalankan workflow |
| `requirements.txt` | Versi dependensi yang digunakan | Dibaca installer Python | Library yang konsisten |
| `tests/test_*.py` | Verifikasi perilaku loader, transformasi, model, metrik, prediksi/training | Fixture sintetis | Hasil pengujian, bukan performa kakao |
| `tests/test_memory_transfer.py` | Regresi DataLoader dengan kegagalan pinned-memory yang disimulasikan | Tensor CPU dan ketersediaan CUDA simulasi | Semua sampel dipertahankan tanpa memanggil allocator bila pinning dimatikan |
| `.gitignore` / `.gitkeep` | Mengecualikan data/hasil besar dan mempertahankan folder | Pola path | Struktur proyek untuk Git |
| `README.md` | Cara menjalankan, konsep model, dan batas hasil | Dibaca pengguna | Panduan kerja |
| `PENJELASAN_KODE.md` | Fungsi, source, input/output, dan bagian penting kode | Source implementasi | Bahan belajar mahasiswa |

## Membaca hasil dan mengatasi masalah

Hasil utama terdapat dalam tiga lokasi:

- `checkpoints`: last model untuk resume dan best model untuk evaluasi/prediksi.
- `outputs/metrics`: audit dataset, riwayat/grafik training, dan metrik test.
- `outputs/predictions`: gambar dengan box/label/confidence serta hasil hitung.

| File hasil | Isi |
| --- | --- |
| `outputs/metrics/dataset_audit.json` | Jumlah gambar/objek, mapping kategori, dan pemeriksaan split |
| `outputs/metrics/train_val_audit.json` | Audit train/validation yang dipakai training |
| `outputs/metrics/training_history.json` dan `.csv` | Loss total/komponen, learning rate, dan validation mAP setiap epoch |
| `outputs/metrics/training_curves.png` | Grafik training loss dan validation mAP |
| `outputs/metrics/validation_metrics.json` dan `validation_per_class.csv` | Metrik validation pada epoch terakhir |
| `outputs/metrics/best_validation_metrics.json` | Metrik validation saat best checkpoint dipilih |
| `outputs/metrics/test_metrics.json` | Metrik keseluruhan serta per kelas pada test |
| `outputs/metrics/test_per_class.csv` | Tabel kelas, support, TP/FP/FN, Precision, Recall, F1, AP, dan AP50 |
| `outputs/metrics/pipeline_summary.json` | Ringkasan run, checkpoint terpilih, test, dan prediksi |
| `outputs/predictions/<nama>_annotated.png` | Gambar dengan box, label, dan confidence |
| `outputs/predictions/<nama>_predictions.json` | Box xyxy, label, skor, nama kelas, dan jumlah biji |

Prediksi nama gambar yang sama ke folder yang sama memperbarui file hasil
tersebut. Gunakan nama foto yang berbeda atau folder output lain untuk
menyimpan beberapa versi prediksi.

Jika audit gagal karena kategori, periksa nama kelas pada JSON dan mapping
`config.py`. Jika gambar/box tidak valid, perbaiki atau ekspor ulang anotasi
sumber. Jika salinan antar split ditemukan, pisahkan ulang dataset sebelum
melatih model. Jika CUDA tidak tersedia, periksa driver, pilihan indeks wheel,
dan interpreter VS Code, lalu ulangi self-test. Jika prediksi belum menampilkan
biji, periksa kualitas checkpoint dan threshold; jangan menafsirkan tidak adanya
deteksi sebagai bukti bahwa tidak ada biji.

Jika training berhenti dengan `CUDA error: resource already mapped` saat
`DataLoader` memanggil `.pin_memory()`, gunakan `PIN_MEMORY = False` di
`config.py`. Pengaturan ini berlaku pada kedua loader dan tidak mengubah
perangkat model: forward/backward tetap menggunakan CUDA. Pinned memory
hanya optimasi transfer dari CPU; flag ini tidak menjadi bagian syarat
hyperparameter resume, sehingga checkpoint sebelumnya tetap kompatibel.
Restart proses Python setelah error, lalu lanjutkan checkpoint run yang sama
jika sudah ada `last_model.pth`, atau mulai pada folder eksperimen baru.
Peringatan TorchMetrics tentang lebih dari 100 deteksi adalah pesan terpisah
tentang batas pelaporan COCO, bukan error pemetaan memori ini.

Pengujian software:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Pemeriksaan environment memakai bobot COCO pretrained dengan classification
head empat kelas. Pengujian pipeline sintetis juga memeriksa checkpoint dan
resume, dengan 20 objek per gambar. Jalankan bila ingin memverifikasi software
secara lengkap sebelum eksperimen:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

Uji resume ini memakai satu thread CPU agar pembandingan bobot dapat identik;
training normal tetap mengikuti `CPU_NUM_THREADS` pada `config.py`. Operasi
paralel/lintas perangkat bisa menghasilkan selisih numerik kecil meskipun
seed sama. Keduanya berbeda
dari eksperimen kakao. Keberhasilan Windows CUDA dan hasil penelitian Anda
harus dibuktikan oleh run lokal pada dataset nyata. Simpan audit, konfigurasi,
riwayat, checkpoint, metrik test, dan contoh prediksi sebagai artefak skripsi.

Lihat [VALIDASI.md](VALIDASI.md) untuk hasil pemeriksaan yang sudah dieksekusi
di cloud. Windows NVIDIA dan dataset lokal Anda tetap perlu dijalankan lokal.
