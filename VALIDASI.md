# Hasil validasi software

Verifikasi dilakukan pada 12 Oktober 2026, pada Linux x86_64 CPU, Python
3.12.14, PyTorch 2.9.1+cpu, dan TorchVision 0.24.1+cpu.

| Pemeriksaan | Hasil |
| --- | --- |
| Instalasi ulang script cloud dan kompatibilitas dependency | Lulus, exit 0 |
| NMS, pretrained SSDLite dengan head 4 kelas, loss dan backward | Lulus |
| Keluaran boxes/labels/scores, OpenCV dan Matplotlib | Lulus |
| Unit test dataset, augmentasi, metrik, prediksi dan model | 33 lulus; 0 gagal; 0 skipped |
| Pipeline CLI: audit, 1 epoch training, test dan prediksi | Lulus, exit 0 |
| Smoke pretrained: 3 train, 1 valid, 1 test, 20 objek/gambar | Lulus |
| Semua gambar train termasuk batch akhir satu gambar | Diproses |
| Best/last checkpoint, optimizer momentum, dan RNG | Disimpan dan direstore |
| Restore model tanpa unduhan bobot tambahan | Lulus |
| Resume ke epoch 2 dibanding dua epoch tanpa interupsi | History dan bobot identik pada CPU satu thread |
| Smoke offline dengan inisialisasi acak tanpa unduhan | Lulus; hanya self-test software |
| JSON/CSV per kelas, grafik training, PNG/JSON prediksi | Terbentuk dan diverifikasi |

Fixture sintetis dan checkpoint pengujian dibuat dalam direktori sementara
lalu dihapus. Paket ZIP tidak berisi dataset sintetis atau checkpoint yang
dapat disalahartikan sebagai bobot kakao terlatih.

Model pada smoke test tidak diharapkan mendeteksi semua 20 objek setelah satu
epoch. Angka 20 adalah jumlah target fixture; unit test prediksi memakai
detector terkontrol untuk membuktikan seluruh 20 hasil yang lolos threshold
dipertahankan dan dihitung dengan benar.

TorchMetrics dapat memperingatkan ketika model menghasilkan lebih dari 100
deteksi. Model memakai batas 300 deteksi, sedangkan mAP memakai protokol COCO
standar maxDet 100. Peringatan ini tidak dinonaktifkan; batas pelaporan harus
dicatat pada skripsi. Prediksi/visualisasi memakai threshold operasionalnya.

Beberapa thread CPU dapat menghasilkan selisih floating point kecil. Uji
kesetaraan resume memakai satu thread; training normal tetap memakai
CPU_NUM_THREADS pada config.py. Seed tidak menjamin kesamaan bit lintas
hardware, driver, versi library, atau operasi paralel.

Windows PowerShell, driver NVIDIA dan CUDA pada komputer lokal belum
dieksekusi dari cloud Linux ini. Launcher ditinjau secara statis, dan wheel
resmi Windows Python 3.12 untuk cpu, cu126 dan cu128 telah diperiksa tersedia.
Jalankan RUN_WINDOWS.bat lokal untuk membuktikan kesiapan perangkat Anda.

Dataset kakao asli berada pada komputer Anda dan belum diperiksa di sini.
Belum ada klaim performa penelitian, mAP kakao, atau bobot hasil penelitian.
Angka penelitian akan dihasilkan dari train/validation/test nyata lokal.

## Perbaikan transfer memori — 13 Oktober 2026

Default `PIN_MEMORY = False` sekarang berlaku untuk DataLoader train dan
validation. Perubahan ini menghindari alokasi pinned memory yang gagal pada
traceback Windows `CUDA error: resource already mapped`. Pemindahan tensor
dan model ke GPU tetap dilakukan melalui `.to(device)`; pilihan CUDA tetap
dipertahankan. Flag ini tidak dimasukkan ke syarat hyperparameter resume.

Suite terbaru: **36 pengujian lulus, 0 gagal, 0 skipped**. Tiga pengujian baru
memakai DataLoader sesungguhnya, mensimulasikan tersedianya CUDA, dan memaksa
allocator `.pin_memory()` gagal. Pinning mati tetap menghasilkan semua lima
sampel termasuk batch terakhir satu gambar; pinning aktif mereproduksi
kegagalan yang disuntikkan. CPU melewati pinning meskipun flag diaktifkan.

Smoke pretrained CPU juga lulus kembali: training, evaluasi, prediksi,
penyimpanan/restore checkpoint dan kesetaraan resume. Ini menguji software
dan penanganan jalur alokasi memori; CUDA/driver Windows fisik belum diuji
di cloud. Keberhasilan pada mesin pengguna perlu dibuktikan melalui retry
setelah perbaikan.
