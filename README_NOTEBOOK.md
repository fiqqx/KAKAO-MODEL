# Notebook Deteksi Kakao tanpa perintah terminal

Untuk **memulai training dari dataset mentah COCO Roboflow dengan seluruh kode
langsung dalam notebook**, gunakan notebook mandiri
[KAKAO_TRAINING_DARI_NOL.ipynb](KAKAO_TRAINING_DARI_NOL.ipynb).
Panduan di halaman ini membahas notebook website untuk model yang sudah dilatih.

Gunakan **`KAKAO_LENGKAP.ipynb`** untuk menjalankan website Flask dari tombol notebook, atau menjelaskan tahap penelitian saat sidang. Notebook menggunakan model yang sudah Anda latih. Training ulang tidak diperlukan untuk membuka website.

## Persiapan satu kali di Windows

1. Letakkan pembaruan kode pada **folder proyek lama** yang sudah memiliki `.venv`, `dataset`, dan `checkpoints`. Jangan menghapus folder tersebut. Notebook memerlukan `webapp/`, `src/`, `config.py`, dan berkas requirements di folder yang sama.
2. Klik dua kali **`SIAPKAN_NOTEBOOK_WINDOWS.bat`** di File Explorer. Launcher memasang paket web/notebook pada `.venv` yang sudah ada dan mendaftarkan kernel `Python (KAKAO-MODEL)`. PyTorch GPU tidak dipasang ulang; dependensi notebook awal tetap membutuhkan koneksi internet.
3. Buka folder proyek di VS Code. Pasang ekstensi **Python** dan **Jupyter** dari Microsoft melalui menu Extensions jika belum tersedia.
4. Buka **`KAKAO_LENGKAP.ipynb`**. Klik **Select Kernel** di kanan atas, lalu pilih `Python (KAKAO-MODEL)` atau Python Environments → `.venv` milik folder proyek Anda.

Jika memiliki beberapa salinan proyek, pastikan path Python pada Langkah 1 mengarah ke `.venv` folder yang benar. Pilihan kernel berdasarkan nama saja dapat merujuk folder lama.

## Membuka website

Jalankan sel kode **Bagian A Langkah 1–4** melalui tombol ▶:

1. **Pemeriksaan:** menampilkan lokasi proyek, Python, PyTorch, dan status CUDA.
2. **Dependensi:** memeriksa kebutuhan website/notebook. Jika ada yang kurang, klik tombol pemasangan. Sel tidak memasang paket tanpa klik dan tidak memasang ulang PyTorch.
3. **Checkpoint:** pilih `best_model.pth` dari eksperimen yang ingin digunakan. Confidence awal `0.20`; untuk GPU NVIDIA, pilih CUDA bila siap. Checkpoint tidak dipilih otomatis.
4. **Kontrol:** klik **Mulai Website**, tunggu hingga model siap, lalu klik **Buka Website**. Setelah selesai, klik **Hentikan Website**.

Biarkan kernel dan VS Code tetap aktif selama website digunakan. Menutup tab browser tidak menghentikan server. Website memakai Flask/Waitress di latar belakang; notebook tetap dapat digunakan. Jika port sedang dipakai, pilih port berbeda pada Langkah 3, misalnya 5001.

**Run All hanya menyiapkan kontrol.** Ia tidak menginstal paket, memulai server, melatih model, mengevaluasi dataset, atau memprediksi foto secara otomatis. Penelitian tambahan dilakukan melalui tombol masing-masing.

Jika file checkpoint tidak muncul, salin checkpoint Anda ke `checkpoints/<nama_eksperimen>/best_model.pth`, lalu klik **Perbarui Daftar**. Untuk lokasi di luar proyek, isi kolom **Lokasi lain**. Gunakan checkpoint dari sumber yang Anda percaya dan arsitektur/pemetaan kelas proyek ini.

## Isi website

- Home langsung menyediakan **Kamera Langsung** dan **Unggah Gambar**.
- Hasil menampilkan bounding box, kelas, confidence, total, serta jumlah setiap kelas.
- Riwayat memakai **localStorage browser**, maksimal 10 hasil terbaru. Tidak ada database atau SQL.
- Unggahan berhasil disimpan ke riwayat otomatis. Kamera hanya disimpan ketika **Simpan Hasil** ditekan.
- Gambar hasil aktif dapat diunduh; riwayat menyimpan thumbnail terkompresi serta hitungan.

Riwayat berlaku pada browser/perangkat yang sama dan bisa hilang jika data browser dihapus. Riwayat bukan koleksi data pelatihan.

## Ponsel dan kamera

Untuk akses laptop sendiri, biarkan pilihan Akses **Laptop ini**. Jika ingin mengakses dari ponsel, pilih **Jaringan lokal**, lalu ikuti panduan `README_WEB.md`.

Kamera browser ponsel membutuhkan **HTTPS**. Menggunakan HTTP dengan alamat IP laptop saja umumnya tidak memberikan akses kamera. Notebook tidak memasang sertifikat maupun membuka tunnel publik secara otomatis. Unggahan gambar dapat digunakan terlebih dahulu ketika akses kamera belum tersedia.

Android/iPhone mengakses browser website. Ponsel tidak memasang PyTorch; inferensi dijalankan di laptop/server.

## Tahap penelitian opsional

Bagian B menyediakan tombol untuk:

- Audit dataset COCO dan contoh anotasi/horizontal flip.
- Ringkasan arsitektur tanpa mengunduh pretrained weights.
- Training eksperimen baru dengan konfirmasi khusus, epoch/batch yang dapat diatur, serta folder hasil baru.
- Evaluasi checkpoint pada validation, atau test setelah pengaturan final dan konfirmasi terpisah.
- Grafik dari `training_history.csv` yang sudah tersimpan.
- Memilih foto baru dan menjalankan prediksi langsung di notebook.

Hentikan website sebelum training, evaluasi, atau prediksi notebook agar model tidak memakai VRAM secara bersamaan. Training memblokir kernel sampai selesai; gunakan **Interrupt** pada toolbar jika perlu menghentikannya. Checkpoint lama tidak ditimpa oleh training baru.

Augmentasi training tetap horizontal flip dengan peluang 50%. Tampilan contoh flip sengaja memakai peluang 100% agar perubahan posisi bounding box dapat dijelaskan; itu tidak mengubah konfigurasi training.

## Mengatasi kendala umum

| Kendala | Langkah yang dilakukan |
|---|---|
| Pilihan kernel belum ada | Jalankan launcher persiapan, buka ulang VS Code, pilih interpreter `.venv` yang benar. |
| Kontrol tidak tampil | Pastikan ekstensi Jupyter aktif, kernel benar, dan `ipywidgets` terpasang; jalankan ulang sel. |
| PyTorch atau CUDA tidak tersedia | Pilih `.venv` yang dipakai training. Notebook tidak mengganti instalasi GPU secara otomatis. |
| Folder `webapp`/requirements tidak ditemukan | Salin seluruh pembaruan proyek, bukan hanya `.ipynb`. |
| Checkpoint tidak cocok | Pilih checkpoint dari arsitektur dan pemetaan kelas proyek ini. Pesan kesalahan tidak diganti prediksi contoh. |
| Server tidak dapat dimulai | Periksa pesan di notebook; pastikan checkpoint dipilih dan port tidak dipakai aplikasi lain. |
| Mengganti checkpoint/confidence | Hentikan server, ubah pilihan, lalu mulai lagi. |
| Kernel direstart | Jalankan kembali Bagian A. Server lama dari kernel tersebut berhenti. |

Untuk sidang, siapkan model terlebih dahulu dan lakukan latihan **Restart Kernel → Run All → pilih checkpoint → Mulai Website → unggah foto**. Ini memastikan demo tidak bergantung pada variabel dari eksekusi sebelumnya.
