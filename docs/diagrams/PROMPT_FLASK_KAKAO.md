Bantu saya melanjutkan proyek skripsi Teknik Informatika dengan judul:
“Deteksi dan Klasifikasi Tingkat Fermentasi Biji Kakao Menggunakan SSDLite-MobileNetV3-Large Berbasis Analisis Citra Bentuk dan Warna.”

Saya ingin satu notebook utama untuk belajar dan presentasi, serta website Flask untuk deployment. Website harus sederhana, elegan, berbahasa Indonesia, dan nyaman dipakai melalui browser laptop, Android, dan iPhone. Kerjakan implementasi yang benar-benar dapat dijalankan, bukan hanya contoh tampilan.

Susun Tahap 1–8 penelitian pada notebook, lalu Tahap 9 untuk deployment Flask. Pengguna Android/iPhone cukup membuka URL website melalui browser; inferensi diproses pada server.

Konteks proyek:
- Repositori: fiqqx/KAKAO-MODEL. Periksa kode yang tersedia dan gunakan kembali bagian yang sudah benar.
- Komputer lokal: Windows, VS Code, Python 3.12, GPU NVIDIA RTX 2050. Virtual environment .venv dan PyTorch GPU sudah tersedia.
- Dataset COCO JSON hasil ekspor Roboflow sudah tersedia, dengan split train, val/valid, dan test.
- Model sudah pernah dilatih 100 epoch. Website harus bisa menggunakan checkpoint terbaik yang sudah ada, termasuk checkpoint di subfolder eksperimen. Tetapkan lokasi checkpoint secara eksplisit dalam konfigurasi.
- Confidence 0.20 menghasilkan F1 validation tertinggi di antara 0.20, 0.30, 0.40, dan 0.50 yang sudah diuji. Gunakan 0.20 sebagai nilai awal, bukan jaminan semua biji terdeteksi.

1. Model dan notebook
Gunakan implementasi resmi torchvision.models.detection.ssdlite320_mobilenet_v3_large. Pertahankan model ini sebagai model utama penelitian. Tugasnya multi-object detection, bukan satu label untuk seluruh gambar.

Pemetaan label internal:
0 = background
1 = Tidak Fermentasi
2 = Setengah Fermentasi
3 = Full Fermentasi

Gunakan pemetaan kategori COCO yang benar berdasarkan nama kategori; jangan mengasumsikan category_id mentah sama dengan label internal.

Sediakan KAKAO_LENGKAP.ipynb sebagai notebook utama dengan sel Markdown dan kode berurutan: pemeriksaan environment; audit dataset; loader; preprocessing dan augmentasi; model transfer learning dengan head empat kelas; training; evaluasi; dan prediksi foto baru. Fungsi bersama yang dibutuhkan website boleh berada pada modul Python kecil agar kode model dan inferensi tidak diduplikasi.

Pertahankan train/validation/test terpisah. Gunakan validation untuk memilih checkpoint dan pengaturan; test untuk laporan akhir. Gunakan loss bawaan model, yaitu classification dan bbox_regression. Tampilkan loss, grafik training, mAP@0.5, mAP@0.5:0.95, serta precision/recall/F1 keseluruhan dan per kelas. TP/FP/FN harus mempertimbangkan kelas dan IoU.

Augmentasi awal cukup horizontal flip berpeluang 0.5 pada train, dengan bounding box ikut berubah; hindari perubahan warna ekstrem. Pertahankan konfigurasi dan dukungan resume yang sudah benar. Untuk kendala Windows yang pernah terjadi, pertahankan pin_memory=False. Jelaskan alasan perubahan hyperparameter bila diperlukan; jangan otomatis melatih ulang model hanya untuk membuka website.

2. Struktur deployment
Gunakan Flask, HTML, CSS, dan JavaScript sederhana. Gunakan template Jinja, SQLite, dan modul inferensi yang bisa diuji terpisah. Tidak perlu React atau layanan database eksternal. Buat struktur folder yang ringkas dan jelaskan fungsi setiap file.

Muat checkpoint sekali ketika server mulai. Gunakan model.eval() dan torch.inference_mode() untuk prediksi. Gunakan GPU jika tersedia dan sesuai konfigurasi; dukung CPU dengan status yang jelas saat startup. Batasi inferensi bersama agar permintaan beberapa browser tidak bertabrakan. Website hanya melakukan inferensi, tidak menjalankan training.

Gunakan preprocessing, pemetaan kelas, confidence, dan NMS yang konsisten dengan evaluasi. Jika menambahkan penyaringan duplikasi antar kelas, jadikan konfigurasi yang jelas dan evaluasi dahulu pada validation. Jangan menganggap semua kotak bertumpuk adalah duplikasi, dan jangan memaksa jumlah objek menjadi angka tertentu.

3. Beranda dan tampilan
Saat website dibuka, langsung tampilkan dua kartu/tombol utama:
- Kamera Langsung
- Unggah Gambar

Sediakan tautan Riwayat sebagai navigasi sekunder. Tidak perlu login untuk versi awal. Pakai latar terang, aksen hijau dan cokelat kakao, tipografi mudah dibaca, ruang yang cukup, dan tombol besar untuk layar sentuh. Tata letak harus responsif tanpa scroll horizontal. Jangan menampilkan pengaturan teknis model pada alur utama pengguna.

Hasil deteksi harus menampilkan gambar dengan bounding box, nama kelas dan confidence, kartu Total Biji Terdeteksi, serta jumlah Tidak Fermentasi, Setengah Fermentasi, dan Full Fermentasi. Gunakan warna kelas yang konsisten dan selalu sertakan teks. Total harus sama dengan penjumlahan ketiga kelas. Sediakan indikator sedang memproses, kondisi kosong, pesan berhasil, dan pesan error yang mudah dipahami.

4. Kamera langsung
Gunakan navigator.mediaDevices.getUserMedia dari browser pengguna, mengutamakan kamera belakang. Jangan membaca webcam laptop/server sebagai pengganti kamera ponsel. Gunakan atribut video playsinline dan kontrol Mulai Kamera, Hentikan, dan Simpan Hasil.

Kirim frame ke server secara berkala, mulai dari interval sekitar satu detik, dengan paling banyak satu permintaan inferensi aktif per browser. Sesuaikan pengiriman dengan lama pemrosesan; jangan menumpuk permintaan atau menjanjikan FPS tinggi. Jaga gambar hasil dan hitungan berasal dari frame yang sama. Hentikan seluruh track kamera dan pengiriman ketika pengguna berhenti atau meninggalkan halaman.

Hitungan menunjukkan biji pada frame terbaru; jangan menjumlahkan objek antar-frame. Hanya simpan riwayat kamera ketika pengguna menekan Simpan Hasil. Simpan snapshot beserta hasil yang sedang ditampilkan, bukan frame baru yang berbeda. Berikan petunjuk jika izin kamera ditolak, kamera tidak tersedia, atau akses belum memakai HTTPS, serta pilihan beralih ke unggah gambar.

5. Unggah gambar
Sediakan pemilihan gambar dan pratinjau sebelum tombol Deteksi. Dukung JPEG, PNG, WebP, serta HEIC/HEIF untuk foto iPhone dengan decoder yang sesuai, misalnya pillow-heif. Perbaiki orientasi EXIF sebelum inferensi.

Batasi ukuran berkas, misalnya 15 MB, dan jelaskan batas tersebut pada antarmuka. Periksa isi gambar saat decoding, gunakan nama berkas internal yang aman, dan tangani gambar rusak atau tidak didukung dengan pesan jelas. Setelah deteksi berhasil, tampilkan hasil dan simpan satu riwayat secara otomatis. Hasil tanpa objek adalah hasil valid dengan total 0, bukan error. Inferensi yang gagal tidak boleh menghasilkan riwayat seolah-olah berhasil.

6. Riwayat
Gunakan SQLite agar hasil tetap tersedia setelah server dimulai ulang. Pisahkan riwayat setiap browser menggunakan sesi anonim yang konsisten; pengguna hanya dapat melihat, mengunduh, dan menghapus hasil miliknya. Tidak perlu akun pada versi awal.

Simpan ID, identitas sesi, waktu, mode kamera/unggahan, referensi gambar asli dan hasil, boxes/labels/scores, confidence yang dipakai, jumlah setiap kelas, dan total. Simpan waktu secara konsisten dan tampilkan waktu lokal pengguna.

Halaman Riwayat memakai kartu yang nyaman di ponsel, menampilkan thumbnail, waktu, mode, dan total. Sediakan detail hasil, unduh gambar anotasi, serta hapus satu riwayat dengan konfirmasi. Verifikasi kepemilikan di server pada semua akses hasil. Hapus berkas terkait ketika riwayat dihapus, dan beri pengelolaan penyimpanan sementara agar mode kamera tidak memenuhi disk.

7. Menjalankan website dan deployment
Manfaatkan .venv yang sudah ada dan tambahkan hanya dependensi yang diperlukan. Jangan memasang ulang wheel PyTorch GPU berukuran besar jika instalasi yang sesuai sudah tersedia. Sediakan petunjuk memilih kernel .venv di notebook.

Buat launcher Windows yang jelas, misalnya BUKA_WEBSITE_WINDOWS.bat, agar saya dapat membuka website tanpa menghafal perintah terminal. Pisahkan launcher website dari launcher training, dan tampilkan alasan yang mudah dipahami jika checkpoint atau dependensi belum tersedia.

Jelaskan pengujian lokal di laptop serta akses dari ponsel. Kamera browser ponsel membutuhkan HTTPS ketika mengakses server laptop melalui jaringan; alamat localhost pada ponsel menunjuk ponsel itu sendiri. Sediakan panduan HTTPS yang praktis melalui reverse proxy atau tunnel, serta panduan server produksi yang sesuai, misalnya Waitress. Jangan menyarankan menonaktifkan keamanan browser. Simpan secret dan konfigurasi deployment di environment atau konfigurasi lokal yang tidak dipublikasikan. Jangan menjalankan Flask debug pada deployment publik.

8. Diagram UML
Sediakan use case diagram dan activity diagram yang konsisten dengan fitur di atas, dalam PlantUML serta gambar PNG/SVG yang dapat dipakai untuk skripsi.

Aktor utama adalah Pengguna. Use case mencakup deteksi kamera, deteksi gambar unggahan, deteksi biji, menampilkan hasil dan hitungan, menyimpan hasil kamera, menyimpan riwayat, melihat daftar/detail riwayat, mengunduh hasil, serta menghapus satu riwayat. Gunakan include untuk proses wajib dan extend untuk Simpan Hasil kamera yang dilakukan hanya ketika tombol ditekan. Model dan database merupakan komponen internal, bukan aktor pengguna tambahan.

Activity diagram harus menunjukkan dua mode deteksi, izin kamera, validasi unggahan, inferensi, hitungan frame terbaru, penyimpanan riwayat, penghentian kamera, serta melihat/detail/unduh/hapus riwayat. Sertakan alur kegagalan yang relevan, tanpa membuat diagram terlalu rumit.

Tambahkan flowchart sistem secara umum dari persiapan data hingga training, evaluasi, dan penggunaan website. Tambahkan flowchart persiapan dan preprocessing dataset dari pengambilan foto, labeling Roboflow, pembagian split berdasarkan kelompok sumber, ekspor COCO, audit, hingga tensor siap diproses model. Pisahkan penjelasan persiapan data, augmentasi train, dan transformasi internal model.

9. Hasil akhir dan validasi
Berikan implementasi lengkap tanpa placeholder pada fitur inti, struktur folder, penjelasan sederhana tiap file, input/output fungsi penting, dan README berbahasa Indonesia. Sertakan cara demo sidang memakai checkpoint yang sudah dilatih, sehingga tidak perlu menunggu training 100 epoch saat presentasi.

Jalankan pemeriksaan yang relevan: gambar dengan banyak objek dan beberapa kelas; hasil 0 objek; gambar rusak; checkpoint tidak tersedia; penyimpanan riwayat; kesesuaian total dan jumlah per kelas; isolasi sesi; dan penghentian kamera. Uji responsivitas antarmuka, serta lakukan pengujian kamera pada browser Android dan iPhone bila perangkat tersedia. Bedakan hasil yang benar-benar diuji dari yang hanya didokumentasikan. Jangan mengklaim model selalu mendeteksi seluruh biji atau memberi kelas yang selalu benar.
