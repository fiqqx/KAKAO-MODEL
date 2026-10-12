Bantu saya melanjutkan skripsi “Deteksi dan Klasifikasi Tingkat Fermentasi Biji Kakao Menggunakan SSDLite-MobileNetV3-Large Berbasis Analisis Citra Bentuk dan Warna”.

Saya ingin notebook penelitian dan website Flask yang sederhana, elegan, berbahasa Indonesia, nyaman dipakai di laptop, Android, serta iPhone. Kerjakan implementasi yang dapat dijalankan dan mudah dijelaskan saat sidang. Gunakan referensi hanya untuk tingkat kerapian, bukan menyalin desain atau fitur proyek orang lain.

1. Model dan notebook
- Periksa proyek fiqqx/KAKAO-MODEL dan gunakan kembali kode yang sudah benar.
- Model tetap torchvision.models.detection.ssdlite320_mobilenet_v3_large. Ini multi-object detection; satu foto dapat berisi banyak biji dan ketiga kelas.
- Label internal: 0 background; 1 Tidak Fermentasi; 2 Setengah Fermentasi; 3 Full Fermentasi. Petakan kategori COCO berdasarkan nama dengan benar.
- Buat satu notebook utama berisi alur Tahap 1–8: pemeriksaan environment, dataset, loader, preprocessing/augmentasi, model, training, evaluasi, dan prediksi. Beri penjelasan sederhana per sel. Fungsi bersama yang dipakai Flask boleh berada di modul Python kecil agar konsisten.
- Pertahankan train/validation/test terpisah. Checkpoint dan confidence dipilih menggunakan validation; test digunakan untuk laporan akhir.
- Horizontal flip berpeluang 0.5 hanya pada train; bounding box ikut berubah. Resize 320 × 320 dan normalisasi dilakukan model. Jangan menerapkan transformasi itu dua kali.
- Website memakai checkpoint terbaik yang sudah tersedia, dengan lokasi eksplisit pada konfigurasi. Confidence awal 0.20. Gunakan model.eval() dan torch.inference_mode(). Muat model sekali saat server mulai; website tidak menjalankan training.

2. Website tanpa database
- Gunakan Flask, HTML, CSS, dan JavaScript sederhana.
- Jangan gunakan database: tidak perlu SQL, SQLite, MySQL, MongoDB, Firebase, atau IndexedDB. Tidak perlu akun/login.
- Simpan riwayat ringan pada localStorage browser: ID, waktu, mode, thumbnail kecil, total, serta jumlah ketiga kelas. Batasi 10 hasil terbaru dan tangani kapasitas penyimpanan penuh.
- Kompres thumbnail, misalnya JPEG sisi panjang maksimal 480 piksel. Jangan menyimpan seluruh gambar resolusi penuh dalam localStorage. Gambar hasil penuh dapat diunduh langsung saat hasil deteksi masih terbuka.
- Riwayat tersedia hanya pada browser/perangkat yang sama. Tampilkan keterangan singkat bahwa data dapat hilang jika penyimpanan browser dihapus.
- Server memproses gambar di memori atau berkas sementara yang dibersihkan, tanpa membuat database riwayat.

3. Home dan antarmuka
- Home langsung menampilkan dua kartu utama: Kamera Langsung dan Unggah Gambar.
- Navigasi cukup Home dan Riwayat. Pada desktop boleh ada sidebar sederhana; di ponsel gunakan header ringkas, tanpa sidebar yang menyempitkan layar.
- Gunakan desain orisinal dengan latar terang, aksen hijau/cokelat kakao, teks mudah dibaca, tombol minimal 44 × 44 piksel, dan tanpa scroll horizontal.
- Hasil langsung muncul pada halaman kamera/unggahan. Tidak perlu halaman Analisis terpisah.
- Tampilkan gambar berkotak, nama kelas, confidence, Total Biji Terdeteksi, dan jumlah setiap kelas. Kartu hitungan pada mobile memakai grid 2 × 2. Total harus sama dengan penjumlahan ketiga kelas.
- Berikan status sedang mendeteksi, kondisi kosong, serta pesan gagal yang mudah dipahami. Jangan menambahkan peta, koordinat, fuzzy, atau fitur audit proyek lain.

4. Kamera langsung
- Gunakan getUserMedia pada browser pengguna; utamakan kamera belakang dan dukung playsinline di iPhone.
- Tombol: Mulai Kamera, Hentikan, Simpan Hasil, serta Unduh Hasil.
- Kirim frame berkala, awal sekitar satu detik, paling banyak satu request aktif per browser. Sesuaikan dengan waktu inferensi dan batasi inferensi bersama di server.
- Jumlah biji berasal dari frame terbaru, bukan akumulasi antar-frame. Kotak, gambar, dan hitungan harus berasal dari frame yang sama.
- Simpan satu riwayat hanya saat Simpan Hasil ditekan. Hentikan seluruh track kamera dan pengiriman ketika pengguna berhenti atau meninggalkan halaman.
- Tangani izin kamera ditolak dan kamera tidak tersedia. Jelaskan kebutuhan HTTPS saat browser ponsel mengakses server laptop. Berikan alternatif unggah gambar.

5. Unggah gambar dan riwayat
- Sediakan pilih gambar, pratinjau, tombol Deteksi, Ganti Gambar, serta Unduh Hasil.
- Dukung JPEG/PNG/WebP dan HEIC/HEIF iPhone dengan decoder yang diperlukan. Perbaiki orientasi EXIF, batasi berkas sekitar 15 MB, dan periksa isi gambar.
- Setelah inferensi berhasil, simpan satu riwayat otomatis di browser. Gambar valid tanpa deteksi menghasilkan total 0. Inferensi gagal tidak disimpan sebagai hasil berhasil.
- Halaman Riwayat menampilkan thumbnail, waktu, mode, total, dan jumlah setiap kelas. Sediakan Lihat serta Hapus satu item dengan konfirmasi. Jelaskan bahwa gambar pada riwayat adalah pratinjau terkompresi.

6. Menjalankan, diagram, dan hasil akhir
- Gunakan .venv Windows yang sudah ada; jangan memasang ulang PyTorch GPU jika sesuai. Tambahkan hanya dependensi yang diperlukan.
- Sediakan launcher BUKA_WEBSITE_WINDOWS.bat, petunjuk kernel notebook, serta panduan akses laptop/ponsel melalui HTTPS. Gunakan server produksi yang sesuai, misalnya Waitress, dan jangan aktifkan debug pada deployment publik.
- Buat empat diagram ringkas: flowchart sistem umum (input→preprocessing→deteksi→hitungan→hasil→riwayat), flowchart persiapan/preprocessing data (foto→Roboflow→split→COCO→preprocessing), use case satu aktor Pengguna, serta activity Pengguna/Sistem tanpa lajur Database.
- Diagram cukup langkah inti; gunakan hubungan include/extend yang tepat. Simpan sumber yang bisa diedit serta PNG/SVG.
- Buat wireframe Home desktop/mobile, Kamera, Unggah Gambar, dan Riwayat. Jelaskan susunan elemen dan tindakan pengguna; angka pada wireframe harus diberi label sebagai ilustrasi.
- Berikan struktur folder ringkas, kode lengkap untuk fitur inti, penjelasan tiap file, dan README Indonesia. Untuk demo sidang, pakai checkpoint terlatih agar tidak perlu menunggu training ulang.
- Verifikasi multi-objek, total per kelas, hasil 0, gambar rusak, riwayat localStorage, kapasitas penuh, penghapusan, penghentian kamera, dan responsivitas. Bedakan pengujian yang benar-benar dilakukan dari panduan pengujian. Jangan menjanjikan semua biji selalu terdeteksi benar.
