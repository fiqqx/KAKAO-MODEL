# Website Deteksi Kakao dari notebook

Mulai dari [README_NOTEBOOK.md](README_NOTEBOOK.md). Buka `KAKAO_LENGKAP.ipynb`,
jalankan Bagian A Langkah 1–4, pilih checkpoint, lalu klik **Mulai Website** dan
**Buka Website**. Proses ini memakai Flask dan Waitress; Anda tidak perlu
mengetik perintah terminal. Biarkan VS Code dan kernel notebook tetap aktif.

## Memilih model

Pilih `best_model.pth` dari eksperimen yang memang ingin dipakai. Notebook
mendukung checkpoint dalam subfolder eksperimen dan lokasi lain yang Anda
tentukan. Model dibaca sekali ketika server dimulai. Confidence website awal
`0.20` terpisah dari konfigurasi eksperimen training. Hentikan website sebelum
mengganti model, confidence, atau perangkat.

Jika checkpoint belum dipilih atau tidak cocok, notebook menampilkan alasannya.
Tidak ada prediksi contoh yang menggantikan hasil model. Checkpoint terlatih,
dataset, dan `.venv` tidak disertakan dalam ZIP GitHub; gunakan milik Anda yang
sudah ada pada folder proyek lama.

## Menggunakan website

- **Home:** pilih Kamera Langsung atau Unggah Gambar.
- **Unggah Gambar:** pilih foto, tekan Deteksi, lalu lihat kotak, kelas,
  confidence, total, dan jumlah setiap kelas. Hasil berhasil masuk riwayat
  browser otomatis. Foto valid tanpa objek menghasilkan total 0.
- **Kamera Langsung:** izinkan kamera, tekan Mulai Kamera, dan arahkan ke biji.
  Hitungan mengikuti frame terbaru. Tekan Simpan Hasil untuk menyimpan satu
  hasil; Hentikan menutup kamera. Meninggalkan halaman juga menutup kamera.
- **Riwayat:** lihat maksimal 10 hasil terakhir pada browser/perangkat yang
  sama. Tombol Lihat membuka detail dan Hapus meminta konfirmasi satu item.
- **Unduh Hasil:** menyimpan gambar hasil penuh yang masih aktif. Riwayat
  menyimpan thumbnail kecil dan hitungan, bukan gambar resolusi penuh.

Riwayat memakai `localStorage`, tanpa SQL atau database server. Menghapus data
browser atau memakai browser/alamat website lain dapat membuat riwayat tidak
terlihat. Saat penyimpanan penuh, hasil deteksi tetap tampil dan bisa diunduh.

JPEG, PNG, WebP, dan HEIC/HEIF didukung melalui Pillow/pillow-heif. Orientasi
EXIF dikoreksi pada server. Batas berkas 15 MB dan resolusi 50 megapiksel;
gambar rusak atau terlalu besar menampilkan pesan kesalahan.

## Mengakses dari Android atau iPhone

Inferensi tetap berjalan pada laptop. Ponsel cukup membuka browser.

Untuk mencoba **unggahan melalui Wi-Fi lokal**:

1. Hubungkan laptop dan ponsel ke jaringan Wi-Fi yang sama.
2. Pada panel notebook, ubah Akses ke **Jaringan lokal**, lalu mulai website.
3. Lihat alamat IPv4 laptop pada Windows Settings → Network & Internet →
   Properties. Gunakan alamat jaringan laptop, bukan `127.0.0.1`.
4. Buka alamat IPv4 tersebut beserta port notebook pada browser ponsel.
5. Jika Windows meminta izin firewall, izinkan aplikasi pada jaringan pribadi
   yang Anda gunakan. Hindari membuka port ke internet melalui router.

**Kamera browser ponsel memerlukan HTTPS.** Alamat HTTP melalui IP laptop
umumnya hanya dapat dipakai untuk unggahan. Untuk kamera ponsel, gunakan
reverse proxy HTTPS atau tunnel HTTPS tepercaya yang meneruskan koneksi ke
port website. Pada laptop sendiri, browser mengizinkan kamera melalui alamat
loopback lokal.

Contoh jalur deployment: browser ponsel → alamat HTTPS proxy/tunnel → Waitress
pada port notebook. Konfigurasikan proxy/tunnel dengan aplikasi/dokumentasi
resminya, lalu buka alamat HTTPS yang diberikan pada ponsel. Laptop dan kernel
harus tetap aktif. Notebook tidak menerbitkan endpoint publik secara otomatis.
Jangan menonaktifkan keamanan kamera browser untuk melewati kebutuhan HTTPS.

## Struktur kode

| Berkas | Peran |
|---|---|
| `webapp/app.py` | Halaman Flask, validasi gambar, API kesehatan dan prediksi |
| `webapp/inference_service.py` | Checkpoint, satu model bersama, inferensi, gambar hasil dan hitungan |
| `webapp/web_config.py` | Pengaturan website yang terpisah dari eksperimen training |
| `webapp/notebook_server.py` | Memulai dan menghentikan Waitress melalui tombol notebook |
| `webapp/templates/` | Struktur Home, unggahan, kamera, dan riwayat |
| `webapp/static/css/` | Warna, tata letak desktop/mobile, ukuran tombol dan teks |
| `webapp/static/js/` | Unggahan, kamera browser, sinkronisasi hasil, unduhan, riwayat lokal |

`GET /api/health` memeriksa kesiapan model. `POST /api/predict` menerima gambar
melalui field multipart `image` dan mode `upload` atau `camera`; respons berisi
gambar beranotasi, boxes/labels/scores, hitungan, waktu, dan confidence.

## Batas pengujian

API, pemuatan arsitektur, notebook, dan antarmuka diperiksa pada Linux CPU.
Pengujian browser memakai Chromium dengan ukuran layar desktop/ponsel dan
kamera simulasi. Foto/checkpoint penelitian pengguna, CUDA RTX 2050 lokal,
serta kamera Android/iPhone fisik belum tersedia di lingkungan pengujian ini.
