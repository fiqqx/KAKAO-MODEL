@echo off
setlocal
cd /d "%~dp0"
title Menyiapkan Notebook Deteksi Kakao
echo Menyiapkan kernel notebook pada virtual environment proyek.
echo PyTorch GPU tidak dipasang ulang oleh launcher ini.
echo.
if not exist ".venv\Scripts\python.exe" (
    echo GAGAL: .venv belum ditemukan pada folder proyek ini.
    echo Salin pembaruan proyek ke folder KAKAO-MODEL yang sebelumnya dipakai
    echo untuk training dan sudah memiliki .venv, dataset, serta checkpoint.
    echo Panduan lengkap ada pada README_NOTEBOOK.md.
    pause
    exit /b 1
)
if not exist "requirements-notebook.txt" (
    echo GAGAL: requirements-notebook.txt tidak ditemukan.
    echo Salin seluruh pembaruan proyek, bukan file notebook saja.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "requirements-notebook.txt"
if errorlevel 1 (
    echo.
    echo GAGAL: Pemasangan kebutuhan notebook belum selesai.
    echo Periksa pesan di atas lalu jalankan file ini kembali.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m ipykernel install --user --name kakao-model --display-name "Python (KAKAO-MODEL)"
if errorlevel 1 (
    echo GAGAL: Kernel belum berhasil didaftarkan. Periksa pesan di atas.
    pause
    exit /b 1
)
echo.
echo SELESAI. Buka kembali VS Code dan buka KAKAO_LENGKAP.ipynb.
echo Pilih Select Kernel lalu Python (KAKAO-MODEL) atau .venv folder proyek ini.
echo Jalankan Bagian A Langkah 1 sampai 4, pilih checkpoint, lalu Mulai Website.
echo Tidak ada training yang dijalankan oleh file ini.
pause
exit /b 0
