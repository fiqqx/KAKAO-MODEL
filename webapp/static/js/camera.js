import { predictImage, prepareResultImage, downloadResult, setError, setStatus, updateCounts } from "./common.js";
import { saveResult } from "./history-store.js";

const video = document.getElementById("camera-video");
const resultImage = document.getElementById("result-image");
const placeholder = document.getElementById("camera-placeholder");
const start = document.getElementById("start-camera");
const stop = document.getElementById("stop-camera");
const save = document.getElementById("save-result");
const download = document.getElementById("download-result");
const cameraStatus = document.getElementById("camera-status");
const canvas = document.createElement("canvas");
let stream = null;
let active = false;
let runToken = 0;
let timer = null;
let controller = null;
let currentResult = null;
let saving = false;
const savedIds = new Set();

function refreshSaveButton() {
  save.disabled = !currentResult || saving || savedIds.has(currentResult.result_id);
}

function stopCamera(message = "Kamera dihentikan. Hasil terakhir tetap dapat disimpan atau diunduh.") {
  active = false;
  runToken += 1;
  clearTimeout(timer);
  timer = null;
  controller?.abort();
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  video.hidden = true;
  start.disabled = false;
  stop.disabled = true;
  cameraStatus.textContent = "Nonaktif";
  if (!currentResult) placeholder.hidden = false;
  setStatus(message);
}

async function captureFrame(token) {
  if (!active || token !== runToken) return;
  if (!video.videoWidth || !video.videoHeight) {
    timer = setTimeout(() => captureFrame(token), 300);
    return;
  }
  // This is transmission downscaling; the model's own preprocessing remains unchanged.
  const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.88));
  if (!active || token !== runToken) return;
  if (!blob) {
    setError("Frame kamera belum dapat dibaca. Coba hentikan lalu mulai kamera kembali.");
    stopCamera("Pembacaan kamera dihentikan.");
    return;
  }
  const requestController = new AbortController();
  controller = requestController;
  try {
    const result = await predictImage(blob, { mode: "camera", signal: requestController.signal });
    const imageUrl = await prepareResultImage(result);
    if (!active || token !== runToken || requestController.signal.aborted) return;
    // Commit the returned image and its counts together. Never overlay an older frame on live video.
    currentResult = result;
    resultImage.src = imageUrl;
    resultImage.hidden = false;
    video.hidden = true;
    placeholder.hidden = true;
    updateCounts(result);
    refreshSaveButton();
    download.disabled = false;
    setError();
    document.getElementById("result-status").textContent = "Frame terbaru";
    setStatus(result.total === 0 ? "Frame terbaru: tidak ada biji yang terdeteksi pada ambang saat ini." : `Frame terbaru: ${result.total} biji terdeteksi. Jumlah tidak diakumulasi.`);
  } catch (error) {
    if (error.name !== "AbortError" && active && token === runToken) {
      setError(error.message);
      stopCamera("Deteksi kamera dihentikan. Hasil terakhir tetap tersedia; mulai lagi setelah penyebab diperbaiki.");
    }
  } finally {
    // A new frame is scheduled only when this request, decoding and UI update have all finished.
    if (active && token === runToken) timer = setTimeout(() => captureFrame(token), 1000);
  }
}

start.addEventListener("click", async () => {
  if (active) return;
  setError();
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setError("Kamera memerlukan HTTPS pada ponsel dan izin kamera. Buka alamat HTTPS atau gunakan Unggah Gambar.");
    return;
  }
  active = true;
  const token = ++runToken;
  start.disabled = true;
  stop.disabled = false;
  cameraStatus.textContent = "Menghubungkan";
  setStatus("Mengakses kamera. Izinkan akses jika diminta oleh browser.");
  try {
    const requestedStream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } } });
    if (!active || token !== runToken) {
      requestedStream.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = requestedStream;
    video.srcObject = stream;
    await video.play();
    if (!active || token !== runToken) return;
    stream.getTracks().forEach((track) => track.addEventListener("ended", () => {
      if (active && token === runToken) stopCamera("Kamera tidak lagi tersedia. Anda dapat mencoba memulainya kembali.");
    }));
    cameraStatus.textContent = "Aktif";
    placeholder.hidden = true;
    if (!currentResult) video.hidden = false;
    setStatus("Kamera aktif. Menyiapkan hasil deteksi pertama…");
    captureFrame(token);
  } catch (error) {
    if (token !== runToken) return;
    let message = "Kamera belum dapat dibuka. Pastikan kamera tidak sedang digunakan aplikasi lain.";
    if (["NotAllowedError", "PermissionDeniedError"].includes(error.name)) message = "Izin kamera ditolak. Izinkan kamera melalui pengaturan browser, atau gunakan Unggah Gambar.";
    if (["NotFoundError", "DevicesNotFoundError"].includes(error.name)) message = "Kamera tidak ditemukan pada perangkat ini. Gunakan Unggah Gambar untuk melanjutkan.";
    setError(message);
    stopCamera("Kamera belum aktif.");
  }
});

stop.addEventListener("click", () => stopCamera());
download.addEventListener("click", () => downloadResult(currentResult));
save.addEventListener("click", async () => {
  const snapshot = currentResult;
  if (!snapshot || saving || savedIds.has(snapshot.result_id)) return;
  saving = true;
  refreshSaveButton();
  const result = await saveResult(snapshot, "camera");
  saving = false;
  if (result.saved) savedIds.add(snapshot.result_id);
  refreshSaveButton();
  setStatus(result.warning || "Satu hasil kamera telah disimpan ke riwayat browser.");
});

window.addEventListener("pagehide", () => stopCamera("Kamera dihentikan ketika halaman ditinggalkan."));
document.addEventListener("visibilitychange", () => {
  if (document.hidden && active) stopCamera("Kamera dihentikan saat halaman tidak aktif. Tekan Mulai Kamera untuk melanjutkan.");
});
