import { predictImage, prepareResultImage, downloadResult, setError, setStatus, updateCounts } from "./common.js";
import { saveResult } from "./history-store.js";

const input = document.getElementById("image-input");
const choose = document.getElementById("choose-image");
const reset = document.getElementById("reset-button");
const detect = document.getElementById("detect-button");
const preview = document.getElementById("upload-preview");
const resultImage = document.getElementById("result-image");
const placeholder = document.getElementById("upload-placeholder");
const overlay = document.getElementById("processing-overlay");
const download = document.getElementById("download-result");
let selectedFile = null;
let currentResult = null;
let previewUrl = "";
let controller = null;
let generation = 0;

function openPicker() { input.value = ""; input.click(); }
choose.addEventListener("click", openPicker);
reset.addEventListener("click", openPicker);
download.addEventListener("click", () => downloadResult(currentResult));

input.addEventListener("change", () => {
  const file = input.files?.[0];
  if (!file) return;
  generation += 1;
  controller?.abort();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  currentResult = null;
  selectedFile = null;
  updateCounts(null);
  resultImage.hidden = true;
  preview.hidden = true;
  download.disabled = true;
  overlay.hidden = true;
  document.getElementById("result-status").textContent = "Menunggu deteksi";
  setError();
  reset.disabled = false;
  choose.disabled = false;
  detect.disabled = true;
  if (file.size > 15 * 1024 * 1024) {
    setError("Ukuran gambar melebihi 15 MB. Pilih gambar yang lebih kecil.");
    placeholder.hidden = false;
    setStatus("Gambar belum dapat diproses.");
    return;
  }
  selectedFile = file;
  detect.disabled = false;
  document.getElementById("image-state").textContent = "Siap dideteksi";
  setStatus(`${file.name} dipilih. Tekan Deteksi untuk memproses.`);
  const version = generation;
  previewUrl = URL.createObjectURL(file);
  preview.onload = () => {
    if (version !== generation) return;
    preview.hidden = false;
    placeholder.hidden = true;
  };
  preview.onerror = () => {
    if (version !== generation) return;
    preview.hidden = true;
    placeholder.hidden = false;
    placeholder.querySelector("h3").textContent = "Foto siap dikirim";
    placeholder.querySelector("p").textContent = "Pratinjau belum tersedia pada browser ini. Tekan Deteksi untuk memproses gambar di server.";
  };
  preview.src = previewUrl;
});

detect.addEventListener("click", async () => {
  if (!selectedFile || detect.disabled) return;
  const version = generation;
  controller = new AbortController();
  detect.disabled = true;
  reset.disabled = true;
  choose.disabled = true;
  download.disabled = true;
  overlay.hidden = false;
  setError();
  setStatus("Sedang mendeteksi biji kakao…");
  try {
    const result = await predictImage(selectedFile, { mode: "upload", signal: controller.signal });
    const imageUrl = await prepareResultImage(result);
    if (version !== generation || controller.signal.aborted) return;
    currentResult = result;
    resultImage.src = imageUrl;
    resultImage.hidden = false;
    preview.hidden = true;
    placeholder.hidden = true;
    updateCounts(result);
    download.disabled = false;
    document.getElementById("image-state").textContent = "Hasil deteksi";
    document.getElementById("result-status").textContent = "Deteksi selesai";
    const saved = await saveResult(result, "upload");
    if (version !== generation) return;
    const message = result.total === 0 ? "Deteksi selesai. Tidak ada biji yang terdeteksi pada ambang saat ini." : `Deteksi selesai. ${result.total} biji terdeteksi.`;
    setStatus(`${message} ${saved.warning || (saved.saved ? "Hasil tersimpan di riwayat browser." : "")}`);
  } catch (error) {
    if (error.name !== "AbortError" && version === generation) {
      setError(error.message);
      setStatus("Deteksi belum berhasil. Anda dapat mencoba kembali.");
      download.disabled = !currentResult;
    }
  } finally {
    if (version === generation) {
      overlay.hidden = true;
      detect.disabled = !selectedFile;
      reset.disabled = false;
      choose.disabled = false;
    }
  }
});

window.addEventListener("pagehide", () => {
  generation += 1;
  controller?.abort();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
