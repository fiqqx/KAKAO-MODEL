export const CLASS_NAMES = ["Tidak Fermentasi", "Setengah Fermentasi", "Full Fermentasi"];

export function setError(message = "") {
  const target = document.getElementById("error-message");
  if (!target) return;
  target.textContent = message;
  target.hidden = !message;
}

export function setStatus(message) {
  const target = document.getElementById("page-status");
  if (target) target.textContent = message;
}

export function updateCounts(result, container = document) {
  container.querySelectorAll("[data-count]").forEach((element) => {
    const key = element.dataset.count;
    element.textContent = result ? String(key === "total" ? result.total : result.counts[key]) : "—";
  });
}

export function resultDataUrl(result) {
  return `data:${result.image_mime};base64,${result.image_base64}`;
}

export async function prepareResultImage(result) {
  const preview = new Image();
  preview.src = resultDataUrl(result);
  await new Promise((resolve, reject) => {
    if (preview.complete && preview.naturalWidth) resolve();
    else {
      preview.onload = resolve;
      preview.onerror = () => reject(new Error("Gambar hasil tidak dapat ditampilkan. Silakan coba lagi."));
    }
  });
  return preview.src;
}

export async function predictImage(image, { mode = "upload", signal } = {}) {
  const body = new FormData();
  body.append("image", image, image.name || "kamera.jpg");
  body.append("mode", mode);
  let response;
  try {
    response = await fetch("/api/predict", { method: "POST", body, signal });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("Tidak dapat terhubung ke server. Periksa koneksi lalu coba kembali.");
  }
  let result;
  try { result = await response.json(); }
  catch { throw new Error("Server belum dapat mengirim hasil. Silakan coba kembali."); }
  if (!response.ok) throw new Error(result.error || "Deteksi gagal. Silakan coba kembali.");
  const validCounts = result.counts && CLASS_NAMES.every((key) => Number.isSafeInteger(result.counts[key]) && result.counts[key] >= 0);
  if (!validCounts || !Number.isSafeInteger(result.total) || result.total !== CLASS_NAMES.reduce((sum, key) => sum + result.counts[key], 0)
      || typeof result.result_id !== "string" || typeof result.image_base64 !== "string"
      || !["image/jpeg", "image/png", "image/webp"].includes(result.image_mime)) {
    throw new Error("Format hasil server tidak sesuai. Silakan periksa konfigurasi website.");
  }
  return result;
}

export function downloadResult(result) {
  if (!result) return;
  const link = document.createElement("a");
  const extension = result.image_mime === "image/png" ? "png" : result.image_mime === "image/webp" ? "webp" : "jpg";
  link.href = resultDataUrl(result);
  link.download = `hasil-kakao-${result.result_id.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 80)}.${extension}`;
  document.body.append(link);
  link.click();
  link.remove();
}

export function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Waktu tidak tersedia";
  return new Intl.DateTimeFormat("id-ID", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export function modeLabel(mode) {
  return mode === "camera" ? "Kamera Langsung" : "Unggah Gambar";
}
