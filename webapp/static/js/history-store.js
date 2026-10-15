import { CLASS_NAMES, resultDataUrl } from "./common.js";

export const HISTORY_KEY = "deteksi-kakao.history.v1";
export const HISTORY_LIMIT = 10;

function validRecord(item) {
  return item && typeof item.id === "string" && typeof item.created_at === "string"
    && !Number.isNaN(new Date(item.created_at).getTime())
    && ["camera", "upload"].includes(item.mode)
    && item.counts && CLASS_NAMES.every((key) => Number.isSafeInteger(item.counts[key]) && item.counts[key] >= 0)
    && Number.isSafeInteger(item.total) && item.total === CLASS_NAMES.reduce((sum, key) => sum + item.counts[key], 0)
    && typeof item.thumbnail === "string" && item.thumbnail.startsWith("data:image/jpeg;base64,")
    && item.thumbnail.length <= 400000;
}

export function readHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return { items: [], warning: "" };
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) throw new Error("Invalid history");
    const items = parsed.filter(validRecord).slice(0, HISTORY_LIMIT);
    return { items, warning: items.length !== Math.min(parsed.length, HISTORY_LIMIT) ? "Sebagian riwayat tidak dapat dibaca dan tidak ditampilkan." : "" };
  } catch (error) {
    const corrupted = error instanceof SyntaxError || error.message === "Invalid history";
    return { items: [], warning: corrupted ? "Riwayat tersimpan tidak dapat dibaca. Hasil deteksi baru tetap dapat digunakan." : "Penyimpanan browser tidak tersedia. Riwayat tidak dapat disimpan pada perangkat ini." };
  }
}

async function makeThumbnail(result) {
  const image = new Image();
  image.src = resultDataUrl(result);
  await new Promise((resolve, reject) => {
    if (image.complete && image.naturalWidth) resolve();
    else {
      image.onload = resolve;
      image.onerror = () => reject(new Error("Thumbnail tidak dapat dibuat."));
    }
  });
  const scale = Math.min(1, 480 / Math.max(image.naturalWidth, image.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Thumbnail tidak dapat dibuat.");
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.7);
}

export async function saveResult(result, mode) {
  const previous = readHistory();
  if (previous.items.some((item) => item.id === result.result_id)) return { saved: true, warning: "", duplicate: true };
  let thumbnail;
  try { thumbnail = await makeThumbnail(result); }
  catch { return { saved: false, warning: "Hasil berhasil ditampilkan, tetapi pratinjau riwayat tidak dapat dibuat. Anda masih dapat mengunduh hasil." }; }
  const item = {
    id: result.result_id,
    created_at: result.created_at,
    mode,
    total: result.total,
    counts: Object.fromEntries(CLASS_NAMES.map((key) => [key, result.counts[key]])),
    thumbnail,
  };
  // Read again after image decoding so simultaneous completed requests do not replace a newer save.
  const latest = readHistory();
  let items = [item, ...latest.items.filter((entry) => entry.id !== item.id)].slice(0, HISTORY_LIMIT);
  let discarded = false;
  while (items.length) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
      return { saved: true, warning: discarded ? "Hasil tersimpan. Riwayat lama dikurangi karena penyimpanan browser hampir penuh." : previous.warning };
    } catch (error) {
      const quota = error.name === "QuotaExceededError" || error.name === "NS_ERROR_DOM_QUOTA_REACHED" || error.code === 22;
      if (!quota) return { saved: false, warning: "Hasil berhasil ditampilkan, tetapi penyimpanan browser tidak tersedia. Gunakan Unduh Hasil untuk menyimpan gambar." };
      items = items.slice(0, -1);
      discarded = true;
    }
  }
  return { saved: false, warning: "Penyimpanan browser penuh. Hasil tetap tersedia dan dapat diunduh, tetapi belum masuk riwayat." };
}

export function deleteHistory(id) {
  const current = readHistory();
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(current.items.filter((item) => item.id !== id)));
    return { deleted: true, warning: "" };
  } catch {
    return { deleted: false, warning: "Riwayat belum dapat dihapus karena penyimpanan browser tidak tersedia." };
  }
}
