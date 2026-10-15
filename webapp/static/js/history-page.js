import { formatDate, modeLabel, updateCounts } from "./common.js";
import { readHistory, deleteHistory, HISTORY_KEY } from "./history-store.js";

const list = document.getElementById("history-list");
const empty = document.getElementById("history-empty");
const notice = document.getElementById("history-notice");
const dialog = document.getElementById("history-detail");

function showNotice(message) {
  notice.textContent = message;
  notice.hidden = !message;
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function openDetail(item) {
  document.getElementById("detail-image").src = item.thumbnail;
  document.getElementById("detail-meta").textContent = `${modeLabel(item.mode)} · ${formatDate(item.created_at)}`;
  updateCounts(item, document.getElementById("detail-stats"));
  dialog.showModal();
}

function render() {
  const { items, warning } = readHistory();
  list.replaceChildren();
  showNotice(warning);
  empty.hidden = items.length !== 0;
  items.forEach((item) => {
    const card = node("article", "history-card");
    card.dataset.resultId = item.id;
    const summary = node("div", "history-summary");
    const image = node("img", "history-thumbnail");
    image.src = item.thumbnail;
    image.alt = "Pratinjau hasil deteksi biji kakao";
    image.loading = "lazy";
    const copy = node("div", "history-copy");
    const date = node("time", "", formatDate(item.created_at));
    date.dateTime = item.created_at;
    copy.append(node("h2", "", modeLabel(item.mode)), date, node("p", "history-total", `Total: ${item.total} biji`), node("p", "history-counts", `Tidak: ${item.counts["Tidak Fermentasi"]} · Setengah: ${item.counts["Setengah Fermentasi"]} · Full: ${item.counts["Full Fermentasi"]}`));
    summary.append(image, copy);
    const actions = node("div", "history-actions");
    const view = node("button", "button button-secondary", "Lihat");
    view.type = "button";
    view.dataset.action = "view";
    view.setAttribute("aria-label", `Lihat hasil ${formatDate(item.created_at)}`);
    view.addEventListener("click", () => openDetail(item));
    const remove = node("button", "button button-danger", "Hapus");
    remove.type = "button";
    remove.dataset.action = "delete";
    remove.setAttribute("aria-label", `Hapus hasil ${formatDate(item.created_at)}`);
    remove.addEventListener("click", () => {
      if (!window.confirm("Hapus hasil ini dari riwayat browser?")) return;
      const result = deleteHistory(item.id);
      if (result.deleted) render();
      else showNotice(result.warning);
    });
    actions.append(view, remove);
    card.append(summary, actions);
    list.append(card);
  });
}

document.getElementById("close-detail").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});
window.addEventListener("storage", (event) => {
  if (event.key === HISTORY_KEY || event.key === null) { dialog.close(); render(); }
});
render();
