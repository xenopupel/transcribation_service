const state = {
  config: {
    audio_extensions: [".mp3", ".wav", ".ogg", ".flac", ".m4a"],
    max_upload_mb: 500,
  },
  items: [],
  polling: null,
  archiveDownloaded: false,
  notice: "",
};

const dropzone = document.querySelector("#dropzone");
const fileInput = document.querySelector("#fileInput");
const uploadBtn = document.querySelector("#uploadBtn");
const downloadBtn = document.querySelector("#downloadBtn");
const progress = document.querySelector(".progress");
const progressBar = document.querySelector("#progressBar");
const progressLabel = document.querySelector("#progressLabel");
const progressCount = document.querySelector("#progressCount");
const limits = document.querySelector("#limits");
const includeIvr = document.querySelector("#includeIvr");
const selectedCount = document.querySelector("#selectedCount");
const doneCount = document.querySelector("#doneCount");
const queueValue = document.querySelector("#queueValue");
const queueLabel = document.querySelector("#queueLabel");
const errorCount = document.querySelector("#errorCount");
const summary = document.querySelector(".summary");
const notice = document.querySelector("#notice");

function extensionOf(name) {
  const index = name.lastIndexOf(".");
  return index >= 0 ? name.slice(index).toLowerCase() : "";
}

function validateFile(file) {
  const ext = extensionOf(file.name);
  const maxBytes = state.config.max_upload_mb * 1024 * 1024;
  if (!state.config.audio_extensions.includes(ext)) {
    return `Файл "${file.name}" не добавлен: можно загружать только ${state.config.audio_extensions.join(", ")}.`;
  }
  if (file.size > maxBytes) {
    return `Файл "${file.name}" не добавлен: размер больше лимита ${state.config.max_upload_mb} MB.`;
  }
  return "";
}

function resetBatch() {
  stopPolling();
  state.items = [];
  state.archiveDownloaded = false;
  state.notice = "";
  fileInput.value = "";
}

function addFiles(files) {
  if (state.archiveDownloaded && activeItems().length === 0) {
    resetBatch();
  }

  const existing = new Set(state.items.map((item) => `${item.file.name}:${item.file.size}`));
  const rejectedMessages = [];
  const addedItems = [];

  for (const file of files) {
    const key = `${file.name}:${file.size}`;
    if (existing.has(key)) {
      continue;
    }

    const error = validateFile(file);
    if (error) {
      rejectedMessages.push(error);
      continue;
    }

    existing.add(key);
    const item = {
      file,
      jobId: null,
      status: "uploading",
      message: "",
      queuePosition: null,
      uploadedBytes: 0,
    };
    addedItems.push(item);
    state.items.push(item);
  }

  if (rejectedMessages.length > 0) {
    state.notice = rejectedMessages.slice(0, 2).join(" ");
  } else if (addedItems.length > 0) {
    state.notice = "";
  }

  fileInput.value = "";
  render();

  for (const item of addedItems) {
    uploadFileToServer(item);
  }
}

function uploadItems() {
  return state.items.filter((item) => item.status === "uploading");
}

function activeItems() {
  return state.items.filter((item) =>
    ["queued", "processing"].includes(item.status),
  );
}

function doneItems() {
  return state.items.filter((item) => item.status === "done");
}

function failedItems() {
  return state.items.filter((item) => item.status === "failed");
}

function allUploaded() {
  return state.items.length > 0 && state.items.every((item) => item.status === "uploaded");
}

function allDone() {
  return state.items.length > 0 && state.items.every((item) => item.status === "done");
}

function uploadProgressPercent() {
  const totalBytes = state.items.reduce((sum, item) => sum + item.file.size, 0);
  if (totalBytes === 0) {
    return 0;
  }
  const uploadedBytes = state.items.reduce((sum, item) => {
    if (["uploaded", "queued", "processing", "done"].includes(item.status)) {
      return sum + item.file.size;
    }
    return sum + item.uploadedBytes;
  }, 0);
  return Math.max(0, Math.min(100, Math.round((uploadedBytes / totalBytes) * 100)));
}

function queueSummary() {
  if (state.items.some((item) => item.status === "processing")) {
    return { value: "идет", label: "обработка" };
  }

  const queuedPositions = state.items
    .filter((item) => item.status === "queued" && Number.isInteger(item.queuePosition))
    .map((item) => item.queuePosition);

  if (queuedPositions.length === 0) {
    return { value: "-", label: "очередь" };
  }

  const firstPosition = Math.min(...queuedPositions);
  const ahead = Math.max(firstPosition - 1, 0);
  return { value: String(ahead), label: "перед вами" };
}

function render() {
  const total = state.items.length;
  const uploading = uploadItems().length;
  const done = doneItems().length;
  const active = activeItems().length;
  const failed = failedItems().length;
  const finished = done + failed;
  const queue = queueSummary();
  const busy = uploading > 0 || active > 0;

  if (uploading > 0) {
    const percent = uploadProgressPercent();
    progressBar.style.width = `${percent}%`;
    progressCount.textContent = `${percent}%`;
  } else {
    progressBar.style.width = total ? `${Math.round((finished / total) * 100)}%` : "0%";
    progressCount.textContent = `${finished} / ${total}`;
  }

  progress.classList.toggle("active", busy);
  progressLabel.classList.toggle("busy", busy);
  summary.classList.toggle("busy", busy);

  if (total === 0) {
    progressLabel.textContent = "Файлы не выбраны";
  } else if (uploading > 0) {
    progressLabel.textContent = `Загрузка файлов на сервер: ${total - uploading} из ${total} загружено`;
  } else if (allUploaded()) {
    progressLabel.textContent = "Файлы загружены на сервер, можно запускать обработку";
  } else if (active > 0) {
    progressLabel.textContent = `${done} из ${total} обработано. Сервис работает, страницу можно не обновлять`;
  } else if (allDone()) {
    progressLabel.textContent = "Все файлы обработаны, архив готов к скачиванию";
  } else if (failed > 0) {
    progressLabel.textContent = "Обработка завершилась с ошибками";
  } else {
    progressLabel.textContent = `${total} файлов ожидают загрузки`;
  }

  selectedCount.textContent = String(total);
  doneCount.textContent = String(done);
  queueValue.textContent = queue.value;
  queueLabel.textContent = queue.label;
  errorCount.textContent = String(failed);
  notice.textContent = state.notice;

  uploadBtn.disabled = !allUploaded() || busy;
  downloadBtn.disabled = !allDone();
}

function uploadFileToServer(item) {
  const xhr = new XMLHttpRequest();
  const body = new FormData();
  body.append("file", item.file);

  const query = includeIvr.checked ? "&include_ivr=true" : "";
  xhr.open("POST", `/jobs?enqueue=false${query}`);

  xhr.upload.addEventListener("progress", (event) => {
    if (event.lengthComputable) {
      item.uploadedBytes = event.loaded;
      render();
    }
  });

  xhr.addEventListener("load", () => {
    if (xhr.status < 200 || xhr.status >= 300) {
      item.status = "failed";
      item.message = responseError(xhr.responseText, xhr.status);
      render();
      return;
    }

    const payload = JSON.parse(xhr.responseText);
    item.jobId = payload.job_id;
    item.status = payload.status;
    item.uploadedBytes = item.file.size;
    render();
  });

  xhr.addEventListener("error", () => {
    item.status = "failed";
    item.message = "Не удалось загрузить файл на сервер";
    render();
  });

  xhr.send(body);
}

function responseError(text, statusCode) {
  try {
    const payload = JSON.parse(text);
    return payload.detail || `HTTP ${statusCode}`;
  } catch {
    return `HTTP ${statusCode}`;
  }
}

async function startProcessing() {
  state.archiveDownloaded = false;
  state.notice = "";

  const jobIds = state.items
    .filter((item) => item.status === "uploaded" && item.jobId)
    .map((item) => item.jobId);

  const response = await fetch("/jobs/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_ids: jobIds }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    state.notice = payload.detail || `Не удалось запустить обработку: HTTP ${response.status}`;
    render();
    return;
  }

  for (const item of state.items) {
    if (item.status === "uploaded") {
      item.status = "queued";
    }
  }
  render();
  startPolling();
}

async function pollJobs() {
  const active = state.items.filter((item) =>
    item.jobId && ["queued", "processing"].includes(item.status),
  );
  if (active.length === 0) {
    stopPolling();
    render();
    return;
  }

  for (const item of active) {
    try {
      const response = await fetch(`/jobs/${item.jobId}`);
      if (!response.ok) {
        continue;
      }
      const payload = await response.json();
      item.status = payload.status;
      item.message = payload.error_message || "";
      item.queuePosition = payload.queue_position;
    } catch {
      item.message = "Не удалось обновить статус";
    }
  }
  render();
}

function startPolling() {
  stopPolling();
  pollJobs();
  state.polling = window.setInterval(pollJobs, 2000);
}

function stopPolling() {
  if (state.polling) {
    window.clearInterval(state.polling);
    state.polling = null;
  }
}

function downloadArchive() {
  if (!allDone()) {
    return;
  }

  const params = new URLSearchParams();
  for (const item of state.items) {
    if (item.status === "done" && item.jobId) {
      params.append("job_id", item.jobId);
    }
  }
  window.location.href = `/jobs/archive?${params.toString()}`;
  state.archiveDownloaded = true;
  render();
}

async function loadConfig() {
  const response = await fetch("/config");
  if (!response.ok) {
    throw new Error("Не удалось загрузить настройки API");
  }
  state.config = await response.json();
  fileInput.accept = state.config.audio_extensions.join(",");
  limits.textContent = `Форматы: ${state.config.audio_extensions.join(", ")}. Лимит: ${state.config.max_upload_mb} MB на файл. Работает только для стерео звонков.`;
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  addFiles(event.dataTransfer.files);
});
fileInput.addEventListener("change", () => addFiles(fileInput.files));
uploadBtn.addEventListener("click", startProcessing);
downloadBtn.addEventListener("click", downloadArchive);

loadConfig()
  .catch((error) => {
    limits.textContent = error.message;
  })
  .finally(render);
