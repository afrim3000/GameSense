const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropContent = document.getElementById("dropContent");
const previewImg = document.getElementById("previewImg");
const predictBtn = document.getElementById("predictBtn");
const resultCard = document.getElementById("resultCard");
const gameName = document.getElementById("gameName");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const hashtagsEl = document.getElementById("hashtags");
const topPredsList = document.getElementById("topPredsList");
const errorBox = document.getElementById("errorBox");
const loadingBox = document.getElementById("loadingBox");
const copyBtn = document.getElementById("copyBtn");
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");

let selectedFile = null;
let modelReady = false;
let healthPollId = null;

// --- Fetch with a hard timeout, so a stalled request can never spin forever ---
function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
}

// ---- Model status pill (polls /api/health until the model is loaded) ----
async function checkHealth() {
  try {
    const res = await fetchWithTimeout("/api/health", {}, 8000);
    const data = await res.json();
    modelReady = !!data.model_loaded;

    statusPill.classList.remove("checking", "offline");
    if (modelReady) {
      statusText.textContent = "Model ready";
      if (healthPollId) {
        clearInterval(healthPollId);
        healthPollId = null;
      }
    } else {
      statusPill.classList.add("offline");
      statusText.textContent = "Model downloading…";
    }
  } catch (err) {
    modelReady = false;
    statusPill.classList.remove("checking");
    statusPill.classList.add("offline");
    statusText.textContent = "Server unreachable";
  }
}

checkHealth();
healthPollId = setInterval(checkHealth, 5000);

// ---- Selecting a file (click or drag/drop) ----
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please select an image file (PNG, JPG, etc.).");
    return;
  }
  hideError();
  selectedFile = file;
  predictBtn.disabled = false;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    dropContent.hidden = true;
    dropzone.classList.add("has-image");
  };
  reader.readAsDataURL(file);

  resultCard.hidden = true;
}

// ---- Prediction ----
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  if (!modelReady) {
    showError("The model isn't finished loading yet — check the status pill above, and the server terminal for [download_model] progress.");
    return;
  }

  hideError();
  resultCard.hidden = true;
  loadingBox.hidden = false;
  predictBtn.disabled = true;
  dropzone.classList.add("scanning");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    // 60s ceiling: generous for a slow first inference, but guarantees the
    // spinner always resolves into a message instead of spinning forever.
    const res = await fetchWithTimeout("/api/predict", { method: "POST", body: formData }, 60000);
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }
    renderResult(data);
  } catch (err) {
    if (err.name === "AbortError") {
      showError("The request timed out after 60s. The server may still be starting up — try again in a moment.");
    } else {
      showError(err.message || "Something went wrong while contacting the server.");
    }
  } finally {
    loadingBox.hidden = true;
    predictBtn.disabled = false;
    dropzone.classList.remove("scanning");
  }
});

function renderResult(data) {
  gameName.textContent = data.game;

  const pct = Math.round(data.confidence * 100);
  confidenceFill.style.width = pct + "%";
  confidenceText.textContent = `${pct}% confidence`;

  hashtagsEl.innerHTML = "";
  data.hashtags.forEach((tag) => {
    const chip = document.createElement("span");
    chip.className = "hashtag-chip";
    chip.textContent = tag;
    hashtagsEl.appendChild(chip);
  });

  topPredsList.innerHTML = "";
  data.top_predictions.slice(1).forEach((p) => {
    const li = document.createElement("li");
    li.textContent = `${p.game} — ${Math.round(p.confidence * 100)}%`;
    topPredsList.appendChild(li);
  });

  resultCard.hidden = false;
}

copyBtn.addEventListener("click", () => {
  const tags = Array.from(hashtagsEl.children).map((c) => c.textContent).join(" ");
  navigator.clipboard.writeText(tags).then(() => {
    copyBtn.textContent = "Copied!";
    setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
  });
});

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}
function hideError() {
  errorBox.hidden = true;
}
