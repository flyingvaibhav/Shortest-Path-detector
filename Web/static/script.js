const elements = {
  fileInput: document.getElementById("fileInput"),
  dropzone: document.getElementById("dropzone"),
  startBtn: document.getElementById("startBtn"),
  clearBtn: document.getElementById("clearBtn"),
  previewImg: document.getElementById("previewImg"),
  previewVideo: document.getElementById("previewVideo"),
  outputPanel: document.getElementById("outputPanel"),
  logs: document.getElementById("logs"),
  annotatedImg: document.getElementById("annotated"),
  videoResult: document.getElementById("videoResult"),
  downloadLink: document.getElementById("downloadLink"),
  countTag: document.getElementById("count"),
  progressBar: document.getElementById("progressBar"),
  statusLabel: document.getElementById("statusLabel"),
  analysisBtn: document.getElementById("analysisBtn"),
};

const state = {
  currentPreviewUrl: null,
  eventSource: null,
};

const ANALYSIS_STORAGE_KEY = "treesenseUrbanResult";

function init() {
  if (!elements.dropzone || !elements.startBtn) {
    return;
  }

  elements.dropzone.addEventListener("click", () => elements.fileInput.click());
  elements.dropzone.addEventListener("dragover", handleDragOver);
  elements.dropzone.addEventListener("dragleave", handleDragLeave);
  elements.dropzone.addEventListener("drop", handleDrop);
  elements.dropzone.addEventListener("keydown", handleKeyDown);
  elements.fileInput.addEventListener("change", handlePreview);

  elements.startBtn.addEventListener("click", startDetection);
  elements.clearBtn.addEventListener("click", clearAll);

  if (elements.analysisBtn) {
    elements.analysisBtn.addEventListener("click", handleAnalysisNavigation);
    if (hasStoredAnalysis()) {
      enableAnalysisButton();
    } else {
      disableAnalysisButton();
    }
  }
}

function handleDragOver(event) {
  event.preventDefault();
  elements.dropzone.style.background = "rgba(46,224,106,0.15)";
}

function handleDragLeave() {
  elements.dropzone.style.background = "rgba(46,224,106,0.05)";
}

function handleDrop(event) {
  event.preventDefault();
  elements.fileInput.files = event.dataTransfer.files;
  handlePreview();
  handleDragLeave();
}

function handleKeyDown(event) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
}

function handlePreview() {
  const file = elements.fileInput.files[0];
  if (!file) {
    return;
  }

  if (state.currentPreviewUrl) {
    URL.revokeObjectURL(state.currentPreviewUrl);
  }
  state.currentPreviewUrl = URL.createObjectURL(file);

  elements.previewImg.style.display = "none";
  elements.previewVideo.style.display = "none";

  if (file.type.startsWith("image/")) {
    elements.previewImg.src = state.currentPreviewUrl;
    elements.previewImg.style.display = "block";
  } else {
    elements.previewVideo.src = state.currentPreviewUrl;
    elements.previewVideo.style.display = "block";
  }
}

function clearAll() {
  elements.fileInput.value = "";
  if (state.currentPreviewUrl) {
    URL.revokeObjectURL(state.currentPreviewUrl);
    state.currentPreviewUrl = null;
  }

  elements.previewImg.style.display = "none";
  elements.previewVideo.style.display = "none";
  resetOutputPanel();
  if (window.sessionStorage) {
    sessionStorage.removeItem(ANALYSIS_STORAGE_KEY);
  }
  disableAnalysisButton();
}

function resetOutputPanel(hidePanel = true) {
  elements.annotatedImg.style.display = "none";
  elements.videoResult.style.display = "none";
  elements.downloadLink.style.display = "none";
  elements.logs.textContent = "Waiting for detection...";
  elements.countTag.textContent = "";
  setProgress(0);
  setStatus("Idle");
  if (hidePanel) {
    elements.outputPanel.style.display = "none";
  }
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
}

function setProgress(value) {
  const clamped = Math.max(0, Math.min(1, value || 0));
  elements.progressBar.style.width = `${Math.round(clamped * 100)}%`;
}

function setStatus(text) {
  elements.statusLabel.textContent = text;
}

function logMessage(message) {
  const timestamp = new Date().toLocaleTimeString();
  elements.logs.textContent += `[${timestamp}] ${message}\n`;
  elements.logs.scrollTop = elements.logs.scrollHeight;
}

async function startDetection() {
  const file = elements.fileInput.files[0];
  if (!file) {
    alert("Please upload an image or video first.");
    return;
  }

  elements.outputPanel.style.display = "flex";
  resetOutputPanel(false);
  elements.startBtn.disabled = true;
  disableAnalysisButton();
  setStatus("Uploading...");
  logMessage("Uploading file...");

  try {
    const jobId = await uploadFile(file);
    logMessage(`Detection job started (ID: ${jobId})`);
    listenToProgress(jobId);
  } catch (error) {
    handleError(error.message || "Upload failed");
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/upload", { method: "POST", body: formData });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || "Upload failed");
  }

  return payload.job_id;
}

function listenToProgress(jobId) {
  if (!window.EventSource) {
    logMessage("EventSource not supported in this browser.");
    fetchResult(jobId);
    return;
  }

  state.eventSource = new EventSource(`/progress/${jobId}`);
  state.eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.log) {
      logMessage(data.log);
    }
    if (typeof data.progress === "number") {
      setProgress(data.progress);
    }
    if (data.status) {
      setStatus(data.status.toUpperCase());
    }

    if (data.done || data.status === "done" || data.status === "error") {
      state.eventSource.close();
      state.eventSource = null;
      if (data.status === "error") {
        handleError("Detection failed. Check logs for details.");
      } else {
        fetchResult(jobId);
      }
    }
  };

  state.eventSource.onerror = () => {
    logMessage("⚠️ Lost connection to server. Retrying via result endpoint...");
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    fetchResult(jobId);
  };
}

async function fetchResult(jobId) {
  setStatus("Finalizing...");
  logMessage("Fetching results...");

  let payload;
  let response;
  try {
    response = await fetch(`/result/${jobId}`);
    payload = await response.json();
  } catch (error) {
    handleError(error.message || "Unable to reach server");
    return;
  }

  if (!response.ok) {
    handleError(payload.error || "Result retrieval failed");
    return;
  }

  if (payload.type === "image") {
    showImageResult(payload);
  } else if (payload.type === "video") {
    showVideoResult(payload);
  } else {
    logMessage("Unexpected result payload received.");
  }

  storeAnalysisPayload(payload);
  enableAnalysisButton();

  logMessage("✅ Detection complete!");
  setStatus("Done");
  elements.startBtn.disabled = false;
}

function showImageResult(data) {
  elements.annotatedImg.src = data.annotated_image_base64;
  elements.annotatedImg.style.display = "block";
  elements.videoResult.style.display = "none";
  elements.countTag.textContent = `Trees detected: ${data.count}`;
  setupDownload(data.output_image_url, "tree-detections.png");
}

function showVideoResult(data) {
  elements.videoResult.src = data.output_video_url;
  elements.videoResult.style.display = "block";
  elements.annotatedImg.style.display = "none";
  elements.countTag.textContent = `Unique trees tracked: ${data.unique_tree_count}`;
  setupDownload(data.output_video_url, "tree-detections.mp4");
}

function setupDownload(url, filename) {
  if (!url) {
    elements.downloadLink.style.display = "none";
    return;
  }
  elements.downloadLink.href = url;
  elements.downloadLink.download = filename;
  elements.downloadLink.style.display = "inline-block";
}

function handleAnalysisNavigation() {
  if (!hasStoredAnalysis()) {
    alert("Run a detection to generate analysis first.");
    return;
  }
  window.location.href = "/urbananalysis";
}

function storeAnalysisPayload(payload) {
  if (!window.sessionStorage) {
    return;
  }
  try {
    const enriched = {
      ...payload,
      stored_at: new Date().toISOString(),
    };
    sessionStorage.setItem(ANALYSIS_STORAGE_KEY, JSON.stringify(enriched));
  } catch (error) {
    console.warn("Unable to persist analysis payload", error);
  }
}

function enableAnalysisButton() {
  if (!elements.analysisBtn) {
    return;
  }
  elements.analysisBtn.disabled = false;
  elements.analysisBtn.classList.remove("hidden");
  elements.analysisBtn.style.display = "inline-flex";
}

function disableAnalysisButton() {
  if (!elements.analysisBtn) {
    return;
  }
  elements.analysisBtn.disabled = true;
  elements.analysisBtn.classList.add("hidden");
  elements.analysisBtn.style.display = "none";
}

function hasStoredAnalysis() {
  if (!window.sessionStorage) {
    return false;
  }
  try {
    return Boolean(sessionStorage.getItem(ANALYSIS_STORAGE_KEY));
  } catch (error) {
    console.warn("Unable to read stored analysis", error);
    return false;
  }
}

function handleError(message) {
  logMessage(`❌ ${message}`);
  setStatus("Error");
  elements.startBtn.disabled = false;
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
