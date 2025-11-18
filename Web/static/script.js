const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const logsDiv = document.getElementById("logs");
const progressBar = document.getElementById("progressBar");
const countP = document.getElementById("count");
const annotatedImg = document.getElementById("annotated");
const videoResult = document.getElementById("videoResult");

function logMessage(msg) {
  const time = new Date().toLocaleTimeString();
  logsDiv.innerHTML += `[${time}] ${msg}<br>`;
  logsDiv.scrollTop = logsDiv.scrollHeight;
}

uploadBtn.onclick = async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select a file first!");
    return;
  }

  uploadBtn.disabled = true;
  logsDiv.innerHTML = "";
  countP.textContent = "";
  annotatedImg.style.display = "none";
  videoResult.style.display = "none";
  progressBar.style.width = "0%";

  logMessage("Uploading file...");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const { job_id } = await res.json();

    logMessage(`Detection job started (ID: ${job_id})`);

    const evtSource = new EventSource(`/progress/${job_id}`);

    evtSource.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.log) logMessage(data.log);
      if (data.progress) progressBar.style.width = `${Math.round(data.progress * 100)}%`;

      if (data.status === "done" || data.status === "error") {
        evtSource.close();
        fetch(`/result/${job_id}`)
          .then((r) => r.json())
          .then((result) => {
            if (data.status === "error") {
              logMessage("❌ Detection failed. Check logs for details.");
              uploadBtn.disabled = false;
              return;
            }
            logMessage("✅ Detection complete!");
            progressBar.style.width = "100%";

            if (result.type === "image") {
              annotatedImg.src = result.annotated_image_base64;
              annotatedImg.style.display = "block";
              countP.textContent = `Trees detected: ${result.count}`;
            } else if (result.type === "video") {
              videoResult.src = result.output_video_url;
              videoResult.style.display = "block";
              countP.textContent = `Average trees per frame: ${result.avg_tree_count.toFixed(2)}`;
            }
            uploadBtn.disabled = false;
          });
      }
    };

    evtSource.onerror = (err) => {
      logMessage("⚠️ Lost connection to server.");
      evtSource.close();
      uploadBtn.disabled = false;
    };
  } catch (err) {
    logMessage(`Error: ${err.message}`);
    uploadBtn.disabled = false;
  }
};
