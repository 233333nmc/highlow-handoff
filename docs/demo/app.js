async function loadDemoData() {
  try {
    const response = await fetch("data/demos.json");
    if (response.ok) {
      return response.json();
    }
  } catch (error) {
    // Direct file:// opens cannot fetch JSON in some browsers; demos.js
    // keeps the page usable without a local server.
  }
  if (!window.ROBOTWIN_DEMOS) {
    throw new Error("Failed to load demo data.");
  }
  return window.ROBOTWIN_DEMOS;
}

function formatMs(value) {
  return `${Number(value).toFixed(1)} ms`;
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderSummary(data) {
  const el = document.getElementById("summaryMetrics");
  el.innerHTML = [
    metric("Model", data.summary.model),
    metric("Planner modes", data.summary.planner_modes.join(" / ")),
    metric("Overall median", formatMs(data.summary.overall_median_ms)),
    metric("Samples", `${data.summary.samples} rollouts`),
  ].join("");
}

function renderTaskList(data, activeIndex, onSelect) {
  const list = document.getElementById("taskList");
  document.getElementById("taskCount").textContent = `${data.tasks.length}`;
  list.innerHTML = "";
  data.tasks.forEach((task, index) => {
    const button = document.createElement("button");
    button.className = `task-button${index === activeIndex ? " active" : ""}`;
    button.type = "button";
    button.innerHTML = `
      <strong>${task.name}</strong>
      <span>${task.planner_mode} · ${formatMs(task.mean_ms)}</span>
    `;
    button.addEventListener("click", () => onSelect(index));
    list.appendChild(button);
  });
}

function renderTimeline(task) {
  const timeline = document.getElementById("timeline");
  timeline.innerHTML = `
    <div class="step">
      <span>Current subtask</span>
      <strong>${task.current_subtask}</strong>
    </div>
    <div class="step">
      <span>Verifier</span>
      <strong>${task.verifier.action} · ${task.verifier.status}</strong>
    </div>
    <div class="step">
      <span>Next instruction</span>
      <strong>${task.next_subtask}</strong>
    </div>
  `;
}

function renderVideo(task) {
  const video = document.getElementById("demoVideo");
  const placeholder = document.getElementById("videoPlaceholder");
  const placeholderText = document.getElementById("placeholderText");

  video.pause();
  video.removeAttribute("src");
  video.load();

  if (!task.video) {
    video.style.display = "none";
    placeholder.style.display = "grid";
    placeholderText.textContent = `No video assigned yet. Add ${task.name}.mp4 under docs/demo/assets/ and update data/demos.json.`;
    return;
  }

  video.src = task.video;
  video.style.display = "block";
  placeholder.style.display = "none";
  video.onerror = () => {
    video.style.display = "none";
    placeholder.style.display = "grid";
    placeholderText.textContent = `Video not found: ${task.video}`;
  };
}

function renderActiveTask(data, index) {
  const task = data.tasks[index];
  document.getElementById("activeMode").textContent = task.planner_mode;
  document.getElementById("activeTitle").textContent = task.name;
  document.getElementById("activeStatus").textContent = task.verifier.action;
  document.getElementById("boundaryText").textContent = `${task.forced_threshold_s}s -> ${task.actual_trigger}`;
  document.getElementById("latencyText").textContent = formatMs(task.mean_ms);
  document.getElementById("verifierText").textContent = `${task.verifier.action} / ${task.verifier.status}`;
  document.getElementById("cacheText").textContent = task.cache_action;
  document.getElementById("jsonBox").textContent = JSON.stringify(task.verifier.raw_output, null, 2);
  renderTimeline(task);
  renderVideo(task);
}

function renderResults(data) {
  const rows = document.getElementById("resultRows");
  rows.innerHTML = data.results.map((row) => `
    <tr>
      <td><code>${row.task}</code></td>
      <td>${row.forced_threshold_s}s</td>
      <td>${row.actual_trigger}</td>
      <td>${formatMs(row.mean_ms)}</td>
      <td>${formatMs(row.median_ms)}</td>
      <td>${formatMs(row.min_ms)} / ${formatMs(row.max_ms)}</td>
    </tr>
  `).join("");
}

loadDemoData().then((data) => {
  let activeIndex = 0;
  renderSummary(data);
  renderResults(data);
  const selectTask = (index) => {
    activeIndex = index;
    renderTaskList(data, activeIndex, selectTask);
    renderActiveTask(data, activeIndex);
  };
  selectTask(activeIndex);
}).catch((error) => {
  document.body.innerHTML = `<main><pre>${error.message}</pre></main>`;
});
