"use strict";

// ---- state ----
const state = { files: {} }; // slotName -> File
let surfaceEnabled = false;

const views = {
  capture: document.getElementById("capture-view"),
  processing: document.getElementById("processing-view"),
  report: document.getElementById("report-view"),
};

function showView(name) {
  for (const [key, el] of Object.entries(views)) {
    el.hidden = key !== name;
  }
}

const GATE_LABELS = {
  resolution: "Resolution too low",
  card_detection: "Couldn't find the card",
  tilt: "Camera angle too tilted",
  aspect_ratio: "Unexpected aspect ratio",
  glare: "Glare detected",
  uneven_lighting: "Uneven lighting",
};
function gateLabel(name) {
  return GATE_LABELS[name] || name;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

function gradeColor(grade) {
  if (grade === null || grade === undefined) return "var(--muted)";
  if (grade >= 8) return "var(--good)";
  if (grade >= 5) return "var(--warn)";
  return "var(--bad)";
}

// ---- client-side image prep: EXIF orientation fix + downscale ----
async function processImageFile(file) {
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const maxEdge = 3000;
    const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height));
    const w = Math.round(bitmap.width * scale);
    const h = Math.round(bitmap.height * scale);
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0, w, h);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) return file;
    return new File([blob], (file.name || "photo") + ".jpg", { type: "image/jpeg" });
  } catch (e) {
    console.warn("client-side image processing failed, using original file", e);
    return file;
  }
}

// ---- capture slots (picker-mode fallback) ----
function setupSlot(slotEl) {
  const slotName = slotEl.dataset.slot;
  const input = slotEl.querySelector("input[type=file]");
  const thumb = slotEl.querySelector(".thumb");
  const placeholder = slotEl.querySelector(".slot-placeholder");
  const errorEl = slotEl.querySelector(".slot-error");

  input.addEventListener("change", async () => {
    const rawFile = input.files[0];
    if (!rawFile) return;
    const originalPlaceholderText = placeholder.textContent;
    placeholder.textContent = "Processing…";
    placeholder.hidden = false;
    thumb.hidden = true;

    const file = await processImageFile(rawFile);
    state.files[slotName] = file;

    const url = URL.createObjectURL(file);
    thumb.src = url;
    thumb.hidden = false;
    placeholder.hidden = true;
    placeholder.textContent = originalPlaceholderText;
    errorEl.hidden = true;
    errorEl.textContent = "";
    updateSubmitEnabled();
  });
}

document.querySelectorAll(".slot").forEach(setupSlot);

function setSlotError(slotName, message) {
  const slotEl = document.querySelector(`.slot[data-slot="${slotName}"]`);
  if (!slotEl) return;
  const errorEl = slotEl.querySelector(".slot-error");
  errorEl.textContent = message;
  errorEl.hidden = false;
  delete state.files[slotName];
  slotEl.querySelector(".thumb").hidden = true;
  slotEl.querySelector(".slot-placeholder").hidden = false;
  slotEl.querySelector("input[type=file]").value = "";
}

// ---- guided camera capture (primary mode) ----
const STEP_ORDER = ["front", "back", "front_angled", "back_angled"];
const STEP_LABELS = { front: "Front", back: "Back", front_angled: "Front (angled)", back_angled: "Back (angled)" };
const STEP_THUMB_LABELS = { front: "Front", back: "Back", front_angled: "F. angled", back_angled: "B. angled" };

let cameraModeActive = true;
let cameraStream = null;
let precheckTimer = null;
let currentStepIndex = 0;

function activeSteps() {
  return surfaceEnabled ? STEP_ORDER : STEP_ORDER.slice(0, 2);
}

function showCameraUnavailable(message) {
  const banner = document.getElementById("camera-unavailable-banner");
  banner.textContent = message;
  banner.hidden = false;
}

async function startCamera() {
  const video = document.getElementById("camera-video");

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCameraUnavailable(
      "Camera capture needs a secure connection (HTTPS). A plain http://<tailscale-ip> URL doesn't qualify — " +
        "open this page via the Cloudflare Tunnel URL instead, or provision a cert with `tailscale cert`. Using the photo picker for now."
    );
    switchToPickerMode();
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      // Ask for 4K: `ideal` means the browser picks the closest supported
      // mode, so this degrades gracefully on lesser cameras. The previous
      // 1920x1080 request meant captures could never clear the pipeline's
      // 1200px minimum-resolution gate — a full-height crop of a 1080p
      // frame is at most 1080px on its short side.
      video: { facingMode: "environment", width: { ideal: 3840 }, height: { ideal: 2160 } },
      audio: false,
    });
    video.srcObject = cameraStream;
    // #camera-unavailable-banner is this function's own concern (camera
    // access working or not) — safe to clear on success. #camera-error is
    // NOT touched here: it's reportSlotFailures' gate-failure retake
    // message, a completely different concern that happens to live in the
    // same view. This function used to clear #camera-error too, which meant
    // every camera restart — including the one reportSlotFailures triggers
    // right after showing its message — wiped that message out again within
    // a fraction of a second (permission's already granted, so getUserMedia
    // resolves almost instantly). #camera-error is only ever cleared by an
    // explicit user action now (a new shot, or tapping a thumb to retake).
    document.getElementById("camera-unavailable-banner").hidden = true;
    setupZoomControl();
    startPrecheckLoop();
    updateStepIndicator();
  } catch (e) {
    showCameraUnavailable(`Couldn't access the camera (${e.message}). Using the photo picker instead.`);
    switchToPickerMode();
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  document.getElementById("zoom-row").hidden = true;
  stopPrecheckLoop();
}

// Real camera zoom via track constraints — NOT a CSS scale. Phone lenses
// can't focus closer than ~10-15cm, so "fill the guide by moving closer"
// walks straight into permanent blur; sensor zoom fills the guide from a
// distance where focus is sharp, and the zoomed frames are what captureShot
// receives. Shown only when the camera actually supports the zoom
// constraint (iOS 17+ Safari, Android Chrome); hidden otherwise.
const MAX_ZOOM_SHOWN = 5; // beyond ~5x, phone digital zoom is mush

function zoomAvailable() {
  return !document.getElementById("zoom-row").hidden;
}

function setupZoomControl() {
  const row = document.getElementById("zoom-row");
  const slider = document.getElementById("zoom-slider");
  const valueLabel = document.getElementById("zoom-value");
  row.hidden = true;
  const track =
    cameraStream && typeof cameraStream.getVideoTracks === "function"
      ? cameraStream.getVideoTracks()[0]
      : null;
  if (!track || typeof track.getCapabilities !== "function") return;
  const caps = track.getCapabilities();
  if (!caps || !caps.zoom || caps.zoom.max <= caps.zoom.min) return;

  const min = caps.zoom.min;
  const max = Math.min(caps.zoom.max, MAX_ZOOM_SHOWN);
  slider.min = min;
  slider.max = max;
  slider.step = caps.zoom.step || 0.1;
  const current = (track.getSettings && track.getSettings().zoom) || min;
  slider.value = current;
  valueLabel.textContent = `${Number(current).toFixed(1)}×`;
  row.hidden = false;

  slider.oninput = () => {
    const zoom = parseFloat(slider.value);
    valueLabel.textContent = `${zoom.toFixed(1)}×`;
    // fire-and-forget: a rejected constraint just leaves the previous zoom
    track.applyConstraints({ advanced: [{ zoom }] }).catch(() => {});
  };
}

function startPrecheckLoop() {
  stopPrecheckLoop();
  precheckTimer = setInterval(runPrecheck, 300);
}
function stopPrecheckLoop() {
  if (precheckTimer) clearInterval(precheckTimer);
  precheckTimer = null;
}

// Maps the on-screen card-guide box to native video pixel coordinates, then
// to the (smaller) offscreen sample canvas used for the precheck. Reads
// actual layout via getBoundingClientRect so CSS stays the single source of
// truth for where the guide is drawn.
function computeGuideSampleRect(sampleW, sampleH) {
  const wrap = document.querySelector(".viewfinder-wrap");
  const guide = document.getElementById("card-guide");
  const video = document.getElementById("camera-video");
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return { x: 0, y: 0, w: sampleW, h: sampleH };

  const wrapRect = wrap.getBoundingClientRect();
  const guideRect = guide.getBoundingClientRect();

  // object-fit: cover scale factor from native video -> rendered CSS box
  const scale = Math.max(wrapRect.width / vw, wrapRect.height / vh);
  const renderedVideoW = vw * scale;
  const renderedVideoH = vh * scale;
  const cropX = (renderedVideoW - wrapRect.width) / 2;
  const cropY = (renderedVideoH - wrapRect.height) / 2;

  const guideLeftInWrap = guideRect.left - wrapRect.left;
  const guideTopInWrap = guideRect.top - wrapRect.top;

  const nativeX = (cropX + guideLeftInWrap) / scale;
  const nativeY = (cropY + guideTopInWrap) / scale;
  const nativeW = guideRect.width / scale;
  const nativeH = guideRect.height / scale;

  const sx = Math.round((nativeX / vw) * sampleW);
  const sy = Math.round((nativeY / vh) * sampleH);
  const sw = Math.round((nativeW / vw) * sampleW);
  const sh = Math.round((nativeH / vh) * sampleH);

  const x = Math.max(0, Math.min(sx, sampleW - 1));
  const y = Math.max(0, Math.min(sy, sampleH - 1));
  return { x, y, w: Math.max(1, Math.min(sw, sampleW - x)), h: Math.max(1, Math.min(sh, sampleH - y)) };
}

// Below this, the guide region reads as visually flat/uniform — a plain
// background or blank surface, not a printed card. Real cards (even a plain
// solid-color back) carry a border, text, or artwork that pushes local
// luminance variance well above this. A rough starting guess, same as every
// other threshold in this project — expect it to need adjustment once
// tested against a real camera and a real capture setup.
const NO_CARD_VARIANCE_THRESHOLD = 100;

// Catches the case the variance check above can't: something with plenty of
// contrast/texture (so it isn't "flat") but with essentially no color — a
// printed document, a hand, a grayscale object. Real Pokemon cards (front
// borders, back navy) all carry real color. Verified against an actual
// mis-shot: a printed checklist on a gray desk measured ~0.01 mean
// saturation despite high-variance text/table-line detail (586, well past
// the threshold above) — a synthetic colored test card measured ~0.38 in
// the same spot. 0.12 sits with real margin on both sides of that gap, but
// like every threshold here, it's a starting point, not a calibrated value.
// 0.12 originally split the checklist-document false-positive case (measured
// ~0.0104) from a synthetic colorful "card" (~0.38). But a real submitted
// card photo (a muted brown/gray Fighting-type card) measured ~0.124 — right
// on top of that cutoff, and *below* a real non-card scene measured the same
// way (~0.13, a metal staircase). That overlap means 0.12 risked a stable
// false "not a card" warning on genuinely low-saturation real cards, which is
// worse than an occasional missed non-card scene given this check only ever
// hints and never blocks the shutter — so the margin is pushed down to give
// real cards more headroom, at the cost of not reliably catching dull/gray
// non-card scenes anymore.
const LOW_SATURATION_THRESHOLD = 0.07;

// Shared with runPrecheck's severity mapping (red vs. yellow) — kept as
// constants so the two can't drift out of sync. Two distinct messages, two
// distinct severities: "no card detected" gets the red/error treatment
// (nothing there at all), "too little color" stays yellow alongside the
// other technique warnings (dark, glare, too far) — a card IS present, the
// shot just needs adjustment.
const NO_CARD_MESSAGE = "No card detected — center it in the frame";
const NO_COLOR_MESSAGE = "Doesn't look like a card — too little color";

// Pure and DOM-free on purpose: computes the precheck message from raw pixel
// data alone, so it can run in an offscreen sample canvas and be unit tested
// without a real <video>/getUserMedia. Never returns anything that should
// block the shutter — only ever informs the banner text.
function computePrecheckMessage(px, isAngled, nativeGuideHeight, canZoom = false) {
  let sum = 0;
  let sumSq = 0;
  let bright = 0;
  let satSum = 0;
  const count = px.length / 4;
  for (let i = 0; i < px.length; i += 4) {
    const r = px[i];
    const g = px[i + 1];
    const b = px[i + 2];
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    sum += lum;
    sumSq += lum * lum;
    if (lum > 250) bright++;

    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    satSum += maxC > 0 ? (maxC - minC) / maxC : 0;
  }
  const meanLum = count ? sum / count : 0;
  const variance = count ? sumSq / count - meanLum * meanLum : 0;
  const brightFrac = count ? bright / count : 0;
  const meanSaturation = count ? satSum / count : 0;

  if (!isAngled && meanLum < 40) {
    return "Too dark — add more light";
  }
  // Checked before glare/distance: if there's no card-like detail in the
  // guide at all, telling the user to adjust light or move closer is
  // misleading — the real problem is nothing's there to shoot yet.
  if (variance < NO_CARD_VARIANCE_THRESHOLD) {
    return NO_CARD_MESSAGE;
  }
  // Same reasoning, different failure mode: plenty of contrast, but no
  // color — not lighting-technique-specific, so (like the check above) this
  // applies on angled steps too, unlike glare/darkness.
  if (meanSaturation < LOW_SATURATION_THRESHOLD) {
    return NO_COLOR_MESSAGE;
  }
  if (!isAngled && brightFrac > 0.08) {
    return "Glare detected — adjust the light angle";
  }
  if (nativeGuideHeight < 700) {
    // Physically moving closer runs into the lens's minimum focus distance
    // (~10-15cm) and goes blurry — when real camera zoom is available,
    // steer toward that instead.
    return canZoom ? "Card too small in frame — zoom in" : "Move closer to the card";
  }
  return null;
}

// Debounce state for runPrecheck: a borderline scene (real numbers hovering
// right at a threshold, e.g. dull metal/wood ~0.13 saturation vs. a 0.12 cutoff)
// can flip its raw per-frame reading from one 300ms sample to the next due to
// ordinary exposure/compression noise. Requiring a few consecutive matching
// readings before the displayed state changes avoids a flickering warning.
const PRECHECK_CONFIRM_TICKS = 3;
let precheckDisplayedMessage = null;
let precheckPendingMessage = null;
let precheckPendingCount = 0;

function resetPrecheckDebounce() {
  precheckDisplayedMessage = null;
  precheckPendingMessage = null;
  precheckPendingCount = 0;
}

// Coarse, client-side only: warns before an obviously-doomed shot. The
// server-side Stage 1 gates remain the real authority — never blocks the
// shutter.
function runPrecheck() {
  const video = document.getElementById("camera-video");
  if (!video.videoWidth) return;
  const steps = activeSteps();
  const stepName = steps[currentStepIndex];
  if (!stepName) return;
  const isAngled = stepName.endsWith("_angled");

  const canvas = document.getElementById("precheck-canvas");
  const sw = 240;
  const sh = Math.max(1, Math.round((video.videoHeight / video.videoWidth) * sw));
  canvas.width = sw;
  canvas.height = sh;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, sw, sh);

  const rect = computeGuideSampleRect(sw, sh);
  const imgData = ctx.getImageData(rect.x, rect.y, rect.w, rect.h);
  const nativeGuideHeight = rect.h * (video.videoHeight / sh);

  const message = computePrecheckMessage(imgData.data, isAngled, nativeGuideHeight, zoomAvailable());

  if (message === precheckPendingMessage) {
    precheckPendingCount++;
  } else {
    precheckPendingMessage = message;
    precheckPendingCount = 1;
  }
  if (precheckPendingCount >= PRECHECK_CONFIRM_TICKS) {
    precheckDisplayedMessage = message;
  }

  applyPrecheckDisplay(precheckDisplayedMessage);
}

// Severity color-coding: "no card detected" gets its own red treatment
// (there's nothing to grade at all, a step up from the other coarse quality
// warnings), the rest stay yellow, and a confirmed-clean frame turns the
// guide green instead of sitting at the neutral blue default indefinitely.
function applyPrecheckDisplay(message) {
  const banner = document.getElementById("precheck-banner");
  const guide = document.getElementById("card-guide");
  guide.classList.remove("card-guide--good", "card-guide--warning", "card-guide--error");
  banner.classList.remove("precheck-banner--error");

  if (!message) {
    banner.hidden = true;
    guide.classList.add("card-guide--good");
    return;
  }
  banner.textContent = message;
  banner.hidden = false;
  // A text banner at the bottom is easy to miss when you're focused on
  // framing the shot — this puts the warning directly on the thing you're
  // actually looking at.
  if (message === NO_CARD_MESSAGE) {
    guide.classList.add("card-guide--error");
    banner.classList.add("precheck-banner--error");
  } else {
    guide.classList.add("card-guide--warning");
  }
}

// Crops generously around the guide (not just the guide box itself) so
// Stage 1's contour detection has dark background margin to work with —
// the guide is there to help the human frame the shot, not to hand the
// algorithm a pre-cropped card.
function captureShot() {
  const video = document.getElementById("camera-video");
  const guide = document.getElementById("card-guide");
  const wrap = document.querySelector(".viewfinder-wrap");
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const wrapRect = wrap.getBoundingClientRect();
  const guideRect = guide.getBoundingClientRect();
  const scale = Math.max(wrapRect.width / vw, wrapRect.height / vh);
  const renderedVideoW = vw * scale;
  const renderedVideoH = vh * scale;
  const cropX = (renderedVideoW - wrapRect.width) / 2;
  const cropY = (renderedVideoH - wrapRect.height) / 2;
  const guideLeftInWrap = guideRect.left - wrapRect.left;
  const guideTopInWrap = guideRect.top - wrapRect.top;

  const nativeGuideX = (cropX + guideLeftInWrap) / scale;
  const nativeGuideY = (cropY + guideTopInWrap) / scale;
  const nativeGuideW = guideRect.width / scale;
  const nativeGuideH = guideRect.height / scale;

  const margin = 0.35;
  let cropW = nativeGuideW * (1 + margin * 2);
  let cropH = nativeGuideH * (1 + margin * 2);
  let cx = Math.max(0, nativeGuideX - nativeGuideW * margin);
  let cy = Math.max(0, nativeGuideY - nativeGuideH * margin);
  cropW = Math.min(cropW, vw - cx);
  cropH = Math.min(cropH, vh - cy);

  const maxEdge = 3000;
  const outScale = Math.min(1, maxEdge / Math.max(cropW, cropH));
  const outW = Math.round(cropW * outScale);
  const outH = Math.round(cropH * outScale);

  const canvas = document.createElement("canvas");
  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, cx, cy, cropW, cropH, 0, 0, outW, outH);

  const stepName = activeSteps()[currentStepIndex];
  return new Promise((resolve) => {
    canvas.toBlob(
      (blob) => resolve(new File([blob], `${stepName}.jpg`, { type: "image/jpeg" })),
      "image/jpeg",
      0.92
    );
  });
}

function updateStepIndicator() {
  const steps = activeSteps();
  const stepName = steps[currentStepIndex];
  const indicator = document.getElementById("step-indicator");
  if (!stepName) {
    indicator.textContent = "All shots captured — review below, then grade";
    return;
  }
  indicator.textContent = `Shot ${currentStepIndex + 1} of ${steps.length}: ${STEP_LABELS[stepName]}`;
}

function addThumb(stepName, file) {
  const strip = document.getElementById("thumb-strip");
  let thumbEl = strip.querySelector(`[data-step="${stepName}"]`);
  if (!thumbEl) {
    thumbEl = document.createElement("div");
    thumbEl.className = "capture-thumb";
    thumbEl.dataset.step = stepName;
    thumbEl.innerHTML = `<img alt="${STEP_LABELS[stepName]}"><div class="capture-thumb-label">${STEP_THUMB_LABELS[stepName]}</div>`;
    thumbEl.addEventListener("click", () => retakeStep(stepName));
    strip.appendChild(thumbEl);
  }
  thumbEl.querySelector("img").src = URL.createObjectURL(file);
}

function removeThumb(stepName) {
  const thumbEl = document.querySelector(`.capture-thumb[data-step="${stepName}"]`);
  if (thumbEl) thumbEl.remove();
}

function showViewfinder(show) {
  document.querySelector(".viewfinder-wrap").hidden = !show;
  document.getElementById("shutter-btn").hidden = !show;
  if (show) {
    // Clean slate on reopen — otherwise a warning from whatever was last
    // framed (a previous step, or right before a retake) stays visible for
    // up to one precheck tick (300ms) before it's naturally corrected.
    document.getElementById("precheck-banner").hidden = true;
    document.getElementById("precheck-banner").classList.remove("precheck-banner--error");
    document.getElementById("card-guide").classList.remove("card-guide--good", "card-guide--warning", "card-guide--error");
    resetPrecheckDebounce();
  }
}

function retakeStep(stepName) {
  const steps = activeSteps();
  const idx = steps.indexOf(stepName);
  if (idx === -1) return;
  currentStepIndex = idx;
  delete state.files[stepName];
  removeThumb(stepName);
  showViewfinder(true);
  document.getElementById("camera-error").hidden = true;
  updateStepIndicator();
  updateSubmitEnabled();
  if (!cameraStream) startCamera();
}

document.getElementById("shutter-btn").addEventListener("click", async () => {
  const steps = activeSteps();
  const stepName = steps[currentStepIndex];
  if (!stepName) return;
  const file = await captureShot();
  state.files[stepName] = file;
  addThumb(stepName, file);
  // A fresh shot means the user is actively addressing whatever the last
  // error said — an old gate-failure message lingering after they've
  // already retaken the photo just reads as "still broken."
  document.getElementById("camera-error").hidden = true;
  currentStepIndex++;
  if (currentStepIndex >= steps.length) {
    showViewfinder(false);
    stopCamera();
  }
  updateStepIndicator();
  updateSubmitEnabled();
});

// Server rejected one or more captured photos (Stage 1 gate failures) —
// reopen the right slot(s) in whichever mode is currently active.
//
// Takes the whole batch of failures at once rather than being called once
// per failing step: calling it per-step let a second failure's handling
// silently clobber the first's (shared banner + step index), and worse,
// each call's own `if (!cameraStream) startCamera()` could synchronously
// flip cameraModeActive mid-loop (via startCamera's no-camera-API fallback
// to switchToPickerMode()), so a later failure in the same batch would then
// route through the wrong mode entirely.
function reportSlotFailures(failures) {
  if (failures.length === 0) return;

  for (const { step } of failures) {
    delete state.files[step];
    removeThumb(step);
  }

  if (cameraModeActive) {
    const steps = activeSteps();
    const indices = failures.map((f) => steps.indexOf(f.step)).filter((i) => i !== -1);
    currentStepIndex = indices.length ? Math.min(...indices) : 0;
    showViewfinder(true);
    updateStepIndicator();
    const cameraError = document.getElementById("camera-error");
    cameraError.textContent = failures.map((f) => `${STEP_LABELS[f.step]}: ${f.message}`).join("\n");
    cameraError.hidden = false;
    if (!cameraStream) startCamera();
  } else {
    for (const { step, message } of failures) {
      setSlotError(step, message);
    }
  }
}

// ---- mode toggles ----
const pickerModeToggle = document.getElementById("picker-mode-toggle");
pickerModeToggle.addEventListener("click", () => {
  if (cameraModeActive) {
    switchToPickerMode();
  } else {
    switchToCameraMode();
  }
});

function switchToPickerMode() {
  cameraModeActive = false;
  stopCamera();
  document.getElementById("camera-mode").hidden = true;
  document.getElementById("picker-mode").hidden = false;
  pickerModeToggle.textContent = "Use camera instead";
}

async function switchToCameraMode() {
  cameraModeActive = true;
  document.getElementById("camera-mode").hidden = false;
  document.getElementById("picker-mode").hidden = true;
  pickerModeToggle.textContent = "Use photo picker instead";
  showViewfinder(true);
  await startCamera();
}

// ---- surface toggle ----
const surfaceToggleBtn = document.getElementById("surface-toggle");
const surfaceSlots = document.getElementById("surface-slots");
surfaceToggleBtn.addEventListener("click", () => {
  surfaceEnabled = !surfaceEnabled;
  surfaceSlots.hidden = !surfaceEnabled;
  surfaceToggleBtn.textContent = surfaceEnabled ? "− surface analysis (optional)" : "+ surface analysis (optional)";

  // Recompute where the guided sequence stands now that the step count may
  // have changed — works the same whether surface was just turned on or off.
  const steps = activeSteps();
  const nextIdx = steps.findIndex((s) => !state.files[s]);
  if (nextIdx === -1) {
    currentStepIndex = steps.length;
    showViewfinder(false);
    stopCamera();
  } else {
    currentStepIndex = nextIdx;
    if (cameraModeActive) {
      showViewfinder(true);
      if (!cameraStream) startCamera();
    }
  }
  updateStepIndicator();
  updateSubmitEnabled();
});

// ---- submit ----
const submitBtn = document.getElementById("submit-btn");
const submitError = document.getElementById("submit-error");

function updateSubmitEnabled() {
  const hasRequired = state.files.front && state.files.back;
  const hasSurface = !surfaceEnabled || (state.files.front_angled && state.files.back_angled);
  submitBtn.disabled = !(hasRequired && hasSurface);
}

function showSubmitError(message) {
  submitError.textContent = message;
  submitError.hidden = false;
  submitBtn.disabled = false;
  submitBtn.textContent = "Grade card";
}

submitBtn.addEventListener("click", submitJob);

async function submitJob() {
  submitError.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Uploading…";

  const formData = new FormData();
  formData.append("front", state.files.front);
  formData.append("back", state.files.back);
  if (surfaceEnabled) {
    formData.append("front_angled", state.files.front_angled);
    formData.append("back_angled", state.files.back_angled);
  }

  let resp;
  try {
    resp = await fetch("/api/grade", { method: "POST", body: formData });
  } catch (e) {
    showSubmitError("Couldn't reach the server — check the connection and try again.");
    return;
  }

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    showSubmitError(body.detail || `Server error (${resp.status})`);
    return;
  }

  const { job_id } = await resp.json();
  submitBtn.textContent = "Grade card";
  processingMessage.textContent = "Starting…";
  showView("processing");
  pollJob(job_id);
}

// ---- processing / polling ----
const processingMessage = document.getElementById("processing-message");

async function pollJob(jobId) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 800));
    let resp;
    try {
      resp = await fetch(`/api/job/${jobId}`);
    } catch (e) {
      continue; // transient network hiccup — keep polling
    }
    if (resp.status === 404) {
      processingMessage.textContent = "Job expired — please try again.";
      return;
    }
    const data = await resp.json();
    if (data.status === "queued" || data.status === "running") {
      processingMessage.textContent = data.message || "Working…";
      continue;
    }
    if (data.status === "error") {
      processingMessage.textContent = `Something went wrong: ${data.message}`;
      return;
    }
    handleReport(data.report, data.images);
    return;
  }
}

function handleReport(report, images) {
  if (report.centering === null) {
    // capture-quality gate failure(s) — collect every failing slot before
    // acting, so a "both failed" batch reports both, not just the last one
    const failures = [];
    for (const side of ["front", "back"]) {
      const cq = report.capture_quality[side];
      if (!cq.ok) {
        const failed = cq.gates.filter((g) => !g.passed);
        // Only hard (geometry) failures actually forced this retake — soft
        // ones like low resolution would have graded with a warning. A side
        // whose failures are all soft is a usable capture and shouldn't be
        // retaken just because the OTHER side hard-failed. Fall back to the
        // old everything-fails behavior if the server didn't mark hardness.
        const hasHardInfo = failed.some((g) => "hard" in g);
        const hardFailed = failed.filter((g) => g.hard);
        if (hasHardInfo && !hardFailed.length) continue;
        const relevant = hardFailed.length ? hardFailed : failed;
        const reason = relevant.map((g) => `${gateLabel(g.name)} — ${g.detail}`).join("; ");
        failures.push({ step: side, message: reason || "Capture quality check failed" });
      }
    }
    reportSlotFailures(failures);
    updateSubmitEnabled();
    showView("capture");
    return;
  }

  window.__lastReport = { report, images };
  document.getElementById("report-content").innerHTML = buildReportHTML(report, images);
  wireSurfaceSliders();
  showView("report");
}

// ---- report rendering ----
function subgradeTile(label, grade) {
  if (grade === null || grade === undefined) {
    return `<div class="subgrade-tile"><div class="subgrade-label">${label}</div><div class="subgrade-value muted">n/a</div></div>`;
  }
  return `<div class="subgrade-tile">
    <div class="subgrade-label">${label}</div>
    <div class="subgrade-value" style="color:${gradeColor(grade)}">${grade}</div>
  </div>`;
}

function centeringSideHTML(label, sideData, overlayImg) {
  // measurable === false: the border-boundary detection had no confident
  // signal on this side (borderless/full-art card, or the border isn't
  // visible in this capture). Showing the raw ratios would present
  // argmax-of-noise as real measurements. Older reports lack the flag —
  // treat missing as measurable.
  if (sideData.measurable === false) {
    return `<div class="centering-card">
      <div class="centering-card-header">${label}
        <span class="grade-pill" style="color:${gradeColor(null)}">n/a</span>
      </div>
      ${overlayImg ? `<img class="overlay-img" src="${overlayImg}" alt="${label} centering overlay">` : ""}
      <div class="muted" style="font-size:0.82rem">Couldn't measure — borderless/full-art card, or the border isn't visible in this capture.</div>
    </div>`;
  }
  return `<div class="centering-card">
    <div class="centering-card-header">${label}
      <span class="grade-pill" style="color:${gradeColor(sideData.grade)}">grade ${sideData.grade}</span>
    </div>
    ${overlayImg ? `<img class="overlay-img" src="${overlayImg}" alt="${label} centering overlay">` : ""}
    <div class="axis-row"><span class="axis-label">H</span><span class="mono">${escapeHtml(sideData.horizontal.ratio)}</span>
      <span class="grade-pill-sm">g${sideData.horizontal.grade}</span></div>
    <div class="axis-row"><span class="axis-label">V</span><span class="mono">${escapeHtml(sideData.vertical.ratio)}</span>
      <span class="grade-pill-sm">g${sideData.vertical.grade}</span></div>
  </div>`;
}

function regionTileHTML(bareKey, prefixedKey, shortLabel, sideKey, regionData, images) {
  const img = images[`${sideKey}_${prefixedKey}`];
  return `<div class="region-tile">
    ${img ? `<img class="region-thumb" src="${img}" alt="${shortLabel}">` : ""}
    <div class="region-label">${shortLabel}</div>
    <div class="mono region-pct" style="color:${gradeColor(regionData.grade)}">${regionData.whitening_pct.toFixed(2)}%</div>
  </div>`;
}

const CORNER_DEFS = [
  ["top_left", "corner_top_left", "TL"],
  ["top_right", "corner_top_right", "TR"],
  ["bottom_right", "corner_bottom_right", "BR"],
  ["bottom_left", "corner_bottom_left", "BL"],
];
const EDGE_DEFS = [
  ["top", "edge_top", "Top"],
  ["right", "edge_right", "Right"],
  ["bottom", "edge_bottom", "Bottom"],
  ["left", "edge_left", "Left"],
];

function cornersEdgesSideHTML(label, sideKey, sideData, images) {
  let tiles = "";
  for (const [bare, prefixed, short] of CORNER_DEFS) {
    tiles += regionTileHTML(bare, prefixed, short, sideKey, sideData.corners[bare], images);
  }
  for (const [bare, prefixed, short] of EDGE_DEFS) {
    tiles += regionTileHTML(bare, prefixed, short, sideKey, sideData.edges[bare], images);
  }
  return `<div class="ce-card">
    <div class="centering-card-header">${label}
      <span class="grade-pill" style="color:${gradeColor(sideData.grade)}">grade ${sideData.grade}</span>
    </div>
    <div class="region-grid">${tiles}</div>
  </div>`;
}

function surfaceSideHTML(sideKey, label, sideData, images) {
  if (!sideData.aligned) {
    const reasons = (sideData.gates || []).filter((g) => !g.passed).map((g) => `${gateLabel(g.name)} — ${g.detail}`).join("; ");
    return `<div class="surface-card">
      <div class="centering-card-header">${label}</div>
      <div class="banner banner-warn">Couldn't align this angled shot${reasons ? ": " + escapeHtml(reasons) : ""}.
        Retake and resubmit to include this side's surface analysis.</div>
    </div>`;
  }

  const alignedImg = images[`${sideKey}_surface_aligned`];
  const defectMapImg = images[`${sideKey}_surface_defect_map`];
  const sliderId = `surface-slider-${sideKey}`;

  let vjHtml;
  if (sideData.vision_judgment) {
    const vj = sideData.vision_judgment;
    const defects = (vj.defects_found || []).length
      ? `<ul>${vj.defects_found.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}</ul>`
      : `<div class="muted">no defects flagged</div>`;
    vjHtml = `<div class="vision-judgment">
      <span class="grade-pill" style="color:${gradeColor(vj.surface_grade)}">vision judgment: grade ${vj.surface_grade}</span>
      <span class="muted">(confidence: ${escapeHtml(vj.confidence)}${vj.model ? ", " + escapeHtml(vj.model) : ""})</span>
      ${defects}
    </div>`;
  } else {
    vjHtml = `<div class="muted vision-judgment">no vision judgment available (no API credentials configured) — figures below are the raw algorithmic signal only</div>`;
  }

  return `<div class="surface-card">
    <div class="centering-card-header">${label}
      <span class="mono">defect area ${sideData.defect_area_pct.toFixed(2)}%</span>
      <span class="mono muted">holo masked ${sideData.holo_area_pct.toFixed(2)}%</span>
    </div>
    <div class="defect-slider-container">
      ${alignedImg ? `<img class="defect-base" src="${alignedImg}" alt="${label} surface">` : ""}
      ${defectMapImg ? `<img class="defect-overlay" id="${sliderId}-img" src="${defectMapImg}" style="opacity:0.5" alt="${label} defect map">` : ""}
    </div>
    ${defectMapImg ? `<input type="range" min="0" max="100" value="50" class="defect-slider" id="${sliderId}">` : ""}
    ${vjHtml}
  </div>`;
}

// Soft capture-quality gates (resolution, tilt, aspect ratio, glare, uneven
// lighting) no longer block grading — only a genuine "couldn't find the
// card" does (report.centering === null, handled earlier in handleReport).
// When grading proceeded despite a soft gate failing, this surfaces it as a
// warning instead of silently trusting a lower-confidence result. Just the
// gate label, not its numeric detail — "Resolution too low" reads fine
// without "— shortest side=1080px" tacked on.
function captureWarningHTML(report) {
  const parts = [];
  for (const [side, sideLabel] of [["front", "Front"], ["back", "Back"]]) {
    const cq = report.capture_quality[side];
    if (!cq.ok) {
      const labels = cq.gates.filter((g) => !g.passed).map((g) => gateLabel(g.name));
      parts.push(`${sideLabel}: ${labels.join(", ")}`);
    }
  }
  if (!parts.length) return "";
  return `<div class="banner banner-error">The grading might be worse because of:<br>${parts
    .map(escapeHtml)
    .join("<br>")}</div>`;
}

function buildReportHTML(report, images) {
  const ge = report.grade_estimate;
  const centering = report.centering;
  const ce = report.corners_edges;
  const surface = report.surface;

  let html = captureWarningHTML(report);

  html += `<div class="grade-header">
    <div class="overall-grade" style="color:${gradeColor(ge.overall_grade_rounded)}">${ge.overall_grade_rounded}</div>
    <div class="overall-detail">
      <div class="overall-precise mono">${ge.overall_grade.toFixed(2)}</div>
      <div class="overall-note">${escapeHtml(ge.note)}</div>
    </div>
  </div>`;

  html += `<div class="subgrade-row">
    ${subgradeTile("Centering", ge.centering_grade)}
    ${subgradeTile("Corners/Edges", ge.corners_edges_grade)}
    ${subgradeTile("Surface", ge.surface_grade)}
  </div>`;

  html += `<section class="report-section">
    <h2>Centering</h2>
    <div class="side-by-side">
      ${centeringSideHTML("Front", centering.front, images.front_centering_overlay)}
      ${centeringSideHTML("Back", centering.back, images.back_centering_overlay)}
    </div>
  </section>`;

  html += `<section class="report-section">
    <h2>Corners &amp; Edges</h2>
    <div class="side-by-side">
      ${cornersEdgesSideHTML("Front", "front", ce.front, images)}
      ${cornersEdgesSideHTML("Back", "back", ce.back, images)}
    </div>
  </section>`;

  if (surface) {
    html += `<section class="report-section">
      <h2>Surface <span class="indicative-tag">indicative — not a grade by itself</span></h2>
      ${surfaceSideHTML("front", "Front", surface.front, images)}
      ${surfaceSideHTML("back", "Back", surface.back, images)}
    </section>`;
  }

  return html;
}

function wireSurfaceSliders() {
  document.querySelectorAll(".defect-slider").forEach((slider) => {
    const img = document.getElementById(`${slider.id}-img`);
    if (!img) return;
    slider.addEventListener("input", () => {
      img.style.opacity = (slider.value / 100).toFixed(2);
    });
  });
}

// ---- actions: download / reset ----
document.getElementById("download-btn").addEventListener("click", downloadReport);
document.getElementById("reset-btn").addEventListener("click", resetApp);

async function downloadReport() {
  if (!window.__lastReport) return;
  const { report, images } = window.__lastReport;
  const bodyHtml = buildReportHTML(report, images);

  let css = "";
  try {
    css = await (await fetch("/static/style.css")).text();
  } catch (e) {
    console.warn("couldn't inline stylesheet for download", e);
  }

  const sliderScript = `document.querySelectorAll(".defect-slider").forEach(function (slider) {
    var img = document.getElementById(slider.id + "-img");
    if (!img) return;
    slider.addEventListener("input", function () { img.style.opacity = (slider.value / 100).toFixed(2); });
  });`;

  const doc = `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card Grade Report</title>
<style>${css}
body{padding:1rem;}</style>
</head><body>
<main><div id="report-content">${bodyHtml}</div></main>
<script>${sliderScript}<\/script>
</body></html>`;

  const blob = new Blob([doc], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `card-grade-${Date.now()}.html`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function resetApp() {
  state.files = {};
  surfaceEnabled = false;
  surfaceSlots.hidden = true;
  surfaceToggleBtn.textContent = "+ surface analysis (optional)";
  document.querySelectorAll(".slot").forEach((slotEl) => {
    slotEl.querySelector(".thumb").hidden = true;
    slotEl.querySelector(".slot-placeholder").hidden = false;
    slotEl.querySelector("input[type=file]").value = "";
    slotEl.querySelector(".slot-error").hidden = true;
  });
  document.getElementById("thumb-strip").innerHTML = "";
  document.getElementById("camera-error").hidden = true;
  currentStepIndex = 0;
  submitError.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Grade card";
  window.__lastReport = null;
  showView("capture");
  if (cameraModeActive) {
    showViewfinder(true);
    startCamera();
  }
}

showView("capture");
startCamera();
