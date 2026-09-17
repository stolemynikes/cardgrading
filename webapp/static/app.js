"use strict";

// ---- state ----
const state = { files: {} }; // slotName -> File

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

// Formats a scanner produces. A scan needs neither of the things this
// function does — it has no EXIF orientation to correct, and downscaling it
// is the opposite of what the pipeline wants — so it is handed through
// untouched. Left alone, a PNG scan was quietly resampled to 3000px and
// re-encoded as JPEG before upload: the ringing at the border/panel boundary
// that the capture protocol explicitly warns against, applied by us.
// (TIFF escaped only by accident, because createImageBitmap can't decode it.)
const LOSSLESS_CAPTURE = /^image\/(png|tiff)$/i;
const LOSSLESS_EXTENSION = /\.(png|tiff?)$/i;

function isScannerCapture(file) {
  return LOSSLESS_CAPTURE.test(file.type || "") || LOSSLESS_EXTENSION.test(file.name || "");
}

// ---- client-side image prep: EXIF orientation fix + downscale ----
async function processImageFile(file) {
  if (isScannerCapture(file)) return file;
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

// ---- slot thumbnails ----
// A flatbed scan arrives as TIFF, and no browser renders TIFF in an <img> —
// the slot showed a broken-image icon for exactly the capture path the
// pipeline most wants. ScanGear writes uncompressed chunky 8-bit strips, so
// decoding a downsampled preview ourselves is a short, bounded job; anything
// outside that shape falls back to naming the file instead of lying about it.
const TIFF_TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8 };

function readTiffTags(view, le, ifdOffset) {
  const tags = {};
  const entries = view.getUint16(ifdOffset, le);
  for (let i = 0; i < entries; i++) {
    const p = ifdOffset + 2 + i * 12;
    const tag = view.getUint16(p, le);
    const type = view.getUint16(p + 2, le);
    const count = view.getUint32(p + 4, le);
    const unit = TIFF_TYPE_SIZE[type];
    if (!unit || count > 1e6) continue;
    // Values of 4 bytes or fewer live inline in the entry, left-justified;
    // anything larger stores a file offset there instead.
    const base = unit * count > 4 ? view.getUint32(p + 8, le) : p + 8;
    const values = [];
    for (let k = 0; k < count; k++) {
      const o = base + k * unit;
      if (o + unit > view.byteLength) break;
      if (type === 3) values.push(view.getUint16(o, le));
      else if (type === 4) values.push(view.getUint32(o, le));
      else if (type === 1) values.push(view.getUint8(o));
      else break;
    }
    if (values.length) tags[tag] = values;
  }
  return tags;
}

function decodeTiffPreview(buffer, maxEdge) {
  if (buffer.byteLength < 8) return null;
  const view = new DataView(buffer);
  const order = view.getUint16(0, false);
  if (order !== 0x4949 && order !== 0x4d4d) return null;
  const le = order === 0x4949;
  if (view.getUint16(2, le) !== 42) return null;

  const tags = readTiffTags(view, le, view.getUint32(4, le));
  const first = (tag, fallback) => (tags[tag] ? tags[tag][0] : fallback);
  const width = first(256, 0);
  const height = first(257, 0);
  const stripOffsets = tags[273];
  if (!width || !height || !stripOffsets) return null;
  // Uncompressed, 8-bit, chunky only. Everything else returns null and the
  // caller shows a file chip rather than a wrong picture.
  if (first(259, 1) !== 1 || first(258, 8) !== 8 || first(284, 1) !== 1) return null;
  const photometric = first(262, 1);
  if (photometric > 2) return null;
  const samples = first(277, 1);
  const rowsPerStrip = first(278, height) || height;

  const scale = Math.min(1, maxEdge / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const out = ctx.createImageData(w, h);
  const bytes = new Uint8Array(buffer);
  const rowBytes = width * samples;
  const invert = photometric === 0; // WhiteIsZero

  for (let y = 0; y < h; y++) {
    const sy = Math.min(height - 1, Math.floor((y * height) / h));
    const strip = Math.floor(sy / rowsPerStrip);
    const stripStart = stripOffsets[strip];
    if (stripStart === undefined) return null;
    const rowStart = stripStart + (sy - strip * rowsPerStrip) * rowBytes;
    for (let x = 0; x < w; x++) {
      const o = rowStart + Math.floor((x * width) / w) * samples;
      if (o + samples > bytes.length) return null;
      let r = bytes[o];
      let g = samples >= 3 ? bytes[o + 1] : r;
      let b = samples >= 3 ? bytes[o + 2] : r;
      if (invert) {
        r = 255 - r;
        g = 255 - g;
        b = 255 - b;
      }
      const i = (y * w + x) * 4;
      out.data[i] = r;
      out.data[i + 1] = g;
      out.data[i + 2] = b;
      out.data[i + 3] = 255;
    }
  }
  ctx.putImageData(out, 0, 0);
  return canvas;
}

function looksLikeTiff(file) {
  return /^image\/tiff$/i.test(file.type || "") || /\.tiff?$/i.test(file.name || "");
}

// Returns an object URL the <img> can show, or null when this file can't be
// previewed at all. Never throws — a failed preview must not block the upload.
async function thumbnailURL(file) {
  if (!looksLikeTiff(file)) {
    try {
      // Round-trips through createImageBitmap so an undecodable file fails
      // here rather than as a broken <img> the user has to interpret.
      const bitmap = await createImageBitmap(file);
      bitmap.close?.();
      return URL.createObjectURL(file);
    } catch (e) {
      return null;
    }
  }
  try {
    const canvas = decodeTiffPreview(await file.arrayBuffer(), 600);
    if (!canvas) return null;
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    return blob ? URL.createObjectURL(blob) : null;
  } catch (e) {
    console.warn("TIFF preview failed", e);
    return null;
  }
}

function fileSizeLabel(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

// ---- capture slots (picker-mode fallback) ----
function setupSlot(slotEl) {
  const slotName = slotEl.dataset.slot;
  const input = slotEl.querySelector("input[type=file]");
  const thumb = slotEl.querySelector(".thumb");
  const placeholder = slotEl.querySelector(".slot-placeholder");
  const errorEl = slotEl.querySelector(".slot-error");
  placeholder.dataset.defaultText = placeholder.textContent;

  input.addEventListener("change", async () => {
    const rawFile = input.files[0];
    if (!rawFile) return;
    const originalPlaceholderText = placeholder.dataset.defaultText;
    placeholder.textContent = "Processing…";
    placeholder.hidden = false;
    thumb.hidden = true;

    const file = await processImageFile(rawFile);
    state.files[slotName] = file;

    // Preview the file the user actually chose. processImageFile may have
    // re-encoded it, and for a scan it returns the original untouched, so
    // rawFile is the one thing guaranteed to be what they picked.
    const url = await thumbnailURL(rawFile);
    if (thumb.dataset.objectUrl) URL.revokeObjectURL(thumb.dataset.objectUrl);
    if (url) {
      thumb.src = url;
      thumb.dataset.objectUrl = url;
      thumb.hidden = false;
      placeholder.hidden = true;
      placeholder.textContent = originalPlaceholderText;
    } else {
      // No preview available. Name the file rather than leave a broken image:
      // the point of the slot is confirming the right file went in.
      delete thumb.dataset.objectUrl;
      thumb.removeAttribute("src");
      thumb.hidden = true;
      placeholder.textContent = `${rawFile.name} · ${fileSizeLabel(rawFile.size)}`;
      placeholder.hidden = false;
    }
    errorEl.hidden = true;
    errorEl.textContent = "";
    updateSubmitEnabled();
  });
}

document.querySelectorAll(".slot").forEach(setupSlot);

// Put a slot back to its empty state: drop any preview we minted, and restore
// the placeholder's own text (a failed preview leaves the filename in it).
function clearSlotPreview(slotEl) {
  const thumb = slotEl.querySelector(".thumb");
  if (thumb.dataset.objectUrl) {
    URL.revokeObjectURL(thumb.dataset.objectUrl);
    delete thumb.dataset.objectUrl;
  }
  thumb.removeAttribute("src");
  thumb.hidden = true;
  const placeholder = slotEl.querySelector(".slot-placeholder");
  if (placeholder.dataset.defaultText) placeholder.textContent = placeholder.dataset.defaultText;
  placeholder.hidden = false;
}

function setSlotError(slotName, message) {
  const slotEl = document.querySelector(`.slot[data-slot="${slotName}"]`);
  if (!slotEl) return;
  const errorEl = slotEl.querySelector(".slot-error");
  errorEl.textContent = message;
  errorEl.hidden = false;
  delete state.files[slotName];
  clearSlotPreview(slotEl);
  slotEl.querySelector("input[type=file]").value = "";
}

// ---- guided camera capture (primary mode) ----
// Raking-light capture is no longer part of the flow anywhere. Rotating the
// card under a fixed light gives a solved surface-normal map, which
// separates a scratch from printed linework outright — a single angled photo
// could only ever suggest the difference.
const STEP_ORDER = ["front", "back"];
const STEP_LABELS = { front: "Front", back: "Back" };
const STEP_THUMB_LABELS = { front: "Front", back: "Back" };

// Which capture mode the app opens in. Upload is the default: the flatbed
// flow produces files on disk, and a scanner beats a phone at every stage
// except surface, so starting in a viewfinder is wrong for the common case.
// The choice is remembered, so it doesn't have to be flipped every visit.
const CAPTURE_MODE_KEY = "cardgrading.captureMode";

function storedCaptureMode() {
  try {
    return localStorage.getItem(CAPTURE_MODE_KEY);
  } catch (e) {
    // Safari private mode throws on localStorage access.
    return null;
  }
}

function rememberCaptureMode(mode) {
  try {
    localStorage.setItem(CAPTURE_MODE_KEY, mode);
  } catch (e) {
    /* not remembering the preference is not a reason to fail the capture */
  }
}

let cameraModeActive = storedCaptureMode() === "camera";
let cameraStream = null;
let precheckTimer = null;
let currentStepIndex = 0;

function activeSteps() {
  return STEP_ORDER;
}

// The capture advice that matters is completely different per mode — tripod
// and diffuse lighting for a phone, DPI and post-processing switches for a
// scanner — so the hint follows the mode instead of describing both at once.
function setProtocolHint(cameraMode) {
  const hint = document.getElementById("protocol-hint");
  const camera = document.getElementById("protocol-hint-camera");
  const scanner = document.getElementById("protocol-hint-scanner");
  if (!hint || !camera || !scanner) return;
  camera.hidden = !cameraMode;
  scanner.hidden = cameraMode;
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
function computePrecheckMessage(px, nativeGuideHeight, canZoom = false) {
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

  if (meanLum < 40) {
    return "Too dark — add more light";
  }
  // Checked before glare/distance: if there's no card-like detail in the
  // guide at all, telling the user to adjust light or move closer is
  // misleading — the real problem is nothing's there to shoot yet.
  if (variance < NO_CARD_VARIANCE_THRESHOLD) {
    return NO_CARD_MESSAGE;
  }
  // Same reasoning, different failure mode: plenty of contrast, but no
  // color.
  if (meanSaturation < LOW_SATURATION_THRESHOLD) {
    return NO_COLOR_MESSAGE;
  }
  if (brightFrac > 0.08) {
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

  const message = computePrecheckMessage(imgData.data, nativeGuideHeight, zoomAvailable());

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
    switchToPickerMode(true);
  } else {
    switchToCameraMode(true);
  }
});

function switchToPickerMode(remember = false) {
  if (remember) rememberCaptureMode("upload");
  cameraModeActive = false;
  stopCamera();
  document.getElementById("camera-mode").hidden = true;
  document.getElementById("picker-mode").hidden = false;
  // Scanner extras only make sense for files chosen off disk — a live camera
  // has neither a known DPI nor a way to hold still while the light moves.
  // Opened, not just revealed: this is the whole reason to scan rather than
  // shoot, and behind two clicks it may as well not exist.
  document.getElementById("scanner-mode").hidden = false;
  setScannerFieldsOpen(true);
  setProtocolHint(false);
  pickerModeToggle.textContent = "Use camera instead";
}

async function switchToCameraMode(remember = false) {
  if (remember) rememberCaptureMode("camera");
  cameraModeActive = true;
  document.getElementById("scanner-mode").hidden = true;
  setProtocolHint(true);
  document.getElementById("camera-mode").hidden = false;
  document.getElementById("picker-mode").hidden = true;
  pickerModeToggle.textContent = "Use photo picker instead";
  showViewfinder(true);
  await startCamera();
}

// ---- submit ----
const submitBtn = document.getElementById("submit-btn");
const submitError = document.getElementById("submit-error");

function updateSubmitEnabled() {
  // Front and back are the whole requirement now. The rotation set and the
  // DPI are optional refinements, validated on submit rather than gating it.
  submitBtn.disabled = !(state.files.front && state.files.back);
}

function showSubmitError(message) {
  submitError.textContent = message;
  submitError.hidden = false;
  submitBtn.disabled = false;
  submitBtn.textContent = "Grade card";
}

submitBtn.addEventListener("click", submitJob);

// ---- scanner extras ----
// The scan set has to reach the server in the order it was captured: scan k
// is the card turned k*90 degrees, which is what tells the solve where the
// light was. FormData preserves append order, so the file input's own order
// is carried through.
const MIN_PHOTOMETRIC_SCANS = 3;
const MAX_PHOTOMETRIC_SCANS = 6;

function photometricFiles(inputId) {
  const input = document.getElementById(inputId);
  return input && input.files ? [...input.files] : [];
}

function clearPhotometricInputs() {
  for (const id of ["photometric-front-input", "photometric-back-input"]) {
    const input = document.getElementById(id);
    if (input) input.value = "";
  }
  updateScannerToggleLabel();
}

// What's attached, on the collapsed toggle itself. A file input inside a
// closed section is state you can't see, and this is the flow where that
// bit you: four scans that were never picked up look exactly like four that
// were, until the report comes back identical to the last one.
function updateScannerToggleLabel() {
  const toggle = document.getElementById("scanner-toggle");
  if (!toggle) return;
  const open = !document.getElementById("scanner-fields").hidden;
  const counts = [
    ["front", photometricFiles("photometric-front-input").length],
    ["back", photometricFiles("photometric-back-input").length],
  ].filter(([, count]) => count > 0);
  const attached = counts.map(([side, count]) => `${count} ${side}`).join(", ");
  const suffix = attached ? ` · ${attached} rotation scans` : open ? "" : " (optional)";
  toggle.textContent = `${open ? "−" : "+"} surface relief & dimensions${suffix}`;
}

function scannerValidationError() {
  const dpiValue = document.getElementById("dpi-input").value.trim();
  if (dpiValue !== "" && !(Number(dpiValue) > 0)) {
    return "Scan DPI has to be a positive number (or left blank).";
  }
  for (const [inputId, label] of [["photometric-front-input", "Front"], ["photometric-back-input", "Back"]]) {
    const count = photometricFiles(inputId).length;
    if (count === 0) continue;
    if (count < MIN_PHOTOMETRIC_SCANS || count > MAX_PHOTOMETRIC_SCANS) {
      return `${label} relief scans: pick between ${MIN_PHOTOMETRIC_SCANS} and ${MAX_PHOTOMETRIC_SCANS} files (got ${count}).`;
    }
  }
  return null;
}

function appendScannerFields(formData) {
  const dpiValue = document.getElementById("dpi-input").value.trim();
  if (dpiValue !== "") formData.append("dpi", dpiValue);
  // Which way the card was turned between rotation scans. It decides which
  // light direction each frame is solved against, so getting it wrong
  // doesn't fail — it silently produces a normal map lit from the wrong
  // side, with the two middle frames' directions swapped.
  formData.append("rotation", document.getElementById("rotation-input").value);
  for (const [inputId, field] of [
    ["photometric-front-input", "photometric_front"],
    ["photometric-back-input", "photometric_back"],
  ]) {
    for (const file of photometricFiles(inputId)) formData.append(field, file);
  }
}

function setScannerFieldsOpen(open) {
  document.getElementById("scanner-fields").hidden = !open;
  updateScannerToggleLabel();
}

for (const id of ["photometric-front-input", "photometric-back-input"]) {
  const input = document.getElementById(id);
  if (input) input.addEventListener("change", updateScannerToggleLabel);
}

document.getElementById("scanner-toggle").addEventListener("click", () => {
  setScannerFieldsOpen(document.getElementById("scanner-fields").hidden);
});

async function submitJob() {
  const scannerProblem = scannerValidationError();
  if (scannerProblem) {
    showSubmitError(scannerProblem);
    return;
  }

  submitError.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Uploading…";

  const formData = new FormData();
  formData.append("front", state.files.front);
  formData.append("back", state.files.back);
  appendScannerFields(formData);

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
    handleReport(data.report, data.images, data.report_id);
    return;
  }
}

function handleReport(report, images, reportId = null) {
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

  window.__lastReport = { report, images, reportId };
  // A fullscreen adjust panel lives on <body>, not inside report-content, so
  // re-rendering the report would otherwise leave it stranded over the page.
  // Saving a correction re-renders from the server's response, which is
  // exactly when this happens.
  document.querySelectorAll("body > .centering-adjust").forEach((stray) => stray.remove());
  document.body.classList.remove("adjust-fullscreen-open");
  document.getElementById("report-content").innerHTML =
    permalinkHTML(reportId) + buildReportHTML(report, images);
  wireOverlaySliders();
  wirePermalink();
  wireCenteringAdjust();
  showView("report");
}

// ---- permalink ----
// Every finished report is saved on the server under its job id, so the
// report has a stable address. Shown at the top rather than buried in the
// actions row: the id is the only way back to this report later, and a
// report you can't find again may as well not have been saved.
// Inline SVG rather than glyphs or emoji: these have to inherit currentColor,
// scale with the type, and survive the offline export with no icon font.
const ICONS = {
  copy: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>`,
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>`,
  chevron: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>`,
  corners: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4"/></svg>`,
  edges: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="6" y="3" width="12" height="18" rx="2"/><path d="M6 8h12M6 16h12"/></svg>`,
  centering: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><rect x="8" y="8" width="8" height="8" rx="1"/></svg>`,
  surface: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="m5 16 4-4 3 3 3-4 4 5"/></svg>`,
  dimensions: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2" y="8" width="20" height="8" rx="1.5"/><path d="M7 8v3M12 8v4M17 8v3"/></svg>`,
  auto: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 0 0 18z" fill="currentColor" stroke="none"/></svg>`,
  sun: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2M12 19.5v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2.5 12h2M19.5 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"/></svg>`,
  moon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 13.4A8.2 8.2 0 1 1 10.6 4a6.6 6.6 0 0 0 9.4 9.4z"/></svg>`,
  pin: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 21s7-5.6 7-11a7 7 0 1 0-14 0c0 5.4 7 11 7 11Z"/><circle cx="12" cy="10" r="2.5"/></svg>`,
};

function permalinkHTML(reportId) {
  if (!reportId) return "";
  const url = `${location.origin}/r/${reportId}`;
  return `<div class="permalink">
    <span class="permalink-label">Saved report</span>
    <a class="permalink-url mono" href="/r/${escapeHtml(reportId)}">${escapeHtml(url)}</a>
    <button type="button" class="secondary-btn permalink-copy" id="permalink-copy" data-url="${escapeHtml(url)}">
      <span class="btn-icon">${ICONS.copy}</span><span class="btn-text">Copy</span>
    </button>
  </div>`;
}

function wirePermalink() {
  const button = document.getElementById("permalink-copy");
  if (!button) return;
  const label = button.querySelector(".btn-text");
  const icon = button.querySelector(".btn-icon");
  button.addEventListener("click", async () => {
    let ok = true;
    try {
      await navigator.clipboard.writeText(button.dataset.url);
    } catch (e) {
      // No clipboard permission (or no clipboard API over plain HTTP) — the
      // URL is right there as selectable text, so this is not worth an error.
      ok = false;
    }
    label.textContent = ok ? "Copied" : "Copy failed";
    if (ok) icon.innerHTML = ICONS.check;
    button.classList.toggle("is-done", ok);
    setTimeout(() => {
      label.textContent = "Copy";
      icon.innerHTML = ICONS.copy;
      button.classList.remove("is-done");
    }, 1600);
  });
}

// ---- report rendering ----
function subgradeTile(label, grade, reason) {
  if (grade === null || grade === undefined) {
    // "n/a" alone reads the same whether the capture can't support the
    // measurement or the card defeated it, and those want different things
    // done about them — so the tile carries why, and says it has one.
    const why = reason ? ` title="${escapeHtml(reason)}"` : "";
    const mark = reason ? `<span class="why-mark" aria-hidden="true">?</span>` : "";
    return `<div class="subgrade-tile${reason ? " subgrade-tile--explained" : ""}"${why}>
      <div class="subgrade-label">${label}</div>
      <div class="subgrade-value muted">n/a${mark}</div>
    </div>`;
  }
  return `<div class="subgrade-tile">
    <div class="subgrade-label">${label}</div>
    <div class="subgrade-value" style="color:${gradeColor(grade)}">${grade}</div>
  </div>`;
}

// Where each sub-grade's refusal is explained, when it was refused. The
// report carries the reason next to the measurement that produced it rather
// than next to the summary tile, so this is the lookup between them.
function subgradeReason(report, sideKey, attribute) {
  const side = (report.corners_edges || {})[sideKey] || {};
  if (attribute === "corners") return side.corners_reason || null;
  if (attribute === "edges") return side.edges_reason || null;
  if (attribute === "surface") return ((report.surface || {})[sideKey] || {}).reason || null;
  if (attribute === "centering") {
    const centering = (report.centering || {})[sideKey];
    if (centering && centering.measurable === false) {
      return "neither axis's border boundary could be found confidently in this capture — a borderless or full-art card, or a capture where the border isn't visible.";
    }
  }
  return null;
}

// ---- manual centering ----
// The detector refuses an axis it can't find, which is right, but it has a
// failure mode no confidence gate catches: finding *an* edge, confidently
// enough to pass, that isn't the border. Modern cards stack an artwork
// boundary, an inner frame band and a thin rule within a few millimetres of
// each other. So the boundaries can be placed by hand, over the warp, and
// saved back to the report.
//
// Eight lines, not four. The card edge is adjustable as well as the border
// boundary: the warp is *defined* by the detected corners, so the card edge
// is the image edge by construction — but corner detection can be a pixel or
// two out, and at 1500px across a 63mm card two pixels is 0.08mm, which is
// enough to move a border ratio across a grade line.
const EDGE_AXIS = { left: "horizontal", right: "horizontal", top: "vertical", bottom: "vertical" };
const EDGE_OPPOSITE = { left: "right", right: "left", top: "bottom", bottom: "top" };
const MAX_ADJUST_ZOOM = 16;
let centeringTolerances = null;

async function loadCenteringTolerances() {
  if (centeringTolerances) return centeringTolerances;
  const res = await fetch("/api/centering-tolerances");
  if (!res.ok) throw new Error("couldn't load the tolerance tables");
  centeringTolerances = await res.json();
  return centeringTolerances;
}

// Mirrors pipeline.centering._grade_from_ratio. The tables themselves come
// from the server, so there's still one source of truth for the numbers.
function gradeFromRatio(worsePct, tolerances) {
  for (const tier of tolerances) {
    if (worsePct <= tier.max_ratio) return tier.grade;
  }
  return Math.max(1, tolerances[tolerances.length - 1].grade - 2);
}

function axisFromBorders(pxA, pxB, tolerances) {
  const total = pxA + pxB > 0 ? pxA + pxB : 1;
  const pctA = (100 * pxA) / total;
  const pctB = (100 * pxB) / total;
  return {
    pctA,
    pctB,
    ratio: `${pctA.toFixed(0)}/${pctB.toFixed(0)}`,
    grade: gradeFromRatio(Math.max(pctA, pctB), tolerances),
  };
}

function adjustPanelHTML(side, label) {
  const handle = (edge, kind) =>
    `<div class="adjust-line adjust-line--${EDGE_AXIS[edge] === "horizontal" ? "v" : "h"} adjust-line--${kind}"
       data-edge="${edge}" data-kind="${kind}" role="slider" tabindex="0"
       aria-label="${label} ${edge} ${kind === "outer" ? "card edge" : "border boundary"}"><i></i></div>`;
  return `<div class="centering-adjust" data-side="${side}" hidden>
    <div class="adjust-toolbar">
      <button type="button" class="zoom-btn" data-zoom="out" aria-label="Zoom out">&minus;</button>
      <span class="adjust-zoom mono">1.0&times;</span>
      <button type="button" class="zoom-btn" data-zoom="in" aria-label="Zoom in">+</button>
      <button type="button" class="text-toggle" data-zoom="reset">Fit</button>
      <button type="button" class="text-toggle adjust-fullscreen-btn" data-fullscreen>Fullscreen</button>
    </div>
    <div class="adjust-viewport">
      <div class="adjust-canvas">
        <img class="adjust-img" alt="${label} scan, for placing border boundaries">
        ${["left", "right", "top", "bottom"].map((edge) => handle(edge, "outer") + handle(edge, "inner")).join("")}
      </div>
      <div class="adjust-loupe" aria-hidden="true" hidden><span class="adjust-loupe-crosshair"></span></div>
    </div>
    <div class="adjust-side">
      <p class="field-hint">
        <b class="swatch swatch--inner"></b> border meets artwork &nbsp;
        <b class="swatch swatch--outer"></b> card edge &middot;
        drag anywhere near a line to move it, drag the open card to pan,
        scroll to zoom, arrow keys nudge (shift &times;10).
      </p>
      <div class="adjust-readout"></div>
      <div class="adjust-actions">
        <button type="button" class="secondary-btn" data-adjust-cancel>Cancel</button>
        <button type="button" class="primary-btn" data-adjust-save>Save centering</button>
      </div>
      <div class="banner banner-error adjust-error" hidden></div>
    </div>
  </div>`;
}

function wireCenteringAdjust() {
  const stored = window.__lastReport;
  if (!stored || !stored.reportId) return;
  const { report, images, reportId } = stored;

  document.querySelectorAll(".centering-adjust").forEach((panel) => {
    const side = panel.dataset.side;
    const aligned = images[`${side}_aligned`];
    const card = panel.closest(".centering-card");
    const toggle = card && card.querySelector("[data-adjust-open]");
    const overlay = card && card.querySelector(".overlay-img");
    if (!toggle) return;
    if (!aligned) {
      toggle.remove();
      return;
    }

    const viewport = panel.querySelector(".adjust-viewport");
    const canvas = panel.querySelector(".adjust-canvas");
    const img = panel.querySelector(".adjust-img");
    const readout = panel.querySelector(".adjust-readout");
    const zoomLabel = panel.querySelector(".adjust-zoom");
    const loupe = panel.querySelector(".adjust-loupe");
    const errorEl = panel.querySelector(".adjust-error");
    // Zoom into the capture's own detail where it exists. It's a URL, not a
    // data URI, so it costs nothing until this panel is opened — and if it
    // isn't there (a capture at or below canonical resolution) the canonical
    // warp is what there is.
    const detail = images[`${side}_detail`];
    img.src = detail || aligned;
    if (detail) {
      img.addEventListener("error", () => {
        if (!img.src.startsWith("data:")) img.src = aligned;
      });
    }

    const sideData = (report.centering || {})[side] || {};
    const stored_edges = sideData.card_edge_px || {};
    // Positions are distances inward from each side of the warp, which is
    // what the report already stores and what the API takes back.
    const outer = {
      left: Number(stored_edges.left) || 0,
      right: Number(stored_edges.right) || 0,
      top: Number(stored_edges.top) || 0,
      bottom: Number(stored_edges.bottom) || 0,
    };
    // Seed from whatever the detector produced, even on an axis it refused:
    // a wrong line you can drag beats a line at zero.
    const inner = {
      left: outer.left + (Number((sideData.horizontal || {}).side_a_px) || 0),
      right: outer.right + (Number((sideData.horizontal || {}).side_b_px) || 0),
      top: outer.top + (Number((sideData.vertical || {}).side_a_px) || 0),
      bottom: outer.bottom + (Number((sideData.vertical || {}).side_b_px) || 0),
    };

    let zoom = 1;
    let panX = 0;
    let panY = 0;

    const span = (edge) =>
      EDGE_AXIS[edge] === "horizontal"
        ? img.naturalWidth || (centeringTolerances && centeringTolerances.canonical_width_px) || 1500
        : img.naturalHeight || (centeringTolerances && centeringTolerances.canonical_height_px) || 2100;

    // The warp is normalised to the nominal card, so a width in warp pixels
    // converts to millimetres without knowing the scan's DPI. It assumes the
    // card is nominal size — which is what the dimensions stage checks.
    const mm = (edge, px) => (EDGE_AXIS[edge] === "horizontal" ? (px * 63) / span(edge) : (px * 88) / span(edge));

    function applyTransform() {
      canvas.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
      zoomLabel.textContent = `${zoom.toFixed(1)}×`;
      // The whole canvas is scaled, so anything drawn on it is scaled too —
      // a 1px line becomes an 8px bar at 8x, right over the boundary you're
      // zooming in to read. These counter-scale, so the line stays one
      // screen pixel and the grab area stays one finger wide at any zoom.
      panel.style.setProperty("--adjust-hairline", `${1 / zoom}px`);
      panel.style.setProperty("--adjust-halo", `${0.5 / zoom}px`);
      panel.style.setProperty("--adjust-grab", `${14 / zoom}px`);
    }

    // Size the image to *contain* within the viewport rather than leaving it
    // to CSS height:100%, which fits the height and lets the width overflow.
    function fitImage() {
      const vw = viewport.clientWidth;
      const vh = viewport.clientHeight;
      if (!vw || !vh) return;
      const aspect = span("left") / span("top");
      let height = vh;
      let width = height * aspect;
      if (width > vw) {
        width = vw;
        height = width / aspect;
      }
      img.style.width = `${width}px`;
      img.style.height = `${height}px`;
    }

    function setZoom(next, anchorX, anchorY) {
      const box = viewport.getBoundingClientRect();
      const ax = anchorX === undefined ? box.left + box.width / 2 : anchorX;
      const ay = anchorY === undefined ? box.top + box.height / 2 : anchorY;
      const before = canvas.getBoundingClientRect();
      // Fraction of the image sitting under the anchor, held fixed across
      // the zoom so the point you pointed at doesn't slide away.
      const fx = before.width ? (ax - before.left) / before.width : 0.5;
      const fy = before.height ? (ay - before.top) / before.height : 0.5;
      zoom = Math.max(1, Math.min(MAX_ADJUST_ZOOM, next));
      applyTransform();
      const after = canvas.getBoundingClientRect();
      panX += ax - (after.left + fx * after.width);
      panY += ay - (after.top + fy * after.height);
      applyTransform();
      if (zoom >= LOUPE_MAX_ZOOM) hideLoupe();
    }

    function resetView() {
      zoom = 1;
      fitImage();
      // Fit means centred on both axes, and the canvas sits at the viewport's
      // origin, so centring is the pan offset rather than a CSS trick.
      panX = Math.max(0, (viewport.clientWidth - canvas.offsetWidth) / 2);
      panY = Math.max(0, (viewport.clientHeight - canvas.offsetHeight) / 2);
      applyTransform();
      render();
    }

    function render() {
      panel.querySelectorAll(".adjust-line").forEach((line) => {
        const edge = line.dataset.edge;
        const source = line.dataset.kind === "outer" ? outer : inner;
        line.style[edge] = `${(100 * source[edge]) / span(edge)}%`;
        // A card edge sitting exactly on the image edge is the normal case
        // and carries no information; show it, but quietly.
        if (line.dataset.kind === "outer") line.classList.toggle("adjust-line--flush", source[edge] <= 0);
      });
      const tables = centeringTolerances;
      if (!tables) return;
      const tolerances = tables[side] || tables.front;
      const widths = {
        left: inner.left - outer.left,
        right: inner.right - outer.right,
        top: inner.top - outer.top,
        bottom: inner.bottom - outer.bottom,
      };
      const axisH = axisFromBorders(widths.left, widths.right, tolerances);
      const axisV = axisFromBorders(widths.top, widths.bottom, tolerances);
      const row = (label, a, b, axis, edgeA, edgeB) =>
        `<div class="adjust-row"><span class="axis-label">${label}</span>
          <span class="mono">${mm(edgeA, widths[edgeA]).toFixed(2)}mm / ${mm(edgeB, widths[edgeB]).toFixed(2)}mm</span>
          <span class="mono adjust-ratio">${axis.ratio}</span>
          <span class="grade-pill-sm">g${axis.grade}</span></div>`;
      readout.innerHTML =
        row("L/R", widths.left, widths.right, axisH, "left", "right") +
        row("T/B", widths.top, widths.bottom, axisV, "top", "bottom") +
        `<div class="adjust-row adjust-row--total"><span class="axis-label">=</span>
           <span>side grade</span>
           <span class="grade-pill" style="color:${gradeColor(Math.min(axisH.grade, axisV.grade))}">grade ${Math.min(axisH.grade, axisV.grade)}</span></div>`;
    }

    function setLine(edge, kind, value) {
      const limit = span(edge);
      const opposite = EDGE_OPPOSITE[edge];
      if (kind === "outer") {
        // The card edge can't pass its own border boundary, and can't reach
        // the far side's card edge.
        outer[edge] = Math.max(0, Math.min(Math.min(inner[edge], limit - outer[opposite] - 1), value));
      } else {
        // The border boundary can't cross back over its own card edge, nor
        // meet the opposite boundary — past that there's no artwork panel
        // between them and the ratio stops meaning anything.
        inner[edge] = Math.max(outer[edge], Math.min(limit - inner[opposite] - 1, value));
      }
      render();
    }

    // A magnifier pinned to the line being dragged. Zoom-and-then-place works,
    // but it means deciding where the boundary is before you can see it; every
    // comparable tool ships a loupe for exactly this reason.
    const LOUPE_SIZE = 132;
    const LOUPE_MAGNIFICATION = 8;
    // Once the viewport is magnified this far the loupe has nothing left to
    // add — you are already looking at individual pixels — and its anchor,
    // the midpoint of the side, has usually panned off-screen.
    const LOUPE_MAX_ZOOM = 4;

    function showLoupe(edge, kind) {
      if (zoom >= LOUPE_MAX_ZOOM) {
        hideLoupe();
        return;
      }
      // Nothing here divides by the canvas rect — it only offsets into it —
      // so a zero-size rect (a panel not yet laid out) places the loupe
      // harmlessly rather than needing a guard that would skip showing it.
      const rect = canvas.getBoundingClientRect();
      const box = viewport.getBoundingClientRect();
      const position = kind === "outer" ? outer[edge] : inner[edge];
      const fraction = position / span(edge);
      // Centre of the loupe, in viewport coordinates: on the line, and
      // halfway along the side it belongs to.
      const along = 0.5;
      const horizontal = EDGE_AXIS[edge] === "horizontal";
      const fx = horizontal ? (edge === "left" ? fraction : 1 - fraction) : along;
      const fy = horizontal ? along : edge === "top" ? fraction : 1 - fraction;
      const cx = rect.left - box.left + fx * rect.width;
      const cy = rect.top - box.top + fy * rect.height;

      // Clamped into the viewport: a loupe half off the edge shows less than
      // no loupe, because you go looking for the missing half.
      const clamp = (value, limit) => Math.max(0, Math.min(limit - LOUPE_SIZE, value));
      loupe.style.left = `${clamp(cx - LOUPE_SIZE / 2, box.width)}px`;
      loupe.style.top = `${clamp(cy - LOUPE_SIZE / 2, box.height)}px`;
      // Magnify relative to the image's natural size, not its displayed
      // size, so the loupe shows real captured detail rather than the
      // viewport's own scaling of it.
      const naturalW = img.naturalWidth || span("left");
      const naturalH = img.naturalHeight || span("top");
      loupe.style.backgroundImage = `url("${img.src}")`;
      loupe.style.backgroundSize = `${naturalW * LOUPE_MAGNIFICATION}px ${naturalH * LOUPE_MAGNIFICATION}px`;
      loupe.style.backgroundPosition = `${LOUPE_SIZE / 2 - fx * naturalW * LOUPE_MAGNIFICATION}px ${LOUPE_SIZE / 2 - fy * naturalH * LOUPE_MAGNIFICATION}px`;
      loupe.classList.toggle("adjust-loupe--vertical", horizontal);
      loupe.hidden = false;
    }

    function hideLoupe() {
      loupe.hidden = true;
    }

    function positionFor(edge, clientX, clientY) {
      // The canvas rect already carries the zoom and pan, so reading a
      // fraction off it needs no transform maths of its own.
      const rect = canvas.getBoundingClientRect();
      if (edge === "left") return ((clientX - rect.left) / rect.width) * span(edge);
      if (edge === "right") return ((rect.right - clientX) / rect.width) * span(edge);
      if (edge === "top") return ((clientY - rect.top) / rect.height) * span(edge);
      return ((rect.bottom - clientY) / rect.height) * span(edge);
    }

    // A 2px line is a poor thing to have to hit, and zooming in to hit it
    // defeats the point of being able to place it accurately. So a press
    // anywhere within reach of a line grabs that line; a press with no line
    // in reach pans instead.
    const GRAB_RADIUS_PX = 36;

    function lineScreenPosition(line) {
      const rect = canvas.getBoundingClientRect();
      const edge = line.dataset.edge;
      const value = line.dataset.kind === "outer" ? outer[edge] : inner[edge];
      const fraction = value / span(edge);
      if (EDGE_AXIS[edge] === "horizontal") {
        return { axis: "x", at: edge === "left" ? rect.left + fraction * rect.width : rect.right - fraction * rect.width };
      }
      return { axis: "y", at: edge === "top" ? rect.top + fraction * rect.height : rect.bottom - fraction * rect.height };
    }

    function nearestLine(clientX, clientY) {
      let best = null;
      panel.querySelectorAll(".adjust-line").forEach((line) => {
        const { axis, at } = lineScreenPosition(line);
        const distance = Math.abs((axis === "x" ? clientX : clientY) - at);
        if (!best || distance < best.distance) best = { line, distance };
      });
      return best && best.distance <= GRAB_RADIUS_PX ? best.line : null;
    }

    panel.querySelectorAll(".adjust-line").forEach((line) => {
      const edge = line.dataset.edge;
      const kind = line.dataset.kind;
      line.addEventListener("keydown", (event) => {
        const horizontal = EDGE_AXIS[edge] === "horizontal";
        const usesThisKey = horizontal
          ? event.key === "ArrowLeft" || event.key === "ArrowRight"
          : event.key === "ArrowUp" || event.key === "ArrowDown";
        if (!usesThisKey) return;
        event.preventDefault();
        const step = event.shiftKey ? 10 : 1;
        // Keys move the line the way it looks on screen, so the same key
        // means the same direction whichever side the handle is on.
        const screenward = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1;
        const inward = edge === "left" || edge === "top" ? screenward : -screenward;
        setLine(edge, kind, (kind === "outer" ? outer[edge] : inner[edge]) + inward * step);
        // Keyboard placement wants the magnifier just as much as dragging
        // does, and more so: there's no finger in the way to hide it.
        showLoupe(edge, kind);
      });
      line.addEventListener("pointerdown", () => showLoupe(edge, kind));
      line.addEventListener("blur", hideLoupe);
    });

    // One pointer handler for the whole stage. A press either grabs the line
    // nearest it or starts a pan, and the decision is made once, on the way
    // down — so a drag never switches between the two halfway through.
    let drag = null;

    viewport.addEventListener("pointerdown", (event) => {
      if (event.button !== undefined && event.button !== 0) return;
      event.preventDefault();
      const line = event.target.closest(".adjust-line") || nearestLine(event.clientX, event.clientY);
      viewport.setPointerCapture(event.pointerId);
      if (line) {
        drag = { id: event.pointerId, line, edge: line.dataset.edge, kind: line.dataset.kind };
        line.classList.add("adjust-line--dragging");
        line.focus({ preventScroll: true });
        showLoupe(drag.edge, drag.kind);
      } else {
        drag = { id: event.pointerId, pan: true, x: event.clientX, y: event.clientY };
        viewport.classList.add("adjust-viewport--panning");
      }
    });

    viewport.addEventListener("pointermove", (event) => {
      if (!drag || drag.id !== event.pointerId) {
        // Not dragging: hint which line a press would pick up.
        viewport.classList.toggle("adjust-viewport--over-line", !!nearestLine(event.clientX, event.clientY));
        return;
      }
      if (drag.pan) {
        panX += event.clientX - drag.x;
        panY += event.clientY - drag.y;
        drag.x = event.clientX;
        drag.y = event.clientY;
        applyTransform();
        return;
      }
      setLine(drag.edge, drag.kind, Math.round(positionFor(drag.edge, event.clientX, event.clientY)));
      showLoupe(drag.edge, drag.kind);
    });

    const endDrag = (event) => {
      if (!drag || drag.id !== event.pointerId) return;
      if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
      if (drag.line) {
        drag.line.classList.remove("adjust-line--dragging");
        hideLoupe();
      }
      viewport.classList.remove("adjust-viewport--panning");
      drag = null;
    };
    viewport.addEventListener("pointerup", endDrag);
    viewport.addEventListener("pointercancel", endDrag);
    viewport.addEventListener("pointerleave", () => viewport.classList.remove("adjust-viewport--over-line"));

    viewport.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        setZoom(zoom * (event.deltaY < 0 ? 1.18 : 1 / 1.18), event.clientX, event.clientY);
      },
      { passive: false },
    );

    panel.querySelectorAll("[data-zoom]").forEach((button) => {
      button.addEventListener("click", () => {
        const action = button.dataset.zoom;
        if (action === "reset") resetView();
        else setZoom(action === "in" ? zoom * 1.5 : zoom / 1.5);
      });
    });

    // Fullscreen. Placing a boundary to the pixel in a 420px-tall box inside
    // a narrow report column is the wrong shape of problem; this is a CSS
    // overlay rather than the Fullscreen API so the toolbar, readout and
    // Save button come with it instead of being locked out behind it.
    const fullscreenBtn = panel.querySelector("[data-fullscreen]");
    // The panel has to be moved to <body> to go fullscreen. Its own card
    // carries backdrop-filter, and a filtered ancestor becomes the containing
    // block for position:fixed descendants — so the overlay was being trapped
    // inside a 300px-wide card rather than covering the viewport. A marker
    // node holds its place so it goes back exactly where it came from.
    let homeAnchor = null;

    function setFullscreen(on) {
      if (on && !homeAnchor) {
        homeAnchor = document.createComment("centering-adjust");
        panel.parentNode.insertBefore(homeAnchor, panel);
        document.body.appendChild(panel);
      } else if (!on && homeAnchor) {
        homeAnchor.parentNode.insertBefore(panel, homeAnchor);
        homeAnchor.remove();
        homeAnchor = null;
      }
      panel.classList.toggle("centering-adjust--fullscreen", on);
      document.body.classList.toggle("adjust-fullscreen-open", on);
      fullscreenBtn.textContent = on ? "Exit fullscreen" : "Fullscreen";
      hideLoupe();
      // The viewport just changed size, so "fit" means something else now —
      // but only after layout has caught up with the move to <body>.
      const afterLayout = window.requestAnimationFrame || ((fn) => setTimeout(fn, 0));
      afterLayout(resetView);
    }

    fullscreenBtn.addEventListener("click", () => setFullscreen(!panel.classList.contains("centering-adjust--fullscreen")));
    panel.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && panel.classList.contains("centering-adjust--fullscreen")) {
        event.stopPropagation();
        setFullscreen(false);
      }
    });

    function close() {
      panel.hidden = true;
      hideLoupe();
      setFullscreen(false);
      if (overlay) overlay.hidden = false;
      toggle.textContent = "Adjust boundaries";
    }

    toggle.addEventListener("click", async () => {
      const opening = panel.hidden;
      if (!opening) {
        close();
        return;
      }
      panel.hidden = false;
      if (overlay) overlay.hidden = true;
      toggle.textContent = "Done adjusting";
      try {
        await loadCenteringTolerances();
      } catch (e) {
        errorEl.textContent = e.message;
        errorEl.hidden = false;
      }
      render();
      setFullscreen(true);
    });

    panel.querySelector("[data-adjust-cancel]").addEventListener("click", close);

    panel.querySelector("[data-adjust-save]").addEventListener("click", async (event) => {
      const button = event.currentTarget;
      button.disabled = true;
      button.textContent = "Saving…";
      errorEl.hidden = true;
      try {
        const res = await fetch(`/api/report/${reportId}/centering`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            [side]: {
              borders: {
                left: inner.left - outer.left,
                right: inner.right - outer.right,
                top: inner.top - outer.top,
                bottom: inner.bottom - outer.bottom,
              },
              edges: { ...outer },
            },
          }),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(body.detail || "couldn't save the correction");
        // Re-render from the server's response rather than patching the DOM:
        // the grade, sub-grades and dings all move with centering, and the
        // server has just recomputed every one of them.
        handleReport(body.report, body.images, reportId);
      } catch (e) {
        errorEl.textContent = e.message;
        errorEl.hidden = false;
        button.disabled = false;
        button.textContent = "Save centering";
      }
    });

    img.addEventListener("load", () => {
      fitImage();
      render();
    });
    // Entering fullscreen, leaving it, rotating a phone: all change what
    // "fit" means, and none of them fire anything else this panel listens to.
    // Registered on the window, so it has to come off again — every report
    // render builds fresh panels, and without this each one left a listener
    // behind firing against a detached node.
    const onResize = () => {
      if (!panel.isConnected) {
        window.removeEventListener("resize", onResize);
        return;
      }
      if (!panel.hidden) resetView();
    };
    window.addEventListener("resize", onResize);
    fitImage();
    render();
  });
}

// How well each frame of a photometric set lined up. A misaligned set does
// not fail — it renders the card's own print embossed twice and calls it
// damage — so the numbers behind the render have to be readable without
// going to the logs.
function registrationHTML(frames) {
  const row = (frame, index) => {
    const shift = frame.shift_px ? `${frame.shift_px[0]}, ${frame.shift_px[1]}` : "—";
    const scale = frame.scale ? frame.scale.map((v) => `${((v - 1) * 100).toFixed(2)}%`).join(" / ") : "—";
    const residual = frame.residual === undefined ? "—" : frame.residual.toFixed(2);
    const quad = frame.quad_width_px ? `${Math.round(frame.quad_width_px)}×${Math.round(frame.quad_height_px)}` : "—";
    const state = frame.fitted === false ? "refused" : "aligned";
    return `<tr class="${frame.fitted === false ? "align-row--refused" : ""}">
      <th scope="row">${index + 1}${frame.rotation_deg === undefined ? "" : ` · ${frame.rotation_deg}°`}</th>
      <td class="mono">${quad}</td>
      <td class="mono">${shift}</td>
      <td class="mono">${scale}</td>
      <td class="mono">${residual}</td>
      <td>${state}</td>
    </tr>`;
  };
  return `<details class="align-detail">
    <summary>Frame alignment</summary>
    <div class="table-scroll">
      <table class="grader-table">
        <thead><tr>
          <th>Scan</th><th title="The card as the detector found it in that scan">Detected card</th>
          <th title="How far this frame had to move to line up">Shift px</th>
          <th title="How much it had to be resized">Scale</th>
          <th title="How far apart the print edges still are after aligning — near zero is aligned">Residual</th>
          <th>Fit</th>
        </tr></thead>
        <tbody>${frames.map(row).join("")}</tbody>
      </table>
    </div>
    <p class="field-hint">Residual is measured on the high-pass, so it answers whether the print edges
    line up rather than whether the frames are the same brightness — they are not supposed to be.</p>
  </details>`;
}

function adjustControlHTML(label) {
  const side = label.toLowerCase();
  if (side !== "front" && side !== "back") return "";
  return `<button type="button" class="text-toggle" data-adjust-open>Adjust boundaries</button>
  ${adjustPanelHTML(side, label)}`;
}

// The same measurement, read against every grading service's published
// table. Reference only — PSA drives this report's grade. The tables for
// everyone except PSA are third-party transcriptions rather than primary
// sources, which the footnote says out loud, because a table taken on trust
// is how the back tolerances were wrong here before.
function graderComparisonHTML(centering) {
  const byGrader = centering && centering.by_grader;
  if (!byGrader || !Object.keys(byGrader).length) return "";
  const rows = Object.values(byGrader)
    .map(
      (g) => `<tr>
        <th scope="row">${escapeHtml(g.label)}</th>
        <td class="mono">${g.front === null || g.front === undefined ? "—" : g.front}</td>
        <td class="mono">${g.back === null || g.back === undefined ? "—" : g.back}</td>
        <td class="mono" style="color:${gradeColor(g.grade)}">${g.grade === null || g.grade === undefined ? "—" : g.grade}</td>
        <td class="grader-source muted">${escapeHtml(g.source || "")}</td>
      </tr>`,
    )
    .join("");
  return `<details class="grader-compare">
    <summary>Centering under every grading service's table</summary>
    <div class="table-scroll">
      <table class="grader-table">
        <thead><tr><th>Service</th><th>Front</th><th>Back</th><th>Centering</th><th>Table source</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p class="field-hint">Centering only — these are not overall grades, and the other services' tables
    are transcriptions of published standards rather than primary sources.</p>
  </details>`;
}

function centeringSideHTML(label, sideData, overlayImg, knownFullArt = false) {
  // measurable === false: the border-boundary detection had no confident
  // signal on this side (borderless/full-art card, or the border isn't
  // visible in this capture). Showing the raw ratios would present
  // argmax-of-noise as real measurements. Older reports lack the flag —
  // treat missing as measurable.
  if (sideData.measurable === false) {
    const note = knownFullArt
      ? "Couldn't measure — this is a full-art/borderless card, which has no printed border to measure. Expected, not a capture problem."
      : "Couldn't measure — borderless/full-art card, or the border isn't visible in this capture.";
    return `<div class="centering-card">
      <div class="centering-card-header">${label}
        <span class="grade-pill" style="color:${gradeColor(null)}">n/a</span>
      </div>
      ${overlayImg ? `<img class="overlay-img" src="${overlayImg}" alt="${label} centering overlay">` : ""}
      <div class="muted" style="font-size:0.82rem">${note}</div>
      ${adjustControlHTML(label)}
    </div>`;
  }
  // Per-axis rows: a measurable axis shows its real ratio; an unmeasurable
  // one shows n/a — its numbers would be argmax-of-noise. (Real case: a
  // soft capture of a card back where top/bottom measured fine but the
  // low-contrast left/right boundary was invisible — the vertical
  // measurement is real and belongs in the report.) Older reports lack the
  // axis flag — treat missing as measurable.
  const hOk = sideData.horizontal.measurable !== false;
  const vOk = sideData.vertical.measurable !== false;
  // DINGS marker goes to the measurable axis that set this side's grade.
  let worseAxis = null;
  if (hOk && vOk) worseAxis = sideData.horizontal.grade <= sideData.vertical.grade ? "h" : "v";
  else if (hOk) worseAxis = "h";
  else if (vOk) worseAxis = "v";
  const ding = `<span class="ding-marker" title="This axis set the grade">drove the grade</span>`;
  // PSA's published 5% front leeway, and how much the border wandered along
  // the side, are both things a reader has to be able to see: the first
  // explains a grade the bare table wouldn't give, the second says how much
  // to trust a single number for a card that may not be cut square.
  const leewayNote = (axis) =>
    axis.leeway_applied
      ? ` <span class="leeway-chip" title="Published: a 5% leeway is given to the front centering minimum standards for cards which grade PSA 7 or better. Strict table alone: grade ${axis.strict_grade}">+5% leeway</span>`
      : "";
  const wanderNote = (axis) =>
    axis.variation_px >= 6
      ? ` <span class="muted" title="Border width measured at three points down the side; PSA grades the most off-centre part, not the average">±${axis.variation_px.toFixed(0)}px along the side</span>`
      : "";
  const axisRow = (labelChar, axis, ok, isWorse) =>
    ok
      ? `<div class="axis-row"><span class="axis-label">${labelChar}</span><span class="mono" title="${escapeHtml(axis.ratio_conventional || axis.ratio)} in the larger-first convention PSA prints">${escapeHtml(axis.ratio)}</span>
      <span class="grade-pill-sm">g${axis.grade}</span>${isWorse ? ding : ""}${leewayNote(axis)}${wanderNote(axis)}</div>`
      : `<div class="axis-row"><span class="axis-label">${labelChar}</span><span class="muted">n/a — boundary not visible in this capture</span></div>`;
  return `<div class="centering-card">
    <div class="centering-card-header">${label}
      ${sideData.manual ? `<span class="manual-chip" title="Boundaries placed by hand, not detected">set by hand</span>` : ""}
      <span class="grade-pill" style="color:${gradeColor(sideData.grade)}">grade ${sideData.grade}</span>
    </div>
    ${overlayImg ? `<img class="overlay-img" src="${overlayImg}" alt="${label} centering overlay">` : ""}
    ${axisRow("H", sideData.horizontal, hOk, worseAxis === "h")}
    ${axisRow("V", sideData.vertical, vOk, worseAxis === "v")}
    ${adjustControlHTML(label)}
  </div>`;
}

function regionTileHTML(bareKey, prefixedKey, shortLabel, sideKey, regionData, images, isDing) {
  const img = images[`${sideKey}_${prefixedKey}`];
  return `<div class="region-tile${isDing ? " region-tile--ding" : ""}">
    ${img ? `<img class="region-thumb" src="${img}" alt="${shortLabel}">` : ""}
    <div class="region-label">${shortLabel}${isDing ? ` <span class="ding-marker" title="Worst region — set this side's grade">!</span>` : ""}</div>
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
  // TAG's DINGS idea: mark the region(s) that actually set this side's
  // grade — the worst grade among all 8 corners/edges — so the eye goes
  // straight to what matters instead of scanning 8 equal-looking tiles.
  const allRegions = [...Object.values(sideData.corners), ...Object.values(sideData.edges)];
  const worst = Math.min(...allRegions.map((r) => r.grade));
  let tiles = "";
  for (const [bare, prefixed, short] of CORNER_DEFS) {
    tiles += regionTileHTML(bare, prefixed, short, sideKey, sideData.corners[bare], images, sideData.corners[bare].grade === worst);
  }
  for (const [bare, prefixed, short] of EDGE_DEFS) {
    tiles += regionTileHTML(bare, prefixed, short, sideKey, sideData.edges[bare], images, sideData.edges[bare].grade === worst);
  }
  return `<div class="ce-card">
    <div class="centering-card-header">${label}
      <span class="grade-pill" style="color:${gradeColor(sideData.grade)}">grade ${sideData.grade}</span>
    </div>
    <div class="region-grid">${tiles}</div>
  </div>`;
}

function surfaceSideHTML(sideKey, label, sideData) {
  if (!sideData) return "";

  const graded = sideData.grade !== null && sideData.grade !== undefined;
  const gradePill = graded
    ? `<span class="grade-pill" style="color:${gradeColor(sideData.grade)}">grade ${sideData.grade}</span>`
    : `<span class="method-chip">not graded</span>`;

  // What set the grade, and the marks behind it. "surface 4" is not
  // something anyone can check; "a 2.7mm crease at the bottom-left corner
  // caps this at 4" is — and it tells you where to look on the card.
  const kinds = sideData.defect_kinds || {};
  const kindSummary = Object.keys(kinds).length
    ? `<div class="axis-row">
         <span class="axis-label">found</span>
         <span class="mono">${Object.entries(kinds).map(([k, n]) => `${n}&times; ${escapeHtml(k)}`).join(", ")}</span>
       </div>`
    : "";
  const limit =
    graded && sideData.limited_by
      ? `<div class="axis-row">
           <span class="axis-label">limited by</span>
           <span class="mono">${escapeHtml(sideData.limited_by)}</span>
         </div>`
      : "";
  const worst = (sideData.defects || []).filter((d) => d.grade_cap < 10).slice(0, 4);
  const worstList = worst.length
    ? `<ul class="defect-list">${worst
        .map(
          (d) =>
            `<li><span class="defect-kind">${escapeHtml(d.kind)}</span>
               ${d.length_mm}&times;${d.width_mm}mm at ${d.centre_mm[0]}, ${d.centre_mm[1]}mm
               <span class="muted">caps at ${d.grade_cap}</span></li>`
        )
        .join("")}</ul>`
    : "";

  return `<div class="surface-card">
    <div class="centering-card-header">${label} ${gradePill}</div>
    ${limit}
    ${kindSummary}
    <div class="axis-row">
      <span class="axis-label">defect area</span><span class="mono">${sideData.defect_area_pct.toFixed(3)}%</span>
    </div>
    <div class="axis-row">
      <span class="axis-label">defects</span><span class="mono">${sideData.defect_count}</span>
    </div>
    ${worstList}
    <div class="muted cardvision-note">${escapeHtml(sideData.note || "")}</div>
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

// The offline capture-pair check: did the same side get uploaded twice?
// Distinct from the identity warning below, which says the same thing from
// the vision stage — that one needs an API key and is usually skipped, this
// one always runs. A warning, not a block: grading one side against both
// tolerance tables is a legitimate thing to do on purpose.
function capturePairWarningHTML(report) {
  const pair = report.capture_pair;
  if (!pair || !pair.same_side_suspected) return "";
  return `<div class="banner banner-warn">${escapeHtml(pair.note || "The front and back uploads look like the same side.")}</div>`;
}

// Card identity header + capture-pair sanity warnings, from the vision
// identify stage. The commercial graders all identify-first (what card is
// this?) before grading; ours also cross-checks the photo pair itself.
function cardIdHTML(report) {
  const id = report.card_id;
  if (!id) return "";
  let html = "";
  if (id.front_image_side === "back" || id.back_image_side === "front") {
    html += `<div class="banner banner-warn">The two photos may be swapped or show the same side twice — check before trusting this report.</div>`;
  }
  if (id.looks_like_same_card === false) {
    html += `<div class="banner banner-warn">The front and back photos may not be the same card.</div>`;
  }
  const badges = [];
  if (id.is_full_art) badges.push("full-art");
  if (id.is_holo) badges.push("holo");
  const setBits = [id.set_name, id.collector_number ? `#${id.collector_number}` : ""].filter(Boolean).join(" ");
  html += `<div class="card-identity">
    <span class="card-name">${escapeHtml(id.card_name)}</span>
    ${setBits ? `<span class="muted">${escapeHtml(setBits)}</span>` : ""}
    ${badges.map((b) => `<span class="indicative-tag">${b}</span>`).join("")}
    <span class="muted card-id-conf">(id confidence: ${escapeHtml(id.confidence)}${id.model ? ", " + escapeHtml(id.model) : ""})</span>
  </div>`;
  return html;
}

// Raw-card market value for the identified card — the "is this worth
// submitting" context. Raw price only; graded value varies with the grade.
function marketHTML(report) {
  const m = report.market;
  if (!m || !m.prices || !Object.keys(m.prices).length) return "";
  const parts = Object.entries(m.prices).map(([k, v]) => `${escapeHtml(k.replace(/_/g, " "))}: $${Number(v).toFixed(2)}`);
  return `<div class="market-line muted">Raw market value — ${parts.join(" · ")}
    <span class="mono">(${escapeHtml(m.matched_name)}, ${escapeHtml(m.matched_set)} #${escapeHtml(m.matched_number)}, ${escapeHtml(m.source)})</span>
    — ungraded price; graded value varies</div>`;
}

// NXR-style dual-agreement check, adapted: our two independent assessors are
// the pixel measurements and the AI flat-shot opinion. Strong disagreement
// on corners/edges doesn't block anything — it's surfaced so neither number
// gets blind trust.
// ---- card viewer ----
// Modelled on how TAG presents a card: both sides side by side under one
// shared transparency control, so you compare front and back at the same
// blend instead of fiddling with two sliders that drift apart.
function viewerSideHTML(sideKey, label, visionData, images) {
  const baseImg = images[`${sideKey}_aligned`];
  const reliefImg = images[`${sideKey}_card_vision`];
  if (!baseImg) return "";

  const photometric = visionData && visionData.method === "photometric_stereo";
  const methodLabel = photometric
    ? `photometric · ${visionData.light_count} lights`
    : "single-capture approx.";
  // When rotation scans were attached and the solve still didn't run, say so
  // on the chip. Otherwise a set that was uploaded and silently rejected is
  // indistinguishable from one that was never uploaded at all.
  const fallback = !photometric && visionData && visionData.fallback_reason;
  const frames = (visionData && visionData.registration) || [];
  const fallbackNote = fallback
    ? ` title="${escapeHtml(visionData.fallback_reason)}"`
    : "";

  return `<figure class="viewer-side">
    <div class="viewer-frame">
      <img class="defect-base" src="${baseImg}" alt="${label} of the card">
      ${reliefImg ? `<img class="defect-overlay" id="cardvision-${sideKey}-img" src="${reliefImg}" alt="${label} Card Vision relief" style="opacity:0.5">` : ""}
    </div>
    <figcaption class="viewer-caption">
      <span class="viewer-side-label">${label}</span>
      ${reliefImg ? `<span class="method-chip${photometric ? " method-chip--strong" : ""}${fallback ? " method-chip--fell-back" : ""}"${fallbackNote}>${escapeHtml(methodLabel)}${fallback ? " ?" : ""}</span>` : ""}
      ${frames.length ? registrationHTML(frames) : ""}
    </figcaption>
  </figure>`;
}

function cardViewerHTML(report, images, reportId) {
  const cv = report.card_vision || {};
  const front = viewerSideHTML("front", "Front", cv.front, images);
  const back = viewerSideHTML("back", "Back", cv.back, images);
  if (!front && !back) return "";

  const targets = ["front", "back"]
    .filter((side) => images[`${side}_card_vision`])
    .map((side) => `cardvision-${side}-img`)
    .join(",");

  const slider = targets
    ? `<div class="viewer-slider">
         <span class="slider-end">Card Vision</span>
         <input type="range" min="0" max="100" value="50" class="overlay-slider" id="cardvision-slider"
           data-target="${targets}" data-readout="cardvision-readout" data-invert="true"
           aria-label="Card Vision transparency — 0 shows surface relief, 100 shows colour">
         <span class="slider-end">Colour</span>
         <span class="mono slider-readout" id="cardvision-readout">50%</span>
       </div>`
    : "";

  const note = (cv.front && cv.front.note) || (cv.back && cv.back.note) || "";

  return `<section class="report-section viewer">
    <div class="viewer-grid">${front}${back}</div>
    ${slider}
    ${reportId ? `<div class="viewer-cert mono">REPORT #${escapeHtml(reportId.slice(0, 8).toUpperCase())}</div>` : ""}
    ${note ? `<p class="muted viewer-note">${escapeHtml(note)}</p>` : ""}
  </section>`;
}

// ---- attribute strip ----
// The five things a grader actually reports on, each with its front and back
// figure. This is the summary TAG puts directly under the card, and it
// answers "what is wrong with it" before any of the detail sections do.
function attributeStripHTML(report) {
  const dings = report.dings || [];
  const countDings = (attribute, side) =>
    dings.filter((d) => d.attribute === attribute && d.side === side).length;

  const centeringFor = (side) => {
    const sideData = (report.centering || {})[side];
    if (!sideData) return "—";
    const parts = ["horizontal", "vertical"]
      .map((axis) => sideData[axis])
      .filter((axis) => axis && axis.measurable !== false)
      .map((axis) => axis.ratio);
    return parts.length ? parts.join(" · ") : "n/a";
  };

  const dimensions = report.dimensions;
  const dimensionValue = dimensions && dimensions.measurable
    ? `H: ${dimensions.height_mm.toFixed(2)}mm<br>W: ${dimensions.width_mm.toFixed(2)}mm`
    : "requires<br>a scan";

  const columns = [
    ["corners", "Corners", ICONS.corners, `F: ${countDings("corners", "front")} DINGS<br>B: ${countDings("corners", "back")} DINGS`],
    ["edges", "Edges", ICONS.edges, `F: ${countDings("edges", "front")} DINGS<br>B: ${countDings("edges", "back")} DINGS`],
    ["centering", "Centering", ICONS.centering, `F: ${escapeHtml(centeringFor("front"))}<br>B: ${escapeHtml(centeringFor("back"))}`],
    ["surface", "Surface", ICONS.surface, `F: ${countDings("surface", "front")} DINGS<br>B: ${countDings("surface", "back")} DINGS`],
    ["dimensions", "Dimensions", ICONS.dimensions, dimensionValue],
  ];

  return `<section class="report-section">
    <div class="attr-strip">
      ${columns
        .map(
          ([key, label, icon, value]) => `<div class="attr-col" data-attr="${key}">
            <span class="attr-icon">${icon}</span>
            <span class="attr-label">${label}</span>
            <span class="mono attr-value">${value}</span>
          </div>`
        )
        .join("")}
    </div>
  </section>`;
}

// ---- score header ----
// The 1-10 grade is the number people compare against; the score is the
// same estimate without the rounding, so two cards that both land on 9 can
// still be told apart. Both come from this tool's own sub-grades and are
// not any grading company's scale.
function scoreHeaderHTML(ge) {
  return `<div class="grade-header">
    <div class="score-block">
      <div class="overall-score mono" style="color:${gradeColor(ge.overall_grade_rounded)}">${ge.score}</div>
      <div class="score-caption">score / 1000</div>
    </div>
    <div class="overall-detail">
      <div class="overall-grade-inline" style="color:${gradeColor(ge.overall_grade_rounded)}">
        grade ${ge.overall_grade_rounded} <span class="mono muted">(${ge.overall_grade.toFixed(2)})</span>
      </div>
      <div class="overall-note">${escapeHtml(ge.note)}</div>
    </div>
  </div>`;
}

// ---- per-side subgrades ----
// Front and back are separate surfaces with separate wear, and corners and
// edges fail in different ways, so the four attributes are broken out per
// side rather than shown as the three combined numbers that feed the
// overall estimate.
const SUBGRADE_ATTRIBUTES = [
  ["centering", "Centering"],
  ["corners", "Corners"],
  ["edges", "Edges"],
  ["surface", "Surface"],
];

function subgradeMatrixHTML(report) {
  const subgrades = report.subgrades;
  if (!subgrades) return "";
  const row = (sideKey, sideLabel) => {
    const side = subgrades[sideKey] || {};
    const tiles = SUBGRADE_ATTRIBUTES.map(([key, label]) =>
      subgradeTile(label, side[key], subgradeReason(report, sideKey, key)),
    ).join("");
    return `<div class="subgrade-side">
      <div class="subgrade-side-label">${sideLabel}</div>
      <div class="subgrade-row">${tiles}</div>
    </div>`;
  };
  return `<section class="report-section">
    <h2>Sub-grades</h2>
    ${row("front", "Front")}
    ${row("back", "Back")}
  </section>`;
}

// ---- dimensions ----
function dimensionsHTML(report) {
  const dim = report.dimensions;
  if (!dim) return "";
  if (!dim.measurable) {
    return `<section class="report-section">
      <h2>Dimensions</h2>
      <div class="muted">${escapeHtml(dim.note)}</div>
    </section>`;
  }
  // within_tolerance null means the scans disagreed by more than the
  // tolerance they were being judged against, so no verdict was given — that
  // is neither a pass nor a failure and must not be coloured as either.
  const bannerClass =
    dim.within_tolerance === null || dim.within_tolerance === undefined
      ? "banner-warn"
      : dim.within_tolerance
        ? "banner-ok"
        : "banner-error";
  const agreement =
    dim.spread_mm === null || dim.spread_mm === undefined
      ? ""
      : `<span class="muted mono" title="How far apart the ${dim.sample_count} scans of this card were. A single scan has nothing to check itself against.">${dim.sample_count} scans agree to ${dim.spread_mm.toFixed(2)} mm</span>`;
  // The deviations as proportions as well as millimetres. A trim takes
  // roughly equal millimetres off each axis; a capture measured at the wrong
  // scale is off by roughly equal percentages — and in millimetres alone
  // those look the same. Shown side by side rather than judged, because the
  // two cases genuinely overlap.
  const pct =
    dim.width_deviation_pct === null || dim.width_deviation_pct === undefined
      ? ""
      : `<span class="muted mono" title="A trim is off by similar millimetres on both axes; a capture at the wrong scale is off by similar percentages.">off by ${dim.width_deviation_mm.toFixed(2)} &times; ${dim.height_deviation_mm.toFixed(2)} mm (${dim.width_deviation_pct.toFixed(1)}% &times; ${dim.height_deviation_pct.toFixed(1)}%)</span>`;
  const scale =
    report.capture_dpi === null || report.capture_dpi === undefined
      ? ""
      : `<span class="muted mono" title="Every millimetre here is measured against this. A wrong value reads as a miscut card.">measured at ${Math.round(report.capture_dpi)} dpi</span>`;
  return `<section class="report-section">
    <h2>Dimensions</h2>
    <div class="dimension-row">
      <span class="mono dimension-value">${dim.width_mm.toFixed(2)} &times; ${dim.height_mm.toFixed(2)} mm</span>
      <span class="muted mono">nominal ${dim.nominal_width_mm} &times; ${dim.nominal_height_mm}</span>
      <span class="muted mono">out of square ${dim.squareness_deviation_deg.toFixed(2)}&deg;</span>
      ${agreement}
    </div>
    <div class="dimension-row">
      ${pct}
      ${scale}
    </div>
    <div class="banner ${bannerClass}">${escapeHtml(dim.note)}</div>
  </section>`;
}

// ---- DINGS ----
// Defect crops, captioned the way a grader reads them: which side, which
// region, what kind of wear. Each one opens full-screen on tap — the crop is
// the evidence, so it has to be inspectable, not decorative.
const ATTRIBUTE_LABELS = {
  centering: "Centering",
  corners: "Corners",
  edges: "Edges",
  surface: "Surface",
  dimensions: "Dimensions",
};

const DEFECT_TYPES = {
  corners: "Corner wear",
  edges: "Edge wear",
  surface: "Surface defect",
  centering: "Off-centre",
  dimensions: "Cut / size",
};

function dingCardHTML(ding, images) {
  const image = ding.image_key ? images[ding.image_key] : null;
  const grade = ding.grade === null ? "cap" : `g${ding.grade}`;
  const color = ding.grade === null ? "var(--bad)" : gradeColor(ding.grade);
  const region = ding.label.replace(/ (corner|edge|centering)$/i, "");

  return `<figure class="ding-card">
    <div class="ding-crop">
      ${image ? `<img src="${image}" alt="${escapeHtml(ding.side)} ${escapeHtml(ding.label)}">` : `<span class="ding-crop-empty muted">no crop</span>`}
      <span class="ding-grade mono" style="color:${color}">${grade}</span>
    </div>
    <figcaption class="ding-caption">
      <span class="ding-region"><strong>${escapeHtml(ding.side.toUpperCase())}</strong> / ${escapeHtml(region.toUpperCase())}</span>
      <span class="ding-type">${escapeHtml(DEFECT_TYPES[ding.attribute] || ding.attribute)}</span>
      <span class="muted ding-detail">${escapeHtml(ding.detail)}</span>
    </figcaption>
  </figure>`;
}

// Where the dings are, drawn on the card itself.
//
// A gallery of crops shows what each defect looks like and nothing about
// where it is, and "bottom-left corner" is a caption you have to translate
// back onto the card in your head. Every region ding already carries its own
// box in the canonical warp, so the card can just be marked up directly.
function dingMapHTML(report, images) {
  const dings = report.dings || [];
  const located = dings.filter((d) => d.box && d.box.length === 4);
  const sides = ["front", "back"].filter(
    (side) => images[`${side}_aligned`] && located.some((d) => d.side === side),
  );
  if (!sides.length) return "";

  const panel = (side) => {
    const marks = located
      .filter((d) => d.side === side)
      .map((d) => {
        const [x, y, w, h] = d.box;
        const style = `left:${(x * 100).toFixed(3)}%;top:${(y * 100).toFixed(3)}%;width:${(w * 100).toFixed(3)}%;height:${(h * 100).toFixed(3)}%`;
        const grade = d.grade === null || d.grade === undefined ? "?" : `g${d.grade}`;
        return `<span class="ding-mark" style="${style}"
          title="${escapeHtml(d.label)} — ${escapeHtml(d.detail)}"><b>${escapeHtml(grade)}</b></span>`;
      })
      .join("");
    return `<figure class="ding-map-side">
      <div class="ding-map-frame">
        <img src="${images[`${side}_aligned`]}" alt="${side} of the card, with the located defects marked">
        ${marks}
      </div>
      <figcaption class="viewer-side-label">${side}</figcaption>
    </figure>`;
  };

  return `<div class="ding-map">${sides.map(panel).join("")}</div>`;
}

function dingsHTML(report, images) {
  const dings = report.dings;
  if (!dings) return "";

  const heading = `<div class="section-bar">
    <span class="section-bar-icon">${ICONS.pin}</span>
    Defects identified of notable grade significance
  </div>`;

  if (!dings.length) {
    return `<section class="report-section">
      ${heading}
      <p class="muted empty-note">Nothing scored below 10 — no grade-driving defects found.</p>
    </section>`;
  }

  return `<section class="report-section">
    ${heading}
    <p class="muted empty-note">Snapshot of the defects with a notable impact on the grade.</p>
    ${dingMapHTML(report, images)}
    <div class="ding-gallery">${dings.map((ding) => dingCardHTML(ding, images)).join("")}</div>
  </section>`;
}

function buildReportHTML(report, images) {
  const ge = report.grade_estimate;
  const centering = report.centering;
  const ce = report.corners_edges;
  const surface = report.surface;

  let html = captureWarningHTML(report);
  html += capturePairWarningHTML(report);
  html += cardIdHTML(report);

  html += scoreHeaderHTML(ge);
  html += marketHTML(report);
  // Card first, then the five-attribute summary, then the defects behind it —
  // the reading order of a real grading report: what it is, what's wrong,
  // then the evidence.
  html += cardViewerHTML(report, images, window.__lastReport && window.__lastReport.reportId);
  html += attributeStripHTML(report);
  html += dingsHTML(report, images);
  html += subgradeMatrixHTML(report);

  const frontFullArt = !!(report.card_id && report.card_id.is_full_art);
  html += `<section class="report-section">
    <h2>Centering</h2>
    <div class="side-by-side">
      ${centeringSideHTML("Front", centering.front, images.front_centering_overlay, frontFullArt)}
      ${centeringSideHTML("Back", centering.back, images.back_centering_overlay)}
    </div>
    ${graderComparisonHTML(centering)}
  </section>`;

  html += `<section class="report-section">
    <h2>Corners &amp; Edges</h2>
    <div class="side-by-side">
      ${cornersEdgesSideHTML("Front", "front", ce.front, images)}
      ${cornersEdgesSideHTML("Back", "back", ce.back, images)}
    </div>
  </section>`;

  html += dimensionsHTML(report);

  if (surface) {
    html += `<section class="report-section">
      <h2>Surface <span class="method-chip method-chip--strong">measured</span></h2>
      ${surfaceSideHTML("front", "Front", surface.front)}
      ${surfaceSideHTML("back", "Back", surface.back)}
    </section>`;
  }

  return html;
}

function wireOverlaySliders() {
  document.querySelectorAll(".overlay-slider").forEach((slider) => {
    // `data-target` may name more than one overlay: the card viewer drives
    // front and back from a single control, the way TAG's report does, so
    // the two sides can't drift out of sync while you compare them.
    const targets = slider.dataset.target.split(",").map((id) => document.getElementById(id.trim())).filter(Boolean);
    if (!targets.length) return;
    const img = { style: { set opacity(v) { targets.forEach((t) => (t.style.opacity = v)); } } };
    const readout = slider.dataset.readout ? document.getElementById(slider.dataset.readout) : null;
    const invert = slider.dataset.invert === "true";
    const apply = () => {
      const pct = Number(slider.value);
      img.style.opacity = ((invert ? 100 - pct : pct) / 100).toFixed(2);
      if (readout) readout.textContent = `${pct}%`;
    };
    slider.addEventListener("input", apply);
    apply();
  });
}

// ---- actions: download / reset ----
document.getElementById("download-btn").addEventListener("click", downloadReport);
document.getElementById("reset-btn").addEventListener("click", resetApp);

// ---- fullscreen image viewer (TAG "Card Vision" idea, phone-sized) ----
// The report renders 1500x2100 images at thumbnail size; tapping one opens
// it fullscreen with pinch-zoom, one-finger pan, and double-tap 1x/3x.
// Pointer events cover both touch and mouse. In-app only — the downloaded
// HTML export keeps plain images.
let lightboxScale = 1;
let lightboxTx = 0;
let lightboxTy = 0;
const activePointers = new Map();
let pinchStartDist = 0;
let pinchStartScale = 1;
let lastTapTime = 0;

function applyLightboxTransform() {
  const img = document.getElementById("lightbox-img");
  img.style.transform = `translate(${lightboxTx}px, ${lightboxTy}px) scale(${lightboxScale})`;
}

function openLightbox(src) {
  const box = document.getElementById("lightbox");
  const img = document.getElementById("lightbox-img");
  lightboxScale = 1;
  lightboxTx = 0;
  lightboxTy = 0;
  img.src = src;
  applyLightboxTransform();
  box.hidden = false;
}

function closeLightbox() {
  document.getElementById("lightbox").hidden = true;
  document.getElementById("lightbox-img").src = "";
  activePointers.clear();
}

document.getElementById("report-content").addEventListener("click", (e) => {
  const img = e.target.closest("img");
  if (!img || !img.src) return;
  openLightbox(img.src);
});
document.getElementById("lightbox-close").addEventListener("click", closeLightbox);

const lightboxStage = document.getElementById("lightbox-stage");

lightboxStage.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (activePointers.size === 2) {
    const [a, b] = [...activePointers.values()];
    pinchStartDist = Math.hypot(a.x - b.x, a.y - b.y);
    pinchStartScale = lightboxScale;
  } else if (activePointers.size === 1) {
    const now = Date.now();
    if (now - lastTapTime < 300) {
      // double-tap: toggle 1x <-> 3x around the tap point
      if (lightboxScale > 1.5) {
        lightboxScale = 1;
        lightboxTx = 0;
        lightboxTy = 0;
      } else {
        lightboxScale = 3;
      }
      applyLightboxTransform();
    }
    lastTapTime = now;
  }
});

lightboxStage.addEventListener("pointermove", (e) => {
  if (!activePointers.has(e.pointerId)) return;
  const prev = activePointers.get(e.pointerId);
  activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (activePointers.size === 2) {
    const [a, b] = [...activePointers.values()];
    const dist = Math.hypot(a.x - b.x, a.y - b.y);
    if (pinchStartDist > 0) {
      lightboxScale = Math.max(1, Math.min(8, pinchStartScale * (dist / pinchStartDist)));
      applyLightboxTransform();
    }
  } else if (activePointers.size === 1 && lightboxScale > 1) {
    lightboxTx += e.clientX - prev.x;
    lightboxTy += e.clientY - prev.y;
    applyLightboxTransform();
  }
});

function lightboxPointerEnd(e) {
  activePointers.delete(e.pointerId);
  if (activePointers.size < 2) pinchStartDist = 0;
  if (lightboxScale <= 1.01) {
    lightboxScale = 1;
    lightboxTx = 0;
    lightboxTy = 0;
    applyLightboxTransform();
  }
}
lightboxStage.addEventListener("pointerup", lightboxPointerEnd);
lightboxStage.addEventListener("pointercancel", lightboxPointerEnd);

// tapping the dimmed backdrop (not the image) closes, when not zoomed
lightboxStage.addEventListener("click", (e) => {
  if (e.target === lightboxStage && lightboxScale <= 1.01) closeLightbox();
});

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

  // Standalone copy of wireOverlaySliders, in ES5 and without the fullscreen
  // viewer: the download has to work offline from a file:// URL, so it can't
  // reference this file. Kept behaviorally identical to the in-app version —
  // the Card Vision slider is the whole point of the report.
  const sliderScript = `document.querySelectorAll(".overlay-slider").forEach(function (slider) {
    var img = document.getElementById(slider.dataset.target);
    if (!img) return;
    var readout = slider.dataset.readout ? document.getElementById(slider.dataset.readout) : null;
    var invert = slider.dataset.invert === "true";
    function apply() {
      var pct = Number(slider.value);
      img.style.opacity = ((invert ? 100 - pct : pct) / 100).toFixed(2);
      if (readout) readout.textContent = pct + "%";
    }
    slider.addEventListener("input", apply);
    apply();
  });`;

  const doc = `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card Grade Report</title>
<style>${css}</style>
</head><body>
<div class="ambient" aria-hidden="true"><div class="ambient-blob"></div><div class="ambient-blob"></div><div class="ambient-blob"></div></div>
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
  document.querySelectorAll(".slot").forEach((slotEl) => {
    clearSlotPreview(slotEl);
    slotEl.querySelector("input[type=file]").value = "";
    slotEl.querySelector(".slot-error").hidden = true;
  });
  // The rotation scans belong to the card that was just graded, not to the
  // next one. They live in a collapsed section, so leaving them selected
  // meant an invisible stale set rode along with the next submission and
  // produced a byte-identical report six minutes later.
  //
  // Scan DPI and turn direction are deliberately kept: those describe the
  // scanner and the way it's operated, not this particular card.
  clearPhotometricInputs();
  document.getElementById("thumb-strip").innerHTML = "";
  document.getElementById("camera-error").hidden = true;
  currentStepIndex = 0;
  submitError.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Grade card";
  window.__lastReport = null;
  // Grading another card from a /r/<id> deep link would otherwise leave the
  // old report's URL in the address bar, so the next reload reopens the old
  // report instead of the capture flow.
  if (PERMALINK_PATTERN.test(location.pathname)) {
    history.replaceState(null, "", "/");
  }
  showView("capture");
  if (cameraModeActive) {
    showViewfinder(true);
    startCamera();
  }
  loadSavedReports();
}

// ---- theme ----
// Three states, not two. A plain light/dark switch would strand anyone who
// flips it once and then wants the page to follow their phone again at dusk;
// "auto" has to stay reachable, so the control cycles back to it.
const THEME_KEY = "cardgrading.theme";
const THEME_ORDER = ["auto", "light", "dark"];
const THEME_META = {
  auto: { icon: "auto", label: "Theme: auto (following your system)" },
  light: { icon: "sun", label: "Theme: light" },
  dark: { icon: "moon", label: "Theme: dark" },
};

function storedTheme() {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return THEME_ORDER.includes(value) ? value : "auto";
  } catch (e) {
    return "auto";   // private mode
  }
}

function resolvedTheme(choice) {
  if (choice !== "auto") return choice;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(choice) {
  const root = document.documentElement;
  if (choice === "auto") delete root.dataset.theme;
  else root.dataset.theme = choice;

  // The browser chrome is driven by paired media-scoped metas, which stop
  // matching once the page overrides the system. Point it at the resolved
  // theme directly instead.
  const resolved = resolvedTheme(choice);
  document.querySelectorAll('meta[name="theme-color"]').forEach((m) => m.remove());
  const meta = document.createElement("meta");
  meta.name = "theme-color";
  meta.content = resolved === "dark" ? "#0c1219" : "#eaecef";
  document.head.appendChild(meta);

  const button = document.getElementById("theme-toggle");
  if (button) {
    button.innerHTML = ICONS[THEME_META[choice].icon];
    button.title = THEME_META[choice].label;
    button.setAttribute("aria-label", THEME_META[choice].label);
  }
}

function cycleTheme() {
  const next = THEME_ORDER[(THEME_ORDER.indexOf(storedTheme()) + 1) % THEME_ORDER.length];
  try {
    if (next === "auto") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, next);
  } catch (e) {
    /* not remembering the choice is not a reason to refuse it */
  }
  applyTheme(next);
}

applyTheme(storedTheme());
document.getElementById("theme-toggle").addEventListener("click", cycleTheme);

// ---- saved reports ----
const PERMALINK_PATTERN = /^\/r\/([0-9a-f]{32})$/;

async function openSavedReport(reportId) {
  showView("processing");
  processingMessage.textContent = "Loading saved report…";
  let resp;
  try {
    resp = await fetch(`/api/report/${reportId}`);
  } catch (e) {
    processingMessage.textContent = "Couldn't reach the server.";
    return false;
  }
  if (!resp.ok) {
    processingMessage.textContent =
      resp.status === 404 ? "That report doesn't exist (or was deleted)." : `Couldn't load it (${resp.status}).`;
    return false;
  }
  const data = await resp.json();
  handleReport(data.report, data.images, data.report_id);
  return true;
}

function savedReportRowHTML(summary) {
  const when = summary.created_at ? summary.created_at.replace("T", " ").replace("+00:00", "") : "";
  const name = summary.card_name || "Unidentified card";
  const grade = summary.score ? `${summary.score}` : summary.grade !== null ? `g${summary.grade}` : "—";
  return `<li class="saved-row">
    <a class="saved-link" href="/r/${escapeHtml(summary.report_id)}">
      <span class="saved-name">${escapeHtml(name)}</span>
      <span class="mono saved-score">${escapeHtml(String(grade))}</span>
      <span class="muted mono saved-when">${escapeHtml(when)}</span>
      <span class="saved-chevron">${ICONS.chevron}</span>
    </a>
  </li>`;
}

async function loadSavedReports() {
  const container = document.getElementById("saved-reports");
  if (!container) return;
  let summaries = [];
  try {
    const resp = await fetch("/api/reports?limit=25");
    if (!resp.ok) return;
    summaries = (await resp.json()).reports || [];
  } catch (e) {
    return; // the capture flow works fine without the list
  }
  if (!summaries.length) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  container.innerHTML =
    `<div class="saved-heading">Saved reports</div><ul class="saved-list">${summaries.map(savedReportRowHTML).join("")}</ul>`;
}

// Deep-linked report (/r/<id>) skips the capture flow entirely. Anything
// else is a normal visit: show capture, and list what's already been graded.
const permalinkMatch = PERMALINK_PATTERN.exec(location.pathname);
if (permalinkMatch) {
  openSavedReport(permalinkMatch[1]);
} else {
  showView("capture");
  if (cameraModeActive) {
    startCamera();
  } else {
    switchToPickerMode();
  }
  loadSavedReports();
}
