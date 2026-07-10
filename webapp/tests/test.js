const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const REPO = path.join(__dirname, "..", "static");
const html = fs.readFileSync(path.join(REPO, "index.html"), "utf8");
const js = fs.readFileSync(path.join(REPO, "app.js"), "utf8");

function makeDom() {
  const dom = new JSDOM(html, { runScripts: "dangerously", url: "http://localhost/" });
  // createImageBitmap doesn't exist in jsdom - stub it so processImageFile's
  // try/catch falls back to the original file cleanly, matching real-browser
  // fallback behavior for unsupported environments.
  dom.window.createImageBitmap = undefined;
  dom.window.URL.createObjectURL = () => "blob:mock";
  dom.window.fetch = async () => ({ ok: true, text: async () => "" });
  // Inject as a real <script> element (not window.eval) — app.js has "use
  // strict", and strict-mode eval() keeps top-level declarations scoped to
  // the eval call instead of leaking to window, so window.handleReport etc.
  // would never be visible. A real script tag executes at global scope.
  const scriptEl = dom.window.document.createElement("script");
  scriptEl.textContent = js;
  dom.window.document.body.appendChild(scriptEl);
  return dom;
}

let failures = 0;
function assert(cond, message) {
  if (!cond) {
    failures++;
    console.error("FAIL:", message);
  } else {
    console.log("ok:", message);
  }
}

// --- Test 1: success report with surface renders without throwing, and key data appears ---
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  dom.window.handleReport(data.report, data.images);

  const doc = dom.window.document;
  assert(doc.getElementById("report-view").hidden === false, "report view shown on success");
  assert(doc.getElementById("capture-view").hidden === true, "capture view hidden on success");

  const content = doc.getElementById("report-content").innerHTML;
  assert(content.includes("grade-header"), "grade header rendered");
  assert(content.includes(String(data.report.grade_estimate.overall_grade_rounded)), "overall grade value present");
  assert(content.includes("Corners &amp; Edges") || content.includes("Corners & Edges"), "corners/edges section present");
  assert(content.includes("Surface"), "surface section present");
  assert(content.includes("defect-slider"), "defect slider input present");
  assert(
    content.includes("no vision judgment available"),
    "no-credentials vision judgment fallback text shown (matches this test env)"
  );

  // sliders should be wired: check img opacity responds to a manual 'input' event
  const slider = doc.querySelector(".defect-slider");
  assert(slider !== null, "found a defect slider element");
  if (slider) {
    const img = doc.getElementById(slider.id + "-img");
    slider.value = "10";
    slider.dispatchEvent(new dom.window.Event("input"));
    assert(img.style.opacity === "0.10" || img.style.opacity === "0.1", `slider updates image opacity (got ${img.style.opacity})`);
  }
}

// --- Test 1b: soft capture-quality gate failures (resolution, uneven lighting, etc.)
// no longer force a retake — grading proceeds, and the report shows a red
// warning banner instead. Only "couldn't find the card" (centering === null)
// still blocks. Real user report: "Resolution too low — shortest side=1080px;
// Uneven lighting — border-ring brightness gradient=163.1" on both sides —
// grading should still go through, with a banner naming just the gate labels
// (no numeric detail suffix).
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report)); // deep clone, don't mutate the shared fixture
  report.capture_quality.front.ok = false;
  report.capture_quality.front.gates = [
    { name: "resolution", passed: false, detail: "shortest side=1080px", value: 1080 },
    { name: "card_detection", passed: true, detail: "card contour found", value: null },
    { name: "uneven_lighting", passed: false, detail: "border-ring brightness gradient=163.1", value: 163.1 },
  ];
  dom.window.handleReport(report, data.images);

  const doc = dom.window.document;
  assert(doc.getElementById("report-view").hidden === false, "report still renders despite a soft gate failure");
  assert(doc.getElementById("capture-view").hidden === true, "does not bounce back to capture view for a soft gate failure");

  const content = doc.getElementById("report-content").innerHTML;
  assert(content.includes("The grading might be worse because of:"), "soft-gate warning banner is present");
  assert(content.includes("Front: Resolution too low, Uneven lighting"), `banner lists front's failed gates by label (got content snippet: ${content.slice(content.indexOf("might be worse") - 20, content.indexOf("might be worse") + 200)})`);
  assert(!content.includes("shortest side=1080px"), "banner omits the numeric gate detail for resolution");
  assert(!content.includes("border-ring brightness gradient"), "banner omits the numeric gate detail for uneven lighting");
  assert(!content.includes("Back:"), "back passed all gates, so it's not mentioned in the warning banner");
}

// --- Test 1c: no warning banner at all when every capture-quality gate passes ---
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  dom.window.handleReport(data.report, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;
  assert(!content.includes("might be worse"), "no warning banner when all capture-quality gates pass");
}

// --- Test 2: capture-quality failure surfaces a per-slot error and stays on capture view ---
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "failure_result.json"), "utf8"));
  dom.window.handleReport(data.report, data.images);

  const doc = dom.window.document;
  assert(doc.getElementById("capture-view").hidden === false, "capture view shown again on gate failure");
  assert(doc.getElementById("report-view").hidden === true, "report view stays hidden on gate failure");

  const frontError = doc.querySelector('.slot[data-slot="front"] .slot-error');
  const backError = doc.querySelector('.slot[data-slot="back"] .slot-error');
  assert(frontError.hidden === false, "front slot shows an error (it failed resolution+card_detection)");
  assert(frontError.textContent.includes("Resolution too low"), `front error mentions resolution (got: ${frontError.textContent})`);
  assert(backError.hidden === true, "back slot has no error (it passed all gates)");
}

// --- Test 2b: regression — BOTH front and back failing simultaneously must not silently drop one ---
// (This is what the user actually hit: two non-card photos both failed
// capture_quality, but the UI only ever surfaced "back".)
async function runTest2b() {
  const bothFailReport = {
    centering: null,
    capture_quality: {
      front: { ok: false, gates: [{ name: "card_detection", passed: false, detail: "no card contour found" }] },
      back: { ok: false, gates: [{ name: "resolution", passed: false, detail: "shortest side=200px" }] },
    },
  };

  // Picker mode: each slot has its own error element, so this should already
  // work correctly without any special handling.
  {
    const dom = makeDom();
    dom.window.handleReport(bothFailReport, {});
    const doc = dom.window.document;
    const frontError = doc.querySelector('.slot[data-slot="front"] .slot-error');
    const backError = doc.querySelector('.slot[data-slot="back"] .slot-error');
    assert(frontError.hidden === false, "picker mode: front error shown when both fail");
    assert(backError.hidden === false, "picker mode: back error shown when both fail");
    assert(frontError.textContent.includes("card"), "picker mode: front error mentions its own reason");
    assert(backError.textContent.includes("Resolution"), "picker mode: back error mentions its own reason");
  }

  // Camera mode: both failures must be reported together, not just the last
  // one processed, and the sequence must rewind to the earliest failing
  // step (front) rather than stranding the user on the last one (back).
  // This is the realistic path: the camera stops once all shots are taken
  // (see the shutter handler), so by the time a rejected report comes back,
  // cameraStream is null and reportSlotFailures must reacquire it — mocked
  // here as succeeding, same as a real working camera would.
  {
    const dom = makeDom();
    dom.window.navigator.mediaDevices = { getUserMedia: async () => ({ getTracks: () => [] }) };
    dom.window.eval('state.files.front = new File(["x"], "f.jpg"); state.files.back = new File(["x"], "b.jpg");');
    dom.window.eval("cameraModeActive = true; currentStepIndex = 2; cameraStream = null;");
    await dom.window.handleReport(bothFailReport, {});
    // reportSlotFailures's own startCamera() call isn't awaited internally
    // (fire-and-forget, matching real UI code) — give its microtasks a turn.
    await new Promise((r) => setTimeout(r, 0));

    const doc = dom.window.document;
    assert(dom.window.eval("cameraModeActive") === true, "stays in camera mode when the camera can actually be reacquired");
    assert(dom.window.eval("currentStepIndex") === 0, "rewinds to the earliest failing step (front), not the last one (back)");
    const cameraErrorText = doc.getElementById("camera-error").textContent;
    assert(cameraErrorText.includes("Front") && cameraErrorText.includes("card"), `camera error mentions front's failure (got: ${cameraErrorText})`);
    assert(cameraErrorText.includes("Back") && cameraErrorText.includes("Resolution"), `camera error also mentions back's failure (got: ${cameraErrorText})`);
    // Regression: startCamera()'s success path used to unconditionally hide
    // #camera-error as a side effect (it was written for its OWN "couldn't
    // access the camera" message, sharing the element with this gate-failure
    // message). Since reportSlotFailures reacquires the camera right after
    // setting this text, and permission is already granted so that resolves
    // almost instantly, the message was visible for a fraction of a second
    // and then silently hidden again — "flashes then disappears."
    assert(
      doc.getElementById("camera-error").hidden === false,
      "the gate-failure message stays VISIBLE after the camera successfully reacquires, not just present in the DOM"
    );
    assert(dom.window.eval("state.files.front") === undefined, "front's stale file is cleared");
    assert(dom.window.eval("state.files.back") === undefined, "back's stale file is cleared");
  }
}

// --- Test 2c: hard/soft gate split in the retake path ---
// With the server now marking gates hard (geometry: card_detection, tilt,
// aspect_ratio) vs soft (resolution, glare, uneven_lighting), a blocked
// grade should only force retaking the side that hard-failed. A side whose
// failures are all soft is a usable capture; and the retake message should
// name only the hard reasons, not the soft noise. (Validated against real
// sample photos: a misdetected card quad hard-fails tilt/aspect_ratio while
// low thumbnail resolution soft-fails alongside it.)
{
  const mixedReport = {
    centering: null,
    capture_quality: {
      front: {
        ok: false,
        gates: [
          { name: "resolution", passed: false, detail: "shortest side=270px", hard: false },
          { name: "card_detection", passed: true, detail: "card contour found", hard: true },
          { name: "tilt", passed: false, detail: "max corner angle deviation=33.92 deg", hard: true },
          { name: "aspect_ratio", passed: false, detail: "deviation=5.08%", hard: true },
        ],
      },
      back: {
        ok: false,
        gates: [
          { name: "resolution", passed: false, detail: "shortest side=270px", hard: false },
          { name: "card_detection", passed: true, detail: "card contour found", hard: true },
          { name: "tilt", passed: true, detail: "max corner angle deviation=0.46 deg", hard: true },
        ],
      },
    },
  };
  const dom = makeDom();
  dom.window.eval('state.files.front = new File(["x"], "f.jpg"); state.files.back = new File(["x"], "b.jpg");');
  dom.window.eval("cameraModeActive = false;");
  dom.window.handleReport(mixedReport, {});
  const doc = dom.window.document;
  const frontError = doc.querySelector('.slot[data-slot="front"] .slot-error');
  const backError = doc.querySelector('.slot[data-slot="back"] .slot-error');
  assert(frontError.hidden === false, "hard-failing front side shows a retake error");
  assert(frontError.textContent.includes("tilted"), "front retake reason names the hard gate (tilt)");
  assert(!frontError.textContent.includes("Resolution"), "front retake reason omits the soft resolution failure");
  assert(backError.hidden === true, "soft-only back side is NOT asked to retake");
  assert(dom.window.eval("state.files.back") !== undefined, "soft-only back side keeps its captured file");
  assert(dom.window.eval("state.files.front") === undefined, "hard-failing front side has its file cleared");
}

// --- Test 3: gradeColor / gateLabel behave sensibly ---
{
  const dom = makeDom();
  const w = dom.window;
  assert(w.gradeColor(9) === "var(--good)", "grade 9 is good color");
  assert(w.gradeColor(6) === "var(--warn)", "grade 6 is warn color");
  assert(w.gradeColor(3) === "var(--bad)", "grade 3 is bad color");
  assert(w.gradeColor(null) === "var(--muted)", "null grade is muted color");
  assert(w.gateLabel("uneven_lighting") === "Uneven lighting", "gateLabel maps known gate name");
  assert(w.gateLabel("mystery_gate") === "mystery_gate", "gateLabel falls back to raw name for unknown gates");
}

// --- Test 4: submit-enable logic reacts to file state ---
// `state` and `surfaceEnabled` are top-level const/let in app.js, so (unlike
// its function declarations) they aren't exposed as window.* properties --
// that's normal JS scoping, not a bug. eval() against the same document
// shares app.js's global lexical scope, so it can reach them directly.
{
  const dom = makeDom();
  const doc = dom.window.document;
  const submitBtn = doc.getElementById("submit-btn");
  assert(submitBtn.disabled === true, "submit disabled with no files chosen");

  dom.window.eval(`
    state.files.front = new File(["x"], "front.jpg", { type: "image/jpeg" });
    state.files.back = new File(["x"], "back.jpg", { type: "image/jpeg" });
    updateSubmitEnabled();
  `);
  assert(submitBtn.disabled === false, "submit enabled once front+back are set");

  dom.window.eval(`surfaceEnabled = true; updateSubmitEnabled();`);
  assert(submitBtn.disabled === true, "submit disabled again once surface toggle is on but angled files missing");

  dom.window.eval(`
    state.files.front_angled = new File(["x"], "fa.jpg", { type: "image/jpeg" });
    state.files.back_angled = new File(["x"], "ba.jpg", { type: "image/jpeg" });
    updateSubmitEnabled();
  `);
  assert(submitBtn.disabled === false, "submit enabled again once all 4 files are set");
}

// --- Test 6: no getUserMedia (insecure context / jsdom) auto-falls back to picker mode ---
{
  const dom = makeDom(); // jsdom has no navigator.mediaDevices.getUserMedia
  const doc = dom.window.document;
  assert(dom.window.eval("cameraModeActive") === false, "auto-switched away from camera mode when getUserMedia is unavailable");
  assert(doc.getElementById("camera-mode").hidden === true, "camera-mode view hidden after fallback");
  assert(doc.getElementById("picker-mode").hidden === false, "picker-mode view shown after fallback");
  assert(doc.getElementById("camera-unavailable-banner").hidden === false, "explains why camera mode isn't available");
  assert(
    doc.getElementById("camera-unavailable-banner").textContent.includes("HTTPS"),
    "unavailable message mentions the secure-context requirement"
  );
}

// --- Test 7: activeSteps() reflects the surface toggle ---
{
  const dom = makeDom();
  const w = dom.window;
  assert(JSON.stringify(w.activeSteps()) === JSON.stringify(["front", "back"]), "2 steps when surface analysis is off");
  w.eval("surfaceEnabled = true;");
  assert(
    JSON.stringify(w.activeSteps()) === JSON.stringify(["front", "back", "front_angled", "back_angled"]),
    "4 steps when surface analysis is on"
  );
}

// --- Test 8: step sequencing, thumbnails, and retake all stay in sync ---
{
  const dom = makeDom();
  const w = dom.window;
  const doc = w.document;

  // Simulate two captured shots directly (captureShot() itself needs a real
  // video frame, which jsdom can't provide) and confirm the bookkeeping
  // functions around it behave correctly.
  w.eval(`
    state.files.front = new File(["x"], "front.jpg", { type: "image/jpeg" });
    addThumb("front", state.files.front);
    currentStepIndex = 1;
    updateStepIndicator();
  `);
  assert(doc.getElementById("step-indicator").textContent.includes("Shot 2 of 2"), "indicator advances to step 2");
  assert(doc.querySelector('.capture-thumb[data-step="front"]') !== null, "front thumbnail added after capture");

  w.eval(`
    state.files.back = new File(["x"], "back.jpg", { type: "image/jpeg" });
    addThumb("back", state.files.back);
    currentStepIndex = 2;
    updateStepIndicator();
  `);
  assert(doc.getElementById("step-indicator").textContent.includes("All shots captured"), "indicator shows completion after last step");

  // Tapping the front thumbnail should reopen exactly that step
  doc.querySelector('.capture-thumb[data-step="front"]').dispatchEvent(new w.Event("click"));
  assert(w.eval("currentStepIndex") === 0, "retake jumps back to the tapped step's index");
  assert(w.eval("state.files.front") === undefined, "retake clears the old file for that step");
  assert(doc.querySelector('.capture-thumb[data-step="front"]') === null, "retake removes the stale thumbnail");
  assert(w.eval("state.files.back") !== undefined, "retake leaves the other completed step's file alone");
}

// --- Test 9: guide-to-video coordinate mapping (computeGuideSampleRect) ---
{
  const dom = makeDom();
  const w = dom.window;
  const doc = w.document;
  const video = doc.getElementById("camera-video");
  const wrap = doc.querySelector(".viewfinder-wrap");
  const guide = doc.getElementById("card-guide");

  // Fake a 1080x1440 (3:4) video filling a 300x400 CSS box exactly (scale=1,
  // no cropping) with a centered 216x304 guide box — nice round numbers to
  // hand-check the mapping.
  Object.defineProperty(video, "videoWidth", { value: 1080, configurable: true });
  Object.defineProperty(video, "videoHeight", { value: 1440, configurable: true });
  wrap.getBoundingClientRect = () => ({ width: 300, height: 400, left: 0, top: 0 });
  guide.getBoundingClientRect = () => ({ width: 216, height: 304, left: 42, top: 48 });

  const rect = w.computeGuideSampleRect(300, 400);
  // scale = max(300/1080, 400/1440) = max(0.2778, 0.2778) = 0.2778 (no crop)
  // native guide x = 42 / 0.2778 ≈ 151.2, y = 48 / 0.2778 ≈ 172.8
  // sample coords at 300x400 sample == same as CSS since scale cancels out to 1:1 here
  assert(Math.abs(rect.x - 42) <= 1, `guide x maps back to ~42 (got ${rect.x})`);
  assert(Math.abs(rect.y - 48) <= 1, `guide y maps back to ~48 (got ${rect.y})`);
  assert(Math.abs(rect.w - 216) <= 1, `guide w maps back to ~216 (got ${rect.w})`);
  assert(Math.abs(rect.h - 304) <= 1, `guide h maps back to ~304 (got ${rect.h})`);
}

// --- Test 11: computePrecheckMessage — "no card detected" + priority ordering ---
{
  const dom = makeDom();
  const compute = dom.window.computePrecheckMessage;

  function flatPixels(gray, count) {
    const arr = new Uint8ClampedArray(count * 4);
    for (let i = 0; i < count; i++) {
      arr[i * 4] = gray;
      arr[i * 4 + 1] = gray;
      arr[i * 4 + 2] = gray;
      arr[i * 4 + 3] = 255;
    }
    return arr;
  }

  // Colorful on purpose, not grayscale: a real card has color (border,
  // artwork, the navy back), so these need actual saturation to correctly
  // represent "card-like" for the low-saturation check added alongside
  // these tests — a plain R=G=B pattern would have zero saturation and
  // incorrectly trip that check itself.
  function texturedPixels(count, low, high) {
    const arr = new Uint8ClampedArray(count * 4);
    for (let i = 0; i < count; i++) {
      const v = i % 2 === 0 ? low : high;
      arr[i * 4] = v;
      arr[i * 4 + 1] = Math.round(v * 0.6);
      arr[i * 4 + 2] = Math.round(v * 0.3);
      arr[i * 4 + 3] = 255;
    }
    return arr;
  }

  function glarePixels(count, baseGray, glareFrac) {
    const arr = new Uint8ClampedArray(count * 4);
    const glareCount = Math.round(count * glareFrac);
    for (let i = 0; i < count; i++) {
      if (i < glareCount) {
        // genuine glare washes out to near-white regardless of the
        // underlying color — physically accurate, and still leaves the
        // majority of the (colorful) region driving the mean saturation
        arr[i * 4] = 255;
        arr[i * 4 + 1] = 255;
        arr[i * 4 + 2] = 255;
      } else {
        arr[i * 4] = baseGray;
        arr[i * 4 + 1] = Math.round(baseGray * 0.6);
        arr[i * 4 + 2] = Math.round(baseGray * 0.3);
      }
      arr[i * 4 + 3] = 255;
    }
    return arr;
  }

  const N = 400;
  const GOOD_HEIGHT = 900; // well above the 700px "move closer" cutoff

  // Flat/uniform region at a normal brightness — this is the actual
  // regression case: no card present, but not dark enough to already be
  // caught by the darkness check.
  assert(
    compute(flatPixels(130, N), false, GOOD_HEIGHT) === "No card detected — center it in the frame",
    "flat/uniform region with no card-like detail triggers the new warning"
  );

  // Darkness must still win over "no card" when a flat region is ALSO dark
  // — can't tell if a card's there or not if you can't see anything.
  assert(
    compute(flatPixels(20, N), false, GOOD_HEIGHT) === "Too dark — add more light",
    "darkness check takes priority over the no-card check on a dark+flat region"
  );

  // A textured region (real print/border/artwork produces exactly this
  // kind of local contrast) at a normal brightness with no glare must not
  // false-positive on any check.
  assert(
    compute(texturedPixels(N, 80, 180), false, GOOD_HEIGHT) === null,
    "a textured region with a card-like level of contrast produces no warning"
  );

  // High contrast + a genuine glare cluster should still report glare, not
  // get preempted by the (satisfied) no-card check.
  assert(
    compute(glarePixels(N, 150, 0.15), false, GOOD_HEIGHT) === "Glare detected — adjust the light angle",
    "glare is still reported on a textured (card-like) region that also has a bright cluster"
  );

  // Textured, well-lit, but the guide maps to a small native region — move
  // closer should be the only thing left to say.
  assert(
    compute(texturedPixels(N, 80, 180), false, 400) === "Move closer to the card",
    "distance check still fires when nothing else is wrong but the guide region is small"
  );

  // Angled steps skip darkness/glare (raking light triggers them by
  // design) but the no-card check isn't lighting-technique-specific, so it
  // must still fire on a flat/empty region even when isAngled is true.
  assert(
    compute(flatPixels(20, N), true, GOOD_HEIGHT) === "No card detected — center it in the frame",
    "no-card check still applies on angled steps even though darkness/glare don't"
  );

  // And confirm glare truly is suppressed on angled steps (existing
  // behavior, now routed through the refactored pure function).
  assert(
    compute(glarePixels(N, 150, 0.15), true, GOOD_HEIGHT) === null,
    "glare check is suppressed on angled steps, same as before the refactor"
  );

  // Regression: a real mis-shot the user actually hit — pointing the camera
  // at a printed checklist on a gray desk. It has plenty of contrast (black
  // text, table lines: measured variance ~586 on the real photo, so the
  // no-card variance check correctly stays quiet), but almost no color
  // (measured mean saturation ~0.01) — nothing the variance check alone
  // could ever catch. Reproduced here with a grayscale (R=G=B) checkerboard
  // at the same variance level the real photo measured.
  function grayscaleTexturedPixels(count, low, high) {
    const arr = new Uint8ClampedArray(count * 4);
    for (let i = 0; i < count; i++) {
      const v = i % 2 === 0 ? low : high;
      arr[i * 4] = v;
      arr[i * 4 + 1] = v;
      arr[i * 4 + 2] = v;
      arr[i * 4 + 3] = 255;
    }
    return arr;
  }
  assert(
    compute(grayscaleTexturedPixels(N, 170, 250), false, GOOD_HEIGHT) === "Doesn't look like a card — too little color",
    "a high-contrast but colorless region (printed document, the user's actual test case) is now caught"
  );

  // Confirm the colorful synthetic "card" fixtures used throughout this test
  // (measured mean saturation ~0.38, vs. the real document's ~0.01) don't
  // false-positive on the new check — already implied by the "no warning"
  // and "glare" assertions above passing, but worth asserting directly.
  const texturedPx = texturedPixels(N, 80, 180);
  let satSum = 0;
  for (let i = 0; i < texturedPx.length; i += 4) {
    const mx = Math.max(texturedPx[i], texturedPx[i + 1], texturedPx[i + 2]);
    const mn = Math.min(texturedPx[i], texturedPx[i + 1], texturedPx[i + 2]);
    satSum += mx > 0 ? (mx - mn) / mx : 0;
  }
  assert(
    satSum / (texturedPx.length / 4) > dom.window.eval("LOW_SATURATION_THRESHOLD"),
    "sanity check: the colorful test fixture actually has saturation above the threshold"
  );

  // Real-world regression: a genuine submitted card photo (a muted
  // brown/gray Fighting-type Pokemon card) measured mean saturation ~0.1244
  // — only just above the original 0.12 cutoff, and *below* a real non-card
  // scene (a metal staircase) measured the same way at ~0.13. That overlap
  // meant the check could stably false-positive on a real card, not just
  // flicker on a non-card scene. Threshold was lowered to 0.07 to give real
  // cards headroom; assert directly against the measured real-card value so
  // this doesn't regress silently if the threshold ever creeps back up.
  function flatSaturationPixels(count, saturation) {
    const arr = new Uint8ClampedArray(count * 4);
    const max = 200;
    const min = Math.round(max * (1 - saturation));
    for (let i = 0; i < count; i++) {
      arr[i * 4] = max;
      arr[i * 4 + 1] = min;
      arr[i * 4 + 2] = min;
      arr[i * 4 + 3] = 255;
    }
    return arr;
  }
  assert(
    compute(flatSaturationPixels(N, 0.1244), false, GOOD_HEIGHT) !== "Doesn't look like a card — too little color",
    "a real card's measured saturation (~0.1244, the submitted Pangoro card photo) does not false-trigger"
  );
}

// --- Test 12: the precheck never touches the shutter button — it only ever informs, never blocks ---
{
  const dom = makeDom();
  const doc = dom.window.document;
  const shutterBtn = doc.getElementById("shutter-btn");
  const before = shutterBtn.disabled;

  // Force a "no card detected" condition through the real runPrecheck path
  // (not just the pure function) and confirm the shutter is untouched.
  const video = doc.getElementById("camera-video");
  const wrap = doc.querySelector(".viewfinder-wrap");
  const guide = doc.getElementById("card-guide");
  Object.defineProperty(video, "videoWidth", { value: 800, configurable: true });
  Object.defineProperty(video, "videoHeight", { value: 800, configurable: true });
  wrap.getBoundingClientRect = () => ({ width: 300, height: 300, left: 0, top: 0 });
  guide.getBoundingClientRect = () => ({ width: 200, height: 200, left: 50, top: 50 });
  dom.window.HTMLCanvasElement.prototype.getContext = () => ({
    drawImage: () => {},
    getImageData: () => ({ data: new Uint8ClampedArray(200 * 200 * 4).fill(130) }), // flat gray -> no card
  });
  dom.window.eval("currentStepIndex = 0;");
  // Debounce (PRECHECK_CONFIRM_TICKS) requires several consistent readings
  // before the displayed state changes — call it enough times to settle.
  dom.window.runPrecheck();
  dom.window.runPrecheck();
  dom.window.runPrecheck();

  assert(shutterBtn.disabled === before, "shutter's disabled state is unchanged by any precheck outcome");
  assert(doc.getElementById("precheck-banner").hidden === false, "precheck banner is shown for the flat/no-card sample");
  assert(
    doc.getElementById("precheck-banner").textContent.includes("No card detected"),
    "runPrecheck end-to-end (not just the pure function) surfaces the no-card message"
  );

  // The guide-box border marker: a bottom text banner is easy to miss while
  // framing a shot, so a warning also changes the guide's own border —
  // right on the thing the user is actually looking at. "No card detected"
  // gets the red/error treatment specifically (nothing to grade at all),
  // distinct from the yellow used for the other coarse quality warnings.
  assert(
    guide.classList.contains("card-guide--error"),
    "the guide box gets the red error marker for the no-card sample specifically"
  );
  assert(
    !guide.classList.contains("card-guide--warning"),
    "the no-card sample does not also get the yellow warning class"
  );
  assert(
    doc.getElementById("precheck-banner").classList.contains("precheck-banner--error"),
    "the banner itself also gets the red error treatment for the no-card sample"
  );

  // And it must clear again once the frame is actually fine, not just stay
  // stuck on from the last bad tick. Also widen the guide geometry here —
  // the tight 200x200-in-an-800x800-video setup above maps to a native
  // guide height under the 700px "move closer" cutoff, which would trigger
  // that check instead of clearing, independent of the pixel data.
  wrap.getBoundingClientRect = () => ({ width: 300, height: 300, left: 0, top: 0 });
  guide.getBoundingClientRect = () => ({ width: 280, height: 280, left: 10, top: 10 });
  Object.defineProperty(video, "videoWidth", { value: 2000, configurable: true });
  Object.defineProperty(video, "videoHeight", { value: 2000, configurable: true });
  dom.window.HTMLCanvasElement.prototype.getContext = () => ({
    drawImage: () => {},
    getImageData: () => {
      const arr = new Uint8ClampedArray(200 * 200 * 4);
      for (let i = 0; i < arr.length; i += 4) {
        const v = i % 8 === 0 ? 80 : 180; // colorful, textured, well-lit "card"
        arr[i] = v;
        arr[i + 1] = Math.round(v * 0.6);
        arr[i + 2] = Math.round(v * 0.3);
        arr[i + 3] = 255;
      }
      return { data: arr };
    },
  });
  dom.window.runPrecheck();
  dom.window.runPrecheck();
  dom.window.runPrecheck();
  assert(doc.getElementById("precheck-banner").hidden === true, "banner clears once the frame looks card-like");
  assert(!guide.classList.contains("card-guide--warning"), "guide marker clears once the frame looks card-like");
  assert(!guide.classList.contains("card-guide--error"), "error class also clears once the frame looks card-like");
  assert(guide.classList.contains("card-guide--good"), "guide box turns green once a frame is confirmed card-like");
}

// --- Test 14: precheck debounce suppresses flicker on a borderline scene ---
// Real-world root cause: a photo of a metal staircase (not a card) measured
// saturation ~0.128-0.131, right on top of the 0.12 "too little color"
// threshold — ordinary frame-to-frame exposure/compression noise flipped the
// raw per-tick reading above and below the cutoff, so the warning banner and
// guide marker flickered on and off on an unchanging real-world scene. Fix:
// the displayed state only changes after PRECHECK_CONFIRM_TICKS consecutive
// matching raw readings.
{
  const dom = makeDom();
  const doc = dom.window.document;
  const guide = doc.getElementById("card-guide");
  const banner = doc.getElementById("precheck-banner");
  const video = doc.getElementById("camera-video");
  const wrap = doc.querySelector(".viewfinder-wrap");

  Object.defineProperty(video, "videoWidth", { value: 800, configurable: true });
  Object.defineProperty(video, "videoHeight", { value: 800, configurable: true });
  wrap.getBoundingClientRect = () => ({ width: 300, height: 300, left: 0, top: 0 });
  guide.getBoundingClientRect = () => ({ width: 280, height: 280, left: 10, top: 10 });
  dom.window.eval("currentStepIndex = 0;");

  const flatGray = { data: new Uint8ClampedArray(280 * 280 * 4).fill(130) }; // no-card reading
  const cardLike = (() => {
    const arr = new Uint8ClampedArray(280 * 280 * 4);
    for (let i = 0; i < arr.length; i += 4) {
      const v = i % 8 === 0 ? 80 : 180;
      arr[i] = v;
      arr[i + 1] = Math.round(v * 0.6);
      arr[i + 2] = Math.round(v * 0.3);
      arr[i + 3] = 255;
    }
    return { data: arr };
  })();

  function mockFrame(imgData) {
    dom.window.HTMLCanvasElement.prototype.getContext = () => ({
      drawImage: () => {},
      getImageData: () => imgData,
    });
  }

  // A single stray "bad" reading among otherwise-good ones must not flip the
  // displayed state — this is the exact flicker the user reported.
  mockFrame(cardLike);
  dom.window.runPrecheck();
  dom.window.runPrecheck();
  assert(banner.hidden === true, "two consistent good readings: no warning shown yet (sanity baseline)");
  assert(guide.classList.contains("card-guide--good"), "two consistent good readings: guide is green");

  mockFrame(flatGray);
  dom.window.runPrecheck(); // one stray bad tick
  assert(banner.hidden === true, "a single stray bad reading does not flip the displayed state");
  assert(!guide.classList.contains("card-guide--error"), "a single stray bad reading does not add the guide marker");

  mockFrame(cardLike);
  dom.window.runPrecheck(); // back to good before debounce threshold was reached
  assert(banner.hidden === true, "recovering before the confirm threshold keeps the display stable (no flicker)");

  // Now a sustained run of bad readings — this should eventually show.
  mockFrame(flatGray);
  dom.window.runPrecheck();
  dom.window.runPrecheck();
  assert(banner.hidden === true, "still below the confirm threshold after 2 consecutive bad ticks");
  dom.window.runPrecheck();
  assert(banner.hidden === false, "warning shows once bad readings are sustained for the confirm threshold");
  assert(guide.classList.contains("card-guide--error"), "no-card guide marker (red) shows once bad readings are sustained");
  assert(banner.classList.contains("precheck-banner--error"), "no-card banner (red) shows once bad readings are sustained");

  // And a single stray good tick shouldn't instantly clear a confirmed warning.
  mockFrame(cardLike);
  dom.window.runPrecheck();
  assert(banner.hidden === false, "a single stray good reading does not instantly clear a confirmed warning");
  dom.window.runPrecheck();
  dom.window.runPrecheck();
  assert(banner.hidden === true, "warning clears once good readings are sustained for the confirm threshold");
}

// --- Test 13: reopening the viewfinder (retake) resets any stale warning marker/banner ---
{
  const dom = makeDom();
  const doc = dom.window.document;
  const guide = doc.getElementById("card-guide");
  const banner = doc.getElementById("precheck-banner");

  // Simulate a warning left over from whatever was last framed
  guide.classList.add("card-guide--warning");
  banner.hidden = false;
  banner.classList.add("precheck-banner--error");
  banner.textContent = "stale warning from a previous step";

  dom.window.showViewfinder(true);

  assert(!guide.classList.contains("card-guide--warning"), "showViewfinder(true) clears a stale guide marker");
  assert(!guide.classList.contains("card-guide--good"), "showViewfinder(true) clears a stale good marker too");
  assert(!guide.classList.contains("card-guide--error"), "showViewfinder(true) clears a stale error marker too");
  assert(banner.hidden === true, "showViewfinder(true) clears a stale banner");
  assert(!banner.classList.contains("precheck-banner--error"), "showViewfinder(true) clears the stale error banner class");
}

// --- Test 10: switching to camera mode with a mocked getUserMedia actually starts a stream ---
async function runTest10() {
  const dom = makeDom(); // starts in picker mode (no getUserMedia in jsdom)
  const w = dom.window;
  const doc = w.document;

  let stopped = false;
  const fakeStream = { getTracks: () => [{ stop: () => (stopped = true) }] };
  w.navigator.mediaDevices = { getUserMedia: async () => fakeStream };

  await w.switchToCameraMode();
  assert(w.eval("cameraModeActive") === true, "switchToCameraMode flips the mode flag");
  assert(doc.getElementById("camera-mode").hidden === false, "camera view shown after switching to camera mode");
  assert(doc.getElementById("camera-video").srcObject === fakeStream, "video element gets the mocked stream attached");

  w.stopCamera();
  assert(stopped === true, "stopCamera stops every track on the active stream");
}

// --- Test 5: download export produces a complete, self-contained HTML blob ---
(async () => {
  await runTest2b();
  await runTest10();

  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  dom.window.handleReport(data.report, data.images);

  let capturedBlobParts = null;
  dom.window.Blob = class {
    constructor(parts, opts) {
      capturedBlobParts = parts;
      this.type = opts && opts.type;
    }
  };
  dom.window.URL.createObjectURL = () => "blob:mock-download";
  dom.window.URL.revokeObjectURL = () => {};
  let clicked = false;
  dom.window.HTMLAnchorElement.prototype.click = function () {
    clicked = true;
  };

  await dom.window.downloadReport();

  assert(clicked === true, "download triggers an anchor click");
  assert(capturedBlobParts !== null, "a Blob was constructed for the download");
  const downloadedHtml = capturedBlobParts ? capturedBlobParts.join("") : "";
  assert(downloadedHtml.includes("<!doctype html>"), "downloaded file is a full standalone HTML document");
  assert(downloadedHtml.includes(String(data.report.grade_estimate.overall_grade_rounded)), "downloaded file contains the overall grade");
  assert(downloadedHtml.includes("data:image/png;base64"), "downloaded file has images inlined as data URIs (no external references)");
  assert(downloadedHtml.includes("defect-slider"), "downloaded file retains the defect slider markup");
  assert(downloadedHtml.includes("addEventListener(\"input\""), "downloaded file inlines the slider's JS so it works fully offline");

  console.log(failures === 0 ? "\nALL TESTS PASSED" : `\n${failures} TEST(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
})();
