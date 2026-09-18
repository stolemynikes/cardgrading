const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const REPO = path.join(__dirname, "..", "static");
const html = fs.readFileSync(path.join(REPO, "index.html"), "utf8");
const js = fs.readFileSync(path.join(REPO, "app.js"), "utf8");

function makeDom(options = {}) {
  const dom = new JSDOM(html, { runScripts: "dangerously", url: options.url || "http://localhost/" });
  // createImageBitmap doesn't exist in jsdom - stub it so processImageFile's
  // try/catch falls back to the original file cleanly, matching real-browser
  // fallback behavior for unsupported environments.
  dom.window.createImageBitmap = undefined;
  dom.window.URL.createObjectURL = () => "blob:mock";
  dom.window.fetch = options.fetch || (async () => ({ ok: true, text: async () => "" }));
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
  // The fixture predates Card Vision; give it a relief image so the viewer's
  // shared slider renders and the wiring assertion below stays meaningful.
  const images = Object.assign({}, data.images, {
    front_card_vision: "data:image/png;base64,AAAA",
    back_card_vision: "data:image/png;base64,BBBB",
  });
  dom.window.handleReport(data.report, images);

  const doc = dom.window.document;
  assert(doc.getElementById("report-view").hidden === false, "report view shown on success");
  assert(doc.getElementById("capture-view").hidden === true, "capture view hidden on success");

  const content = doc.getElementById("report-content").innerHTML;
  assert(content.includes("grade-header"), "grade header rendered");
  assert(content.includes(String(data.report.grade_estimate.overall_grade_rounded)), "overall grade value present");
  assert(content.includes("Corners &amp; Edges") || content.includes("Corners & Edges"), "corners/edges section present");
  assert(content.includes("Surface"), "surface section present");
  assert(content.includes("overlay-slider"), "overlay slider input present");

  // sliders should be wired: check img opacity responds to a manual 'input' event
  const slider = doc.querySelector(".defect-slider, .overlay-slider");
  assert(slider !== null, "found an overlay slider element");
  if (slider) {
    // data-target may name several overlays (the card viewer drives front and
    // back from one control), so take the first rather than the whole list.
    const firstTarget = (slider.dataset.target || slider.id + "-img").split(",")[0].trim();
    const img = doc.getElementById(firstTarget);
    const inverted = slider.dataset.invert === "true";
    slider.value = "10";
    slider.dispatchEvent(new dom.window.Event("input"));
    const expected = inverted ? 0.9 : 0.1;
    assert(Number(img.style.opacity) === expected, `slider updates image opacity (expected ${expected}, got ${img.style.opacity})`);
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

// --- Test 1e: surface is a measurement, not an opinion. The signal's
// provenance decides whether it grades at all, and the report has to say
// which one produced the number.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  report.surface = {
    front: { grade: 4, defect_area_pct: 0.799, defect_count: 63, longest_defect_px: 957,
             source: "photometric_relief", note: "measured from solved surface normals",
             limited_by: "crease", defect_kinds: { crease: 1, scratch: 9 },
             defects: [
               { kind: "crease", length_mm: 2.7, width_mm: 1.35, depth: 75.1, area_px: 604,
                 centre_mm: [1.2, 87.4], grade_cap: 4 },
               { kind: "scratch", length_mm: 9.9, width_mm: 0.24, depth: 32.7, area_px: 770,
                 centre_mm: [29.1, 57.6], grade_cap: 9 },
             ] },
    back: { grade: 6, defect_area_pct: 1.4, defect_count: 5, longest_defect_px: 300,
            source: "photometric_relief", note: "measured from solved surface normals",
            limited_by: "overall surface wear", defect_kinds: {}, defects: [] },
  };
  dom.window.handleReport(report, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;
  assert(content.includes("grade 4"), "measured surface grade rendered");
  assert(content.includes("0.799%"), "defect area shown to the measured precision");
  assert(content.includes("grade 6"), "the weaker side is rendered too");
  assert(content.includes("measured from solved surface normals"), "the signal's provenance is stated");
  // What capped the grade, and where on the card to look for it — a bare
  // "surface 4" is not something anyone can check.
  assert(content.includes("limited by"), "the report says what set the grade");
  assert(content.includes("crease"), "and names the defect kind that set it");
  assert(content.includes("87.4mm") || content.includes("87.4"), "with the position on the card");
  assert(content.includes("caps at 4"), "and the ceiling that defect carries");
}

// --- Test 1c2: a refused centering axis says why it was refused. "n/a" reads
// the same whether the card has no border to find or the detector found
// something that isn't one — and only the second is worth dragging lines over.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  report.centering.front.vertical = {
    ...report.centering.front.vertical,
    measurable: false,
    variation_px: 159.0,
    reason: "the top/bottom boundary wanders 159px along the side, 3.1x the 52px border it would be measuring — a printed border edge is straight, so this is tracking artwork rather than the border. Place the boundaries by hand to grade this axis.",
  };
  dom.window.handleReport(report, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;
  assert(content.includes("wanders 159px"), "the refusal reason is shown, not a bare n/a");
  assert(content.includes("by hand"), "and points at the manual adjustment");

  // Older reports carry no reason and must fall back to the generic wording.
  const older = JSON.parse(JSON.stringify(data.report));
  older.centering.front.vertical = { ...older.centering.front.vertical, measurable: false };
  dom.window.handleReport(older, data.images);
  assert(
    dom.window.document.getElementById("report-content").innerHTML.includes("boundary not visible"),
    "an older refused axis still explains itself"
  );
}

// --- Test 1d2: a corner tile shows whichever of the two readings actually
// set its grade, and names both. Whitening is blind on a neutral border and
// relief is blind to a stain that hasn't deformed anything.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  const region = (whitening, wear, grade) => ({
    whitening_pct: whitening, relief_wear_pct: wear, blob_count: 1, grade,
    measurable: true, uniformity: 0.9, reason: null, box: [0.01, 0.01, 0.03, 0.02],
  });
  report.corners_edges.front.corners.top_left = region(0.0, 24.2, 4);
  report.corners_edges.front.corners.top_right = region(3.1, 0.0, 8);
  dom.window.handleReport(report, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;
  assert(content.includes("24.20%"), "the relief reading is shown when it is the worse one");
  assert(content.includes("3.10%"), "and the whitening reading when that one is");
  assert(content.includes("surface relief"), "both readings are named on the tile");

  // Reports saved before relief-based wear existed carry no relief_wear_pct.
  const older = JSON.parse(JSON.stringify(data.report));
  dom.window.handleReport(older, data.images);
  const olderContent = dom.window.document.getElementById("report-content").innerHTML;
  assert(olderContent.includes("region-pct"), "an older corners/edges report still renders");
  assert(!olderContent.includes("surface relief"), "and claims no reading it never had");
}

// --- Test 1e2: a report saved before defect classification existed has no
// defects, no kinds and no limited_by, and must still render.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  report.surface = {
    front: { grade: 7, defect_area_pct: 0.799, defect_count: 63, longest_defect_px: 957,
             source: "photometric_relief", note: "measured from solved surface normals" },
    back: null,
  };
  dom.window.handleReport(report, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;
  assert(content.includes("grade 7"), "an older surface report still renders its grade");
  assert(!content.includes("limited by"), "and claims no limit it never recorded");
}

// --- Test 1e3: the three published rubrics disagree, and the report shows
// all three rather than hiding the choice behind one number.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  const byGrader = {
    psa: { label: "PSA", source: "https://www.psacard.com/gradingstandards", grade: 5, limited_by: "crease" },
    tag: { label: "TAG", source: "https://taggrading.com/pages/rubric", grade: 4, limited_by: "crease" },
    cgc: { label: "CGC", source: "https://www.cgccards.com/card-grading/grading-scale/", grade: 4, limited_by: "crease" },
  };
  report.surface = {
    front: { grade: 5, defect_area_pct: 0.15, defect_count: 19, longest_defect_px: 300,
             source: "photometric_relief", note: "measured from solved surface normals",
             grader: "psa", limited_by: "crease", defect_kinds: { crease: 1 },
             defects: [{ kind: "crease", severity: "crease", length_mm: 2.7, width_mm: 1.35,
                         depth: 75.1, area_px: 604, centre_mm: [1.2, 87.4], grade_cap: 5 }],
             by_grader: byGrader },
    back: { grade: 9, defect_area_pct: 0.03, defect_count: 2, longest_defect_px: 40,
            source: "photometric_relief", note: "measured from solved surface normals",
            grader: "psa", limited_by: "pit", defect_kinds: { pit: 2 }, defects: [],
            by_grader: { psa: { label: "PSA", source: "x", grade: 9, limited_by: "pit" },
                         tag: { label: "TAG", source: "y", grade: 9, limited_by: "pit" },
                         cgc: { label: "CGC", source: "z", grade: 9, limited_by: "pit" } } },
  };
  dom.window.handleReport(report, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;
  assert(content.includes("every grading service"), "the surface comparison table is rendered");
  for (const label of ["PSA", "TAG", "CGC"]) {
    assert(content.includes(label), `${label} appears in the surface comparison`);
  }
  assert(content.includes("taggrading.com/pages/rubric"), "each rubric cites its published source");
  assert(content.includes("cgccards.com"), "including CGC's");

  // A report saved before per-grader surface grading has no by_grader and
  // must render without the table rather than throwing.
  const older = JSON.parse(JSON.stringify(report));
  delete older.surface.front.by_grader;
  delete older.surface.back.by_grader;
  dom.window.handleReport(older, data.images);
  const olderContent = dom.window.document.getElementById("report-content").innerHTML;
  assert(olderContent.includes("grade 5"), "an older surface report still renders");
  assert(!olderContent.includes("every grading service's rubric"), "and shows no comparison it never had");
}

// --- Test 1f: the single-capture approximation is measured and shown but
// never graded, because print leaks into that signal.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  report.surface = {
    front: { grade: null, defect_area_pct: 9.1, defect_count: 240, longest_defect_px: 1400,
             source: "single_image_relief", note: "print leaks into this signal" },
    back: null,
  };
  dom.window.handleReport(report, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;
  assert(content.includes("not graded"), "an ungraded surface says so rather than showing a number");
  assert(content.includes("print leaks into this signal"), "the reason it isn't graded is given");
}

// --- Test 1f2: the offline same-side warning. Distinct from the identity
// warning in 1g, which says a similar thing from the vision stage — that one
// needs an API key and is usually skipped, this one always runs.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));

  const clean = JSON.parse(JSON.stringify(data.report));
  clean.capture_pair = { identical_files: false, similarity: 0.07, same_side_suspected: false, note: null };
  dom.window.handleReport(clean, data.images);
  assert(
    !dom.window.document.getElementById("report-content").innerHTML.includes("same side"),
    "a genuine front/back pair gets no warning"
  );

  const duplicated = JSON.parse(JSON.stringify(data.report));
  duplicated.capture_pair = {
    identical_files: true, similarity: 1.0, same_side_suspected: true,
    note: "The front and back uploads are the same file, so the back was graded on the front.",
  };
  dom.window.handleReport(duplicated, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;
  assert(content.includes("the same file"), "the duplicate upload is called out in the report's own words");
  assert(content.includes("banner-warn"), "it warns rather than reading as an error");
  assert(content.includes("grade-header"), "and the report still renders — this never blocks a grade");

  // Reports saved before this check existed have no capture_pair at all.
  const old = JSON.parse(JSON.stringify(data.report));
  delete old.capture_pair;
  dom.window.handleReport(old, data.images);
  assert(
    dom.window.document.getElementById("report-content").innerHTML.includes("grade-header"),
    "an older report with no capture_pair still renders"
  );
}

// --- Test 1g: card identification header, pair warnings, market value, DINGS, disagreement, lightbox ---
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  report.card_id = {
    card_name: "Gothitelle", set_name: "SVP Black Star Promos", collector_number: "211",
    game: "pokemon", is_full_art: true, is_holo: true,
    front_image_side: "front", back_image_side: "back", looks_like_same_card: true,
    confidence: "high", model: "gemini-flash-latest",
  };
  report.market = {
    matched_name: "Gothitelle", matched_set: "SVP Black Star Promos", matched_number: "211",
    prices: { tcgplayer_holofoil: 3.21, cardmarket_trend: 2.8 },
    image_url: "", source: "pokemontcg.io",
  };
  // full-art + unmeasurable front centering -> "expected" note variant
  report.centering.front.measurable = false;
  report.centering.front.grade = null;
  dom.window.handleReport(report, data.images);
  const doc = dom.window.document;
  let content = doc.getElementById("report-content").innerHTML;

  assert(content.includes("Gothitelle"), "card name shown in the report header");
  assert(content.includes("SVP Black Star Promos"), "set name shown");
  assert(content.includes("full-art") && content.includes("holo"), "full-art and holo badges shown");
  assert(content.includes("Raw market value"), "market value line renders");
  assert(content.includes("$3.21"), "a tcgplayer price is shown");
  assert(content.includes("graded value varies"), "market line carries the raw-price caveat");
  assert(!content.includes("may be swapped"), "no swap warning when sides check out");
  assert(
    content.includes("Expected, not a capture problem"),
    "full-art + unmeasurable centering explains it's expected, not a bad capture"
  );

  // DINGS: exactly the worst region(s) get the marker. Compute expected from fixture.
  const ceFront = report.corners_edges.front;
  const regions = [...Object.values(ceFront.corners), ...Object.values(ceFront.edges)];
  const worst = Math.min(...regions.map((r) => r.grade));
  const expectedDings = regions.filter((r) => r.grade === worst).length +
    [...Object.values(report.corners_edges.back.corners), ...Object.values(report.corners_edges.back.edges)]
      .filter((r, _, arr) => r.grade === Math.min(...arr.map((x) => x.grade))).length;
  const dingTiles = (content.match(/region-tile--ding/g) || []).length;
  assert(dingTiles === expectedDings, `DINGS markers land on exactly the worst regions (expected ${expectedDings}, got ${dingTiles})`);
  assert(content.includes("drove the grade"), "centering worse-axis marker present");

  // Lightbox: clicking a report image opens it fullscreen; close button closes.
  const box = doc.getElementById("lightbox");
  assert(box.hidden === true, "lightbox starts hidden");
  const anyImg = doc.querySelector("#report-content img");
  anyImg.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  assert(box.hidden === false, "clicking a report image opens the lightbox");
  assert(doc.getElementById("lightbox-img").src === anyImg.src, "lightbox shows the clicked image");
  doc.getElementById("lightbox-close").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  assert(box.hidden === true, "close button hides the lightbox");
}

// --- Test 1h: pair-sanity warnings from identification (the one remaining
// model call) still surface ---
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  report.card_id = {
    card_name: "Pangoro", set_name: "", collector_number: "", game: "pokemon",
    is_full_art: false, is_holo: false,
    front_image_side: "back", back_image_side: "back", looks_like_same_card: false,
    confidence: "medium", model: "gemini-2.5-flash",
  };
  dom.window.handleReport(report, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;
  assert(content.includes("may be swapped"), "swapped/duplicate-side warning shows");
  assert(content.includes("may not be the same card"), "different-cards warning shows");
}

// --- Test 1i: no card_id/market renders like before ---
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  dom.window.handleReport(report, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;
  assert(!content.includes("card-identity"), "no identity header without card_id");
  assert(!content.includes("Raw market value"), "no market line without market data");
}

// --- Test 1j: per-axis centering — one measurable axis renders its ratio, the other shows n/a ---
// Real case: a soft capture where one axis's boundary is genuinely invisible
// (blue frame melting into blue swirl) while the other axis measures fine.
// The old all-or-nothing rule hid the good measurement behind a blanket n/a.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  report.centering.back.horizontal.measurable = false;
  report.centering.back.vertical.measurable = true;
  report.centering.back.grade = report.centering.back.vertical.grade;
  dom.window.handleReport(report, data.images);
  const content = dom.window.document.getElementById("report-content").innerHTML;

  assert(content.includes("n/a — boundary not visible"), "unmeasurable H axis shows n/a instead of noise numbers");
  const backVRatio = report.centering.back.vertical.ratio;
  assert(content.includes(backVRatio), "measurable V axis still shows its real ratio");
  assert(!content.includes(`>${report.centering.back.horizontal.ratio}<`) ||
         report.centering.back.horizontal.ratio === backVRatio,
         "the unmeasurable H axis's noise ratio is not rendered");
  assert(content.includes(`grade ${report.centering.back.grade}`), "side grade (from the measurable axis) is shown");
  assert(!content.includes("Couldn't measure — borderless"), "no blanket couldn't-measure card when one axis measured");
}

// --- Test 1d: unmeasurable centering (borderless/full-art card) renders honestly ---
// Real case: a full-art promo produced a fake "89/11 grade 3" because the
// border detector returned argmax-of-noise. The server now marks such sides
// measurable:false with grade null — the report must say "couldn't measure"
// instead of showing noise ratios, and still render the rest of the report.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  report.centering.front.measurable = false;
  report.centering.front.grade = null;
  report.centering.overall_grade = report.centering.back.grade;
  report.grade_estimate.centering_grade = null;
  dom.window.handleReport(report, data.images);

  const doc = dom.window.document;
  assert(doc.getElementById("report-view").hidden === false, "report renders with an unmeasurable centering side");
  const content = doc.getElementById("report-content").innerHTML;
  assert(content.includes("Couldn't measure"), "unmeasurable side shows the couldn't-measure note");
  assert(content.includes("borderless/full-art"), "note explains the likely cause");
  const frontRatio = data.report.centering.front.horizontal.ratio;
  assert(!content.includes(`>${frontRatio}<`), "the noise ratio numbers are not shown for the unmeasurable side");
  const backRatio = data.report.centering.back.horizontal.ratio;
  assert(content.includes(backRatio), "the measurable back side still shows its real ratios");
  // centering subgrade tile shows n/a
  assert(content.includes("n/a"), "centering subgrade renders as n/a");
}

// --- Test 1k: Card Vision, the score, per-side sub-grades, dimensions and the
// DINGS list. The whole point of the Card Vision slider is that a viewer can
// tell a photometric solve apart from the single-capture approximation, so
// the method label is asserted, not just the presence of the slider.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  const images = Object.assign({}, data.images, {
    front_card_vision: "data:image/png;base64,AAAA",
    back_card_vision: "data:image/png;base64,BBBB",
  });

  report.card_vision = {
    front: { method: "photometric_stereo", light_count: 4, roughness_pct: 1.2, note: "surface normals solved" },
    back: { method: "single_image", light_count: 1, roughness_pct: 3.4, note: "approximation from a single capture" },
  };
  report.grade_estimate.score = 872;
  report.subgrades = {
    front: { centering: 9, corners: 8, edges: 9, surface: null },
    back: { centering: 10, corners: 9, edges: 9, surface: 8 },
  };
  report.dimensions = {
    measurable: true, width_mm: 62.98, height_mm: 88.02,
    nominal_width_mm: 63.0, nominal_height_mm: 88.0,
    squareness_deviation_deg: 0.12, within_tolerance: true, note: "within 0.75mm of nominal",
  };
  report.dings = [
    { attribute: "corners", side: "front", label: "top-left corner", grade: 8,
      detail: "1.20% whitening, 3 blob(s)", image_key: "front_corner_top_left" },
  ];

  dom.window.handleReport(report, images);
  const doc = dom.window.document;
  const content = doc.getElementById("report-content").innerHTML;

  assert(content.includes("872"), "score is rendered");
  assert(content.includes("Card Vision"), "Card Vision section present");
  assert(content.includes("photometric · 4 lights"), "photometric method labeled with its light count");
  assert(content.includes("single-capture approx."), "the approximated side is labeled as such, not passed off as a solve");
  assert(content.includes("Corner wear"), "DINGS gallery names the defect type");
  assert(content.includes("attr-strip"), "five-attribute summary strip rendered");
  assert(content.includes("F: 1 DINGS"), "attribute strip counts this side's dings");
  assert(content.includes("top-left corner"), "DINGS list names the grade-driving region");
  assert(content.includes("62.98"), "measured dimensions rendered");

  // One inverted slider drives BOTH sides, so front and back can't drift to
  // different blends while you compare them.
  const cvSlider = doc.getElementById("cardvision-slider");
  assert(cvSlider !== null, "shared Card Vision slider exists");
  if (cvSlider) {
    const frontImg = doc.getElementById("cardvision-front-img");
    const backImg = doc.getElementById("cardvision-back-img");
    const opacity = (el) => Number(el.style.opacity);
    assert(opacity(frontImg) === 0.5 && opacity(backImg) === 0.5, "both overlays start at the slider's initial value");
    cvSlider.value = "100";
    cvSlider.dispatchEvent(new dom.window.Event("input"));
    assert(opacity(frontImg) === 0 && opacity(backImg) === 0, "100% colour hides the relief on both sides");
    cvSlider.value = "0";
    cvSlider.dispatchEvent(new dom.window.Event("input"));
    assert(opacity(frontImg) === 1 && opacity(backImg) === 1, "0% shows full Card Vision on both sides");
    assert(doc.getElementById("cardvision-readout").textContent === "0%", "readout tracks the slider");
  }
}

// --- Test 1l: a report from before these fields existed (and one where the
// scan-only measurements couldn't run) must still render — the webapp is
// pointed at whatever backend is deployed, and a phone capture legitimately
// has no dimensions.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const report = JSON.parse(JSON.stringify(data.report));
  delete report.card_vision;
  delete report.subgrades;
  delete report.dings;
  report.dimensions = { measurable: false, note: "not measurable without a known capture scale", within_tolerance: null };

  dom.window.handleReport(report, data.images);
  const doc = dom.window.document;
  assert(doc.getElementById("report-view").hidden === false, "report still renders without the newer fields");
  const content = doc.getElementById("report-content").innerHTML;
  assert(!content.includes("Card Vision"), "no Card Vision section when the backend didn't produce one");
  assert(content.includes("not measurable without a known capture scale"), "unmeasurable dimensions explained rather than shown as a failure");
}

// --- Test 1m: a finished report shows its permalink, so there's a way back
// to it later. Without this the id is saved server-side and unreachable.
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  const reportId = "0123456789abcdef0123456789abcdef";
  dom.window.handleReport(data.report, data.images, reportId);

  const doc = dom.window.document;
  const link = doc.querySelector(".permalink-url");
  assert(link !== null, "permalink rendered for a saved report");
  assert(link && link.getAttribute("href") === `/r/${reportId}`, "permalink points at the report's own URL");
  assert(
    doc.getElementById("permalink-copy") !== null,
    "a copy button is offered (the URL is long and this runs on a phone)"
  );
}

// --- Test 1n: an unsaved report (the store write failed) renders normally
// with no permalink, rather than linking somewhere that doesn't exist ---
{
  const dom = makeDom();
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
  dom.window.handleReport(data.report, data.images, null);
  const doc = dom.window.document;
  assert(doc.querySelector(".permalink-url") === null, "no permalink when the report wasn't saved");
  assert(doc.getElementById("report-view").hidden === false, "report still renders without a permalink");
}

// --- Test 1s: the theme control. Three states, because a plain light/dark
// switch strands anyone who flips it once and then wants the page to follow
// their phone again.
{
  const dom = makeDom();
  const doc = dom.window.document;
  const root = doc.documentElement;
  const button = doc.getElementById("theme-toggle");

  assert(button !== null, "theme control exists in the header");
  assert(!root.dataset.theme, "starts on auto — no override until asked");

  button.click();
  assert(root.dataset.theme === "light", "auto -> light");
  assert(dom.window.localStorage.getItem("cardgrading.theme") === "light", "the choice is remembered");

  button.click();
  assert(root.dataset.theme === "dark", "light -> dark");

  button.click();
  assert(!root.dataset.theme, "dark -> auto, so following the system stays reachable");
  assert(dom.window.localStorage.getItem("cardgrading.theme") === null, "auto clears the stored override");
}

// --- Test 1t: a stored choice is reapplied, and the browser chrome follows
// the resolved theme rather than the now-stale media-scoped metas ---
{
  const dom = makeDom();
  const doc = dom.window.document;
  dom.window.localStorage.setItem("cardgrading.theme", "dark");
  dom.window.eval("applyTheme(storedTheme())");

  assert(doc.documentElement.dataset.theme === "dark", "stored choice reapplied");
  const metas = doc.querySelectorAll('meta[name="theme-color"]');
  assert(metas.length === 1, `exactly one theme-color meta remains (got ${metas.length})`);
  assert(metas[0].content === "#0c1219", `chrome colour matches the resolved theme (got ${metas[0].content})`);

  dom.window.localStorage.setItem("cardgrading.theme", "light");
  dom.window.eval("applyTheme(storedTheme())");
  assert(doc.querySelector('meta[name="theme-color"]').content === "#eaecef", "and follows a switch to light");
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

  // Optional extras never gate submit: front and back are the whole
  // requirement, and the rotation set is validated on submit instead.
  dom.window.eval("updateSubmitEnabled();");
  assert(submitBtn.disabled === false, "submit stays enabled with only front and back");

  dom.window.eval(`
    state.files.photometric_front = [new File(["x"], "r0.tif", { type: "image/tiff" })];
    updateSubmitEnabled();
  `);
  assert(submitBtn.disabled === false, "submit stays enabled with rotation scans attached too");
}

// --- Test 6: the app opens in upload mode (the scanner flow), and only
// complains about the camera once someone actually asks for it ---
{
  const dom = makeDom(); // jsdom has no navigator.mediaDevices.getUserMedia
  const doc = dom.window.document;
  assert(dom.window.eval("cameraModeActive") === false, "opens in upload mode, not the viewfinder");
  assert(doc.getElementById("camera-mode").hidden === true, "camera-mode view hidden on load");
  assert(doc.getElementById("picker-mode").hidden === false, "picker-mode view shown on load");
  assert(doc.getElementById("scanner-mode").hidden === false, "scanner extras available on load");
  assert(doc.getElementById("scanner-fields").hidden === false, "scanner extras are open, not behind a second click");
  assert(doc.getElementById("protocol-hint-scanner").hidden === false, "scan protocol shown, not the tripod advice");
  assert(doc.getElementById("protocol-hint-camera").hidden === true, "camera protocol hidden in upload mode");
  assert(
    doc.getElementById("camera-unavailable-banner").hidden === true,
    "no camera warning when the user never asked for the camera"
  );
}

// --- Test 6b: asking for camera mode where getUserMedia doesn't exist falls
// back to the picker and says why ---
{
  const dom = makeDom();
  const doc = dom.window.document;
  // Not awaited: startCamera()'s no-getUserMedia branch runs synchronously
  // before its first await, and this file is CommonJS (no top-level await).
  dom.window.eval("switchToCameraMode()");
  assert(dom.window.eval("cameraModeActive") === false, "auto-switched back when getUserMedia is unavailable");
  assert(doc.getElementById("picker-mode").hidden === false, "picker-mode view shown after fallback");
  assert(doc.getElementById("camera-unavailable-banner").hidden === false, "explains why camera mode isn't available");
  assert(
    doc.getElementById("camera-unavailable-banner").textContent.includes("HTTPS"),
    "unavailable message mentions the secure-context requirement"
  );
}

// --- Test 7: the guided capture is two steps. Raking-light shots are gone
// from the flow entirely — surface relief comes from rotating the card under
// a fixed light, which is a picker upload, not a capture step.
{
  const dom = makeDom();
  const w = dom.window;
  assert(JSON.stringify(w.activeSteps()) === JSON.stringify(["front", "back"]), "two capture steps: front and back");
  assert(w.document.getElementById("surface-toggle") === null, "no raking-light toggle remains");
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
    compute(flatPixels(130, N), GOOD_HEIGHT) === "No card detected — center it in the frame",
    "flat/uniform region with no card-like detail triggers the new warning"
  );

  // Darkness must still win over "no card" when a flat region is ALSO dark
  // — can't tell if a card's there or not if you can't see anything.
  assert(
    compute(flatPixels(20, N), GOOD_HEIGHT) === "Too dark — add more light",
    "darkness check takes priority over the no-card check on a dark+flat region"
  );

  // A textured region (real print/border/artwork produces exactly this
  // kind of local contrast) at a normal brightness with no glare must not
  // false-positive on any check.
  assert(
    compute(texturedPixels(N, 80, 180), GOOD_HEIGHT) === null,
    "a textured region with a card-like level of contrast produces no warning"
  );

  // High contrast + a genuine glare cluster should still report glare, not
  // get preempted by the (satisfied) no-card check.
  assert(
    compute(glarePixels(N, 150, 0.15), GOOD_HEIGHT) === "Glare detected — adjust the light angle",
    "glare is still reported on a textured (card-like) region that also has a bright cluster"
  );

  // Textured, well-lit, but the guide maps to a small native region — move
  // closer should be the only thing left to say.
  assert(
    compute(texturedPixels(N, 80, 180), 400) === "Move closer to the card",
    "distance check still fires when nothing else is wrong but the guide region is small"
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
    compute(grayscaleTexturedPixels(N, 170, 250), GOOD_HEIGHT) === "Doesn't look like a card — too little color",
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
    compute(flatSaturationPixels(N, 0.1244), GOOD_HEIGHT) !== "Doesn't look like a card — too little color",
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

// --- Test 10b: camera zoom control — real sensor zoom, not CSS scaling ---
// Phone lenses can't focus closer than ~10-15cm, so users who move close to
// "fill the guide" get permanent blur. The zoom slider drives the camera
// track's zoom constraint instead. Only shown when the track supports it.
async function runTest10b() {
  // With zoom capability: slider appears, input applies constraints.
  {
    const dom = makeDom();
    const w = dom.window;
    const doc = w.document;
    const applied = [];
    const fakeTrack = {
      stop: () => {},
      getCapabilities: () => ({ zoom: { min: 1, max: 10, step: 0.1 } }),
      getSettings: () => ({ zoom: 1 }),
      applyConstraints: async (c) => applied.push(c),
    };
    const fakeStream = { getTracks: () => [fakeTrack], getVideoTracks: () => [fakeTrack] };
    w.navigator.mediaDevices = { getUserMedia: async () => fakeStream };

    await w.switchToCameraMode();
    const row = doc.getElementById("zoom-row");
    const slider = doc.getElementById("zoom-slider");
    assert(row.hidden === false, "zoom slider appears when the camera supports the zoom constraint");
    assert(slider.max === "5", `slider max is capped at 5x even when hardware reports 10x (got ${slider.max})`);

    slider.value = "2.5";
    slider.oninput();
    assert(applied.length === 1, "moving the slider applies a track constraint");
    assert(applied[0].advanced[0].zoom === 2.5, "the applied constraint carries the chosen zoom level");
    assert(doc.getElementById("zoom-value").textContent === "2.5×", "the zoom value label updates");

    // Zoom availability changes the "too far" precheck message: telling a
    // user who has a zoom slider to physically move closer walks them into
    // the minimum focus distance — the original blur complaint. Pixel data
    // must be bright/textured/colorful so only the distance check fires.
    const cardPx = new Uint8ClampedArray(400 * 4);
    for (let i = 0; i < cardPx.length; i += 4) {
      const v = i % 8 === 0 ? 80 : 180;
      cardPx[i] = v;
      cardPx[i + 1] = Math.round(v * 0.6);
      cardPx[i + 2] = Math.round(v * 0.3);
      cardPx[i + 3] = 255;
    }
    assert(
      w.computePrecheckMessage(cardPx, 500, true) === "Card too small in frame — zoom in",
      "with zoom available, the distance hint says zoom in"
    );
    assert(
      w.computePrecheckMessage(cardPx, 500, false) === "Move closer to the card",
      "without zoom, the distance hint still says move closer"
    );

    w.stopCamera();
    assert(row.hidden === true, "stopping the camera hides the zoom slider");
  }

  // Without zoom capability: no slider.
  {
    const dom = makeDom();
    const w = dom.window;
    const fakeTrack = { stop: () => {}, getCapabilities: () => ({}) };
    const fakeStream = { getTracks: () => [fakeTrack], getVideoTracks: () => [fakeTrack] };
    w.navigator.mediaDevices = { getUserMedia: async () => fakeStream };
    await w.switchToCameraMode();
    assert(
      w.document.getElementById("zoom-row").hidden === true,
      "zoom slider stays hidden when the camera doesn't support zoom"
    );
  }
}

// --- Test 5: download export produces a complete, self-contained HTML blob ---
(async () => {
  await runTest2b();
  await runTest10b();
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
  assert(downloadedHtml.includes("overlay-slider"), "downloaded file retains the overlay slider markup");
  assert(downloadedHtml.includes("addEventListener(\"input\""), "downloaded file inlines the slider's JS so it works fully offline");

  // --- Test 1o: /r/<id> loads that report straight away and skips capture ---
  {
    const reportId = "0123456789abcdef0123456789abcdef";
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    let requested = null;
    const dom = makeDom({
      url: `http://localhost/r/${reportId}`,
      fetch: async (url) => {
        requested = url;
        return {
          ok: true,
          status: 200,
          json: async () => ({ status: "done", report_id: reportId, report: data.report, images: data.images }),
          text: async () => "",
        };
      },
    });

    await new Promise((r) => setTimeout(r, 0)); // let the deep-link fetch settle
    const doc = dom.window.document;
    assert(requested === `/api/report/${reportId}`, `deep link fetches the saved report (got ${requested})`);
    assert(doc.getElementById("report-view").hidden === false, "deep link lands on the report, not the capture flow");
    assert(doc.getElementById("capture-view").hidden === true, "capture view skipped on a deep link");
  }

  // --- Test 1p: a deep link to a deleted report says so instead of hanging on
  // the spinner or rendering an empty report ---
  {
    const dom = makeDom({
      url: "http://localhost/r/" + "b".repeat(32),
      fetch: async () => ({ ok: false, status: 404, json: async () => ({}), text: async () => "" }),
    });
    await new Promise((r) => setTimeout(r, 0));
    const doc = dom.window.document;
    assert(
      doc.getElementById("processing-message").textContent.includes("doesn't exist"),
      `missing report explained (got: ${doc.getElementById("processing-message").textContent})`
    );
    assert(doc.getElementById("report-view").hidden === true, "no empty report view for a missing report");
  }

  // --- Test 1q: the capture view lists previously saved reports ---
  {
    const summaries = [
      { report_id: "a".repeat(32), created_at: "2026-09-10T09:00:00+00:00", card_name: "Charizard", grade: 9, score: 910 },
      { report_id: "c".repeat(32), created_at: "2026-09-09T09:00:00+00:00", card_name: null, grade: 7, score: 660 },
    ];
    const dom = makeDom({
      fetch: async (url) => ({
        ok: true,
        status: 200,
        json: async () => (url.startsWith("/api/reports") ? { reports: summaries } : {}),
        text: async () => "",
      }),
    });
    await new Promise((r) => setTimeout(r, 0));
    const doc = dom.window.document;
    const container = doc.getElementById("saved-reports");
    assert(container.hidden === false, "saved-report list shown when there are reports");
    assert(container.innerHTML.includes("Charizard"), "identified card listed by name");
    assert(container.innerHTML.includes("Unidentified card"), "a card with no identification still gets a row");
    assert(container.innerHTML.includes(`/r/${"a".repeat(32)}`), "each row links to its report");
  }

  // --- Test 1r: no saved reports means no empty list widget ---
  {
    const dom = makeDom({
      fetch: async () => ({ ok: true, status: 200, json: async () => ({ reports: [] }), text: async () => "" }),
    });
    await new Promise((r) => setTimeout(r, 0));
    assert(dom.window.document.getElementById("saved-reports").hidden === true, "no empty saved-report widget on a first run");
  }

  // --- Test 1s: TIFF previews ---
  // jsdom has no canvas backend, so stand in for one. The assertions are
  // about the decoder's arithmetic — endianness, the >4-byte value-offset
  // branch, strip addressing, channel order — not about rasterizing.
  function withFakeCanvas(dom, fn) {
    const doc = dom.window.document;
    const realCreate = doc.createElement.bind(doc);
    let captured = null;
    doc.createElement = (tag) => {
      if (tag !== "canvas") return realCreate(tag);
      const canvas = { width: 0, height: 0 };
      canvas.getContext = () => ({
        createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
        putImageData: (img) => {
          captured = img;
        },
      });
      canvas.toBlob = (cb) => cb(null);
      return canvas;
    };
    try {
      return { result: fn(), captured };
    } finally {
      doc.createElement = realCreate;
    }
  }

  // A 2x2 uncompressed chunky RGB TIFF, little-endian, with BitsPerSample
  // stored out-of-line (3 SHORTs = 6 bytes, past the 4-byte inline limit).
  function buildTiff({ littleEndian = true, compression = 1 } = {}) {
    const bpsOffset = 8 + 2 + 7 * 12 + 4;
    const pixelOffset = bpsOffset + 6;
    const buf = new ArrayBuffer(pixelOffset + 12);
    const view = new DataView(buf);
    const le = littleEndian;
    view.setUint16(0, le ? 0x4949 : 0x4d4d, false);
    view.setUint16(2, 42, le);
    view.setUint32(4, 8, le);
    view.setUint16(8, 7, le);
    const entries = [
      [256, 3, 1, 2],
      [257, 3, 1, 2],
      [258, 3, 3, bpsOffset],
      [259, 3, 1, compression],
      [262, 3, 1, 2],
      [273, 4, 1, pixelOffset],
      [277, 3, 1, 3],
    ];
    entries.forEach(([tag, type, count, value], i) => {
      const p = 10 + i * 12;
      view.setUint16(p, tag, le);
      view.setUint16(p + 2, type, le);
      view.setUint32(p + 4, count, le);
      const inline = (type === 3 ? 2 : 4) * count <= 4;
      if (inline && type === 3) view.setUint16(p + 8, value, le);
      else view.setUint32(p + 8, value, le);
    });
    for (let i = 0; i < 3; i++) view.setUint16(bpsOffset + i * 2, 8, le);
    const pixels = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
    pixels.forEach((v, i) => view.setUint8(pixelOffset + i, v));
    return buf;
  }

  for (const littleEndian of [true, false]) {
    const dom = makeDom();
    const label = littleEndian ? "little-endian" : "big-endian";
    const { result, captured } = withFakeCanvas(dom, () =>
      dom.window.decodeTiffPreview(buildTiff({ littleEndian }), 600),
    );
    assert(result !== null, `${label} uncompressed RGB TIFF decodes`);
    assert(captured && captured.width === 2 && captured.height === 2, `${label} preview keeps 2x2 at scale 1`);
    const px = captured ? [...captured.data] : [];
    assert(
      JSON.stringify(px.slice(0, 8)) === JSON.stringify([255, 0, 0, 255, 0, 255, 0, 255]),
      `${label} first row is red then green (channel order preserved)`,
    );
    assert(
      JSON.stringify(px.slice(8, 16)) === JSON.stringify([0, 0, 255, 255, 255, 255, 0, 255]),
      `${label} second strip row addressed correctly (blue then yellow)`,
    );
  }

  {
    const dom = makeDom();
    // LZW. Real ScanGear output is uncompressed, but a compressed file must
    // refuse rather than render the compressed bytes as pixels.
    const { result } = withFakeCanvas(dom, () => dom.window.decodeTiffPreview(buildTiff({ compression: 5 }), 600));
    assert(result === null, "a compressed TIFF is refused, not rendered as garbage");

    const notTiff = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0, 0, 0]).buffer;
    assert(dom.window.decodeTiffPreview(notTiff, 600) === null, "a non-TIFF buffer decodes to null");
    assert(dom.window.decodeTiffPreview(new ArrayBuffer(4), 600) === null, "a truncated buffer decodes to null");
  }

  // --- Test 1t: a slot whose file can't be previewed names the file ---
  {
    const dom = makeDom();
    const doc = dom.window.document;
    const pick = async (slotName, fileName) => {
      const slotEl = doc.querySelector(`.slot[data-slot="${slotName}"]`);
      const input = slotEl.querySelector("input[type=file]");
      const file = new dom.window.File([new Uint8Array(2048)], fileName, { type: "image/tiff" });
      Object.defineProperty(input, "files", { value: [file], configurable: true });
      input.dispatchEvent(new dom.window.Event("change"));
      await new Promise((r) => setTimeout(r, 0));
      return slotEl;
    };
    const slot = await pick("front", "IMG_0001.tif");
    const placeholder = slot.querySelector(".slot-placeholder");
    const thumb = slot.querySelector(".thumb");
    assert(thumb.hidden === true, "no broken <img> when the file can't be previewed");
    assert(thumb.hasAttribute("src") === false, "unpreviewable file leaves no dangling src");
    assert(placeholder.hidden === false, "placeholder shown instead");
    assert(placeholder.textContent.includes("IMG_0001.tif"), "placeholder names the chosen file");
    await pick("back", "IMG_0002.tif");
    assert(
      doc.getElementById("submit-btn").disabled === false,
      "files are still staged for upload despite having no preview",
    );

    dom.window.resetApp();
    assert(placeholder.textContent === "Choose file", "reset restores the placeholder's own text");
  }

  // --- Test 1u: manual centering boundaries ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const tolerances = {
      front: [
        { grade: 10, max_ratio: 55 },
        { grade: 9, max_ratio: 60 },
        { grade: 8, max_ratio: 65 },
        { grade: 7, max_ratio: 70 },
      ],
      back: [{ grade: 10, max_ratio: 75 }, { grade: 9, max_ratio: 90 }],
      canonical_width_px: 1500,
      canonical_height_px: 2100,
    };
    let posted = null;
    const dom = makeDom({
      fetch: async (url, options) => {
        if (url === "/api/centering-tolerances") {
          return { ok: true, status: 200, json: async () => tolerances, text: async () => "" };
        }
        if (options && options.method === "POST") {
          posted = { url, body: JSON.parse(options.body) };
          return { ok: true, status: 200, json: async () => ({ report: data.report, images: {} }), text: async () => "" };
        }
        return { ok: true, status: 200, json: async () => ({ reports: [] }), text: async () => "" };
      },
    });
    const images = Object.assign({}, data.images, { front_aligned: "data:image/png;base64,AAAA" });
    dom.window.handleReport(data.report, images, "a".repeat(32));
    await new Promise((r) => setTimeout(r, 0));

    const doc = dom.window.document;
    const panel = doc.querySelector('.centering-adjust[data-side="front"]');
    assert(panel !== null, "front centering card offers a manual adjust panel");
    assert(panel.hidden === true, "adjust panel starts closed");
    assert(panel.querySelectorAll(".adjust-line").length === 8, "eight lines: a card edge and a border boundary per side");
    assert(panel.querySelectorAll('.adjust-line[data-kind="outer"]').length === 4, "four card-edge lines");
    assert(panel.querySelectorAll('.adjust-line[data-kind="inner"]').length === 4, "four border-boundary lines");

    // Opening portals the panel to <body>, so hold the card before clicking.
    const centeringCard = panel.closest(".centering-card");
    const toggle = centeringCard.querySelector("[data-adjust-open]");
    toggle.dispatchEvent(new dom.window.Event("click"));
    await new Promise((r) => setTimeout(r, 0));
    assert(panel.hidden === false, "adjust panel opens on the toggle");
    assert(
      centeringCard.querySelector(".overlay-img").hidden === true,
      "the burned-in overlay hides while adjusting, so two sets of lines never show at once",
    );
    assert(
      panel.parentNode === doc.body,
      "the panel moves to <body> to go fullscreen — its own card has backdrop-filter, " +
        "which would otherwise trap a position:fixed overlay inside it",
    );
    assert(
      panel.classList.contains("centering-adjust--fullscreen"),
      "adjusting opens fullscreen: placing a boundary to the pixel is the whole job",
    );

    // Lines are seeded from the detector's own measurement, as a percentage
    // of the canonical warp, so they land on the boundaries it found.
    const detected = data.report.centering.front.horizontal;
    const leftLine = panel.querySelector('.adjust-line[data-edge="left"][data-kind="inner"]');
    const outerLeft = panel.querySelector('.adjust-line[data-edge="left"][data-kind="outer"]');
    const expectedLeft = `${(100 * detected.side_a_px) / 1500}%`;
    assert(leftLine.style.left === expectedLeft, `border line seeded from the detected boundary (${leftLine.style.left})`);
    assert(outerLeft.style.left === "0%", "card edge starts on the image edge — the warp is defined by the detected corners");
    assert(
      outerLeft.classList.contains("adjust-line--flush"),
      "a card edge resting on the image edge is drawn quietly, since it carries no information",
    );

    const readout = panel.querySelector(".adjust-readout").textContent;
    assert(readout.includes(detected.ratio), `readout opens on the detected ratio (${readout})`);

    // Keyboard nudge: one axis only, and the readout follows.
    leftLine.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "ArrowUp", bubbles: true }));
    assert(leftLine.style.left === expectedLeft, "a vertical key does nothing to a vertical line (wrong axis)");
    leftLine.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "ArrowRight", shiftKey: true, bubbles: true }));
    assert(leftLine.style.left !== expectedLeft, "shift+arrow moves the line inward");

    // The card edge can't be pushed past its own border boundary.
    for (let i = 0; i < 400; i++) {
      outerLeft.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "ArrowRight", shiftKey: true, bubbles: true }));
    }
    assert(
      parseFloat(outerLeft.style.left) <= parseFloat(leftLine.style.left) + 1e-9,
      "the card edge cannot be dragged past its own border boundary",
    );
    for (let i = 0; i < 400; i++) {
      outerLeft.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "ArrowLeft", shiftKey: true, bubbles: true }));
    }
    assert(outerLeft.style.left === "0%", "the card edge cannot be dragged off the image");

    panel.querySelector("[data-adjust-save]").dispatchEvent(new dom.window.Event("click"));
    await new Promise((r) => setTimeout(r, 0));
    assert(posted !== null, "save posts the correction");
    assert(posted.url === `/api/report/${"a".repeat(32)}/centering`, "posted to the report's own centering endpoint");
    assert(
      Object.keys(posted.body).length === 1 && posted.body.front,
      "only the adjusted side is sent, so the other keeps its detected values",
    );
    assert(
      ["left", "right", "top", "bottom"].every((k) => typeof posted.body.front.borders[k] === "number"),
      "all four border widths sent as numbers",
    );
    assert(
      ["left", "right", "top", "bottom"].every((k) => typeof posted.body.front.edges[k] === "number"),
      "card-edge insets sent alongside them",
    );
    assert(
      posted.body.front.borders.left === Math.round(detected.side_a_px) + 10,
      `sent width reflects the nudge (${posted.body.front.borders.left})`,
    );
    assert(posted.body.front.edges.left === 0, "an untouched card edge is sent as a zero inset");
  }

  // --- Test 1v: a report with no id can't be corrected ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const dom = makeDom();
    dom.window.handleReport(data.report, data.images, null);
    await new Promise((r) => setTimeout(r, 0));
    const panel = dom.window.document.querySelector('.centering-adjust[data-side="front"]');
    assert(panel === null || panel.hidden === true, "no open adjust panel without a saved report to write back to");
  }

  // --- Test 1w: hand-set centering is labelled as such ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const report = JSON.parse(JSON.stringify(data.report));
    report.centering.front.manual = true;
    const dom = makeDom();
    dom.window.handleReport(report, data.images, "b".repeat(32));
    await new Promise((r) => setTimeout(r, 0));
    const cards = [...dom.window.document.querySelectorAll(".centering-card")];
    assert(cards[0].querySelector(".manual-chip") !== null, "a hand-set side is marked in the report");
    assert(cards[1].querySelector(".manual-chip") === null, "an untouched side is not");
  }

  // --- Test 1x: PSA's 5% front leeway is shown, never silently applied ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const report = JSON.parse(JSON.stringify(data.report));
    Object.assign(report.centering.front.horizontal, {
      grade: 8,
      strict_grade: 7,
      leeway_applied: true,
      variation_px: 0,
    });
    const dom = makeDom();
    dom.window.handleReport(report, data.images, "c".repeat(32));
    await new Promise((r) => setTimeout(r, 0));
    const front = dom.window.document.querySelectorAll(".centering-card")[0];
    const chip = front.querySelector(".leeway-chip");
    assert(chip !== null, "a grade that used the leeway says so");
    assert(chip.title.includes("grade 7"), `the strict-table grade is in reach (${chip.title}）`.slice(0, 200));
  }

  // --- Test 1y: a card inside the table gets no leeway chip ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const report = JSON.parse(JSON.stringify(data.report));
    Object.assign(report.centering.front.horizontal, { leeway_applied: false, variation_px: 0 });
    Object.assign(report.centering.front.vertical, { leeway_applied: false, variation_px: 0 });
    const dom = makeDom();
    dom.window.handleReport(report, data.images, "c".repeat(32));
    await new Promise((r) => setTimeout(r, 0));
    assert(
      dom.window.document.querySelectorAll(".centering-card")[0].querySelector(".leeway-chip") === null,
      "no leeway chip when the published table alone gave the grade",
    );
  }

  // --- Test 1z: a border that wanders down the side is called out ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const steady = JSON.parse(JSON.stringify(data.report));
    steady.centering.front.horizontal.variation_px = 1.2;
    const wanders = JSON.parse(JSON.stringify(data.report));
    wanders.centering.front.horizontal.variation_px = 24;

    const a = makeDom();
    a.window.handleReport(steady, data.images, "c".repeat(32));
    const b = makeDom();
    b.window.handleReport(wanders, data.images, "c".repeat(32));
    await new Promise((r) => setTimeout(r, 0));
    const text = (dom) => dom.window.document.querySelectorAll(".centering-card")[0].textContent;
    assert(!text(a).includes("along the side"), "a straight cut gets no note");
    assert(text(b).includes("±24px along the side"), "a skewed cut is flagged with how far it wanders");
  }

  // --- Test 2a: both ratio conventions are available ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const report = JSON.parse(JSON.stringify(data.report));
    report.centering.front.horizontal.ratio = "35/65";
    report.centering.front.horizontal.ratio_conventional = "65/35";
    const dom = makeDom();
    dom.window.handleReport(report, data.images, "c".repeat(32));
    await new Promise((r) => setTimeout(r, 0));
    const cell = dom.window.document.querySelectorAll(".centering-card")[0].querySelector(".axis-row .mono");
    assert(cell.textContent === "35/65", "shown left/right, so the direction of the miscut survives");
    assert(cell.title.includes("65/35"), "PSA's larger-first convention is one hover away");
  }

  // --- Test 2b: every grading service's verdict on the same measurement ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const report = JSON.parse(JSON.stringify(data.report));
    report.centering.by_grader = {
      psa: { label: "PSA", front: 8, back: 10, grade: 8, source: "PSA published grading standards" },
      bgs: { label: "BGS", front: 7, back: 9, grade: 7, source: "third-party transcription" },
      tag: { label: "TAG", front: 8.5, back: 10, grade: 8.5, source: "third-party transcription" },
      sgc: { label: "SGC", front: 8, back: null, grade: 8, source: "no back tolerance published" },
    };
    const dom = makeDom();
    dom.window.handleReport(report, data.images, "c".repeat(32));
    await new Promise((r) => setTimeout(r, 0));
    const table = dom.window.document.querySelector(".grader-table");
    assert(table !== null, "the comparison table renders");
    assert(table.querySelectorAll("tbody tr").length === 4, "one row per grading service");
    assert(table.textContent.includes("8.5"), "half grades survive to the page");
    assert(table.textContent.includes("—"), "a service with no published back tolerance shows a dash, not a grade");
    assert(
      table.textContent.includes("third-party transcription"),
      "each table says where it came from — these are transcriptions, not primary sources",
    );
    const section = dom.window.document.querySelector(".grader-compare");
    assert(section.tagName === "DETAILS" && !section.open, "collapsed by default; PSA is still the report's grade");
  }

  // --- Test 2c: no comparison block when there's nothing to compare ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const dom = makeDom();
    dom.window.handleReport(data.report, data.images, "c".repeat(32));
    await new Promise((r) => setTimeout(r, 0));
    assert(
      dom.window.document.querySelector(".grader-table") === null,
      "an older report with no by_grader block renders without an empty table",
    );
  }

  // --- Test 2d: the loupe ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const tolerances = {
      front: [{ grade: 10, max_ratio: 55 }, { grade: 9, max_ratio: 60 }, { grade: 8, max_ratio: 65 }],
      back: [{ grade: 10, max_ratio: 75 }],
      canonical_width_px: 1500,
      canonical_height_px: 2100,
    };
    const dom = makeDom({
      fetch: async (url) =>
        url === "/api/centering-tolerances"
          ? { ok: true, status: 200, json: async () => tolerances, text: async () => "" }
          : { ok: true, status: 200, json: async () => ({ reports: [] }), text: async () => "" },
    });
    const images = Object.assign({}, data.images, { front_aligned: "data:image/png;base64,AAAA" });
    dom.window.handleReport(data.report, images, "d".repeat(32));
    await new Promise((r) => setTimeout(r, 0));

    const panel = dom.window.document.querySelector('.centering-adjust[data-side="front"]');
    const loupe = panel.querySelector(".adjust-loupe");
    assert(loupe !== null, "the adjust panel has a magnifier");
    assert(loupe.hidden === true, "hidden until a line is being placed");

    panel.closest(".centering-card").querySelector("[data-adjust-open]").dispatchEvent(new dom.window.Event("click"));
    await new Promise((r) => setTimeout(r, 0));

    const leftLine = panel.querySelector('.adjust-line[data-edge="left"][data-kind="inner"]');
    leftLine.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    assert(loupe.hidden === false, "placing a line by keyboard raises the magnifier");
    assert(
      loupe.classList.contains("adjust-loupe--vertical"),
      "a left/right boundary gets the vertical crosshair",
    );
    assert(loupe.style.backgroundImage.includes("data:image/png"), "the magnifier samples the scan itself");

    const topLine = panel.querySelector('.adjust-line[data-edge="top"][data-kind="inner"]');
    topLine.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }));
    assert(
      !loupe.classList.contains("adjust-loupe--vertical"),
      "a top/bottom boundary gets the horizontal crosshair",
    );

    panel.querySelector("[data-adjust-cancel]").dispatchEvent(new dom.window.Event("click"));
    assert(loupe.hidden === true, "the magnifier goes away with the panel");
  }

  // --- Test 2e: fullscreen and grab-anywhere dragging ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const tolerances = {
      front: [{ grade: 10, max_ratio: 55 }, { grade: 9, max_ratio: 60 }, { grade: 8, max_ratio: 65 }],
      back: [{ grade: 10, max_ratio: 75 }],
      canonical_width_px: 1500,
      canonical_height_px: 2100,
    };
    const dom = makeDom({
      fetch: async (url) =>
        url === "/api/centering-tolerances"
          ? { ok: true, status: 200, json: async () => tolerances, text: async () => "" }
          : { ok: true, status: 200, json: async () => ({ reports: [] }), text: async () => "" },
    });
    const win = dom.window;
    win.requestAnimationFrame = (fn) => fn();
    const images = Object.assign({}, data.images, { front_aligned: "data:image/png;base64,AAAA" });
    win.handleReport(data.report, images, "e".repeat(32));
    await new Promise((r) => setTimeout(r, 0));

    const panel = win.document.querySelector('.centering-adjust[data-side="front"]');
    const card = panel.closest(".centering-card");
    card.querySelector("[data-adjust-open]").dispatchEvent(new win.Event("click"));
    await new Promise((r) => setTimeout(r, 0));

    const button = panel.querySelector("[data-fullscreen]");
    assert(button !== null, "the adjust toolbar offers fullscreen");
    assert(
      panel.classList.contains("centering-adjust--fullscreen"),
      "adjusting opens straight into fullscreen — placing a boundary to the pixel is the whole job",
    );
    assert(panel.parentNode === win.document.body, "and the panel is portalled out of its backdrop-filtered card");
    assert(win.document.body.classList.contains("adjust-fullscreen-open"), "the page behind it stops scrolling");
    assert(button.textContent === "Exit fullscreen", "the control says how to get back out");
    assert(
      panel.querySelector(".adjust-side").contains(panel.querySelector("[data-adjust-save]")),
      "Save comes into fullscreen with the card, rather than being locked out behind it",
    );

    button.dispatchEvent(new win.Event("click"));
    assert(!panel.classList.contains("centering-adjust--fullscreen"), "the control drops back to the inline panel");
    assert(card.contains(panel), "and the panel goes back exactly where it came from");
    assert(button.textContent === "Fullscreen", "the control offers the way back in");

    button.dispatchEvent(new win.Event("click"));
    panel.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    assert(!panel.classList.contains("centering-adjust--fullscreen"), "Escape leaves fullscreen");
    assert(!win.document.body.classList.contains("adjust-fullscreen-open"), "and gives the page its scroll back");

    // Closing the panel from inside fullscreen must not strand the page.
    button.dispatchEvent(new win.Event("click"));
    panel.querySelector("[data-adjust-cancel]").dispatchEvent(new win.Event("click"));
    assert(
      !win.document.body.classList.contains("adjust-fullscreen-open"),
      "cancelling out of fullscreen restores page scrolling",
    );
    assert(card.contains(panel), "and puts the panel back in its card");

    // Grab-anywhere: a press on the viewport that isn't on a line still
    // picks one up, because a 1px line is a poor thing to have to hit.
    card.querySelector("[data-adjust-open]").dispatchEvent(new win.Event("click"));
    await new Promise((r) => setTimeout(r, 0));
    const viewport = panel.querySelector(".adjust-viewport");
    const press = new win.Event("pointerdown", { bubbles: true });
    Object.assign(press, { pointerId: 1, clientX: 0, clientY: 0, button: 0, preventDefault() {} });
    viewport.setPointerCapture = () => {};
    viewport.releasePointerCapture = () => {};
    viewport.hasPointerCapture = () => true;
    viewport.dispatchEvent(press);
    const grabbed = panel.querySelector(".adjust-line--dragging");
    assert(grabbed !== null, "pressing near a line grabs it without having to hit the line itself");
    const grabbedEdge = grabbed.dataset.edge;
    const before = grabbed.style[grabbedEdge];
    assert(panel.querySelector(".adjust-loupe").hidden === false, "and raises the magnifier immediately");

    const move = new win.Event("pointermove", { bubbles: true });
    Object.assign(move, { pointerId: 1, clientX: 40, clientY: 0, preventDefault() {} });
    viewport.dispatchEvent(move);
    const up = new win.Event("pointerup", { bubbles: true });
    Object.assign(up, { pointerId: 1, clientX: 40, clientY: 0 });
    viewport.dispatchEvent(up);
    assert(panel.querySelector(".adjust-line--dragging") === null, "releasing ends the drag");
    assert(panel.querySelector(".adjust-loupe").hidden === true, "and puts the magnifier away");
    assert(grabbed.style[grabbedEdge] !== before, "the line that was grabbed is the line that moved");
  }

  // --- Test 2f: zoom doesn't thicken the lines, and retires the loupe ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const tolerances = {
      front: [{ grade: 10, max_ratio: 55 }, { grade: 9, max_ratio: 60 }],
      back: [{ grade: 10, max_ratio: 75 }],
      canonical_width_px: 1500,
      canonical_height_px: 2100,
    };
    const dom = makeDom({
      fetch: async (url) =>
        url === "/api/centering-tolerances"
          ? { ok: true, status: 200, json: async () => tolerances, text: async () => "" }
          : { ok: true, status: 200, json: async () => ({ reports: [] }), text: async () => "" },
    });
    const win = dom.window;
    win.requestAnimationFrame = (fn) => fn();
    const images = Object.assign({}, data.images, { front_aligned: "data:image/png;base64,AAAA" });
    win.handleReport(data.report, images, "f".repeat(32));
    await new Promise((r) => setTimeout(r, 0));

    const panel = win.document.querySelector('.centering-adjust[data-side="front"]');
    panel.closest(".centering-card").querySelector("[data-adjust-open]").dispatchEvent(new win.Event("click"));
    await new Promise((r) => setTimeout(r, 0));

    const hairlineAt = () => panel.style.getPropertyValue("--adjust-hairline");
    assert(hairlineAt() === "1px", `unzoomed, a line is one pixel (${hairlineAt()})`);

    const zoomIn = panel.querySelector('[data-zoom="in"]');
    for (let i = 0; i < 4; i++) zoomIn.dispatchEvent(new win.Event("click"));
    const zoom = parseFloat(panel.querySelector(".adjust-zoom").textContent);
    assert(zoom > 4, `zoomed in (${zoom}x)`);
    // The whole canvas is transform-scaled, so a fixed 1px line would render
    // as a zoom-thick bar right over the boundary being read.
    const expected = 1 / zoom;
    const actual = parseFloat(hairlineAt());
    assert(
      Math.abs(actual - expected) < 0.01,
      `the line counter-scales to stay one screen pixel (${actual} vs ${expected.toFixed(3)})`,
    );
    assert(parseFloat(panel.style.getPropertyValue("--adjust-grab")) < 14, "and so does the grab area");

    // Past a few times magnification the viewport shows more than the loupe.
    const loupe = panel.querySelector(".adjust-loupe");
    const line = panel.querySelector('.adjust-line[data-edge="left"][data-kind="inner"]');
    line.dispatchEvent(new win.KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    assert(loupe.hidden === true, "no loupe once the viewport itself is magnified past it");

    panel.querySelector('[data-zoom="reset"]').dispatchEvent(new win.Event("click"));
    assert(hairlineAt() === "1px", "Fit puts the line back to one pixel");
    assert(panel.querySelector(".adjust-zoom").textContent === "1.0×", "and the zoom back to 1x");

    line.dispatchEvent(new win.KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    assert(loupe.hidden === false, "the loupe comes back at a zoom where it helps");
  }

  // --- Test 2g: Fit contains the card, rather than fitting height only ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const tolerances = {
      front: [{ grade: 10, max_ratio: 55 }],
      back: [{ grade: 10, max_ratio: 75 }],
      canonical_width_px: 1500,
      canonical_height_px: 2100,
    };
    const dom = makeDom({
      fetch: async (url) =>
        url === "/api/centering-tolerances"
          ? { ok: true, status: 200, json: async () => tolerances, text: async () => "" }
          : { ok: true, status: 200, json: async () => ({ reports: [] }), text: async () => "" },
    });
    const win = dom.window;
    win.requestAnimationFrame = (fn) => fn();
    const images = Object.assign({}, data.images, { front_aligned: "data:image/png;base64,AAAA" });
    win.handleReport(data.report, images, "g".repeat(32));
    await new Promise((r) => setTimeout(r, 0));

    const panel = win.document.querySelector('.centering-adjust[data-side="front"]');
    const viewport = panel.querySelector(".adjust-viewport");
    // A viewport taller than the card's aspect allows: fitting the height
    // alone would make the card wider than the screen and crop it.
    Object.defineProperty(viewport, "clientWidth", { value: 400, configurable: true });
    Object.defineProperty(viewport, "clientHeight", { value: 1200, configurable: true });

    panel.closest(".centering-card").querySelector("[data-adjust-open]").dispatchEvent(new win.Event("click"));
    await new Promise((r) => setTimeout(r, 0));

    const img = panel.querySelector(".adjust-img");
    const width = parseFloat(img.style.width);
    const height = parseFloat(img.style.height);
    assert(width <= 400 + 0.5, `the card is contained by the viewport width (${width} <= 400)`);
    assert(height <= 1200 + 0.5, `and by its height (${height} <= 1200)`);
    assert(
      Math.abs(width / height - 1500 / 2100) < 0.001,
      "at the warp's own aspect, so the lines still land on the pixels they mark",
    );
    assert(width > 390, "and as large as that allows, rather than merely small enough");
  }

  // --- Test 2h: an n/a sub-grade says why ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const report = JSON.parse(JSON.stringify(data.report));
    report.subgrades = {
      front: { centering: 10, corners: 10, edges: null, surface: null },
      back: { centering: 9, corners: 10, edges: 9, surface: null },
    };
    report.corners_edges.front.edges_reason = "this crop isn't uniform border (0.39 against a 0.55 floor)";
    report.corners_edges.front.corners_reason = null;
    for (const side of ["front", "back"]) {
      report.surface[side].grade = null;
      report.surface[side].reason = "Scan the side four times, rotating the card 90 degrees each time.";
    }
    const dom = makeDom();
    dom.window.handleReport(report, data.images, "h".repeat(32));
    await new Promise((r) => setTimeout(r, 0));

    const tiles = [...dom.window.document.querySelectorAll(".subgrade-tile")];
    const find = (label) => tiles.find((t) => t.querySelector(".subgrade-label").textContent === label);
    const edges = find("Edges");
    assert(edges.textContent.includes("n/a"), "a refused sub-grade still reads n/a");
    assert(edges.title.includes("uniform border"), "and carries the reason it was refused");
    assert(
      edges.classList.contains("subgrade-tile--explained"),
      "marked as having an explanation — an unexplained n/a looks identical without it",
    );

    const surface = find("Surface");
    assert(
      surface.title.includes("rotating the card 90"),
      "surface says what capture would make it gradeable, not just that it isn't",
    );

    const corners = find("Corners");
    assert(corners.title === "", "a sub-grade that measured carries no explanation");
    assert(!corners.classList.contains("subgrade-tile--explained"), "and no marker");
  }

  // --- Test 2i: an older report with no reasons still renders ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const report = JSON.parse(JSON.stringify(data.report));
    // An older report: sub-grades refused, but written before any reason was
    // recorded next to the measurement.
    report.subgrades = { front: { centering: 10, corners: null, edges: null, surface: null }, back: {} };
    for (const side of ["front", "back"]) {
      delete report.corners_edges[side].corners_reason;
      delete report.corners_edges[side].edges_reason;
      delete report.surface[side].reason;
    }
    const dom = makeDom();
    dom.window.handleReport(report, data.images, "i".repeat(32));
    await new Promise((r) => setTimeout(r, 0));
    const tiles = [...dom.window.document.querySelectorAll(".subgrade-tile")];
    assert(tiles.length > 0, "the matrix renders without a corners_edges or surface block");
    assert(
      tiles.every((t) => !t.classList.contains("subgrade-tile--explained")),
      "and claims no explanations it doesn't have",
    );
  }

  // --- Test 2j: the turn direction reaches the server ---
  {
    const dom = makeDom();
    const doc = dom.window.document;
    const select = doc.getElementById("rotation-input");
    assert(select !== null, "the scanner fields offer a turn direction");
    assert(select.value === "ccw", "defaults to counter-clockwise, which is how these scans are taken");

    let sent = null;
    dom.window.fetch = async (url, options) => {
      if (options && options.body && options.body.get) sent = options.body;
      // The app polls the job after submitting; end that poll rather than
      // leaving it to fire into a later test with no report to render.
      if (String(url).startsWith("/api/job/")) {
        return { ok: true, status: 200, json: async () => ({ status: "error", message: "stubbed" }), text: async () => "" };
      }
      return { ok: true, status: 200, json: async () => ({ job_id: "x" }), text: async () => "" };
    };
    // Go through the real picker so the app's own state is what's submitted.
    for (const slotName of ["front", "back"]) {
      const input = doc.querySelector(`.slot[data-slot="${slotName}"] input[type=file]`);
      const file = new dom.window.File([new Uint8Array(8)], `${slotName}.png`, { type: "image/png" });
      Object.defineProperty(input, "files", { value: [file], configurable: true });
      input.dispatchEvent(new dom.window.Event("change"));
    }
    await new Promise((r) => setTimeout(r, 0));

    doc.getElementById("submit-btn").dispatchEvent(new dom.window.Event("click"));
    await new Promise((r) => setTimeout(r, 0));
    assert(sent !== null, "submitting posts a form");
    assert(sent.get("rotation") === "ccw", `the turn direction is included (${sent && sent.get("rotation")})`);

    select.value = "cw";
    doc.getElementById("submit-btn").disabled = false;
    doc.getElementById("submit-btn").dispatchEvent(new dom.window.Event("click"));
    await new Promise((r) => setTimeout(r, 0));
    assert(sent.get("rotation") === "cw", "and follows the control");
  }

  // --- Test 2k: rotation scans don't survive into the next card ---
  {
    const dom = makeDom();
    const doc = dom.window.document;
    const input = doc.getElementById("photometric-front-input");
    const toggle = doc.getElementById("scanner-toggle");

    // jsdom doesn't wire `files` to `value` the way a browser does, where
    // setting value to "" empties the selection. Emulate that, since it is
    // exactly the behaviour the fix relies on.
    let backing = [0, 1, 2, 3].map(
      (i) => new dom.window.File([new Uint8Array(4)], `rot${i}.png`, { type: "image/png" }),
    );
    Object.defineProperty(input, "files", { get: () => backing, configurable: true });
    Object.defineProperty(input, "value", {
      get: () => (backing.length ? "C:\\fakepath\\rot0.png" : ""),
      set: (v) => {
        if (v === "") backing = [];
      },
      configurable: true,
    });
    input.dispatchEvent(new dom.window.Event("change"));

    assert(
      toggle.textContent.includes("4 front"),
      `the collapsed toggle says what's attached (${toggle.textContent})`,
    );

    dom.window.resetApp();
    assert(input.value === "", "grading another card clears the rotation scans");
    assert(
      !toggle.textContent.includes("4 front"),
      `and the toggle stops claiming them (${toggle.textContent})`,
    );
  }

  // --- Test 2l: scanner settings survive, card-specific files don't ---
  {
    const dom = makeDom();
    const doc = dom.window.document;
    doc.getElementById("dpi-input").value = "1200";
    doc.getElementById("rotation-input").value = "ccw";
    const input = doc.getElementById("photometric-front-input");
    let backing = [new dom.window.File([new Uint8Array(4)], "r.png", { type: "image/png" })];
    Object.defineProperty(input, "files", { get: () => backing, configurable: true });
    Object.defineProperty(input, "value", {
      get: () => (backing.length ? "C:\\fakepath\\r.png" : ""),
      set: (v) => {
        if (v === "") backing = [];
      },
      configurable: true,
    });
    input.dispatchEvent(new dom.window.Event("change"));

    dom.window.resetApp();
    assert(doc.getElementById("dpi-input").value === "1200", "the scan DPI describes the scanner, so it stays");
    assert(doc.getElementById("rotation-input").value === "ccw", "so does the turn direction");
    assert(input.value === "", "the scans describe the card, so they go");
  }

  // --- Test 2m: an empty section says nothing about attachments ---
  {
    const dom = makeDom();
    const toggle = dom.window.document.getElementById("scanner-toggle");
    assert(
      !toggle.textContent.includes("rotation scans"),
      `a fresh page claims no attachments (${toggle.textContent})`,
    );
  }

  // --- Test 2n: dimensions that disagree with themselves give no verdict ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const withDim = (extra) => {
      const report = JSON.parse(JSON.stringify(data.report));
      report.dimensions = Object.assign(
        {
          measurable: true, width_mm: 62.1, height_mm: 86.7,
          nominal_width_mm: 63.0, nominal_height_mm: 88.0,
          squareness_deviation_deg: 0.2, note: "note",
        },
        extra,
      );
      return report;
    };

    const disagreeing = makeDom();
    disagreeing.window.handleReport(
      withDim({ within_tolerance: null, spread_mm: 1.83, sample_count: 4, note: "they disagree by 1.83mm" }),
      data.images, "j".repeat(32),
    );
    await new Promise((r) => setTimeout(r, 0));
    const doc = disagreeing.window.document;
    const banner = [...doc.querySelectorAll(".banner")].find((b) => b.textContent.includes("disagree"));
    assert(banner !== null && banner !== undefined, "the disagreement is stated");
    assert(
      banner.classList.contains("banner-warn"),
      "no verdict is neither a pass nor a failure, and must not be coloured as either",
    );
    // Scoped to the section: the offline-export feature inlines app.js's own
    // source into the page, so document.body.textContent contains the script.
    const dimensionSection = (d) =>
      [...d.querySelectorAll(".report-section")].find((s) => s.querySelector("h2")?.textContent === "Dimensions");
    assert(
      dimensionSection(doc).textContent.includes("4 scans agree to 1.83 mm"),
      "the spread is shown alongside the size",
    );

    const agreeing = makeDom();
    agreeing.window.handleReport(
      withDim({ within_tolerance: true, spread_mm: 0.12, sample_count: 4, note: "within 0.75mm of nominal" }),
      data.images, "k".repeat(32),
    );
    await new Promise((r) => setTimeout(r, 0));
    const ok = [...agreeing.window.document.querySelectorAll(".banner")].find((b) => b.textContent.includes("within 0.75mm"));
    assert(ok.classList.contains("banner-ok"), "scans that agree still give a verdict");

    const single = makeDom();
    single.window.handleReport(
      withDim({ within_tolerance: false, spread_mm: null, sample_count: 1, note: "width off by -0.90mm — miscut or trimmed" }),
      data.images, "l".repeat(32),
    );
    await new Promise((r) => setTimeout(r, 0));
    const singleSection = [...single.window.document.querySelectorAll(".report-section")].find(
      (s) => s.querySelector("h2")?.textContent === "Dimensions",
    );
    assert(
      !singleSection.textContent.includes("scans agree"),
      "one scan has nothing to compare itself against, and claims nothing",
    );
  }

  // --- Test 2o: the dings are marked on the card, not only cropped out ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const report = JSON.parse(JSON.stringify(data.report));
    report.dings = [
      { attribute: "corners", side: "front", label: "top-left corner", grade: 7,
        detail: "1.2% whitening", image_key: "front_corner_top_left", box: [0.005, 0.005, 0.07, 0.05] },
      { attribute: "edges", side: "back", label: "left edge", grade: 8,
        detail: "0.8% whitening", image_key: "back_edge_left", box: [0.005, 0.06, 0.04, 0.88] },
      { attribute: "surface", side: "front", label: "surface", grade: 6,
        detail: "whole card", image_key: "front_card_vision", box: null },
    ];
    const images = Object.assign({}, data.images, {
      front_aligned: "data:image/png;base64,AAAA",
      back_aligned: "data:image/png;base64,BBBB",
    });
    const dom = makeDom();
    dom.window.handleReport(report, images, "m".repeat(32));
    await new Promise((r) => setTimeout(r, 0));

    const doc = dom.window.document;
    const map = doc.querySelector(".ding-map");
    assert(map !== null, "the dings section shows the card with its defects marked");
    assert(map.querySelectorAll(".ding-map-side").length === 2, "one panel per side that has a located ding");
    assert(map.querySelectorAll(".ding-mark").length === 2, "a mark per located ding — the whole-card one has no place to be");

    const mark = map.querySelector(".ding-mark");
    assert(parseFloat(mark.style.left) === 0.5, `positioned from its own box (${mark.style.left})`);
    assert(parseFloat(mark.style.width) === 7, `sized from its own box (${mark.style.width})`);
    assert(mark.textContent.includes("g7"), "labelled with the grade it scored");
    assert(mark.title.includes("top-left corner"), "and says which defect it is on hover");

    assert(
      doc.querySelectorAll(".ding-gallery .ding-card").length === 3,
      "the crops are still there — the map says where, the crops say what",
    );
  }

  // --- Test 2p: nothing to locate means no map ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const report = JSON.parse(JSON.stringify(data.report));
    report.dings = [
      { attribute: "surface", side: "front", label: "surface", grade: 6, detail: "whole card", box: null },
    ];
    const dom = makeDom();
    dom.window.handleReport(report, data.images, "n".repeat(32));
    await new Promise((r) => setTimeout(r, 0));
    assert(
      dom.window.document.querySelector(".ding-map") === null,
      "no empty card outline when nothing has a location",
    );
  }

  // --- Test 2q: an older report without boxes still renders ---
  {
    const data = JSON.parse(fs.readFileSync(path.join(__dirname, "success_result.json"), "utf8"));
    const dom = makeDom();
    dom.window.handleReport(data.report, data.images, "o".repeat(32));
    await new Promise((r) => setTimeout(r, 0));
    assert(dom.window.document.getElementById("report-view").hidden === false, "renders without box data");
  }

  // --- Test 2r: scans are not re-encoded on the way to the server ---
  {
    const dom = makeDom();
    const win = dom.window;
    const file = (name, type) => new win.File([new Uint8Array(64)], name, { type });

    for (const [name, type] of [["scan.png", "image/png"], ["scan.tif", "image/tiff"], ["scan.tiff", "image/tiff"]]) {
      const original = file(name, type);
      const processed = await win.processImageFile(original);
      assert(processed === original, `${name} is handed through untouched, not resampled and JPEG'd`);
    }

    // A file whose type the browser didn't fill in is judged by its name.
    const byName = file("scan.PNG", "");
    assert((await win.processImageFile(byName)) === byName, "extension decides when the MIME type is missing");

    // Camera captures still get the EXIF fix and downscale they exist for.
    // createImageBitmap is stubbed out in jsdom, so this falls through its
    // catch and returns the original — what matters is that it tried.
    let attempted = false;
    win.createImageBitmap = async () => {
      attempted = true;
      throw new Error("no canvas in jsdom");
    };
    await win.processImageFile(file("photo.jpg", "image/jpeg"));
    assert(attempted, "a phone photo still goes through orientation and downscale");

    attempted = false;
    await win.processImageFile(file("scan.png", "image/png"));
    assert(!attempted, "a scan does not");
  }

  console.log(failures === 0 ? "\nALL TESTS PASSED" : `\n${failures} TEST(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
})();
