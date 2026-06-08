// Pearlscape Explorer — loads a binary PLY point cloud and renders every bead as
// a glass-bead point sprite in a single THREE.Points draw call. Colour is
// recomputed on the GPU from a per-bead field value, so palettes switch instantly.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { parsePLY } from "./plyparse.js";
import { vertexShader, fragmentShader, MAX_PALETTE } from "./beads.glsl.js";

const FOV = 50;
const DEFAULT_BEAD_DIAMETER = 6.0; // mm, fallback if the PLY has no comment

// --- Scene -----------------------------------------------------------------
const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111317);

const camera = new THREE.PerspectiveCamera(FOV, 1, 1, 1_000_000);
camera.up.set(0, 0, 1); // Rhino is Z-up

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

const material = new THREE.ShaderMaterial({
  uniforms: {
    beadDiameter: { value: DEFAULT_BEAD_DIAMETER },
    sizeScale: { value: 1.0 },
    palette: { value: Array.from({ length: MAX_PALETTE }, () => new THREE.Vector3(1, 1, 1)) },
    paletteSize: { value: 1 },
    ditherAmt: { value: 1.0 },
    useField: { value: false },
  },
  vertexShader,
  fragmentShader,
  transparent: true,
  depthTest: true,
  depthWrite: true,
});

let points = null;
let palettesData = null; // { default, palettes:[{name, colors}] }

// --- Sizing ----------------------------------------------------------------
function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const bufH = renderer.getDrawingBufferSize(new THREE.Vector2()).y;
  material.uniforms.sizeScale.value = bufH / (2 * Math.tan((FOV * Math.PI) / 180 / 2));
}
window.addEventListener("resize", resize);

// --- Palettes --------------------------------------------------------------
function applyPalette(entry) {
  const m = Math.min(entry.colors.length, MAX_PALETTE);
  for (let i = 0; i < m; i++) {
    const [r, g, b] = entry.colors[i];
    material.uniforms.palette.value[i].set(r / 255, g / 255, b / 255);
  }
  material.uniforms.paletteSize.value = m;
}

function applySelectedPalette() {
  if (!palettesData) return;
  const entry = palettesData.palettes[parseInt(paletteSelect.value, 10)] || palettesData.palettes[0];
  applyPalette(entry);
}

// Palette switching needs both the JSON and a field-carrying cloud.
function refreshPaletteState() {
  const canSwitch = !!palettesData && material.uniforms.useField.value;
  paletteSelect.disabled = !canSwitch;
  if (canSwitch) applySelectedPalette();
}

async function loadPalettes() {
  try {
    const res = await fetch("data/palettes.json", { cache: "no-cache" });
    if (!res.ok) return;
    palettesData = await res.json();
    paletteSelect.innerHTML = "";
    palettesData.palettes.forEach((p, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = p.name;
      if (p.name === palettesData.default) opt.selected = true;
      paletteSelect.appendChild(opt);
    });
  } catch {
    /* no palettes.json; dropdown stays disabled */
  }
}

// --- Loading ---------------------------------------------------------------
function frameBounds(geometry) {
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1000;
  const dist = (maxDim / (2 * Math.tan((FOV * Math.PI) / 180 / 2))) * 1.4;
  controls.target.copy(center);
  // Oblique angle so the Z-up stack reads as depth.
  camera.position.set(center.x + dist, center.y - dist, center.z + dist * 0.6);
  camera.near = Math.max(1, dist / 1000);
  camera.far = dist * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

function loadFromBuffer(arrayBuffer) {
  let ply;
  try {
    ply = parsePLY(arrayBuffer);
  } catch (err) {
    setStatus(`Failed to parse PLY: ${err.message}`, true);
    return;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(ply.position, 3));
  if (ply.color) geometry.setAttribute("color", new THREE.BufferAttribute(ply.color, 3, true));
  if (ply.field) geometry.setAttribute("field", new THREE.BufferAttribute(ply.field, 1));
  if (ply.dither) geometry.setAttribute("ditherRand", new THREE.BufferAttribute(ply.dither, 1));

  const hasField = !!(ply.field && ply.dither);
  material.uniforms.useField.value = hasField;
  material.uniforms.beadDiameter.value = ply.beadDiameter || DEFAULT_BEAD_DIAMETER;
  material.uniforms.ditherAmt.value = ply.colorDither != null ? ply.colorDither : 1.0;
  sizeSlider.value = String(material.uniforms.beadDiameter.value);
  sizeVal.textContent = parseFloat(sizeSlider.value).toFixed(1);

  if (points) {
    scene.remove(points);
    points.geometry.dispose();
  }
  points = new THREE.Points(geometry, material);
  points.frustumCulled = false; // one cloud; skip per-frame bbox cull
  scene.add(points);
  frameBounds(geometry);
  refreshPaletteState();

  setStatus(`Loaded ${ply.count.toLocaleString()} beads (bead Ø ${material.uniforms.beadDiameter.value} mm).`);
  countEl.textContent = ply.count.toLocaleString();
}

async function tryAutoLoad() {
  try {
    const res = await fetch("data/pearlscape.ply", { cache: "no-cache" });
    if (!res.ok) return;
    setStatus("Loading data/pearlscape.ply …");
    loadFromBuffer(await res.arrayBuffer());
  } catch {
    /* no auto file; wait for the user */
  }
}

// --- HUD -------------------------------------------------------------------
const fileInput = document.getElementById("file");
const sizeSlider = document.getElementById("size");
const sizeVal = document.getElementById("sizeVal");
const paletteSelect = document.getElementById("palette");
const shotBtn = document.getElementById("shot");
const countEl = document.getElementById("count");
const fpsEl = document.getElementById("fps");
const statusEl = document.getElementById("status");
const hud = document.getElementById("hud");
const toggleBtn = document.getElementById("toggle");

toggleBtn.addEventListener("click", () => {
  const collapsed = hud.classList.toggle("collapsed");
  toggleBtn.textContent = collapsed ? "+" : "–";
  toggleBtn.title = collapsed ? "Expand menu" : "Minimize menu";
});

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.style.color = isError ? "#ff7a7a" : "#9aa0a6";
}

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  setStatus(`Loading ${file.name} …`);
  loadFromBuffer(await file.arrayBuffer());
});

sizeSlider.addEventListener("input", () => {
  const v = parseFloat(sizeSlider.value);
  material.uniforms.beadDiameter.value = v;
  sizeVal.textContent = v.toFixed(1);
});

paletteSelect.addEventListener("change", applySelectedPalette);

function timestamp() {
  // Local time, filesystem-safe (no colons): YYYY-MM-DD_HH-MM-SS.
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_` +
    `${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

shotBtn.addEventListener("click", () => {
  // Render synchronously, then read the buffer before it's cleared. The HUD is
  // separate DOM, so the canvas holds only the model view.
  renderer.render(scene, camera);
  const url = canvas.toDataURL("image/png");
  const a = document.createElement("a");
  a.href = url;
  a.download = `pearlscape_${timestamp()}.png`;
  a.click();
});

// Drag-and-drop a .ply anywhere on the window.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", async (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (!file) return;
  setStatus(`Loading ${file.name} …`);
  loadFromBuffer(await file.arrayBuffer());
});

// --- Render loop -----------------------------------------------------------
let frames = 0;
let fpsT0 = performance.now();
function animate(now) {
  requestAnimationFrame(animate);
  frames++;
  const elapsed = now - fpsT0;
  if (elapsed >= 1000) {
    fpsEl.textContent = String(Math.round((frames * 1000) / elapsed));
    frames = 0;
    fpsT0 = now;
  }
  controls.update();
  renderer.render(scene, camera);
}

resize();
sizeVal.textContent = parseFloat(sizeSlider.value).toFixed(1);
loadPalettes().then(tryAutoLoad);
requestAnimationFrame(animate);
