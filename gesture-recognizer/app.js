// Live Hand Gesture Recognizer
// Webcam frame -> MediaPipe hand landmarks (21 points per hand) -> gesture classifier -> UI

import {
  GestureRecognizer,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision_bundle.mjs";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

// The 7 gestures the pretrained model can recognise
const GESTURES = {
  Thumb_Up:    { emoji: "👍", label: "Thumbs up" },
  Thumb_Down:  { emoji: "👎", label: "Thumbs down" },
  Victory:     { emoji: "✌️", label: "Peace" },
  Open_Palm:   { emoji: "🖐️", label: "Open palm" },
  Closed_Fist: { emoji: "✊", label: "Fist" },
  Pointing_Up: { emoji: "☝️", label: "Pointing up" },
  ILoveYou:    { emoji: "🤟", label: "I love you" },
};

const HAND_COLORS = { Left: "#6aa8ff", Right: "#7cf0b5" };
const STABLE_FRAMES = 8; // a gesture must be held this many frames before it's logged

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const video = $("webcam");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const stage = $("stage");
const startBtn = $("startBtn");
const statusEl = $("status");
const statusText = $("statusText");
const fpsEl = $("fps");
const handCountEl = $("handCount");
const handsEl = $("hands");
const guideEl = $("guide");
const logEl = $("log");
const thresholdInput = $("threshold");
const thresholdValue = $("thresholdValue");
const skeletonToggle = $("showSkeleton");
const reactionsEl = $("reactions");

// ---------- State ----------
let recognizer = null;
let drawingUtils = null;
let stream = null;
let running = false;
let rafId = null;
let lastVideoTime = -1;
let threshold = Number(thresholdInput.value) / 100;

let candidate = null;      // gesture currently being "held"
let candidateFrames = 0;
let lastLogged = null;     // last gesture added to the log
let emptyFrames = 0;

let fpsFrames = 0;
let fpsLastTime = performance.now();

// ---------- Setup ----------
buildGuide();

thresholdInput.addEventListener("input", () => {
  threshold = Number(thresholdInput.value) / 100;
  thresholdValue.textContent = `${thresholdInput.value}%`;
});

$("clearLog").addEventListener("click", () => (logEl.innerHTML = ""));
startBtn.addEventListener("click", () => (running ? stopCamera() : startCamera()));

loadModel();

async function loadModel() {
  setStatus("loading", "Loading model…");
  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    try {
      recognizer = await createRecognizer(vision, "GPU");
    } catch (gpuErr) {
      console.warn("GPU delegate failed, falling back to CPU", gpuErr);
      recognizer = await createRecognizer(vision, "CPU");
    }
    drawingUtils = new DrawingUtils(ctx);
    setStatus("ready", "Model ready");
    startBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("error", "Couldn't load the model. Check your connection and refresh.");
  }
}

function createRecognizer(vision, delegate) {
  return GestureRecognizer.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

// ---------- Camera ----------
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("error", "This browser doesn't support camera access");
    return;
  }
  startBtn.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise((resolve) => {
      if (video.readyState >= 2) resolve();
      else video.addEventListener("loadeddata", resolve, { once: true });
    });

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;

    running = true;
    lastVideoTime = -1;
    stage.classList.add("live");
    startBtn.textContent = "Stop camera";
    startBtn.classList.add("stop");
    setStatus("live", "Live");
    loop();
  } catch (err) {
    console.error(err);
    const msg =
      err.name === "NotAllowedError"
        ? "Camera permission denied. Allow it in your browser settings."
        : err.name === "NotFoundError"
        ? "No camera found"
        : "Couldn't start the camera";
    setStatus("error", msg);
  } finally {
    startBtn.disabled = false;
  }
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(rafId);
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  stage.classList.remove("live");
  startBtn.textContent = "Start camera";
  startBtn.classList.remove("stop");
  renderHands([]);
  highlightGuide(new Set());
  fpsEl.textContent = "— FPS";
  handCountEl.textContent = "0 hands";
  setStatus("ready", "Camera off");
}

// ---------- Main loop ----------
function loop() {
  if (!running) return;
  const now = performance.now();

  // Only run the model when the webcam has a new frame
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = recognizer.recognizeForVideo(video, now);
    handleResults(results);
    tickFps(now);
  }
  rafId = requestAnimationFrame(loop);
}

function handleResults(results) {
  const hands = results.landmarks.map((landmarks, i) => {
    const top = results.gestures[i]?.[0];
    const side = results.handedness?.[i]?.[0]?.categoryName ?? "Hand";
    const known = top && GESTURES[top.categoryName] && top.score >= threshold;
    return {
      landmarks,
      side,
      gesture: known ? top.categoryName : null,
      score: top?.score ?? 0,
    };
  });

  drawSkeletons(hands);
  renderHands(hands);
  highlightGuide(new Set(hands.map((h) => h.gesture).filter(Boolean)));
  handCountEl.textContent = `${hands.length} hand${hands.length === 1 ? "" : "s"}`;

  // Pick the most confident recognised gesture this frame
  const best = hands
    .filter((h) => h.gesture)
    .sort((a, b) => b.score - a.score)[0];
  trackStability(best?.gesture ?? null);
}

// ---------- Drawing ----------
function drawSkeletons(hands) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!skeletonToggle.checked) return;

  for (const hand of hands) {
    const color = HAND_COLORS[hand.side] ?? "#ffffff";
    drawingUtils.drawConnectors(hand.landmarks, GestureRecognizer.HAND_CONNECTIONS, {
      color,
      lineWidth: 4,
    });
    drawingUtils.drawLandmarks(hand.landmarks, {
      color: "#ffffff",
      fillColor: color,
      lineWidth: 1,
      radius: 4,
    });
  }
}

// ---------- UI ----------
function renderHands(hands) {
  if (hands.length === 0) {
    handsEl.innerHTML = `<div class="hand-empty">No hands in view</div>`;
    return;
  }
  handsEl.innerHTML = hands
    .map((h) => {
      const g = h.gesture ? GESTURES[h.gesture] : null;
      const pct = Math.round(h.score * 100);
      const color = HAND_COLORS[h.side] ?? "#ffffff";
      return `
        <div class="hand ${g ? "" : "unknown"}" style="--hand-color:${color}">
          <div class="hand-emoji">${g ? g.emoji : "🤔"}</div>
          <div>
            <div class="hand-name">${g ? g.label : "No gesture"}</div>
            <div class="hand-side">${h.side} hand</div>
            <div class="bar"><span style="width:${g ? pct : 0}%"></span></div>
            <div class="hand-score mono">${g ? `${pct}% confident` : "Try one from the list below"}</div>
          </div>
        </div>`;
    })
    .join("");
}

function buildGuide() {
  guideEl.innerHTML = Object.entries(GESTURES)
    .map(
      ([key, g]) => `
      <div class="guide-item" data-gesture="${key}">
        <span class="e">${g.emoji}</span>
        <span class="l">${g.label}</span>
      </div>`
    )
    .join("");
}

function highlightGuide(active) {
  guideEl.querySelectorAll(".guide-item").forEach((el) => {
    el.classList.toggle("active", active.has(el.dataset.gesture));
  });
}

// A gesture only counts once it's held steadily, which stops the log flickering
function trackStability(gesture) {
  if (!gesture) {
    candidate = null;
    candidateFrames = 0;
    if (++emptyFrames > 10) lastLogged = null; // hand dropped, allow the same gesture again
    return;
  }
  emptyFrames = 0;

  if (gesture === candidate) {
    candidateFrames++;
  } else {
    candidate = gesture;
    candidateFrames = 1;
  }

  if (candidateFrames === STABLE_FRAMES && gesture !== lastLogged) {
    lastLogged = gesture;
    addLog(gesture);
    burst(GESTURES[gesture].emoji);
  }
}

function addLog(gesture) {
  const g = GESTURES[gesture];
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const li = document.createElement("li");
  li.innerHTML = `<span>${g.emoji} ${g.label}</span><span class="t">${time}</span>`;
  logEl.prepend(li);
  while (logEl.children.length > 30) logEl.lastChild.remove();
}

function burst(emoji) {
  for (let i = 0; i < 6; i++) {
    const el = document.createElement("span");
    el.className = "reaction";
    el.textContent = emoji;
    el.style.left = `${10 + Math.random() * 80}%`;
    el.style.animationDelay = `${i * 0.08}s`;
    el.style.setProperty("--rot", `${(Math.random() - 0.5) * 60}deg`);
    el.addEventListener("animationend", () => el.remove());
    reactionsEl.appendChild(el);
  }
}

function tickFps(now) {
  fpsFrames++;
  if (now - fpsLastTime >= 500) {
    fpsEl.textContent = `${Math.round((fpsFrames * 1000) / (now - fpsLastTime))} FPS`;
    fpsFrames = 0;
    fpsLastTime = now;
  }
}

function setStatus(state, text) {
  statusEl.dataset.state = state;
  statusText.textContent = text;
}
