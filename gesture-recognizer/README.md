# 👍 Live Hand Gesture Recognizer

Real-time hand gesture recognition that runs entirely in the browser. Point your webcam at your hand, show a thumbs up, and it knows.

**[▶ Try it live](https://YOUR-USERNAME.github.io/gesture-recognizer/)**

<!-- Record a short GIF of yourself using it and drop it here: -->
<!-- ![demo](demo.gif) -->

## Features

- **Live webcam recognition** at 30+ FPS on most laptops
- **7 gestures**: 👍 thumbs up, 👎 thumbs down, ✌️ peace, 🖐️ open palm, ✊ fist, ☝️ pointing up, 🤟 I love you
- **Two hands at once**, each labelled left or right with its own confidence score
- **Hand skeleton overlay** showing the 21 tracked landmarks per hand
- **Adjustable confidence threshold** to trade sensitivity for accuracy
- **Stability filter** so a gesture only counts after it's held for a few frames (no flickering)
- **Gesture log** and emoji reactions when a gesture is recognised
- **Private by design**: no server, no uploads. Every frame is processed on your device

## How it works

```
Webcam frame
   │
Hand detector -> finds hands in the frame
   │
Landmark model -> 21 (x, y, z) keypoints per hand: wrist, knuckles, fingertips
   │
Gesture classifier -> small neural network that maps the 21 keypoints to a gesture + confidence
   │
Stability filter + UI -> only show/log a gesture once it's held for 8 frames above the threshold
```

The model is Google's [MediaPipe Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer), compiled to WebAssembly and run with GPU acceleration (falls back to CPU if the GPU isn't available).

Working from landmarks instead of raw pixels is what makes it fast and robust: the classifier only sees 21 points, so lighting, skin tone and background matter much less than they would for a model looking at the whole image.

## Tech stack

- Vanilla JavaScript (ES modules), HTML, CSS, with no build step
- [`@mediapipe/tasks-vision`](https://www.npmjs.com/package/@mediapipe/tasks-vision) for hand tracking and gesture classification
- WebRTC `getUserMedia` for the webcam, Canvas 2D for the overlay

## Run it locally

Browsers only allow camera access on `https://` or `localhost`, so opening `index.html` directly as a file won't work. Start any local server in the project folder:

```bash
# Python
python -m http.server 8000

# or Node
npx serve .
```

Then open http://localhost:8000 and click **Start camera**.

## Deploy to GitHub Pages

1. Push this folder to a GitHub repo
2. Go to **Settings → Pages**
3. Under **Source**, pick **Deploy from a branch**, choose `main` and `/ (root)`, then **Save**
4. After a minute your site is live at `https://YOUR-USERNAME.github.io/REPO-NAME/`
5. Replace the "Try it live" link at the top of this README

## Project structure

```
├── index.html   # Layout: video stage, controls, side panel
├── style.css    # Styling (dark theme, responsive down to phone width)
└── app.js       # Model loading, webcam loop, drawing, stability filter, UI
```

## What I learned

- **Detect first, then classify.** The model finds 21 points on each hand before guessing the gesture, so lighting and background barely matter.
- **Real-time output needs smoothing.** Predictions flicker frame to frame, so a gesture only counts once it's held for 8 frames.
- **ML can run fully in the browser.** No server is needed, video stays on your device, and hosting is free.

## License

MIT
