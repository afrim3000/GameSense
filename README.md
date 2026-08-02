# GameSense — Automated Video Game Recognition & Hashtag Suggestion

A web app that identifies which video game is being played from a gameplay screenshot and suggests
matching hashtags — built as the working prototype for the project synopsis *"Automated Video Game
Recognition and Hashtag Suggestion for Live Streaming Platforms Using Image Classification"* (SE334 — AI Lab).

**🔗 Live demo:** [gamesense-h456.onrender.com](https://gamesense-h456.onrender.com/)

<img width="1427" alt="GameSense upload screen" src="https://raw.githubusercontent.com/nihal4/GameSense/main/ss/Screenshot%202026-07-26%20at%202.24.07%E2%80%AFPM.png" />

<img width="1427" alt="GameSense prediction result" src="https://raw.githubusercontent.com/nihal4/GameSense/main/ss/Screenshot%202026-07-26%20at%202.24.15%E2%80%AFPM.png" />

## Overview

GameSense wraps a fine-tuned EfficientNet-B0 game classifier — trained on gameplay screenshots and
exported to ONNX — behind a FastAPI backend and a lightweight static frontend. A user uploads a
screenshot, the backend predicts which of 10 games it's from, and the result is matched against a
hardcoded hashtag dictionary so the page can display ready-to-use tags for that game.

For model architecture, training data, and evaluation details, see the
[model card](https://huggingface.co/nihal4/Game_Detection)

## Features

- Upload-and-classify web UI — drag in a screenshot, get an instant prediction
- ONNX Runtime inference — no GPU required to serve predictions
- Auto-downloading model weights — no manual setup needed on first run
- Hashtag suggestions per predicted game, easy to edit/extend
- Simple FastAPI backend, easy to self-host or extend

## Repo Structure

```
GameSense/
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── model/                  # empty until first run — downloaded files land here
│   └── .gitkeep
├── ss/
│   ├── Screenshot 2026-07-26 at 2.24.07 PM.png
│   └── Screenshot 2026-07-26 at 2.24.15 PM.png
├── LICENSE
├── README.md
├── download_model.py       # auto-downloads the model from Hugging Face on startup
├── main.py                 # FastAPI app: /api/predict, /api/health, serves frontend
└── requirements.txt
```

## Running Locally

1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/nihal4/GameSense.git
   cd GameSense
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python main.py
   ```
   (equivalently: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`)

5. **Open the app**
   Go to **http://localhost:8000** — the same server serves both the API and the frontend, so there's
   nothing else to start.

### Model download

You don't need to run anything extra to get the model — the first time you run `python main.py`, if
`model/efficientnet_b0_gameplay.onnx` and `model/efficientnet_b0_gameplay.onnx.data` aren't already
present, the server automatically downloads them from the
[Hugging Face model repo](https://huggingface.co/nihal4/Game_Detection) into `model/`.
This only happens once — later restarts skip it since the files are already on disk. Watch the terminal
for `[download_model]` log lines the first time you run it.

If your environment has no internet access (or the download fails for any other reason), the server
still starts — `GET /api/health` will report `"model_loaded": false` and `/api/predict` returns a 503
with instructions — just download the two files manually from the link above and drop them into
`model/`, then restart.

## How Predictions Work

- Preprocessing mirrors the training notebook's `eval_transform` exactly: resize to **180×320 (H×W)**,
  scale to `[0,1]`, normalize with ImageNet mean/std (`[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`),
  `HWC → CHW`.
- The ONNX graph's raw output is treated as logits (the trained model's head is a plain `nn.Linear`, no
  softmax) — softmax is applied in the backend before ranking classes.
- Input/output tensor names are read dynamically from the ONNX graph
  (`session.get_inputs()[0].name` / `get_outputs()[0].name`), so it doesn't matter what your exporter
  named them.

## Classes & Hashtags

The 10 supported classes:
`Among Us, Apex Legends, Fortnite, Forza Horizon, Free Fire, Genshin Impact, God of War, Minecraft,
Roblox, Terraria`

Each maps to a fixed 5-hashtag set in `HASHTAGS` in `main.py` — edit that dictionary directly to
change the suggested tags.

## Scope Note

Per the synopsis (Section 7.3), this is the self-contained demo of the core classification engine — it
does not integrate with Twitch/YouTube/Facebook APIs, which is explicitly out of scope for the course
project.

## License

This project is licensed under the MIT License — see [`LICENSE`](./LICENSE) for details.

## Authors & Contributors

- Afrim Hossen Khan
