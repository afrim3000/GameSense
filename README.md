# GameSense — Automated Video Game Recognition & Hashtag Suggestion

Web prototype for the project synopsis *"Automated Video Game Recognition and
Hashtag Suggestion for Live Streaming Platforms Using Image Classification"*
(SE334 — AI Lab).

Implements the exact system flow from the synopsis (Section 7.2):
1. User opens the web page and uploads a screenshot.
2. The image is sent to the FastAPI backend via an API request.
3. The trained CNN (EfficientNet-B0, ONNX) predicts the game class.
4. The predicted game is matched against a hardcoded hashtag dictionary.
5. The web page displays the predicted game and its hashtags.

## Project structure

```
gamesense/
├── backend/
│   ├── main.py             # FastAPI app: /api/predict, /api/health, serves frontend
│   ├── download_model.py   # auto-downloads the model from Hugging Face on startup
│   ├── requirements.txt
│   └── model/               # empty until first run — downloaded files land here
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

## 2. Run

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On startup, if `backend/model/efficientnet_b0_gameplay.onnx` and
`efficientnet_b0_gameplay.onnx.data` aren't already present, the server
automatically downloads them from
[huggingface.co/nihal4/Game_Detection](https://huggingface.co/nihal4/Game_Detection)
into `backend/model/`. This only happens once — later restarts skip it since
the files are already on disk. Watch the terminal for `[download_model]` log
lines the first time you run it.

If your environment has no internet access (or the download fails for any
other reason), the server still starts — `GET /api/health` will report
`"model_loaded": false` and `/api/predict` returns a 503 with instructions —
just download the two files manually from the link above and drop them into
`backend/model/`, then restart.

To trigger the download on its own without starting the server:

```bash
cd backend
python download_model.py
```

Open **http://localhost:8000** — the same server serves both the API and the
frontend, so there's nothing else to start.

## How predictions work

- Preprocessing mirrors the notebook's `eval_transform` exactly: resize to
  **180×320 (H×W)**, scale to `[0,1]`, normalize with ImageNet mean/std
  (`[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`), `HWC → CHW`.
- The ONNX graph's raw output is treated as logits (the trained model's head
  is a plain `nn.Linear`, no softmax) — softmax is applied in the backend
  before ranking classes.
- Input/output tensor names are read dynamically from the ONNX graph
  (`session.get_inputs()[0].name` / `get_outputs()[0].name`), so it doesn't
  matter what your exporter named them.

## Classes & hashtags

The 10 classes (from `CONFIG["class_names"]` in the notebook):
`Among Us, Apex Legends, Fortnite, Forza Horizon, Free Fire, Genshin Impact,
God of War, Minecraft, Roblox, Terraria`

Each maps to a fixed 5-hashtag set in `HASHTAGS` in `backend/main.py` — edit
that dictionary directly to change the suggested tags.

## Scope note

Per the synopsis (Section 7.3), this is the self-contained demo of the core
classification engine — it does not integrate with Twitch/YouTube/Facebook
APIs, which is explicitly out of scope for the course project.
