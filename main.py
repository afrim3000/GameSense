"""
GameSense backend — FastAPI service that serves the EfficientNet-B0 ONNX
gameplay classifier and a static frontend, as described in the project
synopsis (Section 6: Methodology / Workflow, Section 7: Implementation Process).

System flow:
  1. User uploads a screenshot on the web page.
  2. Image is POSTed to /api/predict.
  3. ONNX Runtime runs the EfficientNet-B0 model and returns class probabilities.
  4. The predicted game is matched against a hardcoded hashtag dictionary.
  5. JSON result (game + hashtags) is returned to the frontend for display.
"""
import asyncio
import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from download_model import ensure_model_files

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "efficientnet_b0_gameplay.onnx"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# ------------------------------------------------------------------
# Model / preprocessing config — must match the training notebook exactly
# (gameplay_efficientnet_b0_pytorch.ipynb, CONFIG["img_size"] = (180, 320))
# ------------------------------------------------------------------
IMG_H, IMG_W = 180, 320  # (Height, Width) — half of native 640x360, 16:9 preserved
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = [
    "Among Us", "Apex Legends", "Fortnite", "Forza Horizon", "Free Fire",
    "Genshin Impact", "God of War", "Minecraft", "Roblox", "Terraria",
]

# Fixed hashtag lookup table (Section 6, step 6: "Hashtag Mapping")
HASHTAGS = {
    "Among Us": ["#AmongUs", "#AmongUsGame", "#Impostor", "#SocialDeduction", "#Sus"],
    "Apex Legends": ["#ApexLegends", "#Apex", "#BattleRoyale", "#ApexLegendsGameplay", "#RespawnEntertainment"],
    "Fortnite": ["#Fortnite", "#FortniteBattleRoyale", "#FortniteGameplay", "#EpicGames", "#FortniteChapter5"],
    "Forza Horizon": ["#ForzaHorizon", "#Forza", "#RacingGame", "#OpenWorldRacing", "#XboxGamePass"],
    "Free Fire": ["#FreeFire", "#GarenaFreeFire", "#FreeFireBattleRoyale", "#MobileGaming", "#FreeFireIndia"],
    "Genshin Impact": ["#GenshinImpact", "#Genshin", "#HoYoverse", "#OpenWorldRPG", "#Teyvat"],
    "God of War": ["#GodOfWar", "#Kratos", "#PlayStation", "#ActionAdventure", "#GoW"],
    "Minecraft": ["#Minecraft", "#MinecraftGameplay", "#Mojang", "#Sandbox", "#MinecraftBuilds"],
    "Roblox": ["#Roblox", "#RobloxGame", "#RobloxDev", "#RobloxGameplay", "#RobloxCommunity"],
    "Terraria": ["#Terraria", "#TerrariaGameplay", "#Sandbox2D", "#IndieGame", "#TerrariaBuilds"],
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Auto-download the ONNX model + external data file from Hugging Face
    into backend/model/ on first run, if they aren't already there. Runs in
    a worker thread so it doesn't block the event loop while streaming."""
    ready = await asyncio.to_thread(ensure_model_files)
    if not ready:
        print(
            "[startup] Model files are not fully in place — /api/predict will "
            "return 503 until they are. See the log lines above for manual "
            "download instructions.",
            flush=True,
        )
    yield


app = FastAPI(title="GameSense — Gameplay Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session: ort.InferenceSession | None = None


def get_session() -> ort.InferenceSession:
    """Lazily load the ONNX Runtime session (avoids paying model-load cost on import)."""
    global _session
    if _session is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. "
                "It should auto-download on server startup — check the server "
                "logs. If it failed (e.g. no internet access), download it "
                "manually from https://huggingface.co/nihal4/Game_Detection "
                "and place efficientnet_b0_gameplay.onnx (and "
                "efficientnet_b0_gameplay.onnx.data) inside backend/model/."
            )
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def preprocess(image: Image.Image) -> np.ndarray:
    """Mirror the notebook's eval_transform: Resize -> ToTensor -> Normalize(ImageNet)."""
    image = image.convert("RGB")
    image = image.resize((IMG_W, IMG_H), Image.BILINEAR)  # PIL takes (W, H)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0).astype(np.float32)  # NCHW
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    try:
        session = get_session()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_tensor = preprocess(image)
    logits = session.run([output_name], {input_name: input_tensor})[0]
    probs = softmax(logits)[0]

    top_idx = int(np.argmax(probs))
    top_game = CLASS_NAMES[top_idx]

    ranked = sorted(
        [{"game": CLASS_NAMES[i], "confidence": float(probs[i])} for i in range(len(CLASS_NAMES))],
        key=lambda d: d["confidence"],
        reverse=True,
    )

    return {
        "game": top_game,
        "confidence": float(probs[top_idx]),
        "hashtags": HASHTAGS.get(top_game, []),
        "top_predictions": ranked[:3],
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


# Serve the frontend (index.html, style.css, script.js) as static files.
# Mounted last so it doesn't shadow the /api routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    # Lets `python main.py` work directly, in addition to the normal
    # `uvicorn main:app --reload` command from the README.
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
