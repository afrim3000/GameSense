"""
Downloads the trained model files from Hugging Face Hub into backend/model/
if they aren't already present. Safe to import/run repeatedly — existing
files are left untouched.

Can also be run standalone:
    python download_model.py
"""
import sys
import urllib.request
import urllib.error
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_URLS = {
    "efficientnet_b0_gameplay.onnx":
        "https://huggingface.co/nihal4/Game_Detection/resolve/main/efficientnet_b0_gameplay.onnx",
    "efficientnet_b0_gameplay.onnx.data":
        "https://huggingface.co/nihal4/Game_Detection/resolve/main/efficientnet_b0_gameplay.onnx.data",
}


def _download(url: str, dest: Path) -> None:
    """Stream a URL to disk, reporting progress, writing to a .part file first
    so a crash mid-download never leaves a corrupt file at the final path."""
    tmp_path = dest.with_name(dest.name + ".part")
    last_pct_shown = -1

    def _hook(block_num, block_size, total_size):
        nonlocal last_pct_shown
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            if pct != last_pct_shown and pct % 10 == 0:
                last_pct_shown = pct
                print(f"    {dest.name}: {pct}%", flush=True)

    urllib.request.urlretrieve(url, tmp_path, _hook)
    tmp_path.rename(dest)


def ensure_model_files() -> bool:
    """Downloads any missing model files. Returns True if the model is ready
    to load (i.e. all files exist) afterward, False otherwise."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_ready = True

    for filename, url in MODEL_URLS.items():
        dest = MODEL_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            continue

        print(f"[download_model] Fetching {filename} ...", flush=True)
        try:
            _download(url, dest)
            print(f"[download_model] Saved {dest}", flush=True)
        except (urllib.error.URLError, OSError) as e:
            all_ready = False
            print(
                f"[download_model] Could not download {filename}: {e}\n"
                f"                  Download it manually from {url}\n"
                f"                  and place it at {dest}",
                flush=True,
            )

    return all_ready


if __name__ == "__main__":
    ok = ensure_model_files()
    sys.exit(0 if ok else 1)
