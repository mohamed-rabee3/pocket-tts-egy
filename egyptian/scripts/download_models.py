"""Download the Egyptian Pocket TTS and Tashkeel weights into models/.

Reads the token from the HF_TOKEN environment variable or from --token.
The token is not stored in the repo.

    $env:HF_TOKEN = "hf_..."
    uv run --with huggingface_hub python egyptian/scripts/download_models.py
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

REPOS = (
    ("egyptian-pocket-tts", "Rabe3/egyptian-pocket-tts"),
    ("tashkeel-v4-egyptian", "Rabe3/Tashkeel-v4-Egyptian"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to the HF_TOKEN environment variable.",
    )
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("Set HF_TOKEN or pass --token. Do not commit the token.")

    for folder, repo_id in REPOS:
        dest = MODELS / folder
        dest.mkdir(parents=True, exist_ok=True)
        print(f"downloading {repo_id} -> {dest}", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=dest, token=args.token)
        print(f"saved {dest}", flush=True)


if __name__ == "__main__":
    main()
