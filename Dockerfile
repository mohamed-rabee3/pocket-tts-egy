FROM ghcr.io/astral-sh/uv:debian

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# Dependencies first so code and model changes reuse this layer. No dev group (training
# only); the audio extra brings soundfile, needed to read the FLAC voice.
COPY ./pyproject.toml ./uv.lock ./README.md ./.python-version ./
RUN uv sync --frozen --no-dev --extra audio --no-install-project

COPY ./pocket_tts ./pocket_tts
RUN uv sync --frozen --no-dev --extra audio

# Egyptian 6-layer model (from egyptian/scripts/download_models.py, run on the host before
# building). The config's relative paths resolve against /app, the project root.
COPY ./egyptian/configs/egy_inference_6l.yaml ./egyptian/configs/
COPY ./models/egyptian-pocket-tts/student-6l/model.safetensors ./models/egyptian-pocket-tts/student-6l/
COPY ./models/egyptian-pocket-tts/tokenizer.json ./models/egyptian-pocket-tts/mark_lexicon.json.gz ./models/egyptian-pocket-tts/
COPY ./models/egyptian-pocket-tts/voices ./models/egyptian-pocket-tts/voices

# Everything the server loads is in the image: never reach Hugging Face at runtime.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
ENTRYPOINT ["uv", "run", "--no-sync", "--no-dev", "--extra", "audio", "pocket-tts"]
CMD ["serve", "--config", "egyptian/configs/egy_inference_6l.yaml", "--default-voice", "models/egyptian-pocket-tts/voices/sample_voice_female.flac", "--host", "0.0.0.0", "--port", "8000"]
