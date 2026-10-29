# ruff: noqa: ANN001, ANN201
"""WavLM-SV x-vectors for every sampled clip, then per-dataset voice clustering.

Writes spk_emb.npy (+ spk_emb_index.parquet) and voices.parquet: ds, key, voice (cluster id).
"""

import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy.cluster.hierarchy import fcluster, linkage
from transformers import AutoFeatureExtractor, WavLMForXVector

ROOT = "/workspace/egy"
SIM_THRESHOLD = 0.80  # average-linkage cosine similarity that merges two clips into one voice


def path_of(ds, key):
    ext = ".flac" if ds in ("masri", "mostafa", "dahab", "halasamir") else ".wav"
    return f"{ROOT}/sample/{ds}/{key}{ext}"


def main():
    s = pd.read_parquet(f"{ROOT}/gender_sample.parquet")
    s["path"] = [path_of(d, k) for d, k in zip(s.ds, s.key)]
    s = s[s.path.map(os.path.exists)].reset_index(drop=True)
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()
    embs = []
    with torch.no_grad():
        for i, p in enumerate(s.path):
            wav, sr = sf.read(p, dtype="float32", always_2d=True)
            x = AF.resample(torch.from_numpy(wav.mean(1)), sr, 16000)[: 16000 * 10]
            inp = fe(x.numpy(), sampling_rate=16000, return_tensors="pt").to("cuda")
            e = model(**inp).embeddings
            embs.append(torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy())
            if i % 10000 == 0:
                print(i, file=sys.stderr)
    embs = np.stack(embs)
    np.save(f"{ROOT}/spk_emb.npy", embs)
    s.drop(columns="path").to_parquet(f"{ROOT}/spk_emb_index.parquet")

    voices = np.empty(len(s), dtype=object)
    for ds, idx in s.groupby("ds").indices.items():
        if len(idx) == 1:
            voices[idx] = f"{ds}:0"
            continue
        z = linkage(embs[idx], method="average", metric="cosine")
        lab = fcluster(z, t=1 - SIM_THRESHOLD, criterion="distance")
        voices[idx] = [f"{ds}:{v}" for v in lab]
    s["voice"] = voices
    s.drop(columns="path").to_parquet(f"{ROOT}/voices.parquet")


if __name__ == "__main__":
    main()
