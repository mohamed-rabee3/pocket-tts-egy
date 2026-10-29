# ruff: noqa: ANN001, ANN201, BLE001
"""Gender of every sampled clip: audeering wav2vec2 age/gender head + Praat F0 median.

Writes gender_pred.parquet: ds, key, p_female, p_male, p_child, age, f0_med.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

ROOT = "/workspace/egy"
MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"


class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)  # female, male, child
        self.post_init()

    def forward(self, x):
        h = self.wav2vec2(x)[0].mean(dim=1)
        return self.age(h), torch.softmax(self.gender(h), dim=1)


def path_of(ds, key):
    ext = ".flac" if ds in ("masri", "mostafa", "dahab", "halasamir") else ".wav"
    return f"{ROOT}/sample/{ds}/{key}{ext}"


def load16k(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    return AF.resample(wav, sr, 16000) if sr != 16000 else wav


def f0_median(path):
    try:
        wav, sr = sf.read(path, dtype="float64", always_2d=True)
        pitch = parselmouth.Sound(wav.mean(1), sr).to_pitch(pitch_floor=60, pitch_ceiling=500)
        f = pitch.selected_array["frequency"]
        f = f[f > 0]
        return float(np.median(f)) if len(f) > 20 else float("nan")
    except Exception:
        return float("nan")


def main():
    s = pd.read_parquet(f"{ROOT}/gender_sample.parquet")
    s["path"] = [path_of(d, k) for d, k in zip(s.ds, s.key)]
    s = s[s.path.map(os.path.exists)].reset_index(drop=True)
    print(f"{len(s)} clips on disk", file=sys.stderr)

    with ProcessPoolExecutor(64) as ex:
        f0 = list(ex.map(f0_median, s.path, chunksize=64))

    model = AgeGenderModel.from_pretrained(MODEL).cuda().eval().to(torch.bfloat16)
    out = []
    with torch.no_grad():
        for i, p in enumerate(s.path):
            x = load16k(p)[: 16000 * 12]
            x = (x - x.mean()) / (x.std() + 1e-7)
            age, g = model(x[None].cuda().to(torch.bfloat16))
            out.append([*g[0].float().tolist(), float(age[0, 0]) * 100])
            if i % 5000 == 0:
                print(i, file=sys.stderr)
    out = np.array(out)
    s["p_female"], s["p_male"], s["p_child"], s["age"] = out.T
    s["f0_med"] = f0
    s.drop(columns="path").to_parquet(f"{ROOT}/gender_pred.parquet")


if __name__ == "__main__":
    main()
