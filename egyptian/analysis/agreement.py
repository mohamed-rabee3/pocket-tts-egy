# ruff: noqa: ANN001, ANN201, SIM115
"""CER between jais plain text and the .diacritized file's skeleton, per dataset."""

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = "/workspace/egy/meta"
MARKS = re.compile(r"[ً-ْٰـ]")
PUN = re.compile(r"[^\w\s]|_")


def norm(t):
    t = MARKS.sub("", t)
    t = PUN.sub(" ", t)
    t = re.sub("[أإآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return " ".join(t.lower().split())


def key(r):
    p = r.get("audio_file") or r.get("uid") or r.get("flac")
    for pre in ("flac/", "wavs/"):
        p = p.removeprefix(pre)
    return os.path.splitext(p)[0]


def run(ds):
    a = f"{ROOT}/{ds}/transcripts.jais.jsonl"
    b = f"{ROOT}/{ds}/transcripts.jais.diacritized.jsonl"
    if not (os.path.exists(a) and os.path.exists(b)):
        return []
    plain = {key(r): r.get("text") or "" for r in map(json.loads, open(a))}
    out = []
    for line in open(b):
        r = json.loads(line)
        k = key(r)
        if k not in plain:
            continue
        x, y = norm(plain[k]), norm(r.get("text") or "")
        cer = Levenshtein.distance(x, y) / max(1, len(x))
        out.append((ds, k, cer, r.get("text", "")))
    return out


dss = [d for d in sorted(os.listdir(ROOT)) if d != "drama3_synthetic"]
with ProcessPoolExecutor(24) as ex:
    rows = [r for res in ex.map(run, dss) for r in res]
df = pd.DataFrame(rows, columns=["ds", "key", "cer_jd", "text_diac"])
df.to_parquet("/workspace/egy/agreement.parquet")
g = df.groupby("ds").cer_jd
print(
    pd.DataFrame(
        {
            "n": g.size(),
            "median": g.median(),
            "p90": g.quantile(0.9),
            ">0.10": g.apply(lambda x: (x > 0.1).mean()),
            ">0.25": g.apply(lambda x: (x > 0.25).mean()),
        }
    ).round(3)
)
