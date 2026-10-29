# ruff: noqa: ANN001, ANN201, C408, SIM115
"""Join each dataset's transcripts.jais.jsonl to durations/speakers -> inventory.parquet"""

import csv
import glob
import json
import os
import sys

import pandas as pd

ROOT = "/workspace/egy/meta"
EXCLUDE = {"drama3_synthetic"}


def key(path):
    p = path.replace("\\", "/")
    for pre in ("flac/", "wavs/", "clips/", "flac_split/"):
        p = p.removeprefix(pre)
    return os.path.splitext(p)[0]


def jl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


rows = []
for ds in sorted(os.listdir(ROOT)):
    if ds in EXCLUDE:
        continue
    tj = f"{ROOT}/{ds}/transcripts.jais.jsonl"
    if not os.path.exists(tj):
        print(f"[skip] {ds}: no transcripts.jais.jsonl", file=sys.stderr)
        continue
    meta = {}
    for mf in [
        f"{ROOT}/{ds}/metadata.jsonl",
        *sorted(glob.glob(f"{ROOT}/{ds}/parts/*/metadata.jsonl")),
    ]:
        if os.path.exists(mf):
            for r in jl(mf):
                if "audio_file" in r:
                    meta[key(r["audio_file"])] = (
                        r.get("duration"),
                        r.get("speaker"),
                        r.get("channel"),
                        r.get("video_id"),
                    )
    tk = f"{ROOT}/{ds}/transcripts.tashkeel.jsonl"
    if os.path.exists(tk):
        for r in jl(tk):
            k = key(r.get("flac") or r["uid"])
            if k not in meta:
                meta[k] = (r.get("duration"), r.get("speaker"), None, None)
    mc = f"{ROOT}/{ds}/metadata.csv"
    if os.path.exists(mc):
        for r in csv.DictReader(open(mc)):
            k = key(r["file_name"])
            if k not in meta:
                meta[k] = (
                    float(r["duration"]),
                    r.get("channel"),
                    r.get("channel"),
                    r.get("video_id"),
                )
    n = miss = 0
    for r in jl(tj):
        n += 1
        k = key(r.get("audio_file") or r.get("uid") or r.get("flac"))
        m = meta.get(k)
        if m is None:
            miss += 1
            m = (None, None, None, None)
        rows.append(
            dict(
                ds=ds,
                key=k,
                text=r.get("text", ""),
                refused=r.get("refused"),
                truncated=r.get("truncated"),
                language=r.get("language"),
                dur=m[0],
                speaker=m[1],
                channel=m[2],
                video_id=m[3],
            )
        )
    print(f"{ds:45s} rows={n:7d} missing_dur={miss}", file=sys.stderr)

df = pd.DataFrame(rows)
df.to_parquet("/workspace/egy/inventory.parquet")
