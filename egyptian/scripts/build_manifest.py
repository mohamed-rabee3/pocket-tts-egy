"""Build the Egyptian training segments and manifests from the egy-tts datasets.

Text: transcripts.jais.jsonl (Jais-ASR, Egyptian, code-switched English kept in
Latin script), normalized with pocket_tts.egyptian_text.normalize_egyptian
(diacritics stripped, numbers kept as written).

Filters, per clip:
- no text, refused or truncated Jais output;
- Jais disagrees with the independent acoustic transcript (the
  .diacritized file, compared on the letter skeleton) by more than
  --max-cer, or --max-cer-latin for code-switched clips (the Arabic-only
  diacritizer writes English words in Arabic letters, which inflates the CER);
- implausible speaking rate (--min-cps / --max-cps characters per second).

Consecutive clips of one diarized speaker are joined into segments of up to
--max-seg-sec when nothing was cut between them (consecutive chunk index, gap
<= --max-gap-sec), as in saudi/scripts/build_segments.py: gaps are filled with
noise at the clips' own floor level and the per-clip loudness gain is undone.

Each segment gets a "weight" (training/dataloader/loader.py draws an entry
floor(w) + Bernoulli(frac(w)) times per epoch):
- a channel with more than --cap-hours is scaled down to --cap-hours per epoch
  (hala samir and mustafa hosny are ~690 h each, about half of the corpus);
- then the two genders are scaled to --female-share of the drawn hours.

Run per dataset (so raw audio can be deleted after rendering), align each
<ds>.segments.jsonl into <ds>.aligned.jsonl, then merge (egyptian/scripts/prepare_all.sh):
    python egyptian/scripts/build_manifest.py render --ds besohat --raw /data/egy/raw --out /data/egy/seg
    python egyptian/scripts/build_manifest.py merge --out /data/egy/seg

v2 (diacritized text, egyptian/scripts/prepare_v2.sh) reuses the rendered audio:
`retext` puts the diacritized Jais text (transcripts.jais.diacritized.jsonl, the
marks Jais-ASR heard) on the v1 segments, `gemini` adds gemini_synthetic (its own
QA'd transcripts, 4 labelled voices), both are re-aligned, then `merge`, `lexicon`
(ambiguous words, for the hint form of the tashkeel augmentation), `corpus`
(tokenizer text) and `latents` (the v1 Mimi latents, matched by audio path).
"""

import argparse
import collections
import difflib
import gzip
import json
import logging
import os
import random
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from pocket_tts.egyptian_text import (
    ARABIC_WORD,
    DIACRITICS,
    SUKUN,
    TAGS,
    normalize_egyptian,
    skeleton_key,
    split_edges,
    strip_marks,
    tashkeel_augment,
)

SR = 24000
PEAK = 10 ** (-1 / 20)
FLAC_DATASETS = {"masri", "mostafa", "dahab", "halasamir"}
EXCLUDED = {"drama3_synthetic"}
_LATIN = re.compile(r"[A-Za-z]")
_MARKS = re.compile(r"[ً-ْٰـ]")

logger = logging.getLogger("build_manifest")


def _key(path: str) -> str:
    for pre in ("flac/", "wavs/", "clips/", "flac_split/"):
        path = path.removeprefix(pre)
    return os.path.splitext(path)[0]


def _jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _skeleton(t: str) -> str:
    t = _MARKS.sub("", t)
    t = re.sub(r"[^\w\s]|_", " ", t)
    t = re.sub("[أإآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return " ".join(t.lower().split())


def _cer(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b) / max(1, len(a))


def load_clips(meta: Path, ds: str, stats: collections.Counter) -> list[dict]:
    """Jais rows of one dataset joined to timing, speaker and the acoustic transcript."""
    d = meta / ds
    timing = {}
    for mf in [d / "metadata.jsonl", *sorted(d.glob("parts/*/metadata.jsonl"))]:
        if mf.exists():
            for r in _jsonl(mf):
                if "audio_file" in r:
                    timing[_key(r["audio_file"])] = r
    durations = {}
    if (d / "transcripts.tashkeel.jsonl").exists():
        for r in _jsonl(d / "transcripts.tashkeel.jsonl"):
            durations[_key(r.get("flac") or r["uid"])] = r.get("duration")
    if (d / "metadata.csv").exists():
        for r in pd.read_csv(d / "metadata.csv").to_dict("records"):
            durations.setdefault(_key(r["file_name"]), r["duration"])
    acoustic = {}
    if (d / "transcripts.jais.diacritized.jsonl").exists():
        for r in _jsonl(d / "transcripts.jais.diacritized.jsonl"):
            acoustic[_key(r.get("audio_file") or r.get("uid") or r["flac"])] = r.get("text") or ""

    sources = [("transcripts.jais.jsonl", None)]
    if (d / "transcripts.jais.split.jsonl").exists():  # mostafa: long clips re-cut, extra rows
        sources.append(("transcripts.jais.split.jsonl", "flac_split"))
    rows = []
    for name, subdir in sources:
        for r in _jsonl(d / name):
            rel = r.get("audio_file") or r.get("flac")
            k = _key(rel)
            text = r.get("text") or ""
            if not text.strip() or r.get("refused") or r.get("truncated"):
                stats["drop_no_text"] += 1
                continue
            t = timing.get(k, {})
            dur = t.get("duration") or durations.get(k) or r.get("duration")
            if dur is None:
                stats["drop_no_duration"] += 1
                continue
            norm = normalize_egyptian(text)
            if k in acoustic:
                cer = _cer(_skeleton(norm), _skeleton(acoustic[k]))
                if cer > (args.max_cer_latin if _LATIN.search(norm) else args.max_cer):
                    stats["drop_asr_disagreement"] += 1
                    continue
            cps = len(norm) / float(dur)
            if not args.min_cps <= cps <= args.max_cps:
                stats["drop_speaking_rate"] += 1
                continue
            if subdir is not None or rel.startswith(("flac/", "wavs/")):
                rel_audio = rel  # already relative to the dataset root
            else:
                rel_audio = f"{'flac' if ds in FLAC_DATASETS else 'wavs'}/{rel}"
            spk = t.get("speaker") or f"{ds}:{k.split('/')[0].split('_chunk_')[0]}"
            rows.append(
                {
                    "ds": ds,
                    "key": k,
                    "audio": rel_audio,
                    "text": norm,
                    "duration": float(dur),
                    "video": t.get("video_id") or spk,
                    "idx": t.get("chunk_index"),
                    "start": t.get("start"),
                    "end": t.get("end"),
                    "gain_db": float(t.get("gain_db") or 0.0),
                    "speaker": spk,
                    "speaker_conf": t.get("speaker_conf"),
                }
            )
    return rows


def group_segments(rows: list[dict]) -> list[list[dict]]:
    by_video = collections.defaultdict(list)
    for r in rows:
        by_video[r["video"]].append(r)
    segments = []
    for clips in by_video.values():
        if any(r["idx"] is None or r["start"] is None for r in clips):
            segments += [[r] for r in clips]  # no timing: single clips
            continue
        clips.sort(key=lambda r: int(r["idx"]))
        cur: list[dict] = []
        for r in clips:
            if cur:
                prev = cur[-1]
                gap = r["start"] - prev["end"]
                if (
                    r["speaker"] == prev["speaker"]
                    and int(r["idx"]) == int(prev["idx"]) + 1
                    and 0 <= gap <= args.max_gap_sec
                    and r["end"] - cur[0]["start"] <= args.max_seg_sec
                ):
                    cur.append(r)
                    continue
                segments.append(cur)
            cur = [r]
        if cur:
            segments.append(cur)
    return segments


def _floor_rms(x: np.ndarray, from_end: bool) -> float:
    edge = x[-SR // 5 :] if from_end else x[: SR // 5]
    n = SR // 50
    frames = [edge[i : i + n] for i in range(0, max(1, len(edge) - n + 1), n // 2)]
    rms = [float(np.sqrt(np.mean(f**2))) for f in frames if len(f)]
    return min(rms) if rms else 0.0


def render(job: tuple[Path, Path, list[dict], int]) -> dict | None:
    raw, out_root, clips, seed = job
    rng = np.random.default_rng(seed)
    parts = []
    try:
        for i, r in enumerate(clips):
            x, sr = sf.read(raw / r["ds"] / r["audio"], dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                g = np.gcd(sr, SR)
                x = resample_poly(x, SR // g, sr // g).astype(np.float32)
            x *= 10 ** (-r["gain_db"] / 20)
            if i > 0:
                gap = round((r["start"] - clips[i - 1]["end"]) * SR)
                if gap > 0:
                    level = min(_floor_rms(parts[-1], True), _floor_rms(x, False))
                    fill = (rng.standard_normal(gap) * level).astype(np.float32)
                    ramp = min(len(fill) // 2, SR // 200)
                    if ramp:
                        w = np.linspace(0, 1, ramp, dtype=np.float32)
                        fill[:ramp] *= w
                        fill[-ramp:] *= w[::-1]
                    parts.append(fill)
            parts.append(x)
    except (OSError, RuntimeError, sf.LibsndfileError) as e:
        logger.warning(f"{clips[0]['audio']}: {e}")
        return None
    y = np.concatenate(parts)
    y *= 10 ** (float(np.median([r["gain_db"] for r in clips])) / 20)
    peak = float(np.abs(y).max())
    if peak > PEAK:
        y *= PEAK / peak
    first = clips[0]
    seg_id = f"{Path(first['key']).name}_{len(clips):02d}"
    path = out_root / first["ds"] / Path(first["key"]).parent / f"{seg_id}.flac"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, SR, subtype="PCM_16")
    confs = [float(r["speaker_conf"]) for r in clips if r["speaker_conf"] is not None]
    return {
        "path": str(path),
        "duration": round(len(y) / SR, 3),
        "transcript": " ".join(r["text"] for r in clips),
        # Reference voice prompts come from other entries of the same speaker: only
        # trust the diarization when it is confident (a wrong reference teaches the
        # model to ignore the voice prompt).
        "speaker": f"{first['ds']}/{first['speaker']}"
        if not confs or min(confs) >= args.min_speaker_conf
        else None,
        "cluster": f"{first['ds']}/{first['speaker']}",
        "ds": first["ds"],
        "n_clips": len(clips),
        "clips": [r["key"] for r in clips],
    }


def cmd_render() -> None:
    stats: collections.Counter = collections.Counter()
    rows = load_clips(args.meta, args.ds, stats)
    segments = group_segments(rows)
    logger.info(f"{args.ds}: {len(rows)} clips -> {len(segments)} segments, dropped {dict(stats)}")
    jobs = [(args.raw, args.out, s, args.seed + i) for i, s in enumerate(segments)]
    out = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(render, jobs, chunksize=64):
            if row:
                out.append(row)
    out.sort(key=lambda r: r["path"])
    with (args.out / f"{args.ds}.segments.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.out / f"{args.ds}.stats.json").write_text(
        json.dumps({"clips": len(rows), "segments": len(out), "dropped": dict(stats)}, indent=2)
    )


def cluster_genders(path: Path) -> dict[str, str]:
    """cluster -> "F"/"M" from the sampled clips (egyptian/scripts/classify_gender.py)."""
    if not path.exists():
        raise SystemExit(f"{path} missing: run egyptian/analysis/classify_gender.py first")
    g = pd.read_parquet(path)
    g["female"] = g.p_female > g.p_male
    votes = g.groupby(["ds", "cluster"]).female.mean()
    return {f"{ds}/{c}": ("F" if v >= 0.5 else "M") for (ds, c), v in votes.items()}


def set_weights(train: list[dict]) -> tuple[collections.Counter, collections.Counter]:
    """Cap each channel, then rebalance the genders on the capped hours."""
    hours = collections.Counter()
    for r in train:
        hours[r["ds"]] += r["duration"] / 3600
    for r in train:
        r["weight"] = min(1.0, args.cap_hours / hours[r["ds"]])
    by_gender = collections.Counter()
    for r in train:
        by_gender[r["gender"]] += r["weight"] * r["duration"] / 3600
    f, m = by_gender["F"], by_gender["M"]
    if f > 0 and m > 0:
        target_f = args.female_share * (f + m)
        scale = {"F": target_f / f, "M": (f + m - target_f) / m}
        top = max(r["weight"] * scale.get(r["gender"], 1.0) for r in train)
        for r in train:
            # Never upsample past 1 draw per epoch: repeated segments overfit; scale
            # the majority down instead.
            r["weight"] = round(r["weight"] * scale.get(r["gender"], 1.0) / max(1.0, top), 4)
    drawn = collections.Counter()
    for r in train:
        drawn[(r["ds"], r["gender"])] += r["weight"] * r["duration"] / 3600
    return hours, drawn


def transfer_marks(plain: str, marked: str) -> str:
    """plain's words, each replaced by its diacritized form in marked where the two
    texts have the same word (difflib over letter skeletons): gemini's transcript keeps
    its digits, and its ambiguous words get the marks they were spoken with."""
    a, b = plain.split(), marked.split()
    sm = difflib.SequenceMatcher(None, a, [strip_marks(w) for w in b], autojunk=False)
    out = list(a)
    for i, j, n in sm.get_matching_blocks():
        out[i : i + n] = b[j : j + n]
    return " ".join(out)


def load_diacritized(ds: str) -> dict[str, str]:
    """clip key -> normalized diacritized Jais text (incl. mostafa's re-cut clips)."""
    out = {}
    for name in ("transcripts.jais.diacritized.jsonl", "transcripts.jais.split.diacritized.jsonl"):
        if (args.meta / ds / name).exists():
            for r in _jsonl(args.meta / ds / name):
                text = r.get("text") or ""
                if text.strip() and not r.get("refused") and not r.get("truncated"):
                    key = _key(r.get("audio_file") or r.get("flac") or r["uid"])
                    out[key] = normalize_egyptian(text)
    return out


def cmd_retext() -> None:
    """v2 text: each rendered segment (<ds>.segments.jsonl in --seg) with the diacritized
    Jais text of its clips, into --out. A segment with a clip Jais did not diacritize
    keeps its plain text ("diacritized": false)."""
    for f in sorted(args.seg.glob("*.segments.jsonl")):
        ds = f.name.split(".")[0]
        if ds == "gemini_synthetic":
            continue
        diac = load_diacritized(ds)
        n = full = 0
        with (args.out / f.name).open("w") as fh:
            for r in _jsonl(f):
                texts = [diac.get(k) for k in r["clips"]]
                r["diacritized"] = all(texts)
                if r["diacritized"]:
                    r["transcript"] = " ".join(texts)
                n += 1
                full += r["diacritized"]
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"{ds}: {full} / {n} segments diacritized")


def cmd_gemini() -> None:
    """gemini_synthetic: single clips, their own transcripts (QA'd against an ASR at
    generation time) with the marks of their TTS input, and labelled voices. Audio is
    used in place (24 kHz mono)."""
    root = args.seg / "gemini_synthetic"
    stats: collections.Counter = collections.Counter()
    tagged = 0
    out = []
    for mf in sorted(root.glob("part_*/manifest.jsonl")):
        for r in _jsonl(mf):
            # [laughs], [sighs], [chuckles] stay in the text (one token each,
            # normalize_egyptian): the model learns to produce them.
            tagged += any(t in r["transcript"] for t in TAGS)
            text = transfer_marks(
                normalize_egyptian(r["transcript"]), normalize_egyptian(r["tts_input"])
            )
            dur = float(r["duration"])
            spoken = strip_marks(text)
            for t in TAGS:
                spoken = spoken.replace(t, "")
            cps = len(spoken) / dur
            if not text or dur > args.max_seg_sec or not args.min_cps <= cps <= args.max_cps:
                stats["drop_text_or_rate"] += 1
                continue
            path = mf.parent / r["path"]
            if not path.exists():
                stats["drop_missing_audio"] += 1
                continue
            spk = f"gemini_synthetic/{r['speaker']}"
            out.append(
                {
                    "path": str(path),
                    "duration": dur,
                    "transcript": text,
                    "speaker": spk,
                    "cluster": spk,
                    "ds": "gemini_synthetic",
                    "n_clips": 1,
                    "clips": [r["seg_id"]],
                    "gender": {"female": "F", "male": "M"}[r["gender"]],
                    "diacritized": True,
                }
            )
    out.sort(key=lambda r: r["path"])
    with (args.out / "gemini_synthetic.segments.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.out / "gemini_synthetic.stats.json").write_text(
        json.dumps({"segments": len(out), "with_tags": tagged, "dropped": dict(stats)}, indent=2)
    )
    logger.info(
        f"gemini_synthetic: {len(out)} segments ({tagged} with tags), dropped {dict(stats)}"
    )


def cmd_merge() -> None:
    # <ds>.aligned.jsonl: the rendered segments after training/scripts/align_data.py.
    rows = [r for f in sorted(args.out.glob("*.aligned.jsonl")) for r in _jsonl(f)]
    genders = cluster_genders(args.gender_pred)
    for r in rows:
        r["gender"] = genders.get(r["cluster"]) or r.get("gender")  # gemini: labelled voices
    rng = random.Random(args.seed)

    # Held-out speakers for the new-voice eval: 3-15 min clusters, half female half male.
    secs = collections.Counter()
    for r in rows:
        secs[r["cluster"]] += r["duration"]
    pool = {"F": [], "M": []}
    for c, s in sorted(secs.items()):
        g = genders.get(c)
        if g and 180 <= s <= 900:
            pool[g].append(c)
    unseen = set()
    for clusters in pool.values():
        unseen |= set(rng.sample(clusters, min(args.n_unseen // 2, len(clusters))))
    seen = [r for r in rows if r["cluster"] not in unseen]
    rng.shuffle(seen)
    valid_seen = [r for r in seen if 4 <= r["duration"] <= 20][: args.n_valid_seen]
    held = {r["path"] for r in valid_seen}
    train = sorted((r for r in seen if r["path"] not in held), key=lambda r: r["path"])
    valid_unseen = [r for r in rows if r["cluster"] in unseen and 4 <= r["duration"] <= 20]
    rng.shuffle(valid_unseen)
    valid_unseen = valid_unseen[: args.n_valid_unseen]

    hours, drawn = set_weights(train)

    def dump(name: str, rs: list[dict]) -> None:
        with (args.out / name).open("w") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def plain(r: dict) -> dict:
        words = [{**w, "word": strip_marks(w["word"])} for w in r.get("words") or []]
        return {**r, "transcript": strip_marks(r["transcript"]), "words": words}

    dump("train_aligned.jsonl", train)
    # Validation tracks the main use, plain text; the diacritized copy is for evals.
    dump("valid_seen_aligned.jsonl", [plain(r) for r in valid_seen])
    dump("valid_unseen_aligned.jsonl", [plain(r) for r in valid_unseen])
    dump("valid_aligned.jsonl", [plain(r) for r in valid_seen + valid_unseen])
    dump("valid_diacritized_aligned.jsonl", valid_seen + valid_unseen)
    summary = {
        "train_segments": len(train),
        "train_hours": round(sum(r["duration"] for r in train) / 3600, 1),
        "hours_drawn_per_epoch": round(sum(drawn.values()), 1),
        "drawn_by_gender": {
            g: round(sum(v for (_, gg), v in drawn.items() if gg == g), 1) for g in ("F", "M", None)
        },
        "drawn_by_dataset": {
            ds: round(sum(v for (d, _), v in drawn.items() if d == ds), 1) for ds in sorted(hours)
        },
        "valid_seen": len(valid_seen),
        "valid_unseen": len(valid_unseen),
        "unseen_speakers": sorted(unseen),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(json.dumps({k: v for k, v in summary.items() if k != "unseen_speakers"}, indent=1))


_FINAL_MARKS = re.compile(f"[{DIACRITICS}]+$")


def cmd_lexicon() -> None:
    """Ambiguous words: letter skeletons that the training text reads two ways or more
    (each reading >= 10% of the word's diacritized uses and >= 5 times). A reading
    ignores sukun and final marks, where Jais is inconsistent; an unmarked use is no
    reading. Feeds the hint form of the tashkeel augmentation."""
    readings: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in _jsonl(args.out / "train_aligned.jsonl"):
        if not r.get("diacritized"):
            continue
        for w in r["transcript"].split():
            key = skeleton_key(w)
            core = _FINAL_MARKS.sub("", w.strip("،؟؛.!:,-")).replace(SUKUN, "")
            if key and core != key:
                readings[key][core] += 1
    ambiguous = sorted(
        k
        for k, c in readings.items()
        if sum(v >= 5 and v >= 0.1 * sum(c.values()) for v in c.values()) >= 2
    )
    (args.out / "ambiguous_words.json").write_text(json.dumps(ambiguous, ensure_ascii=False))
    examples = [w for w in ("يركب", "انت", "علم", "درس", "كتبت") if w in set(ambiguous)]
    logger.info(f"{len(ambiguous)} ambiguous words of {len(readings)}; e.g. {examples}")


def cmd_marks() -> None:
    """Inference mark lexicon (pocket_tts.egyptian_text.canonicalize_marks, the
    config's mark_lexicon): the marked forms of every word of the training text
    (>= --min-uses uses, the 12 most used), the words it has at all (>= --min-uses),
    and the ambiguous words."""
    forms: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    uses: collections.Counter = collections.Counter()
    for r in _jsonl(args.out / "train_aligned.jsonl"):
        for token in normalize_egyptian(r["transcript"]).split(" "):
            core = split_edges(token)[1]
            if not ARABIC_WORD.match(core):
                continue
            skeleton = strip_marks(core)
            uses[skeleton] += 1
            if core != skeleton:
                forms[skeleton][core] += 1
    lexicon = {
        "forms": {
            k: [[f, n] for f, n in c.most_common(12) if n >= args.min_uses]
            for k, c in forms.items()
            if c.most_common(1)[0][1] >= args.min_uses
        },
        "known": sorted(k for k, n in uses.items() if n >= args.min_uses),
        "ambiguous": json.loads((args.out / "ambiguous_words.json").read_text()),
    }
    with gzip.open(args.out / "mark_lexicon.json.gz", "wt") as fh:
        json.dump(lexicon, fh, ensure_ascii=False)
    logger.info(
        f"{len(lexicon['forms'])} marked words, {len(lexicon['known'])} known words "
        f"(>= {args.min_uses} uses), {len(lexicon['ambiguous'])} ambiguous"
    )


def cmd_corpus() -> None:
    """Tokenizer text: every training transcript plain, fully marked and in one random
    partial or hint form, so the pieces cover all three."""
    ambiguous = set(json.loads((args.out / "ambiguous_words.json").read_text()))
    rng = random.Random(args.seed)
    with (args.out / "tokenizer_corpus.txt").open("w") as fh:
        for r in _jsonl(args.out / "train_aligned.jsonl"):
            t = r["transcript"]
            fh.write(strip_marks(t) + "\n")
            if r.get("diacritized"):
                fh.write(t + "\n")
                fh.write(tashkeel_augment(t, rng, [0, 0, 1, 1], ambiguous) + "\n")


def encode_latents(rows: list[dict]) -> tuple[dict[str, str], int]:
    """Mimi latents for rows v1 never encoded (the tagged gemini clips), into the same
    store: (audio path -> latents_file, stitch_frames)."""
    import torch

    from training.args import load_args
    from training.modules.builders import load_model_config
    from training.scripts.precompute_latents import load_frozen_mimi, precompute_manifest

    manifest = args.out / "v2_extra.jsonl"
    with manifest.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    train_args = load_args(args.latents_config)
    config = load_model_config(train_args.model_config, train_args.model_overrides)
    mimi = load_frozen_mimi(config).to("cuda")
    precompute_manifest(manifest, mimi, torch.device("cuda"), 32, 16, str(config.weights_path))
    out = {r["path"]: r["latents_file"] for r in _jsonl(args.out / "v2_extra_latents.jsonl")}
    meta = json.loads((args.out / "v2_extra_latents.meta.json").read_text())
    return out, int(meta["stitch_frames"])


def cmd_latents() -> None:
    """train_aligned_latents.jsonl from the v1 latents (--seg), matched by audio path
    (the audio is unchanged); only rows v1 never had (the tagged gemini clips) are
    encoded."""
    latents = args.out / "latents"
    if not latents.exists():
        latents.symlink_to(args.seg / "latents")  # latents_file is relative to the manifest
    by_path = {}
    for f in sorted(args.seg.glob("*_latents.jsonl")):
        if f.name.startswith("train_mix"):
            continue  # the concatenation of the others
        for r in _jsonl(f):
            by_path[r["path"]] = r["latents_file"]
    rows = list(_jsonl(args.out / "train_aligned.jsonl"))
    meta = json.loads((args.seg / "train_aligned_latents.meta.json").read_text())
    missing = [r for r in rows if r["path"] not in by_path]
    if missing:
        logger.info(f"encoding {len(missing)} segments without v1 latents")
        extra, stitch = encode_latents(missing)
        by_path |= extra
        # One stitch window for the whole manifest: the larger one is safe for both.
        meta["stitch_frames"] = max(meta["stitch_frames"], stitch)
    with (args.out / "train_aligned_latents.jsonl").open("w") as fh:
        for r in rows:
            fh.write(
                json.dumps({**r, "latents_file": by_path[r["path"]]}, ensure_ascii=False) + "\n"
            )
    (args.out / "train_aligned_latents.meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"{len(rows)} segments, {len(missing)} newly encoded")


def main() -> None:
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=["render", "retext", "gemini", "merge", "lexicon", "marks", "corpus", "latents"],
    )
    ap.add_argument("--ds", help="dataset to render")
    ap.add_argument("--meta", type=Path, default=Path("/workspace/egy/meta"))
    ap.add_argument("--raw", type=Path, default=Path("/data/egy/raw"))
    ap.add_argument("--out", type=Path, default=Path("/data/egy/seg"))
    ap.add_argument(
        "--seg", type=Path, default=Path("/data/egy/seg"), help="v1 segments (v2 commands)"
    )
    ap.add_argument(
        "--latents-config",
        default="egyptian/configs/egy_teacher_b32.yaml",
        help="training config whose Mimi encodes missing latents (latents)",
    )
    ap.add_argument("--gender-pred", type=Path, default=Path("/workspace/egy/gender_pred.parquet"))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-cer", type=float, default=0.25)
    ap.add_argument("--max-cer-latin", type=float, default=0.45)
    ap.add_argument("--min-cps", type=float, default=4.0)
    ap.add_argument("--max-cps", type=float, default=22.0)
    ap.add_argument("--max-gap-sec", type=float, default=1.0)
    ap.add_argument("--max-seg-sec", type=float, default=30.0)
    ap.add_argument("--min-speaker-conf", type=float, default=0.7)
    ap.add_argument("--cap-hours", type=float, default=150.0)
    ap.add_argument("--female-share", type=float, default=0.5)
    ap.add_argument("--n-unseen", type=int, default=40)
    ap.add_argument("--n-valid-seen", type=int, default=300)
    ap.add_argument("--n-valid-unseen", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--min-uses", type=int, default=3, help="marks: rarer words and forms are left out"
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "render":
        assert args.ds and args.ds not in EXCLUDED, "--ds: a dataset other than drama3"
        cmd_render()
    else:
        commands = {"retext": cmd_retext, "gemini": cmd_gemini, "merge": cmd_merge}
        commands |= {"lexicon": cmd_lexicon, "marks": cmd_marks, "corpus": cmd_corpus}
        commands |= {"latents": cmd_latents}
        commands[args.cmd]()


args: argparse.Namespace

if __name__ == "__main__":  # precompute's decode pool spawns workers that re-import this file
    main()
