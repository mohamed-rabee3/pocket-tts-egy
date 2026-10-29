"""Homograph pronunciation accuracy: does the model read an undiacritized word the
way its context requires (يركب العربية = yirkab, يركب اللمبة = yirakkib)?

WER cannot see this: any ASR writes both readings the same. Here every
generation goes through an ASR that writes the diacritics it hears
(NAMAA-Space/Cohere-Speech-Tashkeel-2B, diacritized from the audio, not from the
text), two ways:

- forced choice (the reported accuracy): the ASR's transcript with the target
  word set to the expected reading vs. the other reading
  (egyptian/eval/homographs.jsonl: "reading", "other_reading"); the one the
  audio makes more likely wins. Free decoding of short vowels is too noisy to
  grade single words (on real speech of known reading it agreed 14/19 times);
- free decoding, as a second opinion: the target word is found by its letter
  skeleton and the marks that tell the readings apart are checked ("checks":
  ["ك", "shadda"] = the kaf must carry a shadda). "missing" = the ASR did not
  write the word at all (dropped or garbled).

    # a checkpoint, both genders:
    uv run --with librosa python egyptian/scripts/eval_homographs.py \\
        --config egyptian/configs/egy_inference_24l.yaml --checkpoint runs/egy/checkpoint_00200000.pt \\
        --voices female.wav male.wav --out evals/homographs_200k
    # the same with the ambiguous word marked, as an LLM front-end would send it:
    ... --text hint --out evals/homographs_200k_hint
    # existing recordings named <id>.wav (validates the scorer on human speech):
    uv run --with librosa python egyptian/scripts/eval_homographs.py --audio-dir recordings/ --out evals/human
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

MODEL = "NAMAA-Space/Cohere-Speech-Tashkeel-2B"
ITEMS = Path(__file__).resolve().parents[1] / "eval" / "homographs.jsonl"
MARKS = {"fatha": "َ", "damma": "ُ", "kasra": "ِ", "sukun": "ْ", "shadda": "ّ"}
_ALL_MARKS = re.compile("[ً-ْٰ]")


def skeleton(word: str) -> str:
    w = _ALL_MARKS.sub("", unicodedata.normalize("NFC", word))
    w = re.sub("[أإآٱ]", "ا", w).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[^\w]", "", w)


def marks_after(word: str, letter: str) -> str | None:
    """The combining marks on the first `letter` of `word`, None if absent."""
    i = word.find(letter)
    if i < 0:
        return None
    j = i + 1
    while j < len(word) and unicodedata.combining(word[j]):
        j += 1
    return word[i + 1 : j]


def check(word: str, checks: list[list[str]]) -> bool:
    for letter, mark in checks:
        m = marks_after(word, letter)
        if m is None:
            return False
        if mark == "no_shadda":
            if MARKS["shadda"] in m:
                return False
        elif MARKS[mark] not in m:
            return False
    return True


def score(item: dict, transcript: str) -> tuple[str, str]:
    target = skeleton(item["target"])
    words = unicodedata.normalize("NFC", transcript).split()
    found = [w for w in words if skeleton(w) == target]
    if not found:
        return "missing", ""
    return ("ok" if check(found[0], item["checks"]) else "wrong"), found[0]


class TashkeelASR:
    def __init__(self, device: str = "cuda") -> None:
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.model = (
            CohereAsrForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16)
            .to(device)
            .eval()
        )

    def _inputs(self, y: np.ndarray, sr: int) -> tuple[Any, Any]:
        y = y if y.ndim == 1 else y.mean(1)
        if sr != 16000:
            g = np.gcd(sr, 16000)
            y = resample_poly(y, 16000 // g, sr // g)
        inputs = self.proc(
            audio=y.astype(np.float32), sampling_rate=16000, return_tensors="pt", language="ar"
        )
        idx = inputs.pop("audio_chunk_index", None)
        inputs.to(self.model.device, dtype=self.model.dtype)
        return inputs, idx

    @torch.no_grad()
    def __call__(self, y: np.ndarray, sr: int) -> str:
        inputs, idx = self._inputs(y, sr)
        out = self.model.generate(**inputs, max_new_tokens=256)
        text = self.proc.decode(out, skip_special_tokens=True, audio_chunk_index=idx, language="ar")
        return text[0] if isinstance(text, list) else text

    @torch.no_grad()
    def log_likelihood(self, y: np.ndarray, sr: int, texts: list[str]) -> list[float]:
        """Sum log p(text | audio) under the decoder for each candidate transcript."""
        inputs, _ = self._inputs(y, sr)
        prompt = inputs.pop("decoder_input_ids")[0].tolist()
        tok = self.proc.tokenizer
        scores = []
        for text in texts:
            ids = prompt + tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id]
            dec = torch.tensor([ids[:-1]], device=self.model.device)
            logits = self.model(**inputs, decoder_input_ids=dec).logits[0].float()
            logp = logits.log_softmax(-1)[torch.arange(len(ids) - 1), torch.tensor(ids[1:])]
            scores.append(float(logp[len(prompt) - 1 :].sum()))
        return scores


def with_reading(transcript: str, item: dict, reading: str) -> str:
    """`transcript` with its target word (found by skeleton) replaced by `reading`; the
    item text is used when the ASR did not write the word."""
    words = unicodedata.normalize("NFC", transcript).split()
    target = skeleton(item["target"])
    for i, w in enumerate(words):
        if skeleton(w) == target:
            tail = re.search(r"[^\w\u064b-\u0652\u0670]*$", w).group(0)
            return " ".join(words[:i] + [reading + tail] + words[i + 1 :])
    words = item["text"].split()
    return " ".join(reading if w == item["target"] else w for w in words)


def forced_choice(asr: TashkeelASR, y: np.ndarray, sr: int, transcript: str, item: dict) -> bool:
    good, bad = asr.log_likelihood(
        y,
        sr,
        [
            with_reading(transcript, item, item["reading"]),
            with_reading(transcript, item, item["other_reading"]),
        ],
    )
    return good > bad


def hinted(item: dict) -> str:
    """The item text with its target word marked as the expected reading: the hint an
    agent's LLM would add. The training text never puts a sukun on a word's first
    letter (يْرَكِّب is written يرَكِّب there), so neither does the hint."""
    reading = re.sub("^(.)\u0652", r"\1", unicodedata.normalize("NFC", item["reading"]))
    return " ".join(reading if w == item["target"] else w for w in item["text"].split())


def generate(args: argparse.Namespace, items: list[dict]) -> list[tuple[dict, str, Path]]:
    from pocket_tts import TTSModel

    model = TTSModel.load_model(config=args.config, checkpoint=args.checkpoint)
    out = []
    for voice in args.voices:
        state = model.get_state_for_audio_prompt(voice)
        tag = Path(voice).stem
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            for item in items:
                # The model normalizes its input (the config's text_normalizer).
                text = hinted(item) if args.text == "hint" else item["text"]
                audio = model.generate_audio(state, text, copy_state=True)
                path = args.out / "audio" / f"{item['id']}_{tag}_{seed}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, audio.numpy(), model.sample_rate)
                out.append((item, f"{tag}/{seed}", path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--voices", nargs="+", default=[])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument(
        "--text",
        choices=["plain", "hint"],
        default="plain",
        help="hint: the target word carries the marks of its expected reading",
    )
    ap.add_argument("--audio-dir", type=Path, help="score existing <id>.wav files instead")
    ap.add_argument("--items", type=Path, default=ITEMS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in args.items.open()]
    if args.audio_dir:
        runs = [(i, "audio", args.audio_dir / f"{i['id']}.wav") for i in items]
        runs = [r for r in runs if r[2].exists()]
    else:
        runs = generate(args, items)

    asr = TashkeelASR()
    results = []
    for item, run, path in runs:
        y, sr = sf.read(path, dtype="float32")
        text = asr(y, sr)
        free, heard = score(item, text)
        verdict = "ok" if forced_choice(asr, y, sr, text, item) else "wrong"
        results.append({"id": item["id"], "run": run, "sense": item["sense"], "verdict": verdict,
                        "free_decoding": free, "heard": heard, "transcript": text})  # fmt: skip
    with (args.out / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_run = Counter((r["run"], r["verdict"]) for r in results)
    total = Counter(r["verdict"] for r in results)
    free = Counter(r["free_decoding"] for r in results)
    n = len(results)
    summary = {
        "n": n,
        "accuracy": round(total["ok"] / max(1, n), 3),
        "free_decoding": dict(free),
        "by_run": {f"{k[0]}:{k[1]}": v for k, v in sorted(by_run.items())},
        "failed_items": sorted({r["id"] for r in results if r["verdict"] != "ok"}),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
