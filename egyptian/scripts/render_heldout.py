"""Render egyptian/eval/heldout_tts.jsonl with one checkpoint, female and male voice, for
listening tests: short items as one chunk (VoiceAgentTTS.say), long ones streamed sentence by
sentence (speak_stream, the voice-agent mode). CFG 2 for the teacher as its training samples,
--cfg 1 for a distilled student (CFG baked in, egy_inference_6l.yaml).

    uv run python egyptian/scripts/render_heldout.py --checkpoint runs/.../checkpoint_00290000.pt \\
        --out /workspace/egy-data/runs/teacher_v2/heldout/step290k
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from pocket_tts import TTSModel
from pocket_tts.continuous import VoiceAgentTTS

VOICES = {
    "female": "/data/egy/voices/sample_voice_female.flac",
    "male": "/data/egy/voices/sample_voice_male.flac",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="egyptian/configs/egy_inference_24l.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--items", default="egyptian/eval/heldout_tts.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=2.0)
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.items)]
    model = TTSModel.load_model(config=args.config, checkpoint=args.checkpoint)
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    clips = []
    for vname, vpath in VOICES.items():
        voice = VoiceAgentTTS(model, vpath, interleaved=True, cfg_coef=args.cfg)
        for item in items:
            text = item.get("text_m", item["text"]) if vname == "male" else item["text"]
            torch.manual_seed(args.seed)
            turn = voice.new_turn()
            chunks = turn.speak_stream(iter([text])) if item.get("long") else turn.say(text)
            audio = torch.cat([c.flatten().cpu() for c in chunks])
            name = f"{item['id']}__{item['mode']}__{vname}.wav"
            sf.write(args.out / "audio" / name, audio.numpy(), model.sample_rate)
            clips.append({**item, "text": text, "voice": vname, "file": name,
                          "sec": round(audio.numel() / model.sample_rate, 2)})
            print(name, clips[-1]["sec"], flush=True)
    clips = [{k: v for k, v in c.items() if k != "text_m"} for c in clips]
    json.dump({"checkpoint": args.checkpoint, "clips": clips},
              open(args.out / "clips.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
