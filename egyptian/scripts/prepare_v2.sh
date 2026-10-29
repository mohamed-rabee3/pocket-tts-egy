#!/usr/bin/env bash
# Egyptian v2 data: the v1 rendered audio with diacritized text, for a model that reads
# plain, partially and fully diacritized input (pocket_tts/egyptian_text.py).
#   retext   v1 segments + transcripts.jais.diacritized.jsonl  -> $V2/<ds>.segments.jsonl
#   gemini   gemini_synthetic, marks from its TTS input, its [laughs] [sighs] [chuckles]
#            tags kept                                            -> $V2/gemini_synthetic.segments.jsonl
#   align    word timings on the letter skeleton                -> $V2/<ds>.aligned.jsonl
#   merge, lexicon (ambiguous words), latents (v1 Mimi latents by audio path),
#   tokenizer (8,000 pieces over plain + full + partial text; marks, digits, punctuation
#   always get a piece).
# Every step skips itself when its output already exists, so the script can be rerun.
set -euo pipefail
SEG=${SEG:-/data/egy/seg}
V2=${V2:-/data/egy/v2}
META=${META:-/workspace/egy/meta}
GENDER_PRED=${GENDER_PRED:-/workspace/egy/gender_pred.parquet}
ALIGN_PROCS=${ALIGN_PROCS:-3}  # 6 runs out of GPU memory on 30 s segments
VOCAB=${VOCAB:-8000}
ALIGN_MODEL=jonatasgrosman/wav2vec2-large-xlsr-53-arabic
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MANIFEST=(uv run --with pandas --with pyarrow python egyptian/scripts/build_manifest.py)
log() { echo "[$(date '+%F %T')] $*"; }
cd "$REPO"
mkdir -p "$V2"

if [ ! -f "$V2/.retext_done" ]; then
  log "retext: diacritized Jais text on the v1 segments"
  "${MANIFEST[@]}" retext --seg "$SEG" --out "$V2" --meta "$META"
  touch "$V2/.retext_done"
fi
if [ ! -f "$V2/gemini_synthetic.segments.jsonl" ]; then
  "${MANIFEST[@]}" gemini --seg "$SEG" --out "$V2"
fi

for seg in "$V2"/*.segments.jsonl; do
  ds=$(basename "$seg" .segments.jsonl)
  [ -f "$V2/$ds.aligned.jsonl" ] && continue
  log "$ds: aligning $(wc -l < "$seg") segments"
  split -n "l/$ALIGN_PROCS" -d "$seg" "$V2/$ds.align_in_"
  pids=()
  for part in "$V2/$ds.align_in_"??; do
    # --resume: a retry (e.g. after a transient CUDA OOM) continues where it stopped.
    (for attempt in 1 2 3; do
      uv run python training/scripts/align_data.py "$part" "$part.out" --model "$ALIGN_MODEL" \
        --strip-marks --batch-size 32 --resume >> "$part.log" 2>&1 && exit 0
      log "$part: align attempt $attempt failed"; sleep 30
    done; exit 1) & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  cat "$V2/$ds.align_in_"??.out > "$V2/$ds.aligned.jsonl.tmp"
  mv "$V2/$ds.aligned.jsonl.tmp" "$V2/$ds.aligned.jsonl"
  rm -f "$V2/$ds.align_in_"?? "$V2/$ds.align_in_"??.out
  log "$ds: aligned $(wc -l < "$V2/$ds.aligned.jsonl") / $(wc -l < "$seg")"
done

if [ ! -f "$V2/train_aligned.jsonl" ]; then
  log "merging: weights, splits"
  "${MANIFEST[@]}" merge --out "$V2" --gender-pred "$GENDER_PRED"
fi
[ -f "$V2/ambiguous_words.json" ] || "${MANIFEST[@]}" lexicon --out "$V2"
[ -f "$V2/mark_lexicon.json.gz" ] || "${MANIFEST[@]}" marks --out "$V2"
[ -f "$V2/train_aligned_latents.meta.json" ] || "${MANIFEST[@]}" latents --seg "$SEG" --out "$V2"

if [ ! -f "$V2/tokenizer.json" ]; then
  log "training the tokenizer ($VOCAB pieces, plain + diacritized text)"
  "${MANIFEST[@]}" corpus --out "$V2"
  REQUIRED="ًٌٍَُِّْٰ0123456789٠١٢٣٤٥٦٧٨٩.،؟!:؛%-"
  REQUIRED+="ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيپچڤگ"
  REQUIRED+="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  uv run python -m training.scripts.train_tokenizer "$V2/tokenizer" "$V2/tokenizer_corpus.txt" \
    --vocab-size "$VOCAB" --character-coverage 0.9999 --required-chars "$REQUIRED" \
    --user-defined-symbols "[laughs],[sighs],[chuckles]"
  # The converted tokenizer must match sentencepiece on plain and diacritized text.
  shuf -n 20000 --random-source=<(yes) "$V2/tokenizer_corpus.txt" > "$V2/tokenizer_check.txt"
  cat "$V2/valid_diacritized_aligned.jsonl" >> "$V2/tokenizer_check.txt"
  uv run python -m training.scripts.convert_tokenizer "$V2/tokenizer.model" \
    --out "$V2/tokenizer.json" --texts "$V2/tokenizer_check.txt"
fi

log "data ready: $(wc -l < "$V2/train_aligned.jsonl") train segments"
