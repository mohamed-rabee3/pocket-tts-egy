#!/usr/bin/env bash
# Egyptian data preparation, per dataset: download -> render segments -> delete raw
# -> word alignment (on the GPU, in the background while the next dataset downloads);
# then merge (filters, weights, splits) and train the tokenizer.
# Every step skips itself when its output already exists, so the script can be rerun.
#
# Disk: the raw clips of all datasets (~410 GB) and the rendered segments (~240 GB)
# do not fit together on 630 GB, so each dataset's raw audio is deleted once its
# segments are written (peak ~ segments + the largest dataset, heya-wbs ~130 GB).
#
# Download: rsync over SSH (the Storage Box serves it on port 23) streams many small
# files per connection: ~19 MB/s per stream and ~45 MB/s for 6, against ~5 MB/s for
# SFTP. The account allows ~10 connections, so keep FETCH_STREAMS <= 8. The password
# is read from $HZ_CREDENTIALS ("pass: ..." line) at run time.
set -euo pipefail

RAW=${RAW:-/data/egy/raw}
SEG=${SEG:-/data/egy/seg}
META=${META:-/workspace/egy/meta}
GENDER_PRED=${GENDER_PRED:-/workspace/egy/gender_pred.parquet}
HZ_CREDENTIALS=${HZ_CREDENTIALS:-/workspace/apis.txt}
HZ_REMOTE=${HZ_REMOTE:?set HZ_REMOTE to user@host of the Storage Box}
FETCH_STREAMS=${FETCH_STREAMS:-6}
ALIGN_PROCS=${ALIGN_PROCS:-3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ALIGN_MODEL=jonatasgrosman/wav2vec2-large-xlsr-53-arabic
# Every dataset with transcripts.jais.jsonl; drama3_synthetic is excluded on purpose.
DATASETS=(besohat dahab daheeh drmohamedelnazer droos eqkawkab espitalia fatma-abuhaty
  halasamir heya-wbs jacklin-bahgat masri mostafa mustafa-hosny nourhan-elsheikh
  nourhan-kandil playlist-PLRCzrSHS5u_FFoGXKpBoG7kJ8yZNgNwLK sarahhany somayasserscience)
# ONLY="a b" restricts the run to those datasets (tests).
if [ -n "${ONLY:-}" ]; then read -ra DATASETS <<< "$ONLY"; fi
FLAC=" masri mostafa dahab halasamir "
SSH="ssh -p 23 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
SSHPASS=$(sed -n 's/^pass: //p' "$HZ_CREDENTIALS")
export SSHPASS
log() { echo "[$(date '+%F %T')] $*"; }
cd "$REPO"
mkdir -p "$SEG" "$RAW"

rsync_part() {  # remote dir, dest dir, file list: retried, rsync resumes what it has
  for attempt in 1 2 3 4 5 6; do
    if sshpass -e rsync -a -r --timeout=120 --files-from="$3" -e "$SSH" "$HZ_REMOTE:$1/" "$2/"; then
      return 0
    fi
    log "rsync $1 ($3) failed, attempt $attempt"; sleep 20
  done
  return 1
}

fetch() {  # dataset, subdir: the top-level entries split over FETCH_STREAMS rsyncs
  local remote="egy-tts/datasets/$1/$2" dest="$RAW/$1/$2" list="$RAW/$1.$2.list"
  mkdir -p "$dest"
  sshpass -e rsync --list-only -e "$SSH" "$HZ_REMOTE:$remote/" | awk '$NF != "." {print $NF}' > "$list"
  split -n "r/$FETCH_STREAMS" "$list" "$list.part_"
  local pids=()
  for part in "$list.part_"*; do
    rsync_part "$remote" "$dest" "$part" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  rm -f "$list" "$list.part_"*
}

align_dataset() {  # <ds>.segments.jsonl -> <ds>.aligned.jsonl, ALIGN_PROCS processes on the GPU
  local ds=$1 pids=()
  split -n "l/$ALIGN_PROCS" -d "$SEG/$ds.segments.jsonl" "$SEG/$ds.align_in_"
  for part in "$SEG/$ds.align_in_"??; do
    uv run python training/scripts/align_data.py "$part" "$part.out" --model "$ALIGN_MODEL" \
      --strip-marks --batch-size 32 --resume > "$part.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  cat "$SEG/$ds.align_in_"??.out > "$SEG/$ds.aligned.jsonl.tmp"
  mv "$SEG/$ds.aligned.jsonl.tmp" "$SEG/$ds.aligned.jsonl"
  rm -f "$SEG/$ds.align_in_"?? "$SEG/$ds.align_in_"??.out
  log "$ds: aligned $(wc -l < "$SEG/$ds.aligned.jsonl") / $(wc -l < "$SEG/$ds.segments.jsonl") segments"
}

align_pid=""
for ds in "${DATASETS[@]}"; do
  if [ ! -f "$SEG/$ds.segments.jsonl" ]; then
    sub=wavs; [[ $FLAC == *" $ds "* ]] && sub=flac
    log "$ds: downloading $sub"
    fetch "$ds" "$sub"
    [ "$ds" = mostafa ] && fetch "$ds" flac_split
    log "$ds: rendering segments ($(du -sh "$RAW/$ds" | cut -f1) raw)"
    # render writes <ds>.segments.jsonl only once every segment is on disk.
    uv run --with pandas --with pyarrow python egyptian/scripts/build_manifest.py render \
      --ds "$ds" --meta "$META" --raw "$RAW" --out "$SEG"
    rm -rf "${RAW:?}/$ds"
    log "$ds: $(cat "$SEG/$ds.stats.json" | tr -d '\n ')"
  fi
  if [ ! -f "$SEG/$ds.aligned.jsonl" ]; then
    [ -n "$align_pid" ] && wait "$align_pid"  # one dataset aligning at a time
    align_dataset "$ds" & align_pid=$!
  fi
done
[ -n "$align_pid" ] && wait "$align_pid"

if [ ! -f "$SEG/train_aligned.jsonl" ]; then
  log "merging: filters, weights, splits"
  uv run --with pandas --with pyarrow python egyptian/scripts/build_manifest.py merge \
    --out "$SEG" --gender-pred "$GENDER_PRED"
fi

if [ ! -f "$SEG/tokenizer.json" ]; then
  log "training the tokenizer (4000 pieces, digits split, Latin kept)"
  uv run python -m training.scripts.train_tokenizer "$SEG/tokenizer" "$SEG/train_aligned.jsonl" \
    --character-coverage 0.9999
  uv run python -m training.scripts.convert_tokenizer "$SEG/tokenizer.model" \
    --out "$SEG/tokenizer.json" --texts "$SEG/valid_aligned.jsonl"
fi

log "data ready: $(wc -l < "$SEG/train_aligned.jsonl") train segments"
