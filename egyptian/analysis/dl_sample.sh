#!/bin/bash
# Download the gender-sample clips, one dataset at a time (the Storage Box caps SFTP connections at ~10).
cd /workspace/egy
for f in lists/*.txt; do
  ds=$(basename $f .txt)
  case $ds in masri|mostafa|dahab|halasamir) sub=flac;; *) sub=wavs;; esac
  rclone copy hz:egy-tts/datasets/$ds/$sub sample/$ds --files-from $f --no-traverse --transfers 8 --checkers 2 --retries 5 --low-level-retries 20 2>&1 | tail -3
  echo "done $ds $(find sample/$ds -type f | wc -l) / $(wc -l < $f)"
done
