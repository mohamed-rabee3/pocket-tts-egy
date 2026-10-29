# Egyptian Arabic Pocket TTS: training kit

Trains an Egyptian-dialect Pocket TTS (24-layer teacher, then a 6-layer CPU
student) on the `egy-tts/datasets` corpora (all but `drama3_synthetic`), with
the diacritized Jais-ASR transcripts (`transcripts.jais.diacritized.jsonl`) plus
`gemini_synthetic`. The model reads **plain, partially or fully diacritized**
text (a mark on an ambiguous word is a hint, never required), numbers as
written, the non-speech tags `[laughs]` `[sighs]` `[chuckles]`, and runs in the
continuous voice-agent mode of the Saudi kit (`pocket_tts/continuous.py`,
interleaved training).

v1 (undiacritized `transcripts.jais.jsonl`, `runs/teacher`, stopped at step 78.5k)
mispronounced homographs whose rare reading the data barely has (يركب اللمبة:
~270 install readings against ~1,830 rides); v2 (`runs/teacher_v2`) is trained
from scratch so that an LLM front-end can settle any homograph with a mark.

This branch is upstream Pocket TTS 4.0.1 plus the Saudi commit replayed on top
of it, so it carries the upstream fixes that matter for Arabic data (#394: a
final unaligned word, typically a number or an English word, is no longer cut
from the audio; #396: no weight decay on the text table; #397), REFL and the
release checks.

## Pipeline

1. **Analysis** (`analysis/`, already run once; outputs under `/workspace/egy`):
   `inventory.py` (hours, speakers), `agreement.py` (Jais vs. the acoustic
   diacritized transcript), `classify_gender.py` (audeering wav2vec2 age/gender
   + Praat F0, one to three clips per diarized speaker), `embed_speakers.py`
   (WavLM-SV voice clusters).
2. **Audio** (`scripts/prepare_all.sh`, v1), per dataset: rsync download, render
   segments (`scripts/build_manifest.py render`), delete the raw audio; latents
   precomputed by the v1 run (`/data/egy/seg/latents`).
   **v2 data** (`scripts/prepare_v2.sh`, into `/data/egy/v2`): the diacritized
   Jais text on those segments (`retext`), `gemini_synthetic` with the marks of its
   TTS input and its tags (`gemini`), word alignment on the letter skeleton,
   `merge` (weights, splits; validation text plain), `lexicon` (ambiguous words),
   `marks` (how training marks each word, for inference),
   `latents` (the v1 latents matched by audio path: nothing re-encoded) and an
   8,000-piece tokenizer over plain + full + partial text (every mark, digit and
   punctuation sign has a piece, each tag is one token).
3. **Teacher** (`configs/egy_teacher.yaml`, run by `scripts/train_supervisor.sh`,
   which waits for the data, restarts after crashes or hangs and falls back to
   `egy_teacher_b32.yaml` after an OOM; v2 runs the b32 config, batch 64 OOMs on
   30 s rows): fine-tuned from `english_2026-09_24l`
   with a fresh text embedding, cosine 2e-4 -> 2e-5 over 300k steps, effective
   batch 64. ~3.2 steps/s on one RTX PRO 6000 (96 GB): ~26 h.
4. **Pick the teacher checkpoint** on WER/CER (unseen speakers, both genders),
   `scripts/eval_homographs.py`, speaker similarity and listening. v2: **300k**
   (held-out listening test, `eval/heldout_tts.jsonl` rendered by `scripts/render_heldout.py`:
   mean 4.31 against 4.16 for 290k over the same 86 clips; validation loss flat from 260k).
5. **Student** (`configs/egy_distill_6l.yaml`, `egy_distill_6l_b32.yaml` after an OOM;
   `train_supervisor.sh` with RUN, CONFIG and LOW_MEM_CONFIG set): teacher weights
   `kept/checkpoint_00300000.pt` (EMA), 200k steps, CFG baked at 3.0 (`runs/distill_6l_v2_cfg3`);
   batch 64 peaks at 27 GB, ~5 steps/s (~11 h). The first student (`runs/distill_6l_v2`) baked
   1.0, the teacher's unguided output: female voices came out higher and thinner than their
   prompt (189 Hz against ~170), while the teacher at cfg 3 holds the voice and scored 4.83
   against 4.32 at cfg 2 on the held-out listening test.

## Data decisions

- **Text** (`pocket_tts/egyptian_text.normalize_egyptian`, applied again at
  inference through the config's `text_normalizer: egyptian`): the marks Jais heard
  are kept, except a word-final sukun (written on ~40% of words, left off on as
  many: one sound, two spellings); final vowels stay (أنتَ / أنتِ is enta / enti, the
  only way to say who is addressed). Tatweel removed, NFC (shadda/vowel order),
  Latin punctuation to Arabic, numbers and English words as written, Jais's
  `[EN: share]` to `share`, tags kept, other bracketed text removed.
- **Tashkeel augmentation** (`tashkeel_augment`, `data.tashkeel_mode_probs`): each
  time a transcript is drawn it is shown plain (40%), fully marked (20%), with a
  random 10-90% of its words marked (25%), or with marks on its ambiguous words
  only (15%, the `lexicon`: skeletons the data reads two ways, e.g. يركب); marked
  words lose some of their marks 30% of the time (a lone shadda). Validation is
  measured on plain text.
- **Filters**: no text / refused / truncated; Jais disagreeing with the
  independent acoustic transcript by > 25% CER on the letter skeleton (> 45%
  for code-switched clips, which the Arabic-only diacritizer transliterates);
  implausible speaking rate. ~1% of clips.
- **Segments**: consecutive clips of one diarized speaker (consecutive chunk
  index, gap <= 1 s) are joined up to 30 s: 59% of the hours end up in
  multi-sentence segments of 15-30 s, which long-form and turn-to-turn speech
  need (and which keep the continuous mode's 25 s state inside trained lengths).
- **Speakers**: a segment only carries `speaker` (used to draw reference voice
  prompts from another segment of the same speaker) when every clip's
  diarization confidence is >= 0.7.
- **Balancing** (`weight` per segment, drawn floor(w) + Bernoulli(frac(w)) times
  per epoch by the loader, so nothing is thrown away): every channel is capped
  at 150 h per epoch (Hala Samir and Mustafa Hosny are ~690 h each: about half
  of the corpus is two people), then female and male hours are equalised.
- **gemini_synthetic**: ~295 h, 4 studio voices (2 F, 2 M), its own QA'd
  transcripts (CER-checked at generation; numbers as digits, spoken in the audio)
  with the marks of its diacritized TTS input on the words it marked (its
  ambiguous words: 63% of clips), and 3,836 clips with `[sighs]` / `[laughs]` /
  `[chuckles]`, the only tagged data (expect the tags to work best in those 4
  voices). Weighted like one more channel: 150 h per epoch, ~12% of the draws.
  Training samples use one of its female voices (`sample_voice`).
- **Splits**: 40 held-out diarized speakers (half female, half male, 3-15 min
  each) for the new-voice evaluation, 300 seen-speaker segments.

## Gender and speakers (measured)

`analysis/classify_gender.py` on 64,311 clips (one per diarized speaker, three
for speakers over 5 min; the classifier agrees with the F0 median on 92%):
**1,647 h female (59%), 1,155 h male (41%)**. By channel the split is stark:
Hala Samir, dahab, fatma-abuhaty, somayasserscience, nourhan-elsheikh and
heya-wbs (80%) are female; Mustafa Hosny, droos, mostafa, daheeh, masri and
the playlist are male. Mustafa Hosny alone is 58% of the male hours.

Speakers are the bigger issue: nearly every channel is one host. The three
largest voices (Hala Samir, Mustafa Hosny, the heya-wbs host) carry 1,450-1,950 h
and the top five 57-75% of all hours (the range is the voice-clustering
threshold, `analysis/embed_speakers.py`: WavLM-SV separates speakers poorly on
this Demucs-processed audio). The diversity is in the long tail of guests. A
model fine-tuned mostly on a few people clones *new* voices worse and pulls
them towards those people, so the unseen-speaker split is the evaluation to
watch, per gender; the channel cap above is the main mitigation in the data
(more multi-speaker Egyptian data is the real one).

Pocket TTS clones the voice from the prompt; it does not pick a gender, so an
unbalanced corpus does not make the model "male" or "female". What imbalance
does is make the minority side worse: lower speaker similarity, more artifacts,
and, in long continuous generations that condition on their own output, a drift
of pitch and timbre towards the majority. The CFG null branch (voice dropout)
also learns the majority voice. Equalising the hours per epoch (above) and
evaluating every metric per gender is the fix; speaker *diversity* matters as
much as hours, which is the reason for the channel cap.

## Homographs (يركب العربية / يركب اللمبة)

Plain text: every audio frame attends to the whole chunk's text, so context can
decide a reading, but only readings the data has: the install reading of يركب is
~13% of its uses, and v1 flipped between readings from one checkpoint to the
next. No TTS can learn every homograph from context (some are undecidable from the
text: كتابك is ketābak or ketābik), so v2 takes marks as hints: the agent's LLM
marks the ambiguous words, plain text keeps working.

- Measure it: `scripts/eval_homographs.py` (35 sentences in `eval/`, both
  readings of 16 ambiguous skeletons; forced choice between the two readings
  under an acoustic diacritizing ASR), with `--text plain` (context alone) and
  `--text hint` (the target word marked). The scorer agrees with the corpus labels
  on 75% of real clips: compare checkpoints with it, confirm by ear.
- Improve the plain-text side with data: sentences that use the rare readings,
  spoken from a diacritized input (as `gemini_synthetic` was), verified by the
  acoustic ASR.

## Inference

`egyptian/configs/egy_inference_{24l,6l}.yaml` set `text_normalizer: egyptian`
(every entry point, `generate_audio`, the CLI and `VoiceAgentTTS`, normalizes the
text as the training text was), `mark_lexicon` (below), `capitalize_first_letter: false`
and `min_tokens_per_second` (the per-chunk generation budget; the English 3.0 cuts slow
Egyptian speech before its EOS).

**Marks in the training text's conventions** (`mark_lexicon`, built by
`build_manifest.py marks`: every marked form of each training word, 3.9 MB). The model
learned Jais's way of marking each word, not phonetics: a mark another writer would
put (بَعْتِذِر, where all 294 marked uses in training are بَعْتَذِر) or marks on a word
training almost never marked (لِلْأُقْصُر: 2 marked uses, 22 plain) reach it as a
token sequence it rarely saw, and the word comes out slightly off (held-out listening
test, 290k). `canonicalize_marks` rewrites each marked word to the attested form that
agrees with the caller's marks (partial marks completed, another vowel inside the word
replaced), or plain when training had the word but not marked; it keeps what the
caller chose: a shadda (أَجَّهَّز), a vowel against a sukun, the final vowel (إنتِ,
كُبَّايَةِ قَهْوَة), any vowel on an ambiguous word (حَبّ / حُبّ), and new words.

Front-end for a voice agent: let the LLM write plain Egyptian text and mark only
the words it judges ambiguous, with Egyptian pronunciation and no MSA case endings
(the model pronounces the marks it is given): "عشان يرَكِّب اللمبة", "إنتِ عاملة إيه".
Tags go inline: "ده كان يوم طويل [sighs] بس الحمد لله". Streaming chunks start
before the rest of the sentence is known (`first_chunk_words`), so a homograph at
the end of a first chunk cannot use the context after it; the mark settles it.

## Continuous mode

`VoiceAgentTTS(model, voice, interleaved=True, normalize_text=normalize_egyptian)`.
It is not the upstream default (upstream generates each sentence from the bare
voice prompt with a fresh decoder); it is this kit's extension, and the
interleaved training layout is tested to match it step for step
(`training/tests/test_interleaved.py`). Compare it against the default mode on
long paragraphs (WER, speaker similarity over time, seams) when picking a
checkpoint.
