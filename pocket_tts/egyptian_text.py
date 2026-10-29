"""Egyptian Arabic text normalization and tashkeel augmentation, shared by training
and inference.

The Egyptian models read plain, partially or fully diacritized text. Training text
is the diacritized Jais-ASR transcript (transcripts.jais.diacritized.jsonl: the
marks of what was said), and `tashkeel_augment` shows each sentence plain, fully
marked, partially marked or with marks on its ambiguous words only, so every form
a caller may send is in distribution. Marks are hints, never required: plain text
is read from context, as a native reader does; a mark on يرَكِّب settles a homograph
the context cannot.

Every transcript, and every text sent to the model at inference, goes through
`normalize_egyptian`:

- Tatweel and Quranic annotation marks are removed, then NFC, so a vowel and a
  shadda on one letter always come in the same order.
- A word-final sukun is removed: a final consonant without a mark already means
  "no vowel" (Jais writes the sukun on ~40% of words and leaves it off on as many,
  two spellings of one sound). Every other mark is kept, final vowels included
  (كتبتِ is katabti).
- Non-speech tags: [laughs], [sighs] and [chuckles] (the gemini_synthetic tags, one
  token each) are kept, and variants ([Laugh], [laughing]) map onto them; Jais's
  [EN: share] becomes the English word it marks; anything else in brackets is not
  spoken and is removed.
- Latin , ? ; become the Arabic ، ؟ ؛ (a comma between digits, "1,500", is
  kept); quotes and parentheses are dropped (the words inside are kept);
  whitespace is collapsed.
- Numbers and Latin-script words (code switching) are kept as written.

At inference, `canonicalize_marks` then writes the marks of each word the way the
training text writes that word (the Jais conventions the model learned): marks a
caller adds are read through the tokens the training text made of them, so a
spelling the training text never used (بَعْتِذِر for its بَعْتَذِر) or marks on a word
it almost never marked (لِلْأُقْصُر) reach the model as sequences it rarely saw.
"""

import functools
import gzip
import json
import random
import re
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass

SUKUN = "ْ"
SHADDA = "ّ"
DIACRITICS = "ًٌٍَُِّْٰ"  # tanween, vowels, shadda, sukun, dagger alif
_LETTERS = "ء-يٱ-ۓ"
_REMOVE = re.compile("[ـؐ-ؚۖ-ۭ]")
_FINAL_SUKUN = re.compile(f"{SUKUN}(?![{_LETTERS}{DIACRITICS}])")
_MARKS = re.compile(f"[{DIACRITICS}]")
_DROP = re.compile("[\"«»“”„‘’'()\\[\\]{}<>]")
_COMMA = re.compile(r"(?<!\d),|,(?!\d)")
_PUNCT = str.maketrans({"?": "؟", ";": "؛"})
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([،؟؛.!:])")
_EDGE_PUNCT = "،؟؛.!:,-"

TAGS = ("[laughs]", "[sighs]", "[chuckles]")
_TAG_VARIANTS = {
    f"{stem}{end}": tag
    for tag, stem in zip(TAGS, ("laugh", "sigh", "chuckl"), strict=True)
    for end in (["", "s", "ing"] if stem != "chuckl" else ["e", "es", "ing"])
}
# Tags pass the punctuation rules as private-use characters, restored at the end.
_TAG_CODE = {tag: chr(0xE000 + i) for i, tag in enumerate(TAGS)}
_BRACKETED = re.compile(r"\[([^\[\]]*)\]")
_ENGLISH = re.compile(r"^\s*EN\s*:\s*", re.IGNORECASE)

# tashkeel_augment modes, in the order of its mode_probs.
MODES = ("plain", "full", "partial", "hint")


def _bracketed(m: re.Match) -> str:
    inner = m.group(1).strip()
    if _ENGLISH.match(inner):
        return f" {_ENGLISH.sub('', inner)} "
    tag = _TAG_VARIANTS.get(inner.lower())
    return f" {_TAG_CODE[tag]} " if tag else " "


def normalize_egyptian(text: str) -> str:
    # Tatweel first: between a shadda and a vowel it blocks NFC's reordering.
    text = unicodedata.normalize("NFC", _REMOVE.sub("", text))
    text = _BRACKETED.sub(_bracketed, text)
    text = _FINAL_SUKUN.sub("", text)
    text = _DROP.sub(" ", text)
    text = _COMMA.sub("،", text).translate(_PUNCT)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", _SPACES.sub(" ", text))
    for tag, code in _TAG_CODE.items():
        text = text.replace(code, tag)
    return text.strip()


def strip_marks(text: str) -> str:
    """The letter skeleton: every diacritic removed."""
    return _MARKS.sub("", text)


def skeleton_key(word: str) -> str:
    """A word's lookup key in the ambiguous-word lexicon: no marks, no edge punctuation."""
    return strip_marks(word).strip(_EDGE_PUNCT)


def _thin(word: str, rng: random.Random, keep: float) -> str:
    return _MARKS.sub(lambda m: m.group(0) if rng.random() < keep else "", word)


def tashkeel_augment(
    text: str,
    rng: random.Random,
    mode_probs: Sequence[float],
    ambiguous: Collection[str] = (),
    thin_prob: float = 0.3,
) -> str:
    """Training augmentation of a diacritized text, mode drawn from mode_probs
    (plain, full, partial, hint):

    - plain: every mark removed;
    - full: the text as transcribed;
    - partial: each word keeps its marks with a per-sentence probability in 0.1-0.9;
    - hint: the words of the ambiguous lexicon keep their marks (90%), others
      rarely (5%), as an LLM asked to mark only ambiguous words would write.

    A word that keeps its marks loses some of them with probability thin_prob
    (half-marked words, a lone shadda), in every mode but plain and full.
    """
    mode = MODES[rng.choices(range(len(MODES)), weights=mode_probs)[0]]
    if mode == "plain":
        return strip_marks(text)
    if mode == "full":
        return text
    words = text.split(" ")
    if mode == "partial":
        p = rng.uniform(0.1, 0.9)
        marked = [rng.random() < p for _ in words]
    else:
        marked = [rng.random() < (0.9 if skeleton_key(w) in ambiguous else 0.05) for w in words]
    out = []
    for w, m in zip(words, marked, strict=True):
        if not m:
            out.append(strip_marks(w))
        elif rng.random() < thin_prob:
            out.append(_thin(w, rng, keep=rng.uniform(0.3, 0.8)))
        else:
            out.append(w)
    return " ".join(out)


ARABIC_WORD = re.compile(f"^[{_LETTERS}{DIACRITICS}]+$")
_EDGES = re.compile(f"^([{re.escape(_EDGE_PUNCT)}]*)(.*?)([{re.escape(_EDGE_PUNCT)}]*)$", re.DOTALL)


def split_edges(token: str) -> tuple[str, str, str]:
    """A space-separated token as (leading punctuation, word, trailing punctuation)."""
    lead, core, trail = _EDGES.match(token).groups()  # type: ignore[union-attr]
    return lead, core, trail


@dataclass(frozen=True)
class MarkLexicon:
    """How the training text marks its words (build_manifest.py marks):
    forms: skeleton -> its marked forms, most used first, with their counts;
    known: skeletons the training text has at all (plain or marked);
    ambiguous: skeletons read two ways or more (the hint lexicon)."""

    forms: dict[str, list[tuple[str, int]]]
    known: frozenset[str]
    ambiguous: frozenset[str]


@functools.lru_cache(maxsize=4)
def load_mark_lexicon(path: str) -> MarkLexicon:
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as fh:
        d = json.load(fh)
    return MarkLexicon(
        {k: [(f, n) for f, n in v] for k, v in d["forms"].items()},
        frozenset(d["known"]),
        frozenset(d["ambiguous"]),
    )


def _letter_marks(word: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ch in word:
        if ch not in DIACRITICS:
            out.append((ch, ""))
        elif out:
            out[-1] = (out[-1][0], out[-1][1] + ch)
    return out


def _compare(
    given: list[tuple[str, str]], form: list[tuple[str, str]], strict: bool
) -> tuple[float, int]:
    """(agreement, contradictions) of the marks a caller wrote against an attested
    form of the same skeleton, letter by letter.

    Contradictions, which the caller meant: a mark the form does not have at all
    (the form would drop it: the shadda of أَجَّهَّز, the vowel that makes the ة of
    كُبَّايَةِ قَهْوَة a t), a sukun aside, which Jais writes on some words and not
    others; a shadda the form lacks; a vowel against a sukun (another syllable
    structure); a different final vowel (إنتَ / إنتِ); any different vowel on an
    ambiguous word (حَبّ / حُبّ). A different vowel inside any other word is how two
    transcribers write one word (بَعْتِذِر / بَعْتَذِر): it only lowers the agreement,
    as does a shadda the caller left out."""
    agree, hard = 0.0, 0
    last = len(given) - 1
    for i, ((_, a), (_, b)) in enumerate(zip(given, form, strict=True)):
        if a and not b and a != SUKUN:
            hard += 1
        if not a or not b:
            continue
        if SHADDA in a and SHADDA not in b:
            hard += 1
            continue
        agree += 2 if SHADDA in a else -0.5 if SHADDA in b else 0
        va, vb = a.replace(SHADDA, ""), b.replace(SHADDA, "")
        if not va or not vb:
            continue
        if va == vb:
            agree += 1
        elif strict or i == last or (SUKUN in va) != (SUKUN in vb):
            hard += 1
        else:
            agree -= 0.5
    return agree, hard


def _canonical(word: str, lexicon: MarkLexicon) -> str:
    skeleton = strip_marks(word)
    forms = lexicon.forms.get(skeleton)
    if not forms:
        # The training text (almost) never marked this word: plain is how the model
        # knows it. A word it never had at all keeps the caller's marks.
        return skeleton if skeleton in lexicon.known else word
    given = _letter_marks(word)
    strict = skeleton in lexicon.ambiguous
    best: tuple[tuple[float, int], str] | None = None
    for form, count in forms:
        agree, hard = _compare(given, _letter_marks(form), strict)
        if hard == 0 and (best is None or (agree, count) > best[0]):
            best = ((agree, count), form)
    # No attested form agrees: a reading the training text never wrote (أَجَّهَّز), as
    # the caller wrote it.
    return best[1] if best else word


def canonicalize_marks(text: str, lexicon: MarkLexicon) -> str:
    """Marks rewritten in the training text's conventions, word by word (text already
    through normalize_egyptian; unmarked words, numbers, Latin and tags unchanged):

    - the attested form that agrees with the caller's marks, the most used one on a
      tie: partial marks are completed (يرَكّب -> يرَكِّب), another transcriber's vowel
      inside a word is replaced (بَعْتِذِر -> بَعْتَذِر), and the reading the caller chose
      is kept (see `_compare` for what counts as a choice);
    - plain if the training text has the word but (almost) never marked it;
    - as written if no attested form agrees, or the word is new."""
    out = []
    for token in text.split(" "):
        lead, core, trail = split_edges(token)
        if core != strip_marks(core) and ARABIC_WORD.match(core):
            core = _canonical(core, lexicon)
        out.append(lead + core + trail)
    return " ".join(out)
