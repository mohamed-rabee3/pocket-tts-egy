import gzip
import json
import random
from pathlib import Path

from pocket_tts.egyptian_text import (
    MarkLexicon,
    canonicalize_marks,
    load_mark_lexicon,
    normalize_egyptian,
    skeleton_key,
    strip_marks,
    tashkeel_augment,
)


def test_marks_kept_final_sukun_and_tatweel_removed() -> None:
    assert normalize_egyptian("السّلام عليكم، كيف الحالـــ؟") == "السّلام عليكم، كيف الحال؟"
    assert normalize_egyptian("يِرْكَبْ العَرَبِيَّة") == "يِرْكَب العَرَبِيَّة"
    assert normalize_egyptian("يِرَكِّبْ، اللَّمْبَةْ.") == "يِرَكِّب، اللَّمْبَة."
    assert normalize_egyptian("كَتَبْتِ") == "كَتَبْتِ"  # a final vowel is pronounced


def test_shadda_vowel_order_is_canonical() -> None:
    assert normalize_egyptian("يِرَكّـِب") == normalize_egyptian("يِرَكِّب")


def test_punctuation_quotes_and_latin() -> None:
    assert normalize_egyptian('قال "تمام" , ماشي ?') == "قال تمام، ماشي؟"
    assert normalize_egyptian("اعمل subscribe  و share") == "اعمل subscribe و share"


def test_numbers_kept_as_written() -> None:
    assert normalize_egyptian("جاب 895 جنيه") == "جاب 895 جنيه"
    assert normalize_egyptian("جاب ٨٩٥ جنيه") == "جاب ٨٩٥ جنيه"
    assert (
        normalize_egyptian("1,500 جنيه, و 3.5 كيلو, خصم 20%") == "1,500 جنيه، و 3.5 كيلو، خصم 20%"
    )


TEXT = "الكَهْرَبائِي جاي بُكْرَة عَشان يِرَكِّب اللَّمْبَة، 3 مَرّات."


def test_augment_modes() -> None:
    rng = random.Random(0)
    assert tashkeel_augment(TEXT, rng, [1, 0, 0, 0]) == strip_marks(TEXT)
    assert tashkeel_augment(TEXT, rng, [0, 1, 0, 0]) == TEXT
    hinted = tashkeel_augment(TEXT, random.Random(1), [0, 0, 0, 1], {"يركب"}, thin_prob=0)
    assert "يِرَكِّب" in hinted.split()
    for _ in range(200):
        out = tashkeel_augment(TEXT, rng, [1, 1, 1, 1], {"يركب"})
        # Only marks ever change: the skeleton, digits and punctuation stay.
        assert strip_marks(out) == strip_marks(TEXT)


def test_skeleton_key() -> None:
    assert skeleton_key("يِرَكِّب،") == "يركب"


def test_model_applies_its_config_normalizer() -> None:
    from types import SimpleNamespace

    from pocket_tts.models.tts_model import TTSModel

    model = SimpleNamespace(config=SimpleNamespace(text_normalizer="egyptian", mark_lexicon=None))
    assert TTSModel.normalize_text(model, "عايز يِرْكَبْ?") == "عايز يِرْكَب؟"  # type: ignore[arg-type]
    model.config.text_normalizer = None
    assert TTSModel.normalize_text(model, "x ?") == "x ?"  # type: ignore[arg-type]


def test_loader_applies_tashkeel_augmentation() -> None:
    from training.dataloader.loader import DataLoader

    seen: list[str] = []
    loader = DataLoader.__new__(DataLoader)
    loader.rng = random.Random(0)
    loader.tokenize = lambda text: seen.append(text) or [0]
    loader.tashkeel_mode_probs = [1.0, 0.0, 0.0, 0.0]
    loader._tokens(TEXT)
    assert seen == [strip_marks(TEXT)]


def test_tags_kept_variants_mapped_other_brackets_removed() -> None:
    assert normalize_egyptian("حلوة. [sighs] بس ضهري") == "حلوة. [sighs] بس ضهري"
    assert normalize_egyptian("تمام [Laughing]!") == "تمام [laughs]!"
    assert normalize_egyptian("قال [chuckle] بجد") == "قال [chuckles] بجد"
    assert normalize_egyptian("كح [cough] خلاص") == "كح خلاص"
    assert normalize_egyptian("اعمل [EN: share] للفيديو") == "اعمل share للفيديو"
    assert strip_marks("يِرْكَب [laughs]") == "يركب [laughs]"


# How the training text marks these words (counts as in /data/egy/v2/mark_lexicon.json.gz).
LEXICON = MarkLexicon(
    forms={
        "بعتذر": [("بَعْتَذِر", 294)],
        "يضبط": [("يَضْبِط", 12), ("يَضْبُط", 4)],
        "يركب": [("يِرْكَب", 167), ("يِرَكِّب", 25)],
        "أجهز": [("أَجَهِّز", 91), ("أجهّز", 31), ("أُجَهِّز", 18)],
        "ضحكت": [("ضَحِكْت", 82), ("ضَحِكَت", 40)],
        "كباية": [("كُبَايَة", 1183), ("كُبَّايَة", 17), ("كُبَايَةِ", 6)],
        "إنت": [("إِنْتَ", 1227), ("إِنْتِ", 85)],
        "حب": [("حُبّ", 2903), ("حَبّ", 463)],
    },
    known=frozenset({"بعتذر", "يضبط", "يركب", "أجهز", "ضحكت", "كباية", "إنت", "حب", "للأقصر"}),
    ambiguous=frozenset({"يركب", "أجهز", "ضحكت", "حب"}),
)


def _canon(text: str) -> str:
    return canonicalize_marks(normalize_egyptian(text), LEXICON)


def test_marks_rewritten_in_training_conventions() -> None:
    assert _canon("وَبَعْتِذِر.") == "وَبَعْتِذِر."  # a prefixed form the lexicon lacks: as written
    assert _canon("بَعْتِذِر لِحَضْرِتَك") == "بَعْتَذِر لِحَضْرِتَك"  # another transcriber's vowel
    assert _canon("يِضْبَط الضغط") == "يَضْبِط الضغط"
    assert _canon("رحنا لِلْأُقْصُر،") == "رحنا للأقصر،"  # known word, (almost) never marked: plain
    assert _canon("عشان يرَكّب اللمبة") == "عشان يِرَكِّب اللمبة"  # partial marks completed


def test_marks_that_choose_a_reading_are_kept() -> None:
    assert _canon("بيحب يِرْكَب العجلة") == "بيحب يِرْكَب العجلة"
    assert _canon("ضَحِكَت وقالتلي") == "ضَحِكَت وقالتلي"
    assert _canon("أَنَا ضَحِكْت") == "أَنَا ضَحِكْت"
    assert _canon("إِنْتِ جاهزة") == "إِنْتِ جاهزة"  # final vowel
    assert _canon("حَبّ بطيخ") == "حَبّ بطيخ"  # a vowel on an ambiguous word
    assert _canon("عشان أَجَّهَّز.") == "عشان أَجَّهَّز."  # a shadda no attested form has
    assert _canon("كُبَّايَةِ قَهْوَة") == "كُبَّايَةِ قَهْوَة"  # the vowel that makes the ة a t
    assert _canon("كَلِمَة جَدِيدَة") == "كَلِمَة جَدِيدَة"  # words the training text lacks


def test_canonicalize_leaves_unmarked_text_alone() -> None:
    text = "ابعتلي ال email، رقم 58213 [sighs] بعتذر يركب"
    assert canonicalize_marks(normalize_egyptian(text), LEXICON) == normalize_egyptian(text)


def test_load_mark_lexicon(tmp_path: Path) -> None:
    path = tmp_path / "marks.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"forms": {"بعتذر": [["بَعْتَذِر", 294]]}, "known": ["بعتذر"], "ambiguous": []}, fh)
    lexicon = load_mark_lexicon(str(path))
    assert canonicalize_marks("بَعْتِذِر", lexicon) == "بَعْتَذِر"
