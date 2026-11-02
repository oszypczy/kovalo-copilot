"""Komentarze z przeglądu ręcznego: `review.yaml` (kontrakt F2, 14.3)."""

from __future__ import annotations

import hashlib
import re

import pytest

VALID = """\
version: 1
reviewed_on: 2026-11-02
intro: |
  Przegląd T2.
items:
  - variants: [ours-ctx]
    id: M04
    answer_digest: {ours-ctx: 3f2a9c0d1b7e}
    comment: |
      Odmowa uczciwa.
      Druga linia.
  - variants: [ours-ctx, ours-hybrid]
    id: M09
    comment: Przesadzone odpowiedzi.
notes:
  - Pozostałe uzasadnienia sędziego.
"""


def _write(tmp_path, text: str):
    path = tmp_path / "review.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_review_ok(tmp_path):
    from kovalo.eval.review import REVIEW_FILE, load_review

    assert REVIEW_FILE == "review.yaml"
    review = load_review(_write(tmp_path, VALID))
    assert review is not None and review.version == 1
    assert str(review.reviewed_on) == "2026-11-02"
    assert review.intro.strip() == "Przegląd T2."
    assert [i.id for i in review.items] == ["M04", "M09"]
    assert review.notes == ["Pozostałe uzasadnienia sędziego."]
    assert review.find("ours-hybrid", "M09") is review.items[1]
    assert review.find("ours-hybrid", "M04") is None
    assert review.find("ours-ctx", "M99") is None
    assert review.items[0].answer_digest == {"ours-ctx": "3f2a9c0d1b7e"}


def test_load_review_missing_returns_none(tmp_path):
    from kovalo.eval.review import load_review

    assert load_review(tmp_path / "review.yaml") is None


@pytest.mark.parametrize(
    "text",
    [
        "version: [1\n  : :",  # zły YAML
        VALID.replace("version: 1", "version: 2"),
        VALID.replace("notes:", "nieznany_klucz: 1\nnotes:"),
        VALID.replace("      Odmowa uczciwa.\n      Druga linia.\n", "      \n"),  # pusty comment
        VALID.replace(
            "notes:",
            "  - {variants: [ours-ctx], id: M04, comment: Drugi raz.}\nnotes:",
        ),  # zduplikowana para (ours-ctx, M04)
        # klucz answer_digest spoza variants
        VALID.replace("{ours-ctx: 3f2a9c0d1b7e}", "{ours-hybrid: 3f2a9c0d1b7e}"),
    ],
    ids=["bad_yaml", "version_2", "unknown_key", "empty_comment", "duplicate_pair", "digest_key"],
)
def test_load_review_rejects_bad_files(tmp_path, text):
    from kovalo.config import ConfigError
    from kovalo.eval.review import load_review

    path = _write(tmp_path, text)
    with pytest.raises(ConfigError, match=re.escape(path.name)):
        load_review(path)


def test_load_review_rejects_duplicate_pair_across_items(tmp_path):
    from kovalo.config import ConfigError
    from kovalo.eval.review import load_review

    text = (
        "version: 1\nreviewed_on: 2026-11-02\nitems:\n"
        "  - {variants: [ours-ctx], id: M04, comment: a}\n"
        "  - {variants: [ours-hybrid, ours-ctx], id: M04, comment: b}\n"
    )
    path = _write(tmp_path, text)
    with pytest.raises(ConfigError, match=r"review\.yaml"):
        load_review(path)


def test_load_review_error_names_field(tmp_path):
    from kovalo.config import ConfigError
    from kovalo.eval.review import load_review

    text = (
        "version: 1\nreviewed_on: 2026-11-02\nitems:\n"
        "  - {variants: [ours-ctx], id: M04, comment: a}\n"
        "  - {variants: [ours-ctx], id: M05, comment: a}\n"
        "  - {variants: [ours-ctx], id: M06, comment: '  '}\n"
    )
    with pytest.raises(ConfigError, match=r"items\.2\.comment"):
        load_review(_write(tmp_path, text))


def test_check_review_keys(tmp_path):
    from kovalo.config import ConfigError
    from kovalo.eval.review import check_review_keys, load_review

    review = load_review(_write(tmp_path, VALID))
    variants = ["ours-ctx", "ours-hybrid", "reference-ctx"]
    ids = ["M04", "M09", "M10"]
    check_review_keys(review, variants=variants, ids=ids)  # poprawne: bez wyjątku

    with pytest.raises(ConfigError, match="ours-hybrid") as info:
        check_review_keys(review, variants=["ours-ctx", "reference-ctx"], ids=ids)
    assert "ours-ctx" in str(info.value)  # lista znanych etykiet

    with pytest.raises(ConfigError, match="M09") as info:
        check_review_keys(review, variants=variants, ids=["M04", "M10"])
    assert "M10" in str(info.value)  # lista znanych id


def test_answer_digest_stable():
    from kovalo.eval.review import answer_digest

    first = answer_digest("Odpowiedź z źródłem [POL-001].")
    assert first == answer_digest("Odpowiedź z źródłem [POL-001].")
    assert re.fullmatch(r"[0-9a-f]{12}", first)
    assert first == hashlib.sha256("Odpowiedź z źródłem [POL-001].".encode()).hexdigest()[:12]
    assert first != answer_digest("Inna odpowiedź.")
    # brak odpowiedzi: skrót błędu; ignorowany błąd, gdy odpowiedź jest
    assert answer_digest(None, "ValueError: x") == hashlib.sha256(b"ValueError: x").hexdigest()[:12]
    assert answer_digest("tekst", "ValueError: x") == answer_digest("tekst")
    assert answer_digest(None) == hashlib.sha256(b"").hexdigest()[:12]


def test_review_models_forbid_extra_fields():
    from pydantic import ValidationError

    from kovalo.eval.review import Review, ReviewItem

    with pytest.raises(ValidationError):
        ReviewItem(variants=["ours-ctx"], id="M01", comment="x", nieznane=1)
    with pytest.raises(ValidationError):
        Review(version=1, reviewed_on="2026-11-02", nieznane=1)
    with pytest.raises(ValidationError):
        ReviewItem(variants=["Zła Etykieta"], id="M01", comment="x")
    with pytest.raises(ValidationError):
        ReviewItem(variants=["ours-ctx"], id="M01", comment="x", answer_digest={"ours-ctx": "XYZ"})
