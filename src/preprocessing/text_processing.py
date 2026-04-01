"""
Text preprocessing for OpenSanctions entities.

Key design decision from the learning notebook (Section 2d):
- Entity names (name, alias): normalize only — do NOT lemmatize.
  Lemmatizing "Kremlin Finance Group" would corrupt the name.
- Free text (notes, description): normalize + lemmatize for better recall.
  "evading" → "evade", "sanctions" → "sanction".
- Identifiers (imoNumber, mmsi): never touch — exact-match only.
"""

import re
import unicodedata
from functools import lru_cache
from typing import Optional

import nltk
import spacy

# Download required NLTK data on first import (silent)
for _pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{_pkg}" if _pkg != "punkt" else f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords as _nltk_sw


def _load_spacy() -> spacy.language.Language:
    """Load spaCy model, downloading if absent."""
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True, capture_output=True,
        )
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])


class TextProcessor:
    """
    Stateful text processor — initialised once per pipeline run to avoid
    reloading spaCy and NLTK on every document.

    Usage
    -----
    tp = TextProcessor()
    blob = tp.build_name_text(["Viktor Petrov", "Виктор Петров"])
    desc = tp.build_desc_text("Russian oligarch evading OFAC sanctions.")
    """

    def __init__(
        self,
        extra_stopwords: Optional[set[str]] = None,
        min_token_length: int = 2,
    ):
        self._nlp = _load_spacy()
        self._stop_words: set[str] = set(_nltk_sw.words("english"))
        if extra_stopwords:
            self._stop_words |= extra_stopwords
        self._min_len = min_token_length

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_latin_base(char: str) -> bool:
        """Return True if *char* is a Latin-script base character."""
        try:
            return "LATIN" in unicodedata.name(char, "")
        except ValueError:
            return False

    @staticmethod
    def normalize(text: str) -> str:
        """
        Canonical normalization applied to ALL text:
          1. NFC compose (ensure consistent starting form)
          2. NFD decompose (separate base characters from combining marks)
          3. Selectively strip combining marks (category Mn) that follow a
             Latin base character — this removes accents from é, ñ, ü etc.
             Combining marks on non-Latin bases (Cyrillic й = и + breve,
             ё = е + diaeresis) are preserved.
          4. NFC recompose (restore precomposed forms, e.g. и + breve → й)
          5. Lowercase
          6. Remove punctuation, keep alphanumeric + spaces
          7. Collapse whitespace
        """
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = unicodedata.normalize("NFD", text)
        # Walk the decomposed characters, tracking the most recent base
        out: list[str] = []
        last_base_is_latin = False
        for c in text:
            cat = unicodedata.category(c)
            if cat == "Mn":
                # Only strip combining marks following a Latin base character
                if last_base_is_latin:
                    continue
            else:
                # Update base-script tracker for non-combining characters
                last_base_is_latin = TextProcessor._is_latin_base(c)
            out.append(c)
        text = "".join(out)
        text = unicodedata.normalize("NFC", text)
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_valid_token(self, token: str) -> bool:
        return (
            len(token) >= self._min_len
            and token not in self._stop_words
            and not token.isdigit()
        )

    # ------------------------------------------------------------------
    # Public processing methods
    # ------------------------------------------------------------------

    def tokenize_name(self, text: str) -> list[str]:
        """
        Normalize a name string and split into tokens.
        No lemmatization — entity name integrity must be preserved.
        No stopword removal — only filtered by minimum token length.
        """
        normalized = self.normalize(text)
        return [t for t in normalized.split() if len(t) >= self._min_len]

    def tokenize_and_lemmatize(self, text: str) -> list[str]:
        """
        Full NLP pipeline for free-text fields (notes, description):
        normalize → spaCy tokenize → lemmatize → filter stop words.
        """
        normalized = self.normalize(text)
        if not normalized:
            return []
        doc = self._nlp(normalized)
        return [
            token.lemma_
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and self._is_valid_token(token.lemma_)
        ]

    # ------------------------------------------------------------------
    # Field-level builders (used by document_builder.py)
    # ------------------------------------------------------------------

    def build_name_text(self, values: list[str]) -> str:
        """
        Produce a normalized (non-lemmatized) string from name/alias values.
        All name variants are concatenated so BM25 can match any of them.
        """
        tokens = []
        for v in values:
            tokens.extend(self.tokenize_name(v))
        return " ".join(tokens)

    def build_desc_text(self, values: list[str]) -> str:
        """
        Produce a lemmatized string from free-text fields (notes, description).
        """
        tokens = []
        for v in values:
            tokens.extend(self.tokenize_and_lemmatize(v))
        return " ".join(tokens)

    def build_keyword_text(self, values: list[str]) -> str:
        """
        Normalize a flat list of keyword strings (topics, sector, legalForm).
        These are short terms so we normalize but skip heavy NLP.
        """
        tokens = []
        for v in values:
            norm = self.normalize(v)
            if norm:
                tokens.append(norm)
        return " ".join(tokens)

    def build_sanctions_text(self, sanctions: list[dict]) -> str:
        """
        Flatten nested sanctions sub-objects into searchable text.
        Extracts only free-text fields: authority, reason.
        programId is excluded — it is a structured identifier routed to
        metadata by document_builder.py (see Bug 5 in audit).
        """
        parts: list[str] = []
        for s in sanctions:
            if not isinstance(s, dict):
                continue
            props = s.get("properties", {})
            for field in ("authority", "reason"):
                for val in props.get(field, []):
                    lemmatized = self.tokenize_and_lemmatize(str(val))
                    parts.extend(lemmatized)
        return " ".join(parts)

    def build_address_text(self, address_entities: list[dict]) -> str:
        """
        Flatten nested addressEntity sub-objects into searchable text.
        Extracts: full address, city.
        """
        parts: list[str] = []
        for ae in address_entities:
            if not isinstance(ae, dict):
                continue
            props = ae.get("properties", {})
            for field in ("full", "city"):
                for val in props.get(field, []):
                    norm = self.normalize(str(val))
                    if norm:
                        parts.append(norm)
        return " ".join(parts)


# ------------------------------------------------------------------
# Module-level convenience instance (lazy singleton pattern)
# ------------------------------------------------------------------

_default_processor: Optional[TextProcessor] = None


def get_default_processor() -> TextProcessor:
    """Return a module-level TextProcessor, creating it on first call."""
    global _default_processor
    if _default_processor is None:
        _default_processor = TextProcessor()
    return _default_processor


if __name__ == "__main__":
    tp = TextProcessor()

    tests = [
        ("name",  "Viktor Petrov"),
        ("name",  "Виктор Петров"),
        ("name",  "新疆纺织有限公司"),
        ("desc",  "Russian oligarch evading OFAC sanctions since 2015."),
        ("desc",  "Oil tanker suspected of sanctions evasion in the Baltic Sea."),
        ("kwrd",  "sanction"),
    ]

    print(f"{'Mode':<6} {'Input':<55} {'Output'}")
    print("-" * 100)
    for mode, text in tests:
        if mode == "name":
            out = tp.build_name_text([text])
        elif mode == "desc":
            out = tp.build_desc_text([text])
        else:
            out = tp.build_keyword_text([text])
        print(f"{mode:<6} {text:<55} {out}")

    # --- Inline Cyrillic normalization test (not executed) ---
    # result = tp.normalize("Электростальский")
    # assert "й" in result, f"FAIL: й was stripped -> got '{result}'"
    # assert result == "электростальский", f"FAIL: got '{result}'"
    # # Before fix: "Электростальский" -> "электростальскии" (й corrupted to и)
    # # After fix:  "Электростальский" -> "электростальский" (й preserved)
    # print(f"Cyrillic test PASSED: й preserved in '{result}'")
