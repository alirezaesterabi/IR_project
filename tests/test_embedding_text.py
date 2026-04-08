"""Tests for build_embedding_text (dense retrieval string)."""

import unittest
import sys
from pathlib import Path

# Repo root on path for `src.preprocessing`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.preprocessing.embedding_text import build_embedding_text  # noqa: E402


class TestEmbeddingText(unittest.TestCase):
    def test_includes_country_program_dataset(self):
        doc = {
            "caption": "Test Entity",
            "schema": "Person",
            "metadata": {
                "country": ["ru"],
                "programId": ["US-SDN"],
                "datasets": ["us_ofac_sdn"],
            },
            "identifiers": {},
            "text_blob": "extra lexical content",
        }
        text = build_embedding_text(doc)
        self.assertIn("Test Entity", text)
        self.assertIn("Person", text)
        self.assertRegex(text, "Russia|Russian")
        self.assertIn("Listed in", text)

    def test_identifiers(self):
        doc = {
            "caption": "MV Example",
            "schema": "Vessel",
            "metadata": {"country": [], "programId": [], "datasets": []},
            "identifiers": {
                "imoNumber": ["1234567"],
                "callSign": ["ABCD"],
            },
            "text_blob": "",
        }
        text = build_embedding_text(doc)
        self.assertIn("IMO number", text)
        self.assertIn("1234567", text)
        self.assertIn("Call sign", text)

    def test_sparse_fallback_truncates_blob(self):
        doc = {
            "caption": "Only Name",
            "schema": "Company",
            "metadata": {},
            "identifiers": {},
            "text_blob": "x" * 600,
        }
        text = build_embedding_text(doc)
        self.assertIn("Only Name", text)
        self.assertIn("x" * 500, text)
        self.assertNotIn("x" * 600, text)


if __name__ == "__main__":
    unittest.main()
