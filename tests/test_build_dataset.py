"""
test_build_dataset.py
=====================
Tests for build_dataset.py -- text cleaning, HTTP helpers, data lists.

Network calls are mocked so these tests run without internet access.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from build_dataset import (
    DIVERSE_ARTICLES,
    GUTENBERG_BOOKS,
    LGBTQ_ARTICLES,
    WIKIQUOTE_PAGES,
    _get,
    _get_json,
    clean,
    strip_gutenberg_boilerplate,
)


# =============================================================================
#  clean()
# =============================================================================

class TestClean(unittest.TestCase):

    def test_removes_html_tags(self):
        """HTML tags must be stripped."""
        result = clean("<b>bold</b> text that is long enough to pass the 40 char filter here")
        self.assertNotIn("<b>", result)
        self.assertNotIn("</b>", result)

    def test_removes_wiki_links(self):
        """[[link|text]] must become just text."""
        text = "See [[Paris|the city of Paris]] for more information about this topic here."
        result = clean(text)
        self.assertNotIn("[[", result)
        self.assertIn("the city of Paris", result)

    def test_removes_templates(self):
        """{{template}} blocks must be removed."""
        text = "Some content {{cite web|url=example.com}} more content here for length purposes."
        result = clean(text)
        self.assertNotIn("{{", result)

    def test_removes_headings(self):
        """== Heading == syntax must be stripped."""
        text = "== Introduction == This is a long enough line to pass the 40 char minimum filter."
        result = clean(text)
        self.assertNotIn("==", result)

    def test_removes_bold_italic(self):
        """''bold'' and '''italic''' markers must be stripped."""
        text = "This is ''bold'' and '''very bold''' text that is long enough to survive cleaning."
        result = clean(text)
        self.assertNotIn("''", result)

    def test_removes_urls(self):
        """https:// URLs must be stripped."""
        text = "Visit https://example.com for more information about this long enough line here."
        result = clean(text)
        self.assertNotIn("https://", result)

    def test_removes_citation_markers(self):
        """[1], [23] citation markers must be removed."""
        text = "According to research [1] and further studies [23] this is a long enough sentence."
        result = clean(text)
        self.assertNotIn("[1]", result)
        self.assertNotIn("[23]", result)

    def test_drops_short_lines(self):
        """Lines under 40 chars must be dropped."""
        text = "Short\nThis is a line that is definitely longer than forty characters and will be kept.\nAlso short"
        result = clean(text)
        self.assertNotIn("Short", result)
        self.assertNotIn("Also short", result)
        self.assertIn("definitely longer", result)

    def test_keeps_long_lines(self):
        """Lines of 40+ chars must be preserved (after markup removal)."""
        long_line = "This is a line that is definitely long enough to survive the cleaning filter."
        result = clean(long_line)
        self.assertGreater(len(result), 0)

    def test_collapses_multiple_spaces(self):
        """Multiple spaces must collapse to one."""
        text = "This   has   multiple   spaces   but   is   still   long   enough   to   survive."
        result = clean(text)
        self.assertNotIn("  ", result)

    def test_collapses_excessive_newlines(self):
        """3+ consecutive newlines must collapse to 2."""
        long_line = "A" * 50
        text = long_line + "\n\n\n\n" + long_line
        result = clean(text)
        self.assertNotIn("\n\n\n", result)

    def test_empty_input(self):
        """Empty input must return empty output."""
        self.assertEqual(clean(""), "")

    def test_blank_lines_preserved_between_paragraphs(self):
        """Blank lines between paragraphs should be kept (para separator)."""
        line  = "A" * 50
        text  = line + "\n\n" + line
        result = clean(text)
        self.assertIn("\n\n", result)


# =============================================================================
#  strip_gutenberg_boilerplate()
# =============================================================================

class TestStripGutenberg(unittest.TestCase):

    def test_strips_header_and_footer(self):
        """Content between START and END markers must be extracted."""
        text = (
            "Preamble stuff nobody wants\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK ***\n"
            "Header line\n\n"
            "Actual book content here.\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK ***\n"
            "Footer boilerplate"
        )
        result = strip_gutenberg_boilerplate(text)
        self.assertIn("Actual book content", result)
        self.assertNotIn("Preamble", result)
        self.assertNotIn("Footer", result)

    def test_no_markers_returns_full_text(self):
        """Text without markers must be returned unchanged."""
        text = "Just a book without any Gutenberg markers."
        self.assertEqual(strip_gutenberg_boilerplate(text), text)

    def test_only_start_marker(self):
        """Only a start marker: return everything after it."""
        text = "*** START OF THE PROJECT GUTENBERG EBOOK ***\nBook content here."
        result = strip_gutenberg_boilerplate(text)
        self.assertIn("Book content", result)
        self.assertNotIn("START OF", result)

    def test_case_insensitive_matching(self):
        """Marker matching must be case-insensitive."""
        text = (
            "*** start of this project gutenberg ***\n"
            "Content\n"
            "*** end of this project gutenberg ***"
        )
        result = strip_gutenberg_boilerplate(text)
        self.assertIn("Content", result)

    def test_strips_leading_trailing_whitespace(self):
        """Extracted content must have no leading/trailing whitespace."""
        text = (
            "*** START OF THE PROJECT GUTENBERG EBOOK ***\n\n\n"
            "   Content   \n\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK ***"
        )
        result = strip_gutenberg_boilerplate(text)
        self.assertEqual(result, result.strip())


# =============================================================================
#  HTTP helpers
# =============================================================================

class TestGetFunction(unittest.TestCase):

    @patch("build_dataset.requests.get")
    def test_returns_text_on_200(self, mock_get):
        """200 response must return response text."""
        r = MagicMock()
        r.status_code = 200
        r.text = "Hello"
        mock_get.return_value = r
        self.assertEqual(_get("http://example.com"), "Hello")

    @patch("build_dataset.requests.get")
    def test_retries_on_429(self, mock_get):
        """429 followed by 200 must eventually return the text."""
        r429 = MagicMock()
        r429.status_code = 429
        r429.headers = {"Retry-After": "0"}

        r200 = MagicMock()
        r200.status_code = 200
        r200.text = "Success"

        mock_get.side_effect = [r429, r429, r200]
        result = _get("http://example.com")
        self.assertEqual(result, "Success")

    @patch("build_dataset.requests.get")
    def test_returns_none_on_network_error(self, mock_get):
        """Network errors must return None, not raise."""
        mock_get.side_effect = Exception("connection refused")
        self.assertIsNone(_get("http://example.com"))


class TestGetJson(unittest.TestCase):

    @patch("build_dataset._get")
    def test_parses_valid_json(self, mock_get):
        """Valid JSON text must be parsed and returned as dict."""
        mock_get.return_value = '{"key": "value", "num": 42}'
        result = _get_json("http://example.com")
        self.assertEqual(result, {"key": "value", "num": 42})

    @patch("build_dataset._get")
    def test_returns_none_on_invalid_json(self, mock_get):
        """Invalid JSON must return None, not raise."""
        mock_get.return_value = "this is not json {{"
        self.assertIsNone(_get_json("http://example.com"))

    @patch("build_dataset._get")
    def test_returns_none_when_get_fails(self, mock_get):
        """None from _get must propagate as None."""
        mock_get.return_value = None
        self.assertIsNone(_get_json("http://example.com"))


# =============================================================================
#  Data lists
# =============================================================================

class TestDataLists(unittest.TestCase):

    def test_gutenberg_not_empty(self):
        self.assertGreater(len(GUTENBERG_BOOKS), 0)

    def test_lgbtq_not_empty(self):
        self.assertGreater(len(LGBTQ_ARTICLES), 0)

    def test_diverse_not_empty(self):
        self.assertGreater(len(DIVERSE_ARTICLES), 0)

    def test_wikiquote_not_empty(self):
        self.assertGreater(len(WIKIQUOTE_PAGES), 0)

    def test_gutenberg_format(self):
        """Each entry must be (int_id, str_title)."""
        for item in GUTENBERG_BOOKS:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            book_id, title = item
            self.assertIsInstance(book_id, int)
            self.assertGreater(book_id, 0)
            self.assertIsInstance(title, str)
            self.assertGreater(len(title), 0)

    def test_no_duplicate_gutenberg_ids(self):
        """Book IDs must be unique."""
        ids = [bid for bid, _ in GUTENBERG_BOOKS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_lgbtq_no_empty_strings(self):
        for t in LGBTQ_ARTICLES:
            self.assertIsInstance(t, str)
            self.assertGreater(len(t), 0)

    def test_diverse_no_empty_strings(self):
        for t in DIVERSE_ARTICLES:
            self.assertIsInstance(t, str)
            self.assertGreater(len(t), 0)

    def test_lgbtq_contains_key_articles(self):
        """Critical LGBTQ+ articles must always be present."""
        required = [
            "Stonewall riots",
            "Harvey Milk",
            "Marsha P. Johnson",
            "Transgender history",
            "LGBT rights in the United States",
        ]
        for title in required:
            self.assertIn(title, LGBTQ_ARTICLES,
                          msg=f"'{title}' missing from LGBTQ_ARTICLES")

    def test_lgbtq_not_in_diverse(self):
        """LGBTQ articles must only be in LGBTQ_ARTICLES, not duplicated in DIVERSE."""
        # A few key ones should not appear in DIVERSE_ARTICLES
        for title in ["Stonewall riots", "Harvey Milk", "Marsha P. Johnson"]:
            self.assertNotIn(title, DIVERSE_ARTICLES,
                             msg=f"'{title}' duplicated in DIVERSE_ARTICLES")

    def test_diverse_covers_civil_rights(self):
        """DIVERSE_ARTICLES must include civil rights entries."""
        has_cr = any("King" in t or "Rosa Parks" in t or "Civil Rights" in t
                     for t in DIVERSE_ARTICLES)
        self.assertTrue(has_cr)

    def test_diverse_covers_womens_rights(self):
        """DIVERSE_ARTICLES must include women's rights entries."""
        has_wr = any("suffrage" in t.lower() or "feminism" in t.lower()
                     or "women" in t.lower()
                     for t in DIVERSE_ARTICLES)
        self.assertTrue(has_wr)

    def test_diverse_covers_science(self):
        """DIVERSE_ARTICLES must include scientists."""
        has_sci = any("Curie" in t or "Darwin" in t or "Goodall" in t
                      for t in DIVERSE_ARTICLES)
        self.assertTrue(has_sci)


if __name__ == "__main__":
    unittest.main()