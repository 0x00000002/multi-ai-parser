import pytest
from src.Parser import Parser

class TestParser:
    def test_extract_text(self):
        text = "Hello <tag>content</tag> world"
        result = Parser.extract_text(text, "<tag>", "</tag>")
        assert result == "content"

    def test_extract_text_multiple(self):
        text = "Hello <tag>content1</tag> and <tag>content2</tag>"
        result = Parser.extract_text(text, "<tag>", "</tag>")
        assert result == "content1"

    def test_extract_text_no_match(self):
        text = "Hello world"
        result = Parser.extract_text(text, "<tag>", "</tag>")
        assert result == ""

    def test_extract_text_nested(self):
        text = "Hello <outer><inner>content</inner></outer>"
        result = Parser.extract_text(text, "<inner>", "</inner>")
        assert result == "content"
