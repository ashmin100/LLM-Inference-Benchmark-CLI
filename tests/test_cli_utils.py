"""Tests for llmbench.cli — utility functions _parse_list, _parse_thinking."""

import pytest
import typer
from llmbench.cli import _parse_list, _parse_thinking


class TestParseList:
    def test_single_item(self):
        assert _parse_list("qwen3:8b") == ["qwen3:8b"]

    def test_multiple_items(self):
        assert _parse_list("qwen3:8b,qwen3:14b,qwen3:30b-a3b") == [
            "qwen3:8b", "qwen3:14b", "qwen3:30b-a3b"
        ]

    def test_strips_whitespace(self):
        assert _parse_list("model-a , model-b , model-c") == [
            "model-a", "model-b", "model-c"
        ]

    def test_ignores_empty_tokens(self):
        assert _parse_list("model-a,,model-b") == ["model-a", "model-b"]

    def test_single_with_trailing_comma(self):
        assert _parse_list("model-a,") == ["model-a"]

    def test_empty_string_returns_empty_list(self):
        assert _parse_list("") == []


class TestParseThinking:
    def test_standard_returns_false(self):
        assert _parse_thinking("standard") == [False]

    def test_thinking_returns_true(self):
        assert _parse_thinking("thinking") == [True]

    def test_both_returns_both(self):
        assert _parse_thinking("both") == [False, True]

    def test_invalid_raises_bad_parameter(self):
        with pytest.raises(typer.BadParameter):
            _parse_thinking("auto")

    def test_invalid_raises_bad_parameter_case_sensitive(self):
        with pytest.raises(typer.BadParameter):
            _parse_thinking("Standard")
