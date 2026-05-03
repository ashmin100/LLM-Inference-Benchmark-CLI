"""Tests for llmbench.prompts — load_prompts(), CATEGORIES."""

import pytest
from llmbench.prompts import load_prompts, CATEGORIES


class TestLoadPrompts:
    def test_loads_all_categories_by_default(self):
        prompts = load_prompts()
        assert len(prompts) > 0

    def test_each_prompt_has_required_keys(self):
        prompts = load_prompts()
        for p in prompts:
            assert "category" in p, f"Missing 'category' in {p}"
            assert "length" in p, f"Missing 'length' in {p}"
            assert "text" in p, f"Missing 'text' in {p}"

    def test_category_values_are_valid(self):
        prompts = load_prompts()
        for p in prompts:
            assert p["category"] in CATEGORIES

    def test_length_values_are_valid(self):
        prompts = load_prompts()
        valid_lengths = {"short", "medium", "long"}
        for p in prompts:
            assert p["length"] in valid_lengths, f"Unexpected length: {p['length']}"

    def test_text_is_non_empty_string(self):
        prompts = load_prompts()
        for p in prompts:
            assert isinstance(p["text"], str)
            assert len(p["text"].strip()) > 0

    def test_filter_single_category(self):
        for cat in CATEGORIES:
            prompts = load_prompts([cat])
            assert all(p["category"] == cat for p in prompts)

    def test_filter_multiple_categories(self):
        selected = ["qa", "coding"]
        prompts = load_prompts(selected)
        returned_cats = {p["category"] for p in prompts}
        assert returned_cats == set(selected)

    def test_raises_for_unknown_category(self):
        with pytest.raises(FileNotFoundError):
            load_prompts(["nonexistent_category"])

    def test_returns_list(self):
        prompts = load_prompts()
        assert isinstance(prompts, list)
