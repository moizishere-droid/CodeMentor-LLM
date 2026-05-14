"""
Tests for data_formatter.py
"""

import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_formatter import format_preference_pair, SYSTEM_PROMPT


def test_format_preference_pair_basic():
    """Test basic preference pair formatting."""
    result = format_preference_pair(
        prompt="Write a Python function to reverse a string.",
        chosen="def reverse(s): return s[::-1]",
        rejected="Here is a long verbose answer..."
    )
    assert result["prompt"] == "Write a Python function to reverse a string."
    assert result["chosen"] == "def reverse(s): return s[::-1]"
    assert result["rejected"] == "Here is a long verbose answer..."


def test_format_preference_pair_keys():
    """Test preference pair has correct keys."""
    result = format_preference_pair(
        prompt="test prompt",
        chosen="good answer",
        rejected="bad answer"
    )
    assert "prompt" in result
    assert "chosen" in result
    assert "rejected" in result


def test_format_preference_pair_empty_strings():
    """Test preference pair with empty strings."""
    result = format_preference_pair(
        prompt="",
        chosen="",
        rejected=""
    )
    assert result["prompt"] == ""
    assert result["chosen"] == ""
    assert result["rejected"] == ""


def test_system_prompt_exists():
    """Test system prompt is defined."""
    assert SYSTEM_PROMPT is not None
    assert len(SYSTEM_PROMPT) > 0
    assert "coding" in SYSTEM_PROMPT.lower()


def test_format_preference_pair_returns_dict():
    """Test preference pair returns dict."""
    result = format_preference_pair("p", "c", "r")
    assert isinstance(result, dict)