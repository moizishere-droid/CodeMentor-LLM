"""
Tests for data_cleaner.py
"""

import pytest
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_cleaner import (
    remove_duplicates,
    remove_nulls,
    filter_low_quality,
    is_low_quality
)


def test_remove_duplicates_basic():
    """Test duplicate removal."""
    df = pd.DataFrame({"text": ["hello", "hello", "world"]})
    result, removed = remove_duplicates(df)
    assert len(result) == 2
    assert removed == 1


def test_remove_duplicates_no_duplicates():
    """Test with no duplicates."""
    df = pd.DataFrame({"text": ["hello", "world", "python"]})
    result, removed = remove_duplicates(df)
    assert len(result) == 3
    assert removed == 0


def test_remove_nulls_basic():
    """Test null removal."""
    df = pd.DataFrame({"text": ["hello", None, "world"]})
    result, removed = remove_nulls(df)
    assert len(result) == 2
    assert removed == 1


def test_remove_nulls_no_nulls():
    """Test with no nulls."""
    df = pd.DataFrame({"text": ["hello", "world"]})
    result, removed = remove_nulls(df)
    assert len(result) == 2
    assert removed == 0


def test_is_low_quality_short_response():
    """Test low quality detection for short response."""
    text = "<|start_header_id|>assistant<|end_header_id|>\nYes<|eot_id|>"
    assert is_low_quality(text) == True


def test_is_low_quality_good_response():
    """Test low quality detection for good response."""
    text = "<|start_header_id|>assistant<|end_header_id|>\ndef reverse(s): return s[::-1]<|eot_id|>"
    assert is_low_quality(text) == False


def test_filter_low_quality_basic():
    """Test low quality filtering."""
    df = pd.DataFrame({
        "text": [
            "<|start_header_id|>assistant<|end_header_id|>\nYes<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\ndef reverse(s): return s[::-1]<|eot_id|>",
        ]
    })
    result, removed = filter_low_quality(df)
    assert len(result) == 1
    assert removed == 1


def test_remove_duplicates_returns_tuple():
    """Test remove_duplicates returns tuple."""
    df = pd.DataFrame({"text": ["hello"]})
    result = remove_duplicates(df)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_remove_nulls_empty_string():
    """Test null removal with empty strings."""
    df = pd.DataFrame({"text": ["hello", "", "world"]})
    result, removed = remove_nulls(df)
    assert len(result) == 2
    assert removed == 1