"""
Tests for inference pipeline.
"""

import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import validate_input


def test_validate_input_valid():
    """Test valid input passes validation."""
    is_valid, error = validate_input("Write a Python function to reverse a string.")
    assert is_valid == True
    assert error == ""


def test_validate_input_empty():
    """Test empty input fails validation."""
    is_valid, error = validate_input("")
    assert is_valid == False
    assert "empty" in error.lower()


def test_validate_input_whitespace():
    """Test whitespace only input fails validation."""
    is_valid, error = validate_input("   ")
    assert is_valid == False
    assert "whitespace" in error.lower()


def test_validate_input_too_long():
    """Test too long input fails validation."""
    long_input = "a" * 3000
    is_valid, error = validate_input(long_input)
    assert is_valid == False
    assert "long" in error.lower()


def test_validate_input_non_string():
    """Test non-string input fails validation."""
    is_valid, error = validate_input(123)
    assert is_valid == False
    assert "string" in error.lower()


def test_validate_input_normal_question():
    """Test normal coding question passes."""
    is_valid, error = validate_input("What is a decorator in Python?")
    assert is_valid == True
    assert error == ""


def test_validate_input_max_length():
    """Test input at exactly max length passes."""
    input_2048 = "a" * 2048
    is_valid, error = validate_input(input_2048)
    assert is_valid == True


def test_validate_input_returns_tuple():
    """Test validate_input returns tuple."""
    result = validate_input("test")
    assert isinstance(result, tuple)
    assert len(result) == 2
    