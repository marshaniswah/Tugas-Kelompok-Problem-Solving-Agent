"""
Unit tests for validation module.
Verifies item conservation, duplicate rejection, capacity constraint check,
and objective value consistency using item indices.
"""

import pytest
from src.validation import validate_packing_solution


def test_valid_solution():
    weights = [50, 40, 60, 30, 20]
    capacity = 100
    # Bin 0: items 0, 1 (50+40=90 <= 100)
    # Bin 1: items 2, 3 (60+30=90 <= 100)
    # Bin 2: item 4 (20 <= 100)
    bins = [[0, 1], [2, 3], [4]]

    valid, msg = validate_packing_solution(bins, weights, capacity, reported_bins=3)
    assert valid is True
    assert "valid" in msg.lower()


def test_capacity_exceeded():
    weights = [60, 50, 40]
    capacity = 100
    # Bin 0: items 0, 1 (60+50=110 > 100)
    bins = [[0, 1], [2]]

    valid, msg = validate_packing_solution(bins, weights, capacity, reported_bins=2)
    assert valid is False
    assert "exceeds capacity" in msg.lower()


def test_missing_item():
    weights = [30, 40, 50]
    capacity = 100
    # Missing item index 2
    bins = [[0, 1]]

    valid, msg = validate_packing_solution(bins, weights, capacity, reported_bins=1)
    assert valid is False
    assert "mismatch" in msg.lower() or "missing" in msg.lower()


def test_duplicate_item():
    weights = [30, 40, 50]
    capacity = 100
    # Item index 1 duplicated, item index 2 missing
    bins = [[0, 1], [1]]

    valid, msg = validate_packing_solution(bins, weights, capacity, reported_bins=2)
    assert valid is False
    assert "duplicate" in msg.lower() or "mismatch" in msg.lower()


def test_reported_bins_mismatch():
    weights = [30, 40, 50]
    capacity = 100
    bins = [[0, 1], [2]]

    # True bins count is 2, reported is 3
    valid, msg = validate_packing_solution(bins, weights, capacity, reported_bins=3)
    assert valid is False
    assert "mismatch" in msg.lower()
