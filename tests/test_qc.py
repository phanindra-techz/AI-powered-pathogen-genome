"""
Unit tests for src.qc module.
Tests sequence length, GC%, A/T/G/C percentages, ambiguous base counts,
PASS/REVIEW evaluation rules, JSON serialization, and safety disclaimers.
"""

import os
import json
import pytest
from src.qc import calculate_qc_metrics, perform_qc_analysis, save_qc_json


def test_calculate_qc_metrics_basic():
    """Test standard DNA sequence metrics (Length, GC%, A%, T%, G%, C%, ambiguous count)."""
    # 20 A, 30 C, 30 G, 20 T = 100 bp total
    seq = ("A" * 20) + ("C" * 30) + ("G" * 30) + ("T" * 20)
    qc = calculate_qc_metrics(seq, header="Test Sample 100bp")

    assert qc["sequence_length"] == 100
    assert qc["length"] == 100
    assert qc["gc_percentage"] == 60.0
    assert qc["a_percentage"] == 20.0
    assert qc["c_percentage"] == 30.0
    assert qc["g_percentage"] == 30.0
    assert qc["t_percentage"] == 20.0
    assert qc["ambiguous_base_count"] == 0
    assert qc["status"] == "PASS"
    assert qc["qc_status"] == "PASS"
    assert "NOT clinical diagnostic" in qc["disclaimer"]


def test_calculate_qc_metrics_ambiguous_review():
    """Test ambiguous IUPAC bases trigger REVIEW status when exceeding threshold."""
    # 80 bp canonical + 20 bp N (20% ambiguous > 5% threshold)
    seq = ("ATGC" * 20) + ("N" * 20)
    qc = calculate_qc_metrics(seq)

    assert qc["sequence_length"] == 100
    assert qc["ambiguous_base_count"] == 20
    assert qc["ambiguous_percentage"] == 20.0
    assert qc["status"] == "REVIEW"
    assert any("High ambiguous base percentage" in w for w in qc["warnings"])


def test_calculate_qc_metrics_short_length_review():
    """Test short sequences (<100 bp default threshold) trigger REVIEW status."""
    seq = "ATGCGATC" * 5  # 40 bp
    qc = calculate_qc_metrics(seq, min_length=100)

    assert qc["sequence_length"] == 40
    assert qc["status"] == "REVIEW"
    assert any("Short sequence length" in w for w in qc["warnings"])


def test_calculate_qc_metrics_extreme_gc_review():
    """Test extreme GC content outside [20%, 80%] triggers REVIEW status."""
    # 90% GC
    seq = ("G" * 45) + ("C" * 45) + ("A" * 5) + ("T" * 5)
    qc = calculate_qc_metrics(seq, min_length=50)

    assert qc["gc_percentage"] == 90.0
    assert qc["status"] == "REVIEW"
    assert any("Atypical GC content" in w for w in qc["warnings"])


def test_save_qc_json(tmp_path):
    """Test saving QC metrics dictionary as a formatted JSON file."""
    seq = "ATGC" * 30  # 120 bp
    qc = calculate_qc_metrics(seq, header="Json Export Sample")
    
    json_path = str(tmp_path / "qc_results" / "qc_report.json")
    saved_path = save_qc_json(qc, json_path)

    assert os.path.exists(saved_path)
    with open(saved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["sequence_length"] == 120
    assert loaded["gc_percentage"] == 50.0
    assert loaded["status"] == "PASS"
    assert "disclaimer" in loaded


def test_calculate_qc_metrics_with_output_path(tmp_path):
    """Test automatic JSON export via output_json_path parameter."""
    json_path = str(tmp_path / "auto_qc.json")
    seq = "ATGC" * 30
    qc = calculate_qc_metrics(seq, output_json_path=json_path)

    assert os.path.exists(json_path)
    assert qc["status"] == "PASS"


def test_empty_sequence():
    """Test empty sequence handling and REVIEW status."""
    qc = calculate_qc_metrics("")
    assert qc["sequence_length"] == 0
    assert qc["status"] == "REVIEW"
    assert qc["gc_percentage"] == 0.0
