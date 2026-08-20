"""
Unit tests for src.predict module.
Tests the end-to-end inference pipeline:
FASTA -> validation -> QC -> k-mer extraction -> model prediction -> probability distribution,
field extraction (sample ID, length, QC metrics, model version, timestamp, probabilities),
JSON serialization, and computational evidence labeling.
"""

import os
import json
import pytest
from src.predict import predict_fasta, predict_pathogen_class, save_prediction_json, load_model


def test_predict_fasta_pipeline(tmp_path):
    """Test full end-to-end prediction pipeline on a FASTA file."""
    fasta_path = tmp_path / "query.fasta"
    fasta_content = ">isolate_test_001 Influenza sample query\n" + ("ATGCGATCGATCGATC" * 30) + "\n"
    fasta_path.write_text(fasta_content, encoding="utf-8")

    out_json = str(tmp_path / "prediction.json")
    result = predict_fasta(str(fasta_path), output_json_path=out_json)

    # 1. Check required return fields
    assert result["sample_id"] == "isolate_test_001"
    assert result["sequence_length"] == 480
    assert "qc_metrics" in result
    assert result["qc_metrics"]["status"] in ["PASS", "REVIEW"]
    assert "gc_percentage" in result["qc_metrics"]
    assert "predicted_class" in result
    assert result["predicted_class"] in result["prediction_probabilities"]
    assert "model_version" in result
    assert "timestamp" in result
    assert "prediction_probabilities" in result
    assert len(result["prediction_probabilities"]) > 0

    # 2. Check computational evidence labeling
    assert result["is_computational_evidence"] is True
    assert "COMPUTATIONAL PREDICTION EVIDENCE" in result["evidence_label"]
    assert "DEMONSTRATION MODEL" in result["demo_banner"]

    # 3. Check JSON export
    assert os.path.exists(out_json)
    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["sample_id"] == "isolate_test_001"
    assert loaded["predicted_class"] == result["predicted_class"]
    assert loaded["is_computational_evidence"] is True


def test_predict_pathogen_class_direct_sequence():
    """Test predict_pathogen_class on direct sequence string."""
    seq = "ATGCGATCGATCGATC" * 25
    res = predict_pathogen_class(seq)
    assert res["sequence_length"] == len(seq)
    assert res["predicted_class"] is not None
    assert 0.0 <= res["confidence_score"] <= 1.0


def test_save_prediction_json(tmp_path):
    """Test standalone save_prediction_json utility."""
    sample_data = {
        "sample_id": "test_id",
        "predicted_class": "SARS-CoV-2",
        "timestamp": "2026-08-19T00:00:00"
    }
    target_path = str(tmp_path / "results" / "custom_prediction.json")
    saved_path = save_prediction_json(sample_data, target_path)

    assert os.path.exists(saved_path)
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["sample_id"] == "test_id"
