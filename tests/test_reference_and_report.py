"""
Unit tests for src.reference_analysis and src.report_generator modules.
Tests reference comparison, documented similarity metrics, metadata extraction,
similarity_results.csv export, disclaimers, and PDF report generation.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.qc import calculate_qc_metrics
from src.predict import predict_pathogen_class
from src.reference_analysis import (
    compare_with_references,
    cosine_similarity,
    jaccard_kmer_similarity,
    save_similarity_csv,
    SIMILARITY_DISCLAIMER
)
from src.report_generator import generate_pdf_report


def test_similarity_metrics():
    """Test mathematical correctness of cosine and Jaccard similarity functions."""
    v1 = np.array([1.0, 0.0, 1.0])
    v2 = np.array([1.0, 0.0, 1.0])
    v3 = np.array([0.0, 1.0, 0.0])
    
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-5
    assert abs(cosine_similarity(v1, v3) - 0.0) < 1e-5

    j_sim = jaccard_kmer_similarity("ATGCATGC", "ATGCATGC", k=3)
    assert abs(j_sim - 1.0) < 1e-5


def test_compare_with_references_ranking_and_metadata(tmp_path):
    """Test comparing query against curated reference dataset, rankings, and metadata."""
    seq = "ATGCGATCGATCGATC" * 30
    csv_out = str(tmp_path / "similarity_results.csv")
    
    res = compare_with_references(seq, reference_dir="data/reference", output_csv_path=csv_out)
    
    assert "rankings" in res
    assert "best_match" in res
    assert "top_matches" in res
    assert len(res["rankings"]) > 0
    assert len(res["top_matches"]) <= 3

    # Check top match structure
    best = res["best_match"]
    assert "Reference ID" in best
    assert "Reference Name" in best
    assert "Metadata Description" in best
    assert "Composite Score (%)" in best
    assert "Rank" in best
    assert best["Rank"] == 1

    # Check disclaimer
    assert res["disclaimer"] == SIMILARITY_DISCLAIMER
    assert "Similarity does not automatically establish biological identity" in res["disclaimer"]

    # Check CSV export
    assert os.path.exists(csv_out)
    df_csv = pd.read_csv(csv_out)
    assert "Reference ID" in df_csv.columns
    assert "Composite Score (%)" in df_csv.columns
    assert len(df_csv) == len(res["rankings"])


def test_save_similarity_csv(tmp_path):
    """Test standalone save_similarity_csv function."""
    data = [
        {"Rank": 1, "Reference ID": "REF_001", "Reference Name": "Pathogen A", "Composite Score (%)": 98.5},
        {"Rank": 2, "Reference ID": "REF_002", "Reference Name": "Pathogen B", "Composite Score (%)": 42.1}
    ]
    target_csv = str(tmp_path / "results" / "custom_similarity.csv")
    saved_path = save_similarity_csv(data, target_csv)
    
    assert os.path.exists(saved_path)
    df = pd.read_csv(saved_path)
    assert len(df) == 2
    assert df.iloc[0]["Reference ID"] == "REF_001"


def test_generate_pdf_report(tmp_path):
    """Test PDF report generator."""
    seq = "ATGCGATCGATCGATC" * 30
    qc_res = calculate_qc_metrics(seq, header="Test Sequence")
    pred_res = predict_pathogen_class(seq)
    ref_res = compare_with_references(seq, reference_dir="data/reference")

    out_dir = str(tmp_path / "reports")
    pdf_path = generate_pdf_report(qc_res, pred_res, ref_res, output_dir=out_dir)

    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0
