"""
Unit tests for src.dataset_loader module.
Tests pluggable external dataset loading (directory structure, CSV metadata),
synthetic demo dataset generation, and demonstration model disclaimers.
"""

import os
import pytest
import pandas as pd
from src.dataset_loader import (
    load_dataset_from_directory,
    load_dataset_from_csv,
    generate_synthetic_demo_dataset,
    load_labeled_dataset,
    DEMO_PATHOGEN_PROFILES
)


def test_generate_synthetic_demo_dataset():
    """Test generating synthetic demonstration dataset."""
    X, y, f_names, meta = generate_synthetic_demo_dataset(samples_per_class=3, k_list=[3], seed=42)
    assert X.shape == (3 * len(DEMO_PATHOGEN_PROFILES), 64)
    assert len(y) == 3 * len(DEMO_PATHOGEN_PROFILES)
    assert meta["is_synthetic"] is True
    assert "DEMONSTRATION MODEL — NOT CLINICALLY VALIDATED" in meta["disclaimer"]


def test_load_dataset_from_directory(tmp_path):
    """Test pluggable dataset loading from class subdirectories."""
    ds_dir = tmp_path / "custom_dataset"
    class_a_dir = ds_dir / "Pathogen_Alpha"
    class_b_dir = ds_dir / "Pathogen_Beta"
    class_a_dir.mkdir(parents=True)
    class_b_dir.mkdir(parents=True)

    (class_a_dir / "sample1.fasta").write_text(">seqA1\nATGCGATCGATCGATCGATCGATCGATC\n", encoding="utf-8")
    (class_a_dir / "sample2.fa").write_text(">seqA2\nATGCGATCGATCGATCGATCGATCGATC\n", encoding="utf-8")
    (class_b_dir / "sample1.fasta").write_text(">seqB1\nTTTTCCCCAAAAGGGGTTTTCCCCAAAA\n", encoding="utf-8")

    X, y, f_names, meta = load_dataset_from_directory(str(ds_dir), k_list=[3])
    assert X.shape == (3, 64)
    assert len(y) == 3
    assert meta["is_synthetic"] is False
    assert meta["classes"] == ["Pathogen_Alpha", "Pathogen_Beta"]
    assert meta["class_counts"]["Pathogen_Alpha"] == 2
    assert meta["class_counts"]["Pathogen_Beta"] == 1


def test_load_dataset_from_csv(tmp_path):
    """Test pluggable dataset loading from metadata CSV."""
    fasta_1 = tmp_path / "fa1.fasta"
    fasta_2 = tmp_path / "fa2.fasta"
    fasta_1.write_text(">rec1\nATGCGATCGATCGATCGATCGATC\n", encoding="utf-8")
    fasta_2.write_text(">rec2\nCCCCGGGGAAAATTTTCCCCGGGG\n", encoding="utf-8")

    csv_path = tmp_path / "metadata.csv"
    df = pd.DataFrame([
        {"filepath": str(fasta_1), "label": "Class_1"},
        {"filepath": str(fasta_2), "label": "Class_2"}
    ])
    df.to_csv(csv_path, index=False)

    X, y, f_names, meta = load_dataset_from_csv(str(csv_path), k_list=[3])
    assert X.shape == (2, 64)
    assert len(y) == 2
    assert meta["is_synthetic"] is False
    assert "Class_1" in meta["classes"]
    assert "Class_2" in meta["classes"]


def test_load_labeled_dataset_universal_fallback():
    """Test that universal loader falls back to synthetic demonstration data when no external path is given."""
    X, y, f_names, meta = load_labeled_dataset(dataset_dir=None, metadata_csv=None, samples_per_class=2, k_list=[3])
    assert meta["is_synthetic"] is True
    assert "DEMONSTRATION MODEL — NOT CLINICALLY VALIDATED" in meta["disclaimer"]


def test_parse_and_load_ncbi_dataset_monkeypox():
    """Test parsing and feature extraction from NCBI Datasets package for Monkeypox."""
    from src.dataset_loader import parse_ncbi_dataset_package, load_ncbi_dataset
    
    mpox_dir = "monkeypox"
    if os.path.exists(mpox_dir):
        pkg = parse_ncbi_dataset_package(mpox_dir)
        assert pkg["num_records"] >= 2
        assert "genomic.fna" in pkg["genomic_fna_path"]
        assert pkg["organism_name"] == "Monkeypox virus"
        assert len(pkg["report_entries"]) >= 2

        X, y, f_names, meta = load_ncbi_dataset(mpox_dir, k_list=[3])
        assert X.shape[0] >= 2
        assert X.shape[1] == 64
        assert "Monkeypox virus" in y[0]
        assert meta["source_type"] == "ncbi_dataset_package"

