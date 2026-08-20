"""
Unit tests for src.kmer_features and src.predict modules.
Tests overlapping k-mer counts, normalized frequencies, consistent feature ordering,
configurable k values, configuration serialization, and ML prediction pipeline.
"""

import os
import json
import numpy as np
import pytest
from src.kmer_features import (
    generate_kmer_alphabet,
    extract_kmer_counts,
    extract_kmer_frequencies,
    extract_kmer_vector,
    extract_multi_kmer_vector,
    KMerFeatureExtractor,
    save_kmer_configuration,
    load_kmer_configuration,
    get_top_kmers
)
from src.predict import predict_pathogen_class


def test_generate_kmer_alphabet_k3():
    """Test generating 4^3 = 64 lexicographically sorted k-mers for k=3."""
    kmers = generate_kmer_alphabet(k=3)
    assert len(kmers) == 64
    assert kmers[0] == "AAA"
    assert kmers[1] == "AAC"
    assert kmers[-1] == "TTT"
    assert kmers == sorted(kmers)


def test_extract_kmer_counts_overlapping():
    """Test sliding window (step=1) overlapping k-mer count extraction."""
    # Sequence: "AAAA" -> contains 2 overlapping "AAA"
    seq = "AAAA"
    counts = extract_kmer_counts(seq, k=3)
    assert counts["AAA"] == 2
    assert counts["AAC"] == 0
    assert sum(counts.values()) == 2


def test_extract_kmer_frequencies_sum_to_one():
    """Test that extracted frequencies sum to 1.0 for valid sequence."""
    seq = "ATGCGATCGATCGATC"
    freqs = extract_kmer_frequencies(seq, k=3)
    total_freq = sum(freqs.values())
    assert abs(total_freq - 1.0) < 1e-5


def test_extract_kmer_vector_deterministic_order():
    """Test that feature vector maintains identical feature ordering."""
    seq1 = "ATGCGATCGATCGATC" * 5
    vec1, names1 = extract_kmer_vector(seq1, k=3)
    vec2, names2 = extract_kmer_vector(seq1, k=3)

    assert len(vec1) == 64
    assert len(names1) == 64
    assert names1 == names2
    np.testing.assert_array_equal(vec1, vec2)
    assert names1[0] == "3mer_AAA"
    assert names1[-1] == "3mer_TTT"


def test_kmer_feature_extractor_class(tmp_path):
    """Test KMerFeatureExtractor class, configuration saving and loading."""
    extractor = KMerFeatureExtractor(k=3)
    assert extractor.num_features == 64

    seq = "ATGCGATCGATCGATC" * 10
    vec = extractor.extract_vector(seq)
    assert vec.shape == (64,)
    assert abs(np.sum(vec) - 1.0) < 1e-5

    # Batch extraction
    batch_vecs = extractor.extract_batch([seq, seq])
    assert batch_vecs.shape == (2, 64)

    # Save and reload config
    cfg_path = str(tmp_path / "kmer_config.json")
    extractor.save_config(cfg_path)
    assert os.path.exists(cfg_path)

    reloaded_extractor = KMerFeatureExtractor.from_config(cfg_path)
    assert reloaded_extractor.num_features == 64
    assert reloaded_extractor.feature_names == extractor.feature_names


def test_extract_multi_kmer_vector():
    """Test concatenated multi-k vector (k=3 and k=4: 64 + 256 = 320 features)."""
    seq = "ATGCGATCGATCGATC" * 10
    vec, names = extract_multi_kmer_vector(seq, k_list=[3, 4])
    assert len(vec) == 320
    assert len(names) == 320
    assert names[0] == "3mer_AAA"
    assert names[64] == "4mer_AAAA"


def test_get_top_kmers():
    """Test retrieving top N frequent k-mers."""
    seq = "AAAAATTTTT"
    df_top = get_top_kmers(seq, k=3, top_n=5)
    assert len(df_top) <= 5
    assert "k-mer" in df_top.columns
    assert "Frequency" in df_top.columns


def test_model_prediction_pipeline():
    """Test downstream ML model inference integration using k-mer features."""
    seq = "ATGCGATCGATCGATC" * 30
    pred = predict_pathogen_class(seq)
    assert "predicted_class" in pred
    assert "confidence_score" in pred
    assert 0.0 <= pred["confidence_score"] <= 1.0
    assert len(pred["class_probabilities"]) > 0
