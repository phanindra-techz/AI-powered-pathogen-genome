"""
k-Mer Feature Extraction Module.
Extracts normalized overlapping k-mer frequency vectors from input DNA/RNA sequences.
Ensures deterministic, identical feature order across training, inference, and evaluation.
Saves and loads feature configuration metadata.
"""

import os
import json
import itertools
from typing import List, Dict, Tuple, Union, Any, Optional
import numpy as np
import pandas as pd

# Canonical Nucleotide Alphabet
CANONICAL_BASES = ["A", "C", "G", "T"]


def generate_kmer_alphabet(k: int = 3, bases: Optional[List[str]] = None) -> List[str]:
    """
    Generates all possible k-mer combinations of length k in deterministic lexicographical order.
    
    Args:
        k: k-mer length (default: 3).
        bases: Nucleotide base alphabet (default: ['A', 'C', 'G', 'T']).

    Returns:
        Sorted list of 4^k k-mer strings.
    """
    if k < 1:
        raise ValueError(f"k must be a positive integer >= 1, got {k}")
    
    alphabet = bases if bases is not None else CANONICAL_BASES
    # Product produces lexicographically sorted tuples
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]


def extract_kmer_counts(sequence: str, k: int = 3) -> Dict[str, int]:
    """
    Extracts raw counts of overlapping length-k sub-sequences from an input sequence.
    
    Args:
        sequence: Input nucleotide sequence.
        k: Length of k-mers (default: 3).

    Returns:
        Dict mapping each k-mer to its integer occurrence count.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Standardize to uppercase and convert Uracil (RNA) to Thymine (DNA)
    seq_clean = "".join(sequence.upper().split()).replace("U", "T")
    kmers = generate_kmer_alphabet(k)
    kmer_counts = {kmer: 0 for kmer in kmers}

    seq_len = len(seq_clean)
    if seq_len < k:
        return kmer_counts

    # Overlapping sliding window (step = 1)
    for i in range(seq_len - k + 1):
        window = seq_clean[i : i + k]
        if window in kmer_counts:
            kmer_counts[window] += 1

    return kmer_counts


def extract_kmer_frequencies(sequence: str, k: int = 3) -> Dict[str, float]:
    """
    Converts raw k-mer counts into normalized relative frequencies (proportions).
    The sum of frequencies across all 4^k k-mers equals 1.0 (or 0.0 if sequence length < k).

    Args:
        sequence: Input nucleotide sequence.
        k: Length of k-mers (default: 3).

    Returns:
        Dict mapping each k-mer to its normalized float frequency.
    """
    counts = extract_kmer_counts(sequence, k=k)
    total_kmers = sum(counts.values())
    
    if total_kmers == 0:
        return {kmer: 0.0 for kmer in counts}
    
    return {kmer: round(count / total_kmers, 8) for kmer, count in counts.items()}


def extract_kmer_vector(
    sequence: str,
    k: int = 3
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts a 1D NumPy array of normalized k-mer frequencies in strictly deterministic order.

    Args:
        sequence: Input nucleotide sequence.
        k: Length of k-mers (default: 3).

    Returns:
        Tuple of (feature_vector: np.ndarray, feature_names: List[str]).
    """
    kmers = generate_kmer_alphabet(k)
    freqs = extract_kmer_frequencies(sequence, k=k)
    
    vector = [freqs[kmer] for kmer in kmers]
    feature_names = [f"{k}mer_{kmer}" for kmer in kmers]
    
    return np.array(vector, dtype=np.float64), feature_names


def extract_multi_kmer_vector(
    sequence: str,
    k_list: List[int] = [3, 4]
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts a concatenated feature vector of normalized k-mer frequencies across multiple k values.
    
    Args:
        sequence: Input nucleotide sequence.
        k_list: List of integer k values (e.g. [3, 4]).

    Returns:
        Tuple of (feature_vector: np.ndarray, feature_names: List[str]).
    """
    feature_vector: List[float] = []
    feature_names: List[str] = []

    for k in k_list:
        kmers = generate_kmer_alphabet(k)
        freqs = extract_kmer_frequencies(sequence, k=k)
        for kmer in kmers:
            feature_names.append(f"{k}mer_{kmer}")
            feature_vector.append(freqs[kmer])

    return np.array(feature_vector, dtype=np.float64), feature_names


class KMerFeatureExtractor:
    """
    Configurable k-mer frequency feature extractor.
    Enforces deterministic feature ordering and manages feature configuration serialization.
    """

    def __init__(self, k: Union[int, List[int]] = 3, bases: Optional[List[str]] = None):
        if isinstance(k, int):
            self.k_list = [k]
        elif isinstance(k, (list, tuple)):
            self.k_list = list(k)
        else:
            raise TypeError(f"k must be an int or list of ints, got {type(k).__name__}")

        self.bases = bases if bases is not None else list(CANONICAL_BASES)
        
        # Build immutable deterministic feature names list
        self.feature_names: List[str] = []
        for k_val in self.k_list:
            kmers = generate_kmer_alphabet(k_val, bases=self.bases)
            for kmer in kmers:
                self.feature_names.append(f"{k_val}mer_{kmer}")

        self.num_features = len(self.feature_names)

    def extract_vector(self, sequence: str) -> np.ndarray:
        """Extracts normalized feature vector for an input sequence."""
        vec, _ = extract_multi_kmer_vector(sequence, k_list=self.k_list)
        return vec

    def extract_batch(self, sequences: List[str]) -> np.ndarray:
        """Extracts 2D feature matrix (n_samples, n_features) for a list of sequences."""
        return np.vstack([self.extract_vector(seq) for seq in sequences])

    def get_feature_names(self) -> List[str]:
        """Returns the list of feature column names in exact order."""
        return list(self.feature_names)

    def get_config(self) -> Dict[str, Any]:
        """Returns the extractor configuration dictionary."""
        return {
            "k_list": self.k_list,
            "alphabet_bases": self.bases,
            "num_features": self.num_features,
            "feature_names": self.feature_names,
            "normalization": "relative_frequency"
        }

    def save_config(self, config_path: str) -> str:
        """Saves extractor configuration to a JSON file."""
        return save_kmer_configuration(self.k_list, config_path, bases=self.bases)

    @classmethod
    def from_config(cls, config_path: str) -> "KMerFeatureExtractor":
        """Instantiates a KMerFeatureExtractor from a saved JSON configuration file."""
        config = load_kmer_configuration(config_path)
        return cls(k=config["k_list"], bases=config.get("alphabet_bases", CANONICAL_BASES))


def save_kmer_configuration(
    k: Union[int, List[int]],
    output_path: str,
    bases: Optional[List[str]] = None
) -> str:
    """
    Saves k-mer feature extractor configuration schema as JSON.

    Args:
        k: Single k value or list of k values.
        output_path: Target JSON file path.
        bases: Nucleotide base alphabet.

    Returns:
        Saved file path.
    """
    k_list = [k] if isinstance(k, int) else list(k)
    alphabet = bases if bases is not None else CANONICAL_BASES
    
    feature_names: List[str] = []
    for k_val in k_list:
        for kmer in generate_kmer_alphabet(k_val, bases=alphabet):
            feature_names.append(f"{k_val}mer_{kmer}")

    config = {
        "k_list": k_list,
        "alphabet_bases": alphabet,
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "normalization": "relative_frequency",
        "description": "Consistent k-mer frequency feature configuration for genomic classification."
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return output_path


def load_kmer_configuration(config_path: str) -> Dict[str, Any]:
    """Loads a saved k-mer feature configuration JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Feature configuration file not found at: '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_top_kmers(sequence: str, k: int = 3, top_n: int = 10) -> pd.DataFrame:
    """
    Computes k-mer relative frequencies and returns the top N most frequent k-mers.
    
    Args:
        sequence: Input nucleotide sequence.
        k: Length of k-mers (default: 3).
        top_n: Number of top k-mers to return.

    Returns:
        DataFrame with columns ['k-mer', 'Frequency'].
    """
    freqs = extract_kmer_frequencies(sequence, k=k)
    sorted_kmers = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(sorted_kmers, columns=["k-mer", "Frequency"])
