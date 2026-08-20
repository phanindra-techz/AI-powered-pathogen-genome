"""
Reference Sequence Comparison and Distance Analysis Module.
Compares query sequence against an existing curated reference dataset.
Calculates documented computational similarity metrics (k-mer Cosine similarity,
Jaccard set distance, GC delta, and composite evidence score), ranks matches,
displays reference IDs/metadata, and exports similarity_results.csv.

DOCUMENTED SIMILARITY MEASURES:
1. k-mer Vector Cosine Similarity:
   S_cos(u, v) = (u . v) / (||u||_2 * ||v||_2)
2. k-mer Jaccard Set Overlap (k=3):
   J(A, B) = |A ∩ B| / |A ∪ B|
3. GC Proximity Score:
   P_GC = max(0.0, 1.0 - (|GC_query - GC_ref| / 50.0))
4. Documented Composite Match Score (%):
   Composite Score = (0.70 * S_cos + 0.20 * J + 0.10 * P_GC) * 100.0
"""

import os
import glob
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

from src.fasta_reader import parse_fasta
from src.qc import calculate_qc_metrics
from src.kmer_features import extract_kmer_frequencies, extract_multi_kmer_vector

# Mandatory Research Boundary Notice
SIMILARITY_DISCLAIMER = "Similarity does not automatically establish biological identity, phenotype or clinical significance."


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates cosine similarity between two 1D normalized frequency vectors.
    Formula: S_cos(u, v) = (u . v) / (||u||_2 * ||v||_2)
    """
    dot_product = float(np.dot(vec1, vec2))
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def jaccard_kmer_similarity(seq1: str, seq2: str, k: int = 3) -> float:
    """
    Calculates Jaccard set similarity of observed length-k oligomer vocabularies.
    Formula: J(A, B) = |A ∩ B| / |A ∪ B|
    """
    freqs1 = extract_kmer_frequencies(seq1, k=k)
    freqs2 = extract_kmer_frequencies(seq2, k=k)

    set1 = set(kmer for kmer, freq in freqs1.items() if freq > 0)
    set2 = set(kmer for kmer, freq in freqs2.items() if freq > 0)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 0.0
    return float(len(intersection) / len(union))


def parse_reference_metadata(header: str, filename: str) -> Dict[str, str]:
    """
    Extracts structured reference ID, organism name, and description from FASTA header.
    """
    ref_id = "REF_UNKNOWN"
    ref_name = os.path.splitext(filename)[0].replace("_", " ").title()
    ref_desc = header

    if "|" in header:
        parts = [p.strip() for p in header.split("|")]
        ref_name = parts[0].replace(">", "").strip()
        for p in parts[1:]:
            if "RefID:" in p:
                ref_id = p.replace("RefID:", "").strip()
            elif "REF_" in p:
                ref_id = p.strip()
            else:
                ref_desc = p.strip()
    else:
        ref_name = header.replace(">", "").split(None, 1)[0]
        ref_desc = header.replace(">", "").strip()

    return {
        "reference_id": ref_id,
        "reference_name": ref_name,
        "description": ref_desc
    }


def save_similarity_csv(
    rankings: Union[List[Dict[str, Any]], pd.DataFrame],
    output_path: str = "similarity_results.csv"
) -> str:
    """
    Saves ranked reference comparison results to a standardized CSV file.

    Args:
        rankings: List of ranking dictionaries or DataFrame.
        output_path: Target CSV file path.

    Returns:
        Path to the saved CSV file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if isinstance(rankings, pd.DataFrame):
        df = rankings
    else:
        df = pd.DataFrame(rankings)

    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def compare_with_references(
    query_sequence: str,
    reference_dir: str = "data/reference",
    k_list: List[int] = [3, 4],
    output_csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compares an input query sequence against all curated reference FASTA files in reference_dir.
    Calculates documented computational similarity scores, ranks matches, extracts metadata,
    and optionally exports similarity_results.csv.

    Args:
        query_sequence: Validated nucleotide sequence string of the query.
        reference_dir: Directory containing curated reference .fasta / .fa files.
        k_list: k-mer sizes for multi-scale feature vector cosine similarity.
        output_csv_path: Optional file path to save similarity_results.csv.

    Returns:
        Dict containing ranked similarity list, best match, top matches, DataFrame, and disclaimers.
    """
    if not os.path.exists(reference_dir):
        return {
            "best_match": None,
            "top_matches": [],
            "rankings": [],
            "dataframe": pd.DataFrame(),
            "warning": f"Reference directory '{reference_dir}' not found.",
            "disclaimer": SIMILARITY_DISCLAIMER
        }

    ref_files = sorted(glob.glob(os.path.join(reference_dir, "*.fasta"))) + sorted(glob.glob(os.path.join(reference_dir, "*.fa")))
    
    if not ref_files:
        return {
            "best_match": None,
            "top_matches": [],
            "rankings": [],
            "dataframe": pd.DataFrame(),
            "warning": f"No curated reference FASTA files found in '{reference_dir}'.",
            "disclaimer": SIMILARITY_DISCLAIMER
        }

    query_qc = calculate_qc_metrics(query_sequence)
    query_vec, _ = extract_multi_kmer_vector(query_sequence, k_list=k_list)

    comparison_results: List[Dict[str, Any]] = []

    for ref_path in ref_files:
        filename = os.path.basename(ref_path)
        try:
            parsed_ref = parse_fasta(ref_path)
            ref_seq = parsed_ref["primary_sequence"]
            ref_header = parsed_ref["header"]

            meta = parse_reference_metadata(ref_header, filename)
            ref_qc = calculate_qc_metrics(ref_seq)
            ref_vec, _ = extract_multi_kmer_vector(ref_seq, k_list=k_list)

            # 1. Cosine similarity on multi-scale k-mer vectors
            cos_sim = cosine_similarity(query_vec, ref_vec)
            
            # 2. Jaccard similarity on 3-mer presence
            jaccard_sim = jaccard_kmer_similarity(query_sequence, ref_seq, k=3)

            # 3. GC difference & proximity
            gc_diff = abs(query_qc["gc_percentage"] - ref_qc["gc_percentage"])
            gc_score = max(0.0, 1.0 - (gc_diff / 50.0))

            # 4. Documented Composite Match Score (0 - 100%)
            composite_score = round((0.70 * cos_sim + 0.20 * jaccard_sim + 0.10 * gc_score) * 100.0, 2)

            comparison_results.append({
                "Reference File": filename,
                "Reference ID": meta["reference_id"],
                "Reference Name": meta["reference_name"],
                "Header Info": ref_header,
                "Metadata Description": meta["description"],
                "Query Length (bp)": query_qc["sequence_length"],
                "Reference Length (bp)": ref_qc["sequence_length"],
                "Query GC (%)": f"{query_qc['gc_percentage']:.2f}%",
                "Reference GC (%)": f"{ref_qc['gc_percentage']:.2f}%",
                "k-mer Cosine Sim": round(cos_sim, 4),
                "Jaccard Sim": round(jaccard_sim, 4),
                "GC Delta (%)": round(gc_diff, 2),
                "Composite Score (%)": composite_score,
                "Disclaimer": SIMILARITY_DISCLAIMER
            })
        except Exception as e:
            print(f"Skipping unreadable reference '{filename}': {e}")
            continue

    # Sort results by Composite Score descending
    comparison_results.sort(key=lambda x: x["Composite Score (%)"], reverse=True)

    # Assign Rank numbers
    for idx, item in enumerate(comparison_results, start=1):
        item["Rank"] = idx

    df_comp = pd.DataFrame(comparison_results)
    best_match = comparison_results[0] if comparison_results else None
    top_matches = comparison_results[:3] if comparison_results else []

    # Export CSV if requested
    if output_csv_path:
        save_similarity_csv(comparison_results, output_csv_path)

    return {
        "best_match": best_match,
        "top_matches": top_matches,
        "rankings": comparison_results,
        "dataframe": df_comp,
        "similarity_metric_documentation": (
            "Composite Score (%) = (0.70 * Cosine_Similarity + 0.20 * Jaccard_Similarity + 0.10 * GC_Proximity) * 100"
        ),
        "disclaimer": SIMILARITY_DISCLAIMER
    }
