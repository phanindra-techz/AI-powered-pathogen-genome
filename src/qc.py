"""
Quality Control (QC) Analysis Module.
Computes nucleotide composition, GC and individual base percentages,
ambiguous base counts, applies transparent PASS/REVIEW rules, and exports JSON results.
"""

import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd


# Analytical Safety Boundary
QC_DISCLAIMER = (
    "COMPUTATIONAL QUALITY CONTROL ONLY: Metrics are technical sequencing quality checkpoints "
    "for analytical bioinformatics research. They are NOT clinical diagnostic tests and MUST NOT "
    "be used for medical diagnosis, clinical evaluation, or therapeutic decision-making."
)


def calculate_qc_metrics(
    sequence: str,
    header: str = "Query Sequence",
    output_json_path: Optional[str] = None,
    min_length: int = 100,
    max_ambiguous_percent: float = 5.0,
    gc_min: float = 20.0,
    gc_max: float = 80.0
) -> Dict[str, Any]:
    """
    Accepts a validated DNA/RNA sequence and calculates comprehensive QC metrics:
    sequence length, GC percentage, individual base percentages (A, T, G, C),
    ambiguous base counts, transparent PASS/REVIEW status, and optional JSON export.

    Args:
        sequence: Validated nucleotide sequence string.
        header: Optional sequence identifier or description header.
        output_json_path: Optional file path to save results as JSON.
        min_length: Minimum sequence length threshold for PASS (default: 100 bp).
        max_ambiguous_percent: Maximum allowed ambiguous base percentage for PASS (default: 5.0%).
        gc_min: Minimum plausible GC percentage (default: 20.0%).
        gc_max: Maximum plausible GC percentage (default: 80.0%).

    Returns:
        Dict containing all computed QC metrics, PASS/REVIEW status, rules evaluated, and disclaimers.
    """
    seq_upper = sequence.upper().strip()
    # Remove any internal whitespace
    seq_clean = "".join(seq_upper.split())
    total_len = len(seq_clean)

    if total_len == 0:
        qc_dict: Dict[str, Any] = {
            "header": header,
            "sequence_length": 0,
            "length": 0,
            "gc_percentage": 0.0,
            "gc_content_percent": 0.0,
            "a_percentage": 0.0,
            "t_percentage": 0.0,
            "g_percentage": 0.0,
            "c_percentage": 0.0,
            "at_percentage": 0.0,
            "at_content_percent": 0.0,
            "ambiguous_base_count": 0,
            "ambiguous_count": 0,
            "ambiguous_percentage": 0.0,
            "ambiguous_percent": 0.0,
            "base_counts": {"A": 0, "C": 0, "G": 0, "T/U": 0, "N": 0, "Other Ambiguous": 0},
            "base_percentages": {"A": 0.0, "C": 0.0, "G": 0.0, "T/U": 0.0, "N": 0.0, "Other Ambiguous": 0.0},
            "status": "REVIEW",
            "qc_status": "REVIEW",
            "warnings": ["Sequence length is 0 bp (empty sequence)."],
            "rules_applied": {
                "min_length_threshold_bp": min_length,
                "max_ambiguous_threshold_percent": max_ambiguous_percent,
                "gc_plausibility_range_percent": [gc_min, gc_max]
            },
            "disclaimer": QC_DISCLAIMER
        }
        if output_json_path:
            save_qc_json(qc_dict, output_json_path)
        return qc_dict

    # Count canonical bases (include Uracil as T/U for RNA compatibility)
    count_a = seq_clean.count("A")
    count_c = seq_clean.count("C")
    count_g = seq_clean.count("G")
    count_t = seq_clean.count("T") + seq_clean.count("U")
    count_n = seq_clean.count("N")

    # Ambiguous/degenerate bases (non-ACGTU)
    standard_bases = set("ACGTU")
    other_ambiguous = sum(1 for b in seq_clean if b not in standard_bases)
    ambiguous_total = other_ambiguous

    # Compute percentages
    a_pct = round((count_a / total_len) * 100.0, 2)
    c_pct = round((count_c / total_len) * 100.0, 2)
    g_pct = round((count_g / total_len) * 100.0, 2)
    t_pct = round((count_t / total_len) * 100.0, 2)
    gc_pct = round(((count_g + count_c) / total_len) * 100.0, 2)
    at_pct = round(((count_a + count_t) / total_len) * 100.0, 2)
    ambiguous_pct = round((ambiguous_total / total_len) * 100.0, 2)

    base_counts = {
        "A": count_a,
        "C": count_c,
        "G": count_g,
        "T/U": count_t,
        "N": count_n,
        "Other Ambiguous": ambiguous_total - count_n
    }

    base_percentages = {
        "A": a_pct,
        "C": c_pct,
        "G": g_pct,
        "T/U": t_pct,
        "N": round((count_n / total_len) * 100.0, 2),
        "Other Ambiguous": round(((ambiguous_total - count_n) / total_len) * 100.0, 2)
    }

    # Transparent Rules-Based PASS / REVIEW Evaluation
    warnings: List[str] = []
    status = "PASS"

    # Rule 1: Minimum Length
    if total_len < min_length:
        warnings.append(
            f"Short sequence length: {total_len} bp (minimum standard threshold is {min_length} bp)."
        )
        status = "REVIEW"

    # Rule 2: Ambiguous Base Ratio
    if ambiguous_pct > max_ambiguous_percent:
        warnings.append(
            f"High ambiguous base percentage: {ambiguous_pct:.2f}% (acceptable threshold is <= {max_ambiguous_percent:.1f}%)."
        )
        status = "REVIEW"

    # Rule 3: Biological GC Content Range
    if gc_pct < gc_min or gc_pct > gc_max:
        warnings.append(
            f"Atypical GC content: {gc_pct:.2f}% (expected reference range is {gc_min:.1f}% - {gc_max:.1f}%)."
        )
        status = "REVIEW"

    if not warnings:
        warnings.append("Sequence passed all computational quality control thresholds.")

    qc_dict = {
        "header": header,
        "sequence_length": total_len,
        "length": total_len,
        "gc_percentage": gc_pct,
        "gc_content_percent": gc_pct,
        "a_percentage": a_pct,
        "t_percentage": t_pct,
        "g_percentage": g_pct,
        "c_percentage": c_pct,
        "at_percentage": at_pct,
        "at_content_percent": at_pct,
        "ambiguous_base_count": ambiguous_total,
        "ambiguous_count": ambiguous_total,
        "ambiguous_percentage": ambiguous_pct,
        "ambiguous_percent": ambiguous_pct,
        "base_counts": base_counts,
        "base_percentages": base_percentages,
        "status": status,
        "qc_status": status,
        "warnings": warnings,
        "rules_applied": {
            "min_length_threshold_bp": min_length,
            "max_ambiguous_threshold_percent": max_ambiguous_percent,
            "gc_plausibility_range_percent": [gc_min, gc_max]
        },
        "disclaimer": QC_DISCLAIMER
    }

    if output_json_path:
        save_qc_json(qc_dict, output_json_path)

    return qc_dict


# Alias for backward compatibility across modules
perform_qc_analysis = calculate_qc_metrics


def save_qc_json(qc_data: Dict[str, Any], output_path: str) -> str:
    """
    Saves the QC results dictionary as a formatted JSON file.

    Args:
        qc_data: Dictionary returned by calculate_qc_metrics / perform_qc_analysis.
        output_path: Path to target JSON file.

    Returns:
        The output file path.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qc_data, f, indent=2)
    return output_path


def get_qc_summary_dataframe(qc_results: Dict[str, Any]) -> pd.DataFrame:
    """Converts QC base metrics into a display-ready Pandas DataFrame for UI and reporting."""
    df_data = []
    for base, count in qc_results["base_counts"].items():
        pct = qc_results["base_percentages"].get(base, 0.0)
        df_data.append({
            "Nucleotide Base": base,
            "Count (bp)": count,
            "Composition (%)": f"{pct:.2f}%"
        })
    return pd.DataFrame(df_data)
