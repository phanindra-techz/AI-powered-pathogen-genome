"""
Dataset Ingestion & Pluggable Data Loader Module.
Allows plugging in properly sourced, verified labeled pathogen genome datasets
from directories or metadata CSV files. Provides synthetic demonstration dataset
generator for software pipeline validation when verified datasets are not yet available.
"""

import os
import json
import glob
import random
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd

from src.fasta_reader import parse_fasta, validate_sequence
from src.kmer_features import extract_multi_kmer_vector

# Demonstration Pathogen Synthetic Profiles (for demonstration pipeline testing only)
DEMO_PATHOGEN_PROFILES: Dict[str, Dict[str, Any]] = {
    "Influenza A virus": {
        "gc_target": 0.44,
        "kmer_bias": {"AGA": 2.5, "GAA": 2.2, "AAA": 2.0, "AGG": 1.8},
        "description": "Synthetic demonstration profile (Influenza A segment analog)",
        "ref_filename": "influenza_a_ref.fasta"
    },
    "SARS-CoV-2": {
        "gc_target": 0.38,
        "kmer_bias": {"TTT": 2.4, "TTA": 2.1, "ATG": 1.9, "GTT": 1.8},
        "description": "Synthetic demonstration profile (Betacoronavirus analog)",
        "ref_filename": "sars_cov_2_ref.fasta"
    },
    "Dengue virus": {
        "gc_target": 0.46,
        "kmer_bias": {"GAC": 2.3, "CGA": 2.0, "AAG": 1.9, "GAA": 1.7},
        "description": "Synthetic demonstration profile (Flavivirus analog)",
        "ref_filename": "dengue_virus_ref.fasta"
    },
    "Escherichia coli": {
        "gc_target": 0.50,
        "kmer_bias": {"CGC": 2.2, "GCG": 2.2, "GCC": 2.0, "CAG": 1.9},
        "description": "Synthetic demonstration profile (E. coli bacterial analog)",
        "ref_filename": "ecoli_ref.fasta"
    },
    "Staphylococcus aureus": {
        "gc_target": 0.33,
        "kmer_bias": {"AAT": 2.6, "ATT": 2.5, "TAA": 2.3, "TTA": 2.0},
        "description": "Synthetic demonstration profile (S. aureus bacterial analog)",
        "ref_filename": "staph_aureus_ref.fasta"
    },
    "Mycobacterium tuberculosis": {
        "gc_target": 0.65,
        "kmer_bias": {"CGC": 2.8, "CCG": 2.7, "GCC": 2.5, "CGG": 2.4},
        "description": "Synthetic demonstration profile (High-GC Mycobacterial analog)",
        "ref_filename": "m_tuberculosis_ref.fasta"
    },
    "Monkeypox virus": {
        "gc_target": 0.33,
        "kmer_bias": {"TAA": 2.7, "AAT": 2.5, "ATT": 2.4, "TTA": 2.2, "ATA": 2.0},
        "description": "Synthetic demonstration profile (Poxviridae / Orthopoxvirus monkeypox analog)",
        "ref_filename": "monkeypox_ref.fasta"
    }
}


def _generate_synthetic_sequence(
    gc_target: float,
    kmer_bias: Dict[str, float],
    length: int = 1500,
    seed: Optional[int] = None
) -> str:
    """Generates an artificial demonstration sequence matching target GC and k-mer biases."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    bases = ['A', 'C', 'G', 'T']
    gc_half = gc_target / 2.0
    at_half = (1.0 - gc_target) / 2.0
    probs = [at_half, gc_half, gc_half, at_half]
    
    seq_list = list(np.random.choice(bases, size=length, p=probs))

    for kmer, weight in kmer_bias.items():
        num_injections = int(weight * 12)
        for _ in range(num_injections):
            pos = random.randint(0, length - len(kmer) - 1)
            for idx, char in enumerate(kmer):
                seq_list[pos + idx] = char

    return "".join(seq_list)


def load_dataset_from_directory(
    dataset_dir: str,
    k_list: List[int] = [3, 4]
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Loads labeled FASTA files from class subdirectories:
        dataset_dir/
            <class_1>/
                sample1.fasta
                sample2.fa
            <class_2>/
                ...

    Args:
        dataset_dir: Root directory containing labeled class folders.
        k_list: k-mer sizes for feature extraction.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray, feature_names: List[str], metadata: dict)
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    X_list = []
    y_list = []
    feature_names: List[str] = []
    record_counts: Dict[str, int] = {}

    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    if not subdirs:
        raise ValueError(
            f"No class subdirectories found in '{dataset_dir}'. "
            "Expected structure: dataset_dir/<class_name>/<sequence_files>.fasta"
        )

    for class_name in sorted(subdirs):
        class_folder = os.path.join(dataset_dir, class_name)
        fasta_files = (
            glob.glob(os.path.join(class_folder, "*.fasta")) +
            glob.glob(os.path.join(class_folder, "*.fa")) +
            glob.glob(os.path.join(class_folder, "*.fna"))
        )

        class_count = 0
        for fpath in fasta_files:
            try:
                parsed = parse_fasta(fpath)
                seq = parsed["primary_sequence"]
                feat_vec, f_names = extract_multi_kmer_vector(seq, k_list=k_list)
                if not feature_names:
                    feature_names = f_names
                X_list.append(feat_vec)
                y_list.append(class_name)
                class_count += 1
            except Exception as e:
                print(f"Skipping unparseable file {fpath}: {e}")

        record_counts[class_name] = class_count

    if not X_list:
        raise ValueError(f"No valid FASTA sequence records found in '{dataset_dir}'.")

    metadata = {
        "source_type": "directory",
        "dataset_path": dataset_dir,
        "is_synthetic": False,
        "class_counts": record_counts,
        "total_samples": len(y_list),
        "classes": sorted(list(record_counts.keys()))
    }

    return np.array(X_list), np.array(y_list), feature_names, metadata


def load_dataset_from_csv(
    csv_path: str,
    k_list: List[int] = [3, 4]
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Loads labeled FASTA files using a metadata CSV/TSV containing columns:
    'filepath' (or 'file') and 'label' (or 'class' / 'pathogen_class').

    Args:
        csv_path: Path to metadata CSV or TSV file.
        k_list: k-mer sizes for feature extraction.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray, feature_names: List[str], metadata: dict)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV '{csv_path}' does not exist.")

    sep = "\t" if csv_path.endswith(".tsv") else ","
    df = pd.read_csv(csv_path, sep=sep)

    # Resolve column names
    path_col = None
    for col in ["filepath", "file", "path", "fasta_file", "filename"]:
        if col in df.columns:
            path_col = col
            break

    label_col = None
    for col in ["label", "class", "pathogen_class", "organism", "pathogen"]:
        if col in df.columns:
            label_col = col
            break

    if not path_col or not label_col:
        raise ValueError(
            f"CSV must contain a file path column ({['filepath', 'file', 'path']}) "
            f"and a label column ({['label', 'class', 'pathogen_class']}). Found: {list(df.columns)}"
        )

    base_dir = os.path.dirname(csv_path)
    X_list = []
    y_list = []
    feature_names: List[str] = []
    class_counts: Dict[str, int] = {}

    for _, row in df.iterrows():
        fpath = str(row[path_col]).strip()
        label = str(row[label_col]).strip()

        # Handle relative paths relative to CSV location
        if not os.path.isabs(fpath) and not os.path.exists(fpath):
            candidate_path = os.path.join(base_dir, fpath)
            if os.path.exists(candidate_path):
                fpath = candidate_path

        try:
            parsed = parse_fasta(fpath)
            seq = parsed["primary_sequence"]
            feat_vec, f_names = extract_multi_kmer_vector(seq, k_list=k_list)
            if not feature_names:
                feature_names = f_names
            X_list.append(feat_vec)
            y_list.append(label)
            class_counts[label] = class_counts.get(label, 0) + 1
        except Exception as e:
            print(f"Skipping record at '{fpath}': {e}")

    if not X_list:
        raise ValueError(f"No valid FASTA files could be parsed from CSV '{csv_path}'.")

    metadata = {
        "source_type": "metadata_csv",
        "dataset_path": csv_path,
        "is_synthetic": False,
        "class_counts": class_counts,
        "total_samples": len(y_list),
        "classes": sorted(list(class_counts.keys()))
    }

    return np.array(X_list), np.array(y_list), feature_names, metadata


def generate_synthetic_demo_dataset(
    samples_per_class: int = 50,
    k_list: List[int] = [3, 4],
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Generates clearly-labeled synthetic demonstration data to test and validate
    the software ML pipeline without requiring verified pathogen sequence data.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray, feature_names: List[str], metadata: dict)
    """
    random.seed(seed)
    np.random.seed(seed)

    X_list = []
    y_list = []
    feature_names: List[str] = []
    class_counts: Dict[str, int] = {}
    sample_idx = 0

    for p_name, p_info in DEMO_PATHOGEN_PROFILES.items():
        for _ in range(samples_per_class):
            sample_idx += 1
            seq_len = random.randint(1200, 2400)
            seq = _generate_synthetic_sequence(
                gc_target=p_info["gc_target"],
                kmer_bias=p_info["kmer_bias"],
                length=seq_len,
                seed=sample_idx + seed
            )
            feat_vec, f_names = extract_multi_kmer_vector(seq, k_list=k_list)
            if not feature_names:
                feature_names = f_names
            X_list.append(feat_vec)
            y_list.append(p_name)

        class_counts[p_name] = samples_per_class

    metadata = {
        "source_type": "synthetic_demonstration",
        "dataset_path": "synthetic_in_memory",
        "is_synthetic": True,
        "class_counts": class_counts,
        "total_samples": len(y_list),
        "classes": sorted(list(class_counts.keys())),
        "disclaimer": "DEMONSTRATION MODEL — NOT CLINICALLY VALIDATED. Synthetic demonstration sequence profiles."
    }

    return np.array(X_list), np.array(y_list), feature_names, metadata


def parse_ncbi_dataset_package(package_dir: str) -> Dict[str, Any]:
    """
    Parses an NCBI Datasets package directory (from 'datasets download virus genome taxon ...').
    Detects data_report.jsonl, dataset_catalog.json, and genomic.fna / *.fasta files.

    Directory structure:
        <package_dir>/
        ├── README.md
        ├── md5sum.txt
        └── ncbi_dataset/
            └── data/
                ├── data_report.jsonl
                ├── dataset_catalog.json
                └── genomic.fna

    Args:
        package_dir: Root path to the unzipped NCBI Datasets folder.

    Returns:
        Dict containing:
            - 'records': List of sequence records with attached metadata
            - 'report_entries': List of parsed metadata entries from data_report.jsonl
            - 'catalog': Parsed dataset_catalog.json
            - 'genomic_fna_path': Path to the primary FASTA / FNA sequence file
            - 'num_records': Total number of sequence records
            - 'organism_name': Main scientific name if identified
    """
    if not os.path.exists(package_dir):
        raise FileNotFoundError(f"NCBI dataset package directory not found at '{package_dir}'.")

    # Locate the internal data folder
    candidate_data_dirs = [
        os.path.join(package_dir, "ncbi_dataset", "data"),
        os.path.join(package_dir, "data"),
        package_dir
    ]
    data_dir = None
    for c_dir in candidate_data_dirs:
        if os.path.exists(c_dir) and os.path.isdir(c_dir):
            data_dir = c_dir
            break

    if data_dir is None:
        raise ValueError(f"Could not locate data folder in NCBI dataset package '{package_dir}'.")

    # Find genomic.fna or other FASTA files
    fna_candidates = [
        os.path.join(data_dir, "genomic.fna"),
        os.path.join(data_dir, "genomic.fasta"),
        os.path.join(data_dir, "genomic.fa")
    ] + glob.glob(os.path.join(data_dir, "*.fna")) + glob.glob(os.path.join(data_dir, "*.fasta"))

    fna_path = None
    for cand in fna_candidates:
        if os.path.exists(cand) and os.path.isfile(cand) and os.path.getsize(cand) > 0:
            fna_path = cand
            break

    if fna_path is None:
        raise FileNotFoundError(f"No valid genomic sequence file (genomic.fna) found in '{data_dir}'.")

    # Parse data_report.jsonl if available
    report_entries = []
    metadata_by_accession: Dict[str, Dict[str, Any]] = {}
    report_jsonl_path = os.path.join(data_dir, "data_report.jsonl")
    if os.path.exists(report_jsonl_path):
        with open(report_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    entry = json.loads(line_str)
                    report_entries.append(entry)
                    acc = entry.get("accession") or entry.get("assemblyInfo", {}).get("assemblyAccession")
                    if acc:
                        metadata_by_accession[acc] = entry
                except Exception:
                    continue

    # Parse dataset_catalog.json if available
    catalog = {}
    catalog_path = os.path.join(data_dir, "dataset_catalog.json")
    if os.path.exists(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception:
            catalog = {}

    # Parse FASTA records from genomic.fna
    parsed_fasta = parse_fasta(fna_path)
    records = parsed_fasta.get("records", [])

    # Merge metadata per record
    for rec in records:
        rec_id = rec["id"]
        # Match accession prefix (e.g. NC_063383.1 from 'NC_063383.1 Monkeypox virus...')
        matching_meta = None
        for acc, meta in metadata_by_accession.items():
            if acc in rec_id or rec_id in acc:
                matching_meta = meta
                break
        if not matching_meta and metadata_by_accession:
            # If single report entry, attach it
            matching_meta = list(metadata_by_accession.values())[0]

        rec["ncbi_metadata"] = matching_meta or {}

    # Determine default organism name
    primary_organism = "Unknown Pathogen"
    if report_entries:
        first_entry = report_entries[0]
        primary_organism = (
            first_entry.get("virus", {}).get("sciName")
            or first_entry.get("organism", {}).get("sciName")
            or first_entry.get("assemblyInfo", {}).get("organism", {}).get("organismName")
            or "Unknown Pathogen"
        )
    elif records and "Monkeypox" in records[0].get("description", ""):
        primary_organism = "Monkeypox virus"

    return {
        "package_dir": package_dir,
        "data_dir": data_dir,
        "genomic_fna_path": fna_path,
        "records": records,
        "report_entries": report_entries,
        "catalog": catalog,
        "num_records": len(records),
        "organism_name": primary_organism
    }


def load_ncbi_dataset(
    package_dir: str,
    default_label: Optional[str] = None,
    k_list: List[int] = [3, 4]
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Extracts k-mer feature vectors and labels from an NCBI Datasets package directory.

    Args:
        package_dir: Path to unzipped NCBI dataset directory.
        default_label: Optional explicit label. If None, derives from data_report.jsonl or folder name.
        k_list: k-mer sizes for feature extraction.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray, feature_names: List[str], metadata: dict)
    """
    pkg = parse_ncbi_dataset_package(package_dir)
    records = pkg["records"]
    if not records:
        raise ValueError(f"No sequence records found in NCBI dataset package at '{package_dir}'.")

    # Determine label
    label = default_label or pkg.get("organism_name")
    if not label or label == "Unknown Pathogen":
        folder_base = os.path.basename(os.path.normpath(package_dir))
        label = folder_base.replace("_", " ").replace("-", " ").title()

    X_list = []
    y_list = []
    feature_names: List[str] = []

    for rec in records:
        seq = rec["sequence"]
        feat_vec, f_names = extract_multi_kmer_vector(seq, k_list=k_list)
        if not feature_names:
            feature_names = f_names
        X_list.append(feat_vec)
        y_list.append(label)

    metadata = {
        "source_type": "ncbi_dataset_package",
        "dataset_path": package_dir,
        "is_synthetic": False,
        "class_counts": {label: len(y_list)},
        "total_samples": len(y_list),
        "classes": [label],
        "organism_name": pkg.get("organism_name", label),
        "ncbi_report_entries_count": len(pkg.get("report_entries", []))
    }

    return np.array(X_list), np.array(y_list), feature_names, metadata


def load_labeled_dataset(
    dataset_dir: Optional[str] = None,
    metadata_csv: Optional[str] = None,
    samples_per_class: int = 50,
    k_list: List[int] = [3, 4],
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Universal dataset loader.
    Supports:
      1. NCBI Datasets packages (e.g. from 'datasets download virus genome ...')
      2. Metadata CSV/TSV files mapping sequence files to labels
      3. Class subdirectory folders (dataset_dir/<class_name>/<sequences>.fasta)
      4. Clearly-labeled synthetic demonstration data fallback

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray, feature_names: List[str], metadata: dict)
    """
    if metadata_csv and os.path.exists(metadata_csv):
        print(f"Loading external dataset from metadata CSV: {metadata_csv}")
        return load_dataset_from_csv(metadata_csv, k_list=k_list)

    if dataset_dir and os.path.exists(dataset_dir):
        # Check if dataset_dir is an NCBI Datasets package
        if (
            os.path.exists(os.path.join(dataset_dir, "ncbi_dataset"))
            or os.path.exists(os.path.join(dataset_dir, "data_report.jsonl"))
            or os.path.exists(os.path.join(dataset_dir, "genomic.fna"))
            or os.path.exists(os.path.join(dataset_dir, "ncbi_dataset", "data", "genomic.fna"))
        ):
            print(f"Loading external dataset from NCBI Datasets package: {dataset_dir}")
            return load_ncbi_dataset(dataset_dir, k_list=k_list)

        print(f"Loading external dataset from directory: {dataset_dir}")
        return load_dataset_from_directory(dataset_dir, k_list=k_list)

    print("No external verified dataset provided. Utilizing clearly-labeled synthetic demonstration data...")
    return generate_synthetic_demo_dataset(samples_per_class=samples_per_class, k_list=k_list, seed=seed)

