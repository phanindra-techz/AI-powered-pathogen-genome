"""
FASTA File Reader and Validation Module.
Provides robust parsing and validation of genomic FASTA and FA files using Biopython.
Extracts sequence ID, header descriptions, validates nucleotide symbols,
and ensures biological sequence integrity.
"""

import io
import os
from typing import Dict, List, Any, Union, Tuple, Optional
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

# Standard and IUPAC nucleotide alphabets
STANDARD_NUCLEOTIDES = set("ACGTU")
IUPAC_NUCLEOTIDES = set("ACGTURYSWKMBDHVN")


class FASTAValidationError(Exception):
    """Custom exception raised when FASTA file or sequence fails validation."""
    pass


def validate_nucleotide_sequence(
    sequence: str,
    allow_ambiguous: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validates that a sequence string contains only expected nucleotide symbols.
    
    Args:
        sequence: Nucleotide sequence string (case-insensitive).
        allow_ambiguous: If True, allows IUPAC ambiguous codes (R, Y, S, W, K, M, B, D, H, V, N).
                         If False, only allows canonical A, C, G, T, U.

    Returns:
        Tuple of (is_valid: bool, invalid_characters: List[str])
    """
    if not sequence:
        return False, ["<EMPTY>"]

    allowed = IUPAC_NUCLEOTIDES if allow_ambiguous else STANDARD_NUCLEOTIDES
    seq_upper = sequence.upper()
    invalid_chars = sorted(list(set(seq_upper) - allowed))

    return len(invalid_chars) == 0, invalid_chars


def validate_sequence(sequence: str, min_length: int = 50) -> Tuple[bool, List[str]]:
    """
    Validates that a nucleotide sequence meets basic bioinformatics quality checks.
    
    Args:
        sequence: Nucleotide sequence string.
        min_length: Minimum acceptable sequence length.

    Returns:
        Tuple of (is_valid: bool, issues: List[str])
    """
    issues: List[str] = []
    
    if not sequence or len(sequence.strip()) == 0:
        return False, ["Sequence is completely empty."]

    clean_seq = "".join(sequence.split()).upper()

    if len(clean_seq) < min_length:
        issues.append(f"Sequence length ({len(clean_seq)} bp) is shorter than minimum recommended threshold ({min_length} bp).")

    is_valid_nuc, invalid_chars = validate_nucleotide_sequence(clean_seq, allow_ambiguous=True)
    if not is_valid_nuc:
        issues.append(f"Sequence contains invalid non-IUPAC characters: {', '.join(repr(c) for c in invalid_chars)}")

    is_valid = len(invalid_chars) == 0 and len(clean_seq) >= min_length
    return is_valid, issues


def read_fasta(file_path: str, validate_nucleotides: bool = True) -> Dict[str, Any]:
    """
    Reads a FASTA or FA file from a file path using Biopython.
    
    Args:
        file_path: Path to the .fasta or .fa file.
        validate_nucleotides: If True, validates sequence symbols.

    Returns:
        Dict containing parsed sequence information and records.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file not found at path: '{file_path}'")
    
    return parse_fasta(file_path, validate_nucleotides=validate_nucleotides)


def parse_fasta(
    source: Union[str, io.StringIO, io.BytesIO, Any],
    validate_nucleotides: bool = True,
    allow_ambiguous: bool = True
) -> Dict[str, Any]:
    """
    Parses a FASTA/FA file or string content into structured genome record data using Biopython.
    
    Args:
        source: File path (str), StringIO, BytesIO, or file-like object containing FASTA data.
        validate_nucleotides: Whether to enforce valid nucleotide characters.
        allow_ambiguous: Whether IUPAC ambiguous nucleotide codes are accepted.

    Returns:
        Dict containing:
            - 'records': List of dicts per record ('id', 'description', 'sequence', 'length')
            - 'primary_sequence': Uppercase concatenated sequence string
            - 'total_length': Total nucleotide length
            - 'num_records': Count of records
            - 'header': Main header description
            - 'sequence_id': Primary sequence identifier
    """
    raw_text = ""

    try:
        if isinstance(source, str):
            if os.path.exists(source) and os.path.isfile(source):
                # Existing file path
                if os.path.getsize(source) == 0:
                    raise FASTAValidationError(f"FASTA file '{source}' is empty (0 bytes).")
                with open(source, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read()
            elif "\n" not in source and (source.endswith((".fasta", ".fa", ".fna", ".txt")) or "/" in source or "\\" in source):
                if not os.path.exists(source):
                    raise FileNotFoundError(f"FASTA file not found at path: '{source}'")
                raw_text = source
            else:
                # String content
                raw_text = source
        elif isinstance(source, io.BytesIO):
            raw_text = source.getvalue().decode("utf-8", errors="replace")
        elif isinstance(source, io.StringIO):
            raw_text = source.getvalue()
        elif hasattr(source, "read"):
            content = source.read()
            if isinstance(content, bytes):
                raw_text = content.decode("utf-8", errors="replace")
            else:
                raw_text = str(content)
            # Reset seek position if possible
            if hasattr(source, "seek"):
                try:
                    source.seek(0)
                except Exception:
                    pass
        else:
            raise FASTAValidationError(f"Unsupported FASTA source type: {type(source).__name__}")
    except (FileNotFoundError, FASTAValidationError):
        raise
    except Exception as e:
        raise FASTAValidationError(f"Error accessing FASTA source: {str(e)}")

    stripped_text = raw_text.strip()
    if not stripped_text:
        raise FASTAValidationError("FASTA file/content is empty and contains no records.")

    # Check for basic FASTA structure (must begin with '>')
    if not stripped_text.startswith(">"):
        raise FASTAValidationError(
            "Malformed FASTA format: Input does not begin with a valid FASTA header (expected line starting with '>')."
        )

    # Parse with Biopython SeqIO
    handle = io.StringIO(stripped_text)
    try:
        seq_records = list(SeqIO.parse(handle, "fasta"))
    except Exception as e:
        raise FASTAValidationError(f"Biopython failed to parse FASTA stream: {str(e)}")

    if not seq_records:
        raise FASTAValidationError("Malformed FASTA format: No valid sequence records could be extracted.")

    records_data: List[Dict[str, Any]] = []

    for idx, rec in enumerate(seq_records):
        rec_id = (rec.id or f"record_{idx + 1}").strip()
        rec_desc = (rec.description or rec_id).strip()
        
        # Extract sequence and convert to uppercase
        seq_raw = str(rec.seq)
        seq_clean = "".join(seq_raw.split()).upper()

        if not seq_clean:
            raise FASTAValidationError(
                f"Malformed FASTA: Record '{rec_id}' (index {idx + 1}) contains an empty sequence."
            )

        # Validate nucleotide symbols
        if validate_nucleotides:
            is_valid, invalid_chars = validate_nucleotide_sequence(seq_clean, allow_ambiguous=allow_ambiguous)
            if not is_valid:
                char_list = ", ".join(repr(c) for c in invalid_chars)
                expected_desc = "IUPAC DNA/RNA symbols (A, C, G, T, U, N, R, Y, S, W, K, M, B, D, H, V)" if allow_ambiguous else "canonical A, C, G, T, U"
                raise FASTAValidationError(
                    f"Invalid sequence in record '{rec_id}': Contains non-nucleotide characters [{char_list}]. "
                    f"Expected valid {expected_desc}."
                )

        records_data.append({
            "id": rec_id,
            "description": rec_desc,
            "sequence": seq_clean,
            "length": len(seq_clean)
        })

    if not records_data:
        raise FASTAValidationError("FASTA file contains records but no valid biological sequence data.")

    primary_seq = records_data[0]["sequence"] if len(records_data) == 1 else "".join(r["sequence"] for r in records_data)
    total_length = sum(r["length"] for r in records_data)
    main_header = records_data[0]["description"]
    main_id = records_data[0]["id"]

    return {
        "records": records_data,
        "primary_sequence": primary_seq,
        "total_length": total_length,
        "num_records": len(records_data),
        "header": main_header,
        "sequence_id": main_id
    }
