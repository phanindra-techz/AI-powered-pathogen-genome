"""
Unit tests for src.fasta_reader module.
Validates FASTA/FA parsing, ID extraction, uppercase conversion,
nucleotide symbol validation, empty files, malformed files, and error messages.
"""

import os
import pytest
import io
from src.fasta_reader import (
    parse_fasta,
    read_fasta,
    validate_sequence,
    validate_nucleotide_sequence,
    FASTAValidationError
)


def test_parse_fasta_valid_string():
    """Test reading a valid FASTA string with single record."""
    fasta_str = ">seq_001 Pathogen test isolate\nATGCGATCGATCGATCGATCGATCGATC\n"
    res = parse_fasta(fasta_str)
    assert res["num_records"] == 1
    assert res["sequence_id"] == "seq_001"
    assert res["header"] == "seq_001 Pathogen test isolate"
    assert res["primary_sequence"] == "ATGCGATCGATCGATCGATCGATCGATC"
    assert res["total_length"] == 28


def test_read_fasta_and_fa_files(tmp_path):
    """Test reading actual .fasta and .fa files from disk using Biopython."""
    fasta_file = tmp_path / "sample.fasta"
    fasta_file.write_text(">isolate_A Influenza segment 1\nATGGATTCCAACACTGTGTCAAGC\n", encoding="utf-8")
    
    fa_file = tmp_path / "sample.fa"
    fa_file.write_text(">isolate_B SARS-CoV-2 spike partial\nATGTTTGTTTTTCTTGTTTTATTGCCACTA\n", encoding="utf-8")

    res_fasta = read_fasta(str(fasta_file))
    assert res_fasta["sequence_id"] == "isolate_A"
    assert res_fasta["primary_sequence"] == "ATGGATTCCAACACTGTGTCAAGC"

    res_fa = read_fasta(str(fa_file))
    assert res_fa["sequence_id"] == "isolate_B"
    assert res_fa["primary_sequence"] == "ATGTTTGTTTTTCTTGTTTTATTGCCACTA"


def test_sequence_id_and_header_extraction():
    """Test extracting sequence ID and full header information correctly."""
    fasta_content = ">NC_045512.2 Severe acute respiratory syndrome coronavirus 2 isolate Wuhan-Hu-1\nATTAAAGGTTTATACCTTCCCAGGTAACAAACCAACCAACTTTCGATCTCTTGTAGATCTGTTCTCTAAA\n"
    res = parse_fasta(fasta_content)
    assert res["sequence_id"] == "NC_045512.2"
    assert "Wuhan-Hu-1" in res["header"]
    assert res["records"][0]["id"] == "NC_045512.2"


def test_convert_sequence_to_uppercase():
    """Test that lowercase and mixed-case sequences are standardized to uppercase."""
    fasta_content = ">seq_lower Lowercase test\natgcgatcgatc\n"
    res = parse_fasta(fasta_content)
    assert res["primary_sequence"] == "ATGCGATCGATC"


def test_biological_sequence_integrity():
    """Test that biological sequence characters are preserved accurately across multiline FASTA."""
    expected_seq = "ATGCGTACGTTAGCTAGCTAGCTAGCATCGATCGATCGATC"
    fasta_content = f">seq_multi Multiline record\n{expected_seq[:20]}\n{expected_seq[20:]}\n"
    res = parse_fasta(fasta_content)
    assert res["primary_sequence"] == expected_seq
    assert res["total_length"] == len(expected_seq)


def test_valid_nucleotide_symbols():
    """Test that standard DNA/RNA and IUPAC ambiguous nucleotide symbols pass validation."""
    # Canonical DNA/RNA + Ambiguous IUPAC (R, Y, S, W, K, M, B, D, H, V, N)
    valid_seq = "ACGTUNRYSWKMBDHV"
    fasta_content = f">valid_iupac IUPAC ambiguous sample\n{valid_seq}\n"
    res = parse_fasta(fasta_content, validate_nucleotides=True, allow_ambiguous=True)
    assert res["primary_sequence"] == valid_seq


def test_invalid_nucleotide_symbols_raises_error():
    """Test that invalid non-nucleotide characters raise FASTAValidationError with informative message."""
    invalid_fasta = ">invalid_seq Bad characters\nATGC123Z#!\n"
    with pytest.raises(FASTAValidationError) as excinfo:
        parse_fasta(invalid_fasta, validate_nucleotides=True)
    
    err_msg = str(excinfo.value)
    assert "Invalid sequence in record 'invalid_seq'" in err_msg
    assert "non-nucleotide characters" in err_msg


def test_handle_empty_string_and_empty_file(tmp_path):
    """Test that empty inputs raise clear FASTAValidationError."""
    with pytest.raises(FASTAValidationError) as excinfo:
        parse_fasta("")
    assert "empty and contains no records" in str(excinfo.value)

    with pytest.raises(FASTAValidationError) as excinfo:
        parse_fasta("   \n\n  \t ")
    assert "empty and contains no records" in str(excinfo.value)

    empty_file = tmp_path / "empty.fasta"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(FASTAValidationError) as excinfo:
        parse_fasta(str(empty_file))
    assert "is empty (0 bytes)" in str(excinfo.value)


def test_handle_malformed_fasta_no_header():
    """Test that malformed FASTA missing '>' header raises clear error."""
    malformed_data = "ATGCGATCGATCGATCGATC\nGCTAGCTAGCTA\n"
    with pytest.raises(FASTAValidationError) as excinfo:
        parse_fasta(malformed_data)
    assert "Malformed FASTA format" in str(excinfo.value)
    assert "starting with '>'" in str(excinfo.value)


def test_handle_malformed_fasta_empty_sequence():
    """Test that header without sequence raises clear error."""
    malformed_data = ">empty_seq_header\n>next_header\nATGC\n"
    with pytest.raises(FASTAValidationError) as excinfo:
        parse_fasta(malformed_data)
    assert "contains an empty sequence" in str(excinfo.value)


def test_file_not_found():
    """Test that reading a nonexistent file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_fasta("non_existent_path_xyz123.fasta")


def test_validate_sequence_helper():
    """Test validate_sequence helper for length and character checks."""
    is_valid, issues = validate_sequence("ATGC", min_length=50)
    assert not is_valid
    assert any("shorter than minimum" in issue for issue in issues)

    is_valid_ok, issues_ok = validate_sequence("ATGC" * 20, min_length=50)
    assert is_valid_ok
    assert len(issues_ok) == 0

    is_valid_bad, issues_bad = validate_sequence("ATGC123" * 10, min_length=50)
    assert not is_valid_bad
    assert any("invalid non-IUPAC" in issue for issue in issues_bad)
