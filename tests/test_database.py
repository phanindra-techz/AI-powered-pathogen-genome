"""
Unit tests for SQLite Database Layer (src/database.py).
Tests database initialization, CRUD operations, foreign keys, constraints, and error handling.
"""

import os
import tempfile
import pytest
from src.database import (
    init_database,
    check_db_status,
    db_session,
    save_analysis,
    get_analysis,
    get_all_analyses,
    save_reference_result,
    save_reference_results,
    get_reference_results,
    save_difference,
    save_differences,
    get_differences,
    save_model_run,
    get_model_runs,
    delete_analysis,
    get_db_connection
)


@pytest.fixture
def temp_db():
    """Creates a temporary SQLite database file for isolation."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_database(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_database_and_table_creation(temp_db):
    """Test 1 & 2: Database creation and table initialization."""
    assert os.path.exists(temp_db)
    is_connected, msg = check_db_status(temp_db)
    assert is_connected is True
    assert msg == "Connected"

    with db_session(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]
        assert "analyses" in tables
        assert "reference_results" in tables
        assert "differences" in tables
        assert "model_runs" in tables


def test_insert_and_get_analysis(temp_db):
    """Test 3 & 4: Insert analysis and retrieve analysis by analysis_id."""
    analysis_data = {
        "analysis_id": "PGI-20260820-TEST01",
        "sample_id": "SARS_COV_2_SAMPLE_01",
        "filename": "sars_sample.fasta",
        "sequence_length": 29903,
        "gc_percent": 37.95,
        "ambiguous_bases": 0,
        "qc_status": "PASS",
        "prediction": "SARS-CoV-2",
        "confidence": 98.5,
        "model_name": "Random Forest",
        "model_version": "v1.0",
        "created_at": "2026-08-20 20:00:00"
    }
    success = save_analysis(analysis_data, db_path=temp_db)
    assert success is True

    record = get_analysis("PGI-20260820-TEST01", db_path=temp_db)
    assert record is not None
    assert record["analysis_id"] == "PGI-20260820-TEST01"
    assert record["sample_id"] == "SARS_COV_2_SAMPLE_01"
    assert record["sequence_length"] == 29903
    assert record["gc_percent"] == 37.95
    assert record["prediction"] == "SARS-CoV-2"
    assert record["confidence"] == 98.5
    assert record["qc_status"] == "PASS"


def test_get_all_analyses(temp_db):
    """Test 5: Retrieve all analyses ordered descending."""
    for i in range(3):
        save_analysis({
            "analysis_id": f"PGI-20260820-000{i+1}",
            "sample_id": f"SAMPLE_{i+1}",
            "filename": f"sample_{i+1}.fasta",
            "sequence_length": 1000 * (i + 1),
            "gc_percent": 40.0 + i,
            "ambiguous_bases": 0,
            "qc_status": "PASS",
            "prediction": f"Pathogen_{i+1}",
            "confidence": 90.0 + i
        }, db_path=temp_db)

    all_records = get_all_analyses(db_path=temp_db)
    assert len(all_records) == 3
    # Ordered descending
    assert all_records[0]["analysis_id"] == "PGI-20260820-0003"
    assert all_records[2]["analysis_id"] == "PGI-20260820-0001"


def test_insert_and_get_reference_result(temp_db):
    """Test 6 & 7: Insert reference results and retrieve by analysis_id."""
    analysis_id = "PGI-20260820-REF01"
    save_analysis({
        "analysis_id": analysis_id,
        "sample_id": "FLU_SAMPLE",
        "sequence_length": 13000,
        "gc_percent": 44.0,
        "prediction": "Influenza A virus",
        "confidence": 95.0
    }, db_path=temp_db)

    # Save single
    save_reference_result({
        "analysis_id": analysis_id,
        "reference_id": "Influenza A Reference NC_007373",
        "similarity": 96.5,
        "evidence": "Cosine: 0.9650 | Jaccard: 0.9200"
    }, db_path=temp_db)

    # Save batch
    save_reference_results(analysis_id, [
        {"Reference Name": "SARS-CoV-2 Ref", "Composite Score (%)": 82.0, "k-mer Cosine Sim": 0.82, "Jaccard Sim": 0.75},
        {"Reference Name": "E. coli Ref", "Composite Score (%)": 54.0, "k-mer Cosine Sim": 0.54, "Jaccard Sim": 0.40}
    ], db_path=temp_db)

    results = get_reference_results(analysis_id, db_path=temp_db)
    assert len(results) == 2
    assert results[0]["similarity"] == 82.0


def test_insert_and_get_differences(temp_db):
    """Test 8: Insert differences and retrieve by analysis_id."""
    analysis_id = "PGI-20260820-DIFF01"
    save_analysis({
        "analysis_id": analysis_id,
        "sample_id": "MPOX_SAMPLE",
        "sequence_length": 197000,
        "gc_percent": 33.1,
        "prediction": "Monkeypox virus",
        "confidence": 99.0
    }, db_path=temp_db)

    diffs = [
        {"reference_id": "Monkeypox Ref", "difference_summary": "Length Δ: +120 bp, GC Δ: +0.15%"},
        {"reference_id": "Monkeypox Ref", "difference_summary": "k-mer Cosine Distance: 0.0085"}
    ]
    save_differences(analysis_id, diffs, db_path=temp_db)

    retrieved = get_differences(analysis_id, db_path=temp_db)
    assert len(retrieved) == 2
    assert retrieved[0]["reference_id"] == "Monkeypox Ref"


def test_insert_and_get_model_runs(temp_db):
    """Test 9: Insert model metrics and retrieve historical runs."""
    model_data = {
        "model_name": "Random Forest Classifier",
        "model_version": "v1.0",
        "accuracy": 0.945,
        "precision_score": 0.940,
        "recall_score": 0.945,
        "f1_score": 0.942
    }
    success = save_model_run(model_data, db_path=temp_db)
    assert success is True

    runs = get_model_runs(db_path=temp_db)
    assert len(runs) == 1
    assert runs[0]["model_name"] == "Random Forest Classifier"
    assert runs[0]["accuracy"] == 0.945
    assert runs[0]["f1_score"] == 0.942


def test_duplicate_analysis_id_handling(temp_db):
    """Test 10: Duplicate analysis IDs are updated cleanly without error."""
    record = {
        "analysis_id": "PGI-20260820-DUP01",
        "sample_id": "INITIAL_SAMPLE",
        "prediction": "Dengue virus",
        "confidence": 75.0,
        "gc_percent": 45.0,
        "sequence_length": 10000
    }
    assert save_analysis(record, db_path=temp_db) is True

    # Update with same analysis_id
    updated_record = {
        "analysis_id": "PGI-20260820-DUP01",
        "sample_id": "INITIAL_SAMPLE_UPDATED",
        "prediction": "Dengue virus",
        "confidence": 92.0,
        "gc_percent": 45.5,
        "sequence_length": 10000
    }
    assert save_analysis(updated_record, db_path=temp_db) is True

    fetched = get_analysis("PGI-20260820-DUP01", db_path=temp_db)
    assert fetched["sample_id"] == "INITIAL_SAMPLE_UPDATED"
    assert fetched["confidence"] == 92.0

    # Ensure total count is still 1
    all_recs = get_all_analyses(db_path=temp_db)
    assert len(all_recs) == 1


def test_delete_analysis_cascade(temp_db):
    """Test cascade deletion of analysis and child records."""
    analysis_id = "PGI-CASCADE-TEST"
    save_analysis({"analysis_id": analysis_id, "prediction": "Test"}, db_path=temp_db)
    save_reference_result({"analysis_id": analysis_id, "reference_id": "Ref1", "similarity": 80.0}, db_path=temp_db)
    save_difference({"analysis_id": analysis_id, "reference_id": "Ref1", "difference_summary": "Diff"}, db_path=temp_db)

    assert len(get_reference_results(analysis_id, db_path=temp_db)) == 1
    assert len(get_differences(analysis_id, db_path=temp_db)) == 1

    delete_analysis(analysis_id, db_path=temp_db)

    assert get_analysis(analysis_id, db_path=temp_db) is None
    assert len(get_reference_results(analysis_id, db_path=temp_db)) == 0
    assert len(get_differences(analysis_id, db_path=temp_db)) == 0
