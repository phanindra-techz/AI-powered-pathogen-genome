"""
Database Access Layer for AI-Powered Pathogen Genome Intelligence
Uses Python's standard library sqlite3 for storing metadata, predictions,
analysis history, reference comparisons, differences, and model runs.
"""

import os
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Generator

logger = logging.getLogger(__name__)

# Base directory of the repository
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "pathogen_intelligence.db")


def get_default_db_path() -> str:
    """Returns the default absolute SQLite database file path."""
    return DEFAULT_DB_PATH


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Creates and returns a SQLite connection with foreign keys enabled
    and row_factory configured to return dictionary-like sqlite3.Row objects.
    """
    path = db_path or get_default_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=20.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_session(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a connection, commits on success, rolls back on error,
    and guarantees that the connection is closed upon exit.
    """
    conn = get_db_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database(db_path: Optional[str] = None) -> bool:
    """
    Initializes the SQLite database and creates the required tables and indexes
    if they do not already exist.

    Tables created:
    - analyses
    - reference_results
    - differences
    - model_runs
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()

            # 1. analyses table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT UNIQUE NOT NULL,
                sample_id TEXT,
                filename TEXT,
                sequence_length INTEGER,
                gc_percent REAL,
                ambiguous_bases INTEGER,
                qc_status TEXT,
                prediction TEXT,
                confidence REAL,
                model_name TEXT,
                model_version TEXT,
                created_at TEXT
            );
            """)

            # 2. reference_results table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS reference_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                reference_id TEXT,
                similarity REAL,
                evidence TEXT,
                created_at TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
            );
            """)

            # 3. differences table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS differences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                reference_id TEXT,
                difference_summary TEXT,
                created_at TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
            );
            """)

            # 4. model_runs table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                model_version TEXT,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                created_at TEXT
            );
            """)

            # Indexes for high performance querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_analysis_id ON analyses(analysis_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ref_results_analysis_id ON reference_results(analysis_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_differences_analysis_id ON differences(analysis_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_runs_created_at ON model_runs(created_at);")

            return True
    except Exception as e:
        logger.error(f"Error initializing SQLite database at {db_path}: {e}")
        return False


def check_db_status(db_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Checks if the database is accessible and functional.
    Returns (is_connected, status_message).
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.fetchone()
            return True, "Connected"
    except Exception as e:
        return False, f"Unavailable ({e})"


def save_analysis(analysis_data: Dict[str, Any], db_path: Optional[str] = None) -> bool:
    """
    Inserts or updates an analysis record in the analyses table.
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            created_at = analysis_data.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute("""
            INSERT INTO analyses (
                analysis_id, sample_id, filename, sequence_length,
                gc_percent, ambiguous_bases, qc_status, prediction,
                confidence, model_name, model_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(analysis_id) DO UPDATE SET
                sample_id=excluded.sample_id,
                filename=excluded.filename,
                sequence_length=excluded.sequence_length,
                gc_percent=excluded.gc_percent,
                ambiguous_bases=excluded.ambiguous_bases,
                qc_status=excluded.qc_status,
                prediction=excluded.prediction,
                confidence=excluded.confidence,
                model_name=excluded.model_name,
                model_version=excluded.model_version,
                created_at=excluded.created_at;
            """, (
                analysis_data.get("analysis_id"),
                analysis_data.get("sample_id"),
                analysis_data.get("filename"),
                int(analysis_data.get("sequence_length", 0)),
                float(analysis_data.get("gc_percent", 0.0)),
                int(analysis_data.get("ambiguous_bases", 0)),
                analysis_data.get("qc_status", "UNKNOWN"),
                analysis_data.get("prediction", "Unknown"),
                float(analysis_data.get("confidence", 0.0)),
                analysis_data.get("model_name", "Random Forest"),
                analysis_data.get("model_version", "v1.0"),
                created_at
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to save analysis {analysis_data.get('analysis_id')}: {e}")
        return False


def get_analysis(analysis_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single analysis record by its unique analysis_id.
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses WHERE analysis_id = ? LIMIT 1;", (analysis_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    except Exception as e:
        logger.error(f"Failed to get analysis {analysis_id}: {e}")
        return None


def get_all_analyses(limit: int = 100, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves all analyses ordered by creation date descending.
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get all analyses: {e}")
        return []


def save_reference_result(ref_data: Dict[str, Any], db_path: Optional[str] = None) -> bool:
    """
    Inserts a single reference comparison result.
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            created_at = ref_data.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
            INSERT INTO reference_results (
                analysis_id, reference_id, similarity, evidence, created_at
            ) VALUES (?, ?, ?, ?, ?);
            """, (
                ref_data.get("analysis_id"),
                ref_data.get("reference_id"),
                float(ref_data.get("similarity", 0.0)),
                ref_data.get("evidence", ""),
                created_at
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to save reference result: {e}")
        return False


def save_reference_results(
    analysis_id: str,
    results_list: List[Dict[str, Any]],
    db_path: Optional[str] = None
) -> bool:
    """
    Batch inserts reference comparison results for an analysis.
    """
    if not results_list:
        return True
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            # Clean existing records for this analysis_id to allow clean updates
            cursor.execute("DELETE FROM reference_results WHERE analysis_id = ?;", (analysis_id,))
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for item in results_list:
                cursor.execute("""
                INSERT INTO reference_results (
                    analysis_id, reference_id, similarity, evidence, created_at
                ) VALUES (?, ?, ?, ?, ?);
                """, (
                    analysis_id,
                    item.get("reference_id") or item.get("Reference Name") or item.get("Reference ID"),
                    float(item.get("similarity") or item.get("Composite Score (%)") or item.get("k-mer Cosine Sim", 0.0)),
                    item.get("evidence") or f"Cosine: {item.get('k-mer Cosine Sim', 0.0):.4f} | Jaccard: {item.get('Jaccard Sim', 0.0):.4f}",
                    created_at
                ))
            return True
    except Exception as e:
        logger.error(f"Failed to save reference results batch for {analysis_id}: {e}")
        return False


def get_reference_results(analysis_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves all reference comparison results associated with an analysis_id.
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM reference_results WHERE analysis_id = ? ORDER BY similarity DESC;",
                (analysis_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get reference results for {analysis_id}: {e}")
        return []


def save_difference(diff_data: Dict[str, Any], db_path: Optional[str] = None) -> bool:
    """
    Inserts a single observed difference record.
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            created_at = diff_data.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
            INSERT INTO differences (
                analysis_id, reference_id, difference_summary, created_at
            ) VALUES (?, ?, ?, ?);
            """, (
                diff_data.get("analysis_id"),
                diff_data.get("reference_id"),
                diff_data.get("difference_summary"),
                created_at
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to save difference: {e}")
        return False


def save_differences(
    analysis_id: str,
    differences_list: List[Dict[str, Any]],
    db_path: Optional[str] = None
) -> bool:
    """
    Batch inserts observed difference records for an analysis.
    """
    if not differences_list:
        return True
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM differences WHERE analysis_id = ?;", (analysis_id,))
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for item in differences_list:
                cursor.execute("""
                INSERT INTO differences (
                    analysis_id, reference_id, difference_summary, created_at
                ) VALUES (?, ?, ?, ?);
                """, (
                    analysis_id,
                    item.get("reference_id"),
                    item.get("difference_summary"),
                    created_at
                ))
            return True
    except Exception as e:
        logger.error(f"Failed to save differences batch for {analysis_id}: {e}")
        return False


def get_differences(analysis_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves all observed differences associated with an analysis_id.
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM differences WHERE analysis_id = ? ORDER BY id ASC;",
                (analysis_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get differences for {analysis_id}: {e}")
        return []


def save_model_run(model_data: Dict[str, Any], db_path: Optional[str] = None) -> bool:
    """
    Inserts a machine learning model evaluation record in the model_runs table.
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            created_at = model_data.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
            INSERT INTO model_runs (
                model_name, model_version, accuracy,
                precision_score, recall_score, f1_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                model_data.get("model_name", "Random Forest"),
                model_data.get("model_version", "v1.0"),
                float(model_data.get("accuracy", 0.0)),
                float(model_data.get("precision_score") or model_data.get("precision", 0.0)),
                float(model_data.get("recall_score") or model_data.get("recall", 0.0)),
                float(model_data.get("f1_score", 0.0)),
                created_at
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to save model run: {e}")
        return False


def get_model_runs(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves historical model evaluation runs.
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_runs ORDER BY id DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get model runs: {e}")
        return []


def delete_analysis(analysis_id: str, db_path: Optional[str] = None) -> bool:
    """
    Deletes an analysis and cascade deletes its associated reference results and differences.
    """
    try:
        with db_session(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analyses WHERE analysis_id = ?;", (analysis_id,))
            return True
    except Exception as e:
        logger.error(f"Failed to delete analysis {analysis_id}: {e}")
        return False
