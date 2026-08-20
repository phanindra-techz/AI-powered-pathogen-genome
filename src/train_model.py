"""
Machine Learning Pathogen Genome Classifier Training and Model Selection Module.
Supports pluggable real/verified datasets (directory or CSV) and fallback synthetic demonstration data.
Trains and compares Logistic Regression, Random Forest, and Support Vector Machine (SVM)
using multi-scale k-mer frequency features. Evaluates Accuracy, Precision, Recall, F1-score,
and Confusion Matrices across Train/Validation/Test splits.
Selects the best model based strictly on validation performance and saves models/classifier.joblib.
"""

import os
import json
import argparse
import joblib
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.dataset_loader import (
    load_labeled_dataset,
    DEMO_PATHOGEN_PROFILES,
    _generate_synthetic_sequence
)

# Exported profile dictionary
PATHOGEN_PROFILES = DEMO_PATHOGEN_PROFILES

# Prominent Research Safety Banner
DEMO_BANNER = "DEMONSTRATION MODEL — NOT CLINICALLY VALIDATED"
DEMO_DISCLAIMER = (
    "DEMONSTRATION MODEL — NOT CLINICALLY VALIDATED. "
    "This software prototype was evaluated on demonstration sequence profiles for software testing only. "
    "Predictions do NOT constitute real biological or pathogen identification, and MUST NOT be used "
    "for clinical diagnosis, medical evaluation, or therapeutic decision-making."
)


def create_reference_datasets(data_dir: str = "data") -> None:
    """Creates demonstration reference FASTA files in data/reference and data/raw."""
    ref_dir = os.path.join(data_dir, "reference")
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print("Generating reference pathogen genome datasets...")
    for idx, (p_name, p_info) in enumerate(DEMO_PATHOGEN_PROFILES.items()):
        # Generate master reference sequence (2500 bp)
        ref_seq = _generate_synthetic_sequence(
            gc_target=p_info["gc_target"],
            kmer_bias=p_info["kmer_bias"],
            length=2500,
            seed=42 + idx * 10
        )
        
        ref_path = os.path.join(ref_dir, p_info["ref_filename"])
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(f">{p_name} | RefID: REF_{idx+1:03d} | [DEMO_SYNTHETIC] {p_info['description']}\n")
            for i in range(0, len(ref_seq), 70):
                f.write(ref_seq[i:i+70] + "\n")

        # Sample query FASTA files in data/raw for demonstration testing
        if idx in [0, 1, 3, 6]:  # Influenza, SARS-CoV-2, E. coli, Monkeypox
            sample_seq = _generate_synthetic_sequence(
                gc_target=p_info["gc_target"],
                kmer_bias=p_info["kmer_bias"],
                length=1800,
                seed=99 + idx * 15
            )
            raw_filename = f"sample_{p_name.lower().replace(' ', '_').replace('-', '_')}.fasta"
            raw_path = os.path.join(raw_dir, raw_filename)
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(f">{p_name}_Sample_Query | [DEMO_SYNTHETIC] DEMO_QUERY_{idx+1}\n")
                for i in range(0, len(sample_seq), 70):
                    f.write(sample_seq[i:i+70] + "\n")

    print(f"Reference sequences created in: {ref_dir}")


def generate_training_dataset(samples_per_class: int = 50, k_list: List[int] = [3, 4], seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generates synthetic demonstration training dataset."""
    X, y, feature_names, _ = load_labeled_dataset(samples_per_class=samples_per_class, k_list=k_list, seed=seed)
    return X, y, feature_names


def build_and_train_model(
    models_dir: str = "models",
    dataset_dir: Optional[str] = None,
    metadata_csv: Optional[str] = None,
    samples_per_class: int = 50,
    k_list: List[int] = [3, 4],
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains and compares Logistic Regression, Random Forest, and SVM models.
    Accepts external verified datasets or generates labeled synthetic demo profiles.
    Splits dataset into 60% Train, 20% Validation, 20% Test.
    Selects best model based strictly on validation performance.
    Saves winning model and metadata to models/classifier.joblib.

    Returns:
        Dictionary of model metadata, comparative evaluation metrics, and final test score.
    """
    os.makedirs(models_dir, exist_ok=True)

    print("\n============================================================")
    print(f"  {DEMO_BANNER}")
    print("============================================================")
    print("\n--- 1. Ingesting Dataset & Extracting k-Mer Features ---")
    
    X, y, feature_names, dataset_meta = load_labeled_dataset(
        dataset_dir=dataset_dir,
        metadata_csv=metadata_csv,
        samples_per_class=samples_per_class,
        k_list=k_list,
        seed=random_state
    )
    
    classes = sorted(list(np.unique(y)))
    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(classes)} classes.")
    print(f"Classes: {', '.join(classes)}")
    print(f"Dataset Type: {'Synthetic Demo Profiles' if dataset_meta.get('is_synthetic', True) else 'Verified Labeled Dataset'}")

    # --- 2. Train / Validation / Test Split (60% Train, 20% Val, 20% Test) ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp
    )

    print(f"Split sizes: Train={X_train.shape[0]}, Validation={X_val.shape[0]}, Test={X_test.shape[0]}")

    # --- 3. Define Candidate Models ---
    candidate_pipelines: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state, C=1.0))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state))
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=random_state, C=1.0),
                ensemble=False
            ))
        ])
    }

    # --- 4. Train Models & Evaluate on Validation Set (Model Selection) ---
    print("\n--- 2. Training and Comparing Candidate Models (Validation Set) ---")
    val_comparison: Dict[str, Dict[str, Any]] = {}
    best_model_name: Optional[str] = None
    best_val_f1 = -1.0

    for name, pipeline in candidate_pipelines.items():
        print(f"Fitting {name}...")
        pipeline.fit(X_train, y_train)

        # Validation set evaluation (for model selection)
        y_val_pred = pipeline.predict(X_val)
        val_acc = float(accuracy_score(y_val, y_val_pred))
        val_prec = float(precision_score(y_val, y_val_pred, average="weighted", zero_division=0))
        val_rec = float(recall_score(y_val, y_val_pred, average="weighted", zero_division=0))
        val_f1 = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))
        val_cm = confusion_matrix(y_val, y_val_pred, labels=classes).tolist()

        val_comparison[name] = {
            "accuracy": round(val_acc, 4),
            "precision": round(val_prec, 4),
            "recall": round(val_rec, 4),
            "f1_score": round(val_f1, 4),
            "confusion_matrix": val_cm
        }

        print(f"  -> {name} | Val Acc: {val_acc*100:.2f}% | Val Precision: {val_prec*100:.2f}% | Val Recall: {val_rec*100:.2f}% | Val F1: {val_f1*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_name = name

    print(f"\n[BEST MODEL] Selected Model (based on Validation performance): '{best_model_name}' (Val F1: {best_val_f1*100:.2f}%)")

    # --- 5. Evaluate Selected Best Model on Unseen Final Test Set ---
    print("\n--- 3. Final Unbiased Evaluation on Held-Out Test Set ---")
    winning_pipeline = candidate_pipelines[best_model_name]
    y_test_pred = winning_pipeline.predict(X_test)

    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_prec = float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_rec = float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_f1 = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_cm = confusion_matrix(y_test, y_test_pred, labels=classes).tolist()
    test_report_dict = classification_report(y_test, y_test_pred, labels=classes, output_dict=True, zero_division=0)

    print(f"Final Test Accuracy : {test_acc * 100.0:.2f}%")
    print(f"Final Test Precision: {test_prec * 100.0:.2f}%")
    print(f"Final Test Recall   : {test_rec * 100.0:.2f}%")
    print(f"Final Test F1-Score : {test_f1 * 100.0:.2f}%")
    print("\nTest Set Classification Report:\n", classification_report(y_test, y_test_pred, labels=classes, zero_division=0))

    test_metrics = {
        "accuracy": round(test_acc, 4),
        "precision": round(test_prec, 4),
        "recall": round(test_rec, 4),
        "f1_score": round(test_f1, 4),
        "confusion_matrix": test_cm,
        "classification_report": test_report_dict
    }

    # --- 6. Package Metadata and Persist Final Artifacts ---
    is_demo = dataset_meta.get("is_synthetic", True)
    model_metadata = {
        "pipeline": winning_pipeline,
        "best_model_name": best_model_name,
        "classes": classes,
        "k_list": k_list,
        "feature_names": feature_names,
        "accuracy": test_acc,
        "validation_comparison": val_comparison,
        "test_metrics": test_metrics,
        "split_sizes": {
            "train": int(X_train.shape[0]),
            "validation": int(X_val.shape[0]),
            "test": int(X_test.shape[0])
        },
        "dataset_metadata": dataset_meta,
        "is_demo_model": is_demo,
        "demo_banner": DEMO_BANNER if is_demo else "TRAINED ON USER DATASET",
        "pathogen_profiles": DEMO_PATHOGEN_PROFILES if is_demo else {},
        "disclaimer": DEMO_DISCLAIMER if is_demo else "RESEARCH PREDICTION MODEL. NOT FOR CLINICAL DIAGNOSIS."
    }

    # Save joblib model artifact
    joblib_path = os.path.join(models_dir, "classifier.joblib")
    joblib.dump(model_metadata, joblib_path)
    print(f"\nModel artifact saved to: {joblib_path}")

    # Save JSON metadata
    json_metrics = {
        "demo_banner": model_metadata["demo_banner"],
        "is_demo_model": model_metadata["is_demo_model"],
        "best_model_name": best_model_name,
        "classes": classes,
        "k_list": k_list,
        "num_features": len(feature_names),
        "split_sizes": model_metadata["split_sizes"],
        "dataset_metadata": dataset_meta,
        "validation_comparison": val_comparison,
        "test_metrics": test_metrics,
        "disclaimer": model_metadata["disclaimer"]
    }
    json_path = os.path.join(models_dir, "model_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Metrics and metadata saved to: {json_path}")

    return model_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pathogen Genome ML Classifier")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Directory containing class subfolders of FASTA files")
    parser.add_argument("--metadata-csv", type=str, default=None, help="Path to metadata CSV mapping file paths to labels")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save classifier.joblib and metrics")
    parser.add_argument("--samples-per-class", type=int, default=50, help="Demo samples per class (for synthetic mode)")
    args = parser.parse_args()

    create_reference_datasets()
    build_and_train_model(
        models_dir=args.models_dir,
        dataset_dir=args.dataset_dir,
        metadata_csv=args.metadata_csv,
        samples_per_class=args.samples_per_class
    )
