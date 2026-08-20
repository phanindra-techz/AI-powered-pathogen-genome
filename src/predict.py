"""
Pathogen Genome Classification Inference Pipeline Module.
Executes the full end-to-end computational prediction workflow:
    Input FASTA -> Sequence Validation -> Quality Control (QC)
    -> k-mer Feature Transformation (consistent with training)
    -> Trained Classifier Pipeline -> Pathogen Class & Probabilities.

Outputs structured prediction records, persists prediction.json,
and clearly labels all results as statistical computational evidence.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd

from src.fasta_reader import parse_fasta, validate_sequence, FASTAValidationError
from src.qc import calculate_qc_metrics
from src.kmer_features import extract_multi_kmer_vector

# Safety & Evidence Notices
EVIDENCE_LABEL = "COMPUTATIONAL PREDICTION EVIDENCE - IN-SILICO STATISTICAL ESTIMATE"
DEMO_BANNER = "DEMONSTRATION MODEL - NOT CLINICALLY VALIDATED"
RESEARCH_DISCLAIMER = (
    "DEMONSTRATION MODEL - NOT CLINICALLY VALIDATED. "
    "All outputs represent in-silico computational prediction evidence derived from "
    "statistical k-mer feature models. They do NOT constitute real biological pathogen identification, "
    "and are strictly NOT FOR CLINICAL DIAGNOSIS, MEDICAL EVALUATION, OR THERAPEUTIC DECISION-MAKING."
)


def load_model(model_path: str = "models/classifier.joblib") -> Dict[str, Any]:
    """
    Loads saved ML model pipeline and metadata from a joblib file.
    
    Args:
        model_path: Path to the trained model .joblib file.

    Returns:
        Dict containing the pipeline, classes, k_list, and metadata.
    """
    if not os.path.exists(model_path):
        import joblib
        raise FileNotFoundError(
            f"Trained model artifact not found at '{model_path}'. "
            "Please run 'python -m src.train_model' to train the classifier pipeline first."
        )
    import joblib
    return joblib.load(model_path)


def save_prediction_json(prediction_result: Dict[str, Any], output_path: str = "prediction.json") -> str:
    """
    Saves the structured prediction record as a formatted JSON file.

    Args:
        prediction_result: Dictionary returned by predict_fasta / predict_pathogen_class.
        output_path: Target JSON file path (default: 'prediction.json').

    Returns:
        Absolute or relative path to the saved JSON file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prediction_result, f, indent=2)

    return output_path


def predict_fasta(
    source: Union[str, Any],
    model_path: str = "models/classifier.joblib",
    output_json_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the complete end-to-end prediction pipeline for an input FASTA file or stream:
        1. FASTA Parsing & Header Extraction
        2. Sequence Validation
        3. Quality Control (QC) Metrics Computation
        4. Identical Multi-Scale k-Mer Transformation (from training config)
        5. ML Classifier Inference & Probability Estimation
        6. Packaging Structured Result with Evidence Labeling & Timestamp
        7. Optional JSON Export (prediction.json)

    Args:
        source: File path (str), FASTA string, or file-like object.
        model_path: Path to classifier.joblib.
        output_json_path: Optional destination path for prediction.json.

    Returns:
        Structured dictionary containing sample ID, sequence length, QC metrics,
        predicted class, model version, prediction probabilities, timestamp, and evidence labels.
    """
    timestamp = datetime.now().isoformat()

    # Step 1: Parse FASTA
    parsed = parse_fasta(source)
    sample_id = parsed.get("sequence_id", "sample_query")
    header = parsed.get("header", sample_id)
    sequence = parsed.get("primary_sequence", "")
    total_length = parsed.get("total_length", len(sequence))

    # Step 2: Validate Sequence
    is_valid, val_issues = validate_sequence(sequence)
    if not is_valid and not sequence:
        raise ValueError(f"Sequence validation failed for '{sample_id}': {val_issues}")

    # Step 3: Run Quality Control (QC) Analysis
    qc_metrics = calculate_qc_metrics(sequence, header=header)

    # Step 4: Load Trained Model & Retrieve Feature Configuration
    model_meta = load_model(model_path)
    pipeline = model_meta["pipeline"]
    classes = model_meta["classes"]
    k_list = model_meta.get("k_list", [3, 4])
    model_version = model_meta.get("model_version", "1.0.0-demo")
    best_model_name = model_meta.get("best_model_name", "Trained Classifier Pipeline")
    is_demo = model_meta.get("is_demo_model", True)

    # Step 5: Extract Identical k-Mer Transformation Vectors
    feat_vec, _ = extract_multi_kmer_vector(sequence, k_list=k_list)
    feat_vec_2d = feat_vec.reshape(1, -1)

    # Step 6: Classifier Inference & Probability Computation
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(feat_vec_2d)[0]
    else:
        pred_label = pipeline.predict(feat_vec_2d)[0]
        probs = np.array([1.0 if c == pred_label else 0.0 for c in classes])

    pred_idx = int(np.argmax(probs))
    predicted_class = classes[pred_idx]
    confidence_score = float(probs[pred_idx])

    # Construct complete probability distribution dictionary
    prediction_probabilities = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(classes, probs)
    }

    sorted_probs = sorted(prediction_probabilities.items(), key=lambda x: x[1], reverse=True)
    top_3_predictions = sorted_probs[:3]

    banner = model_meta.get("demo_banner", DEMO_BANNER if is_demo else "RESEARCH MODEL")
    disclaimer = model_meta.get("disclaimer", RESEARCH_DISCLAIMER if is_demo else "RESEARCH USE ONLY.")

    # Step 7: Build Structured Return Record
    result: Dict[str, Any] = {
        "sample_id": sample_id,
        "sequence_length": total_length,
        "length": total_length,
        "qc_metrics": qc_metrics,
        "predicted_class": predicted_class,
        "model_version": model_version,
        "prediction_probabilities": prediction_probabilities,
        "class_probabilities": prediction_probabilities,
        "confidence_score": confidence_score,
        "confidence_percent": f"{confidence_score * 100.0:.2f}%",
        "top_3_predictions": top_3_predictions,
        "best_model_name": best_model_name,
        "timestamp": timestamp,
        "evidence_label": EVIDENCE_LABEL,
        "evidence_type": "computational_in_silico_evidence",
        "is_computational_evidence": True,
        "is_demo_model": is_demo,
        "demo_banner": banner,
        "disclaimer": disclaimer
    }

    # Step 8: Save to JSON if requested or default
    if output_json_path:
        save_prediction_json(result, output_json_path)

    return result


def predict_pathogen_class(
    sequence: str,
    model_path: str = "models/classifier.joblib",
    output_json_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Direct sequence string interface to the prediction pipeline.
    Wraps sequence in a formatted FASTA string and executes full pipeline.
    """
    fasta_str = f">Query_Sequence\n{sequence}\n"
    return predict_fasta(fasta_str, model_path=model_path, output_json_path=output_json_path)


def get_prediction_dataframe(class_probs: Dict[str, float]) -> pd.DataFrame:
    """Converts class probabilities dictionary into a display-ready Pandas DataFrame."""
    df_data = [
        {"Pathogen Class": cls_name, "Probability": prob, "Probability (%)": f"{prob*100:.2f}%"}
        for cls_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    ]
    return pd.DataFrame(df_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Pathogen Genome Computational Prediction Pipeline")
    parser.add_argument("--fasta", type=str, required=True, help="Path to input FASTA (.fasta / .fa) file")
    parser.add_argument("--model", type=str, default="models/classifier.joblib", help="Path to trained classifier.joblib")
    parser.add_argument("--output", type=str, default="prediction.json", help="Path to save output prediction.json")
    args = parser.parse_args()

    print(f"\nAnalyzing FASTA sequence: {args.fasta}...")
    res = predict_fasta(args.fasta, model_path=args.model, output_json_path=args.output)

    print("\n============================================================")
    print(f"  {res['demo_banner']}")
    print(f"  {res['evidence_label']}")
    print("============================================================")
    print(f"Sample ID          : {res['sample_id']}")
    print(f"Sequence Length    : {res['sequence_length']:,} bp")
    print(f"QC Status          : {res['qc_metrics']['status']} (GC: {res['qc_metrics']['gc_percentage']}%)")
    print(f"Predicted Class    : {res['predicted_class']}")
    print(f"Confidence Score   : {res['confidence_percent']}")
    print(f"Model Architecture : {res.get('best_model_name', 'Classifier')}")
    print(f"Model Version      : {res['model_version']}")
    print(f"Timestamp          : {res['timestamp']}")
    print("\nTop 3 Predictions (Computational Probabilities):")
    for cls_name, prob in res['top_3_predictions']:
        print(f"  - {cls_name:<30}: {prob * 100:.2f}%")
    print(f"\nPrediction saved to: {os.path.abspath(args.output)}")
    print(f"\nSAFETY DISCLAIMER:\n{res['disclaimer']}")
    print("============================================================\n")
