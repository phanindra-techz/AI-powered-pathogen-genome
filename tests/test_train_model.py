"""
Unit tests for src.train_model module.
Tests multi-model training (Logistic Regression, Random Forest, SVM),
Train/Val/Test splits, metric computation (Acc, Precision, Recall, F1, Confusion Matrix),
validation-based model selection, and joblib/JSON persistence.
"""

import os
import json
import joblib
import pytest
from src.train_model import (
    generate_training_dataset,
    build_and_train_model,
    PATHOGEN_PROFILES
)


def test_generate_training_dataset():
    """Test generating training dataset with multi-scale k-mers."""
    X, y, feature_names = generate_training_dataset(samples_per_class=5, k_list=[3, 4], seed=42)
    expected_samples = 5 * len(PATHOGEN_PROFILES)
    expected_features = 64 + 256  # 320 features

    assert X.shape == (expected_samples, expected_features)
    assert len(y) == expected_samples
    assert len(feature_names) == expected_features
    assert set(y) == set(PATHOGEN_PROFILES.keys())


def test_build_and_train_model(tmp_path):
    """Test full multi-model training, comparison, validation selection, and artifact saving."""
    models_dir = str(tmp_path / "models")
    meta = build_and_train_model(
        models_dir=models_dir,
        samples_per_class=10,
        k_list=[3, 4],
        random_state=42
    )

    # Verify return metadata structure
    assert "pipeline" in meta
    assert "best_model_name" in meta
    assert meta["best_model_name"] in ["Logistic Regression", "Random Forest", "Support Vector Machine"]
    assert "validation_comparison" in meta
    assert "test_metrics" in meta
    assert "split_sizes" in meta

    # Verify all 3 models evaluated on validation set
    val_comp = meta["validation_comparison"]
    for model_name in ["Logistic Regression", "Random Forest", "Support Vector Machine"]:
        assert model_name in val_comp
        metrics = val_comp[model_name]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0

    # Verify test set metrics
    test_met = meta["test_metrics"]
    assert "accuracy" in test_met
    assert "f1_score" in test_met
    assert "confusion_matrix" in test_met

    # Verify saved joblib model artifact
    joblib_file = os.path.join(models_dir, "classifier.joblib")
    assert os.path.exists(joblib_file)
    loaded_joblib = joblib.load(joblib_file)
    assert "pipeline" in loaded_joblib
    assert hasattr(loaded_joblib["pipeline"], "predict_proba")

    # Verify saved JSON metrics
    json_file = os.path.join(models_dir, "model_metrics.json")
    assert os.path.exists(json_file)
    with open(json_file, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["best_model_name"] == meta["best_model_name"]
    assert "validation_comparison" in loaded_json
    assert "test_metrics" in loaded_json
