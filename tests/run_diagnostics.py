"""
Complete Application Verification & Diagnostics Test Suite.
Tests:
- valid FASTA
- empty FASTA
- malformed FASTA
- invalid characters
- missing sequence
- feature generation
- model loading
- prediction
- dashboard visualizers & error handling
- full end-to-end pipeline execution
"""

import os
import sys
import io
import json
import traceback
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fasta_reader import parse_fasta, validate_sequence, validate_nucleotide_sequence, FASTAValidationError
from src.qc import calculate_qc_metrics, get_qc_summary_dataframe
from src.kmer_features import (
    generate_kmer_alphabet,
    extract_kmer_counts,
    extract_kmer_frequencies,
    extract_kmer_vector,
    extract_multi_kmer_vector,
    KMerFeatureExtractor,
    get_top_kmers
)
from src.predict import (
    load_model,
    predict_fasta,
    predict_pathogen_class,
    get_prediction_dataframe,
    save_prediction_json
)
from src.reference_analysis import (
    compare_with_references,
    cosine_similarity,
    jaccard_kmer_similarity,
    save_similarity_csv
)
from src.report_generator import generate_pdf_report
from app.app import (
    plot_base_composition_bar,
    plot_gc_indicator,
    plot_genome_length_display,
    plot_model_performance_chart,
    plot_confusion_matrix_heatmap,
    plot_prediction_probability_chart,
    plot_reference_similarity_ranking_chart,
    plot_pca_feature_space
)
from src.database import (
    init_database,
    check_db_status,
    save_analysis,
    get_analysis,
    get_all_analyses,
    save_reference_results,
    get_reference_results,
    save_differences,
    get_differences,
    save_model_run,
    get_model_runs,
    delete_analysis
)


def test_section(title):
    print("\n" + "=" * 70)
    print(f"  RUNNING TEST: {title}")
    print("=" * 70)


def run_all_tests():
    errors = []

    # -------------------------------------------------------------------------
    # TEST 1: Valid FASTA
    # -------------------------------------------------------------------------
    test_section("1. Valid FASTA Parsing")
    try:
        # Standard DNA (length: 30 + 24 = 54 bp)
        valid_dna = ">Sample_DNA_01 Positive control sequence\nATGCGATCGATCGATCGATCGATCGATCGA\nTCGATCGATCGATCGATCGATCGA"
        res_dna = parse_fasta(valid_dna)
        assert res_dna["sequence_id"] == "Sample_DNA_01", f"Expected Sample_DNA_01, got {res_dna['sequence_id']}"
        assert res_dna["total_length"] == 54, f"Expected 54, got {res_dna['total_length']}"
        assert res_dna["num_records"] == 1
        assert "ATGCGATC" in res_dna["primary_sequence"]
        print(f"  [PASS] Single-record valid DNA parsed accurately ({res_dna['total_length']} bp).")

        # Multi-record FASTA
        multi_fasta = ">rec1 First chunk\nATGCATGC\n>rec2 Second chunk\nCGTACGTA"
        res_multi = parse_fasta(multi_fasta)
        assert res_multi["num_records"] == 2
        assert res_multi["total_length"] == 16
        print("  [PASS] Multi-record valid FASTA parsed accurately.")

        # RNA with Uracil and Ambiguous IUPAC (N, R, Y, S, W, K, M, B, D, H, V)
        rna_iupac = ">RNA_Sample\nAUGCUAGCNNNRY"
        res_rna = parse_fasta(rna_iupac, allow_ambiguous=True)
        assert res_rna["total_length"] == 13
        assert "U" in res_rna["primary_sequence"]
        print("  [PASS] RNA with ambiguous IUPAC codes parsed accurately.")

    except Exception as e:
        print(f"  [FAIL] Valid FASTA test failed: {e}")
        traceback.print_exc()
        errors.append(("Valid FASTA", str(e)))

    # -------------------------------------------------------------------------
    # TEST 2: Empty FASTA
    # -------------------------------------------------------------------------
    test_section("2. Empty FASTA Handling")
    try:
        # Empty string
        try:
            parse_fasta("")
            errors.append(("Empty FASTA", "Empty string did not raise FASTAValidationError"))
            print("  [FAIL] Empty string failed to raise FASTAValidationError.")
        except FASTAValidationError as e:
            print(f"  [PASS] Empty string correctly raised FASTAValidationError: {e}")

        # Whitespace-only string
        try:
            parse_fasta("   \n\n\t  ")
            errors.append(("Empty FASTA", "Whitespace-only did not raise FASTAValidationError"))
            print("  [FAIL] Whitespace-only string failed to raise FASTAValidationError.")
        except FASTAValidationError as e:
            print(f"  [PASS] Whitespace-only string correctly raised FASTAValidationError: {e}")

        # Empty BytesIO / StringIO
        try:
            parse_fasta(io.BytesIO(b""))
            errors.append(("Empty FASTA", "Empty BytesIO did not raise FASTAValidationError"))
            print("  [FAIL] Empty BytesIO failed to raise FASTAValidationError.")
        except FASTAValidationError as e:
            print(f"  [PASS] Empty BytesIO correctly raised FASTAValidationError: {e}")

    except Exception as e:
        print(f"  [FAIL] Unexpected error in Empty FASTA tests: {e}")
        errors.append(("Empty FASTA unexpected", str(e)))

    # -------------------------------------------------------------------------
    # TEST 3: Malformed FASTA
    # -------------------------------------------------------------------------
    test_section("3. Malformed FASTA Handling")
    try:
        # No header '>'
        try:
            parse_fasta("ATGCGATCGATCGATCGATCGATCGATCGA")
            errors.append(("Malformed FASTA", "No '>' header did not raise FASTAValidationError"))
            print("  [FAIL] Sequence with no header failed to raise FASTAValidationError.")
        except FASTAValidationError as e:
            print(f"  [PASS] Missing '>' header correctly raised FASTAValidationError: {e}")

        # Random non-FASTA text
        try:
            parse_fasta("This is a plain text file, not a FASTA file.")
            errors.append(("Malformed FASTA", "Plain text did not raise FASTAValidationError"))
            print("  [FAIL] Plain text failed to raise FASTAValidationError.")
        except FASTAValidationError as e:
            print(f"  [PASS] Plain text correctly raised FASTAValidationError: {e}")

    except Exception as e:
        print(f"  [FAIL] Unexpected error in Malformed FASTA tests: {e}")
        errors.append(("Malformed FASTA unexpected", str(e)))

    # -------------------------------------------------------------------------
    # TEST 4: Invalid Characters
    # -------------------------------------------------------------------------
    test_section("4. Invalid Characters Handling")
    try:
        # Numbers and special characters
        invalid_seq = ">Invalid_Char_Sample\nATGC1234!@#$Z"
        try:
            parse_fasta(invalid_seq, validate_nucleotides=True)
            errors.append(("Invalid Characters", "Invalid characters did not raise FASTAValidationError"))
            print("  [FAIL] Invalid non-nucleotide characters failed to raise FASTAValidationError.")
        except FASTAValidationError as e:
            print(f"  [PASS] Invalid characters correctly raised FASTAValidationError: {e}")

        # Validate sequence helper
        is_val, issues = validate_sequence("ATGCZZXX123")
        assert not is_val
        assert any("invalid" in iss.lower() for iss in issues)
        print(f"  [PASS] validate_sequence accurately detected issues: {issues}")

    except Exception as e:
        print(f"  [FAIL] Unexpected error in Invalid Characters tests: {e}")
        errors.append(("Invalid Characters unexpected", str(e)))

    # -------------------------------------------------------------------------
    # TEST 5: Missing Sequence
    # -------------------------------------------------------------------------
    test_section("5. Missing Sequence Handling")
    try:
        # Header with no sequence
        missing_seq = ">Only_Header\n\n"
        try:
            parse_fasta(missing_seq)
            errors.append(("Missing Sequence", "Header with no sequence did not raise FASTAValidationError"))
            print("  [FAIL] Header with no sequence failed to raise FASTAValidationError.")
        except FASTAValidationError as e:
            print(f"  [PASS] Header with missing sequence correctly raised FASTAValidationError: {e}")

        # Multi-record where second record is missing sequence
        missing_seq_multi = ">rec1\nATGCATGC\n>rec2\n"
        try:
            parse_fasta(missing_seq_multi)
            errors.append(("Missing Sequence Multi", "Empty second record did not raise FASTAValidationError"))
            print("  [FAIL] Multi-record with empty record failed to raise FASTAValidationError.")
        except FASTAValidationError as e:
            print(f"  [PASS] Empty second record correctly raised FASTAValidationError: {e}")

    except Exception as e:
        print(f"  [FAIL] Unexpected error in Missing Sequence tests: {e}")
        errors.append(("Missing Sequence unexpected", str(e)))

    # -------------------------------------------------------------------------
    # TEST 6: Feature Generation
    # -------------------------------------------------------------------------
    test_section("6. Feature Generation (k-mers)")
    try:
        # Standard k-mer extraction
        seq = "ATGCGATCGATCGATCGATCGATCGATCGA"
        vec_k3, names_k3 = extract_kmer_vector(seq, k=3)
        assert len(vec_k3) == 64
        assert len(names_k3) == 64
        assert np.isclose(np.sum(vec_k3), 1.0)
        print(f"  [PASS] k=3 vector generated: shape {vec_k3.shape}, sum = {np.sum(vec_k3):.4f}")

        # Multi k-mer vector (k=3, k=4)
        vec_multi, names_multi = extract_multi_kmer_vector(seq, k_list=[3, 4])
        assert len(vec_multi) == 64 + 256  # 320 features
        assert len(names_multi) == 320
        print(f"  [PASS] Multi-kmer vector (k=3, 4) generated: shape {vec_multi.shape}")

        # Short sequence (< k)
        short_seq = "AT"
        vec_short, _ = extract_multi_kmer_vector(short_seq, k_list=[3, 4])
        assert len(vec_short) == 320
        assert np.sum(vec_short) == 0.0
        print("  [PASS] Short sequence (< k) gracefully returns zero vector without error.")

        # Top k-mers DataFrame
        df_top = get_top_kmers(seq, k=3, top_n=5)
        assert isinstance(df_top, pd.DataFrame)
        assert len(df_top) == 5
        print(f"  [PASS] Top k-mers extracted:\n{df_top.to_string(index=False)}")

    except Exception as e:
        print(f"  [FAIL] Feature Generation test failed: {e}")
        traceback.print_exc()
        errors.append(("Feature Generation", str(e)))

    # -------------------------------------------------------------------------
    # TEST 7: Model Loading & Model Artifacts
    # -------------------------------------------------------------------------
    test_section("7. Model Loading & Verification")
    try:
        model_path = os.path.join(PROJECT_ROOT, "models", "classifier.joblib")
        assert os.path.exists(model_path), f"Model file '{model_path}' does not exist!"
        model_dict = load_model(model_path)
        assert "pipeline" in model_dict
        assert "classes" in model_dict
        assert "k_list" in model_dict
        assert len(model_dict["classes"]) >= 4
        print(f"  [PASS] Model loaded successfully: {model_dict.get('best_model_name')} with classes {list(model_dict['classes'])}")

        metrics_path = os.path.join(PROJECT_ROOT, "models", "model_metrics.json")
        assert os.path.exists(metrics_path), f"Metrics file '{metrics_path}' missing!"
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        assert "validation_comparison" in metrics_data
        assert "test_metrics" in metrics_data
        print(f"  [PASS] Model metrics JSON validated (Validation accuracy: {metrics_data.get('test_metrics', {}).get('accuracy', 0)*100:.1f}%)")

    except Exception as e:
        print(f"  [FAIL] Model loading test failed: {e}")
        traceback.print_exc()
        errors.append(("Model Loading", str(e)))

    # -------------------------------------------------------------------------
    # TEST 8: Prediction Pipeline
    # -------------------------------------------------------------------------
    test_section("8. Prediction Pipeline")
    try:
        sample_fasta = os.path.join(PROJECT_ROOT, "data", "raw", "sample_sars_cov_2.fasta")
        assert os.path.exists(sample_fasta), f"Sample FASTA '{sample_fasta}' not found"
        
        pred_res = predict_fasta(sample_fasta, model_path=os.path.join(PROJECT_ROOT, "models", "classifier.joblib"), output_json_path=os.path.join(PROJECT_ROOT, "reports", "test_prediction.json"))
        assert pred_res["predicted_class"] is not None
        assert "confidence_score" in pred_res
        assert "prediction_probabilities" in pred_res
        assert len(pred_res["prediction_probabilities"]) >= 4
        assert pred_res["qc_metrics"]["status"] in ["PASS", "REVIEW"]
        assert os.path.exists(os.path.join(PROJECT_ROOT, "reports", "test_prediction.json"))
        print(f"  [PASS] Prediction completed: Class='{pred_res['predicted_class']}' ({pred_res['confidence_percent']})")

        # Direct sequence prediction
        pred_direct = predict_pathogen_class("ATGAAACTGTTCAGTCAGCTGATCGATCGATCGATCGATCGATCGATCGATCGA", model_path=os.path.join(PROJECT_ROOT, "models", "classifier.joblib"))
        assert pred_direct["predicted_class"] is not None
        print(f"  [PASS] Direct sequence prediction completed: Class='{pred_direct['predicted_class']}'")

    except Exception as e:
        print(f"  [FAIL] Prediction test failed: {e}")
        traceback.print_exc()
        errors.append(("Prediction", str(e)))

    # -------------------------------------------------------------------------
    # TEST 9: Reference Comparison
    # -------------------------------------------------------------------------
    test_section("9. Reference Comparison & Distance Analysis")
    try:
        sample_fasta = os.path.join(PROJECT_ROOT, "data", "raw", "sample_sars_cov_2.fasta")
        parsed = parse_fasta(sample_fasta)
        query_seq = parsed["primary_sequence"]
        
        ref_res = compare_with_references(query_seq, reference_dir=os.path.join(PROJECT_ROOT, "data", "reference"), output_csv_path=os.path.join(PROJECT_ROOT, "reports", "test_similarity.csv"))
        assert ref_res["best_match"] is not None
        assert len(ref_res["rankings"]) > 0
        assert isinstance(ref_res["dataframe"], pd.DataFrame)
        assert os.path.exists(os.path.join(PROJECT_ROOT, "reports", "test_similarity.csv"))
        print(f"  [PASS] Reference comparison succeeded. Top match: {ref_res['best_match']['Reference Name']} (Composite: {ref_res['best_match']['Composite Score (%)']}%)")

    except Exception as e:
        print(f"  [FAIL] Reference comparison test failed: {e}")
        traceback.print_exc()
        errors.append(("Reference Comparison", str(e)))

    # -------------------------------------------------------------------------
    # TEST 10: Visualizations / Dashboard Plotly Charts
    # -------------------------------------------------------------------------
    test_section("10. Dashboard Plotly Visualization Generators")
    try:
        sample_fasta = os.path.join(PROJECT_ROOT, "data", "raw", "sample_sars_cov_2.fasta")
        parsed = parse_fasta(sample_fasta)
        query_seq = parsed["primary_sequence"]
        qc_res = calculate_qc_metrics(query_seq, header="SARS-CoV-2 Sample")
        pred_res = predict_pathogen_class(query_seq, model_path=os.path.join(PROJECT_ROOT, "models", "classifier.joblib"))
        ref_res = compare_with_references(query_seq, reference_dir=os.path.join(PROJECT_ROOT, "data", "reference"))

        # 1. Base Composition Bar Chart
        fig1 = plot_base_composition_bar(qc_res)
        assert fig1 is not None and len(fig1.data) > 0
        print("  [PASS] 1. Base composition bar chart generated.")

        # 2. GC Indicator
        fig2 = plot_gc_indicator(qc_res["gc_percentage"])
        assert fig2 is not None and len(fig2.data) > 0
        print("  [PASS] 2. GC percentage indicator gauge generated.")

        # 3. Genome Length Display
        fig3 = plot_genome_length_display(qc_res["sequence_length"])
        assert fig3 is not None and len(fig3.data) > 0
        print("  [PASS] 3. Genome length bullet display generated.")

        # 4. Model Performance Chart
        fig4 = plot_model_performance_chart()
        assert fig4 is not None and len(fig4.data) > 0
        print("  [PASS] 4. Model performance comparison chart generated.")

        # 5. Confusion Matrix Heatmap
        fig5 = plot_confusion_matrix_heatmap()
        assert fig5 is not None and len(fig5.data) > 0
        print("  [PASS] 5. Multi-class confusion matrix heatmap generated.")

        # 6. Prediction Probability Chart
        fig6 = plot_prediction_probability_chart(pred_res["prediction_probabilities"])
        assert fig6 is not None and len(fig6.data) > 0
        print("  [PASS] 6. Prediction probability ranking chart generated.")

        # 7. Reference Similarity Ranking Chart
        fig7 = plot_reference_similarity_ranking_chart(ref_res["dataframe"])
        assert fig7 is not None and len(fig7.data) > 0
        print("  [PASS] 7. Reference similarity ranking chart generated.")

        # 8. 2D PCA Feature Space Chart
        fig8 = plot_pca_feature_space(query_seq=query_seq, predicted_class=pred_res["predicted_class"])
        assert fig8 is not None and len(fig8.data) > 0
        print("  [PASS] 8. 2D PCA Feature Space chart generated.")

    except Exception as e:
        print(f"  [FAIL] Visualization generation failed: {e}")
        traceback.print_exc()
        errors.append(("Visualizations", str(e)))

    # -------------------------------------------------------------------------
    # TEST 11: Report Generation (PDF with 11 sections & Figures)
    # -------------------------------------------------------------------------
    test_section("11. Report Generation (PDF with 11 sections & Figures)")
    try:
        figures = {
            "base_composition": fig1,
            "gc_indicator": fig2,
            "genome_length": fig3,
            "model_performance": fig4,
            "confusion_matrix": fig5,
            "prediction_probs": fig6,
            "ref_similarity": fig7
        }
        pdf_path = generate_pdf_report(
            qc_results=qc_res,
            pred_results=pred_res,
            ref_results=ref_res,
            output_dir=os.path.join(PROJECT_ROOT, "reports"),
            sample_id="SARS_CoV_2_Diagnostics_Sample",
            figures=figures
        )
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 1000
        print(f"  [PASS] PDF Report successfully generated at: {pdf_path} (Size: {os.path.getsize(pdf_path):,} bytes)")

    except Exception as e:
        print(f"  [FAIL] Report generation test failed: {e}")
        traceback.print_exc()
        errors.append(("Report Generation", str(e)))

    # -------------------------------------------------------------------------
    # TEST 12: Entire End-to-End Pipeline
    # -------------------------------------------------------------------------
    test_section("12. Entire End-to-End Pipeline Execution")
    try:
        print("Executing pipeline: FASTA upload -> QC -> features -> model -> prediction -> reference comparison -> visualization -> report...")
        raw_fasta_text = """>E_Coli_Isolate_K12 K-12 benchmark sequencing run
AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC
TTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAA
TATAGGCATAGCGCACAGACAGATAAAAATTACAGAGTACACAACATCCATGAAACGCATTAGCACCACC
ATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA"""

        # Step 1: FASTA Upload / Parsing
        parsed_data = parse_fasta(raw_fasta_text)
        seq = parsed_data["primary_sequence"]
        header = parsed_data["header"]
        sample_id = parsed_data["sequence_id"]
        print(f"  [Step 1 - FASTA Ingest] Parsed sample '{sample_id}' ({len(seq)} bp)")

        # Step 2: Quality Control (QC)
        qc = calculate_qc_metrics(seq, header=header)
        assert qc["gc_percentage"] > 0
        print(f"  [Step 2 - Quality Control] GC: {qc['gc_percentage']}%, Status: {qc['status']}")

        # Step 3: Feature Extraction (k-mers)
        feat_vec, feat_names = extract_multi_kmer_vector(seq, k_list=[3, 4])
        assert len(feat_vec) == 320
        print(f"  [Step 3 - Feature Generation] Extracted {len(feat_vec)} k-mer frequency features")

        # Step 4 & 5: Model Loading & Prediction
        pred = predict_fasta(raw_fasta_text, model_path=os.path.join(PROJECT_ROOT, "models", "classifier.joblib"), output_json_path=os.path.join(PROJECT_ROOT, "reports", "e2e_prediction.json"))
        assert pred["predicted_class"] is not None
        print(f"  [Step 4 & 5 - Model & Prediction] Predicted Class: {pred['predicted_class']} ({pred['confidence_percent']})")

        # Step 6: Reference Comparison
        ref_comp = compare_with_references(seq, reference_dir=os.path.join(PROJECT_ROOT, "data", "reference"), output_csv_path=os.path.join(PROJECT_ROOT, "reports", "e2e_similarity.csv"))
        assert ref_comp["best_match"] is not None
        print(f"  [Step 6 - Reference Comparison] Closest Reference: {ref_comp['best_match']['Reference Name']} ({ref_comp['best_match']['Composite Score (%)']}%)")

        # Step 7: Visualization
        e2e_figs = {
            "base_composition": plot_base_composition_bar(qc),
            "gc_indicator": plot_gc_indicator(qc["gc_percentage"]),
            "genome_length": plot_genome_length_display(qc["sequence_length"]),
            "model_performance": plot_model_performance_chart(),
            "confusion_matrix": plot_confusion_matrix_heatmap(),
            "prediction_probs": plot_prediction_probability_chart(pred["prediction_probabilities"]),
            "ref_similarity": plot_reference_similarity_ranking_chart(ref_comp["dataframe"])
        }
        for k_fig, fig in e2e_figs.items():
            assert fig is not None
        print(f"  [Step 7 - Visualization] All 7 presentation visualizations constructed successfully")

        # Step 8: PDF Report
        report_file = generate_pdf_report(
            qc_results=qc,
            pred_results=pred,
            ref_results=ref_comp,
            output_dir=os.path.join(PROJECT_ROOT, "reports"),
            sample_id=sample_id,
            figures=e2e_figs
        )
        assert os.path.exists(report_file)
        print(f"  [Step 8 - Report Generation] Comprehensive PDF report generated at: {report_file}")

        print("  [PASS] Full End-to-End Pipeline Completed Successfully!")

    except Exception as e:
        print(f"  [FAIL] End-to-End Pipeline failed: {e}")
        traceback.print_exc()
        errors.append(("End-to-End Pipeline", str(e)))

    # -------------------------------------------------------------------------
    # 13. SQLITE DATABASE PERSISTENCE & AUDIT TRAIL
    # -------------------------------------------------------------------------
    test_section("13. SQLite Database Layer & Audit Trail")
    try:
        db_path = os.path.join(PROJECT_ROOT, "data", "pathogen_intelligence.db")
        init_ok = init_database(db_path)
        assert init_ok is True
        connected, msg = check_db_status(db_path)
        assert connected is True
        print(f"  [PASS] SQLite database verified at: {db_path} (Status: {msg})")

        # Test insert analysis
        diag_analysis_id = "PGI-DIAGNOSTIC-RUN"
        save_ok = save_analysis({
            "analysis_id": diag_analysis_id,
            "sample_id": "Diagnostic_E_Coli_K12",
            "filename": "diagnostic_sample.fasta",
            "sequence_length": 255,
            "gc_percent": 40.78,
            "ambiguous_bases": 0,
            "qc_status": "PASS",
            "prediction": "Escherichia coli",
            "confidence": 99.2,
            "model_name": "Random Forest",
            "model_version": "v1.0"
        }, db_path=db_path)
        assert save_ok is True

        # Test insert reference results & differences
        save_reference_results(diag_analysis_id, [
            {"Reference Name": "E. coli Ref", "Composite Score (%)": 99.5, "k-mer Cosine Sim": 0.995, "Jaccard Sim": 0.98}
        ], db_path=db_path)

        save_differences(diag_analysis_id, [
            {"reference_id": "E. coli Ref", "difference_summary": "Length delta: 0 bp, GC delta: 0.0%"}
        ], db_path=db_path)

        rec = get_analysis(diag_analysis_id, db_path=db_path)
        assert rec is not None
        assert rec["prediction"] == "Escherichia coli"
        assert len(get_reference_results(diag_analysis_id, db_path=db_path)) == 1
        assert len(get_differences(diag_analysis_id, db_path=db_path)) == 1

        print(f"  [PASS] SQLite persistence, retrieval, reference results, and differences verified successfully!")

    except Exception as e:
        print(f"  [FAIL] SQLite database test failed: {e}")
        traceback.print_exc()
        errors.append(("SQLite Database", str(e)))

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TEST SUITE EXECUTION SUMMARY")
    print("=" * 70)
    if not errors:
        print("  >>> ALL 13 TEST SUITE SECTIONS PASSED WITH ZERO ERRORS! <<<")
        return 0
    else:
        print(f"  >>> {len(errors)} TEST FAILURES DETECTED: <<<")
        for section, err in errors:
            print(f"   - {section}: {err}")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
