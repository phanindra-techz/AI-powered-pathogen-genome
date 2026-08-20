"""
PDF Analysis Report Generator Module.
Creates comprehensive computational research PDF reports using ReportLab.

Sections:
  1.  Project Title and Cover
  2.  Sample Information
  3.  Genome Quality Control
  4.  AI Prediction Results
  5.  Model and Evaluation Information
  6.  Reference Comparison
  7.  Observed Differences
  8.  Charts (embedded Plotly figures)
  9.  Data Provenance
  10. Limitations
  11. Safety Disclaimer

Saves to reports/analysis_report.pdf
"""

import os
import io
from datetime import datetime
from typing import Dict, Any, Optional, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.units import inch

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_NAVY      = colors.HexColor("#0f172a")
_BLUE      = colors.HexColor("#1e40af")
_TEAL      = colors.HexColor("#0284c7")
_INDIGO    = colors.HexColor("#4338ca")
_LIGHT_BG  = colors.HexColor("#f8fafc")
_SLATE     = colors.HexColor("#64748b")
_BORDER    = colors.HexColor("#cbd5e1")
_HEADER_BG = colors.HexColor("#e2e8f0")
_ALERT_RED = colors.HexColor("#b91c1c")
_RED_BG    = colors.HexColor("#fef2f2")
_AMBER     = colors.HexColor("#92400e")
_AMBER_BG  = colors.HexColor("#fffbeb")

PAGE_W = letter[0] - 80   # usable width with 40pt margins each side


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "doc_title": ParagraphStyle(
            "DocTitle", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=22, leading=27,
            textColor=_NAVY, alignment=TA_CENTER, spaceAfter=4
        ),
        "doc_subtitle": ParagraphStyle(
            "DocSubtitle", parent=base["Normal"],
            fontName="Helvetica", fontSize=9, leading=12,
            textColor=_SLATE, alignment=TA_CENTER, spaceAfter=2
        ),
        "section_num": ParagraphStyle(
            "SectionNum", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=11, leading=14,
            textColor=_INDIGO, spaceBefore=14, spaceAfter=5
        ),
        "body": ParagraphStyle(
            "Body_Custom", parent=base["Normal"],
            fontName="Helvetica", fontSize=9, leading=12, textColor=_NAVY
        ),
        "body_sm": ParagraphStyle(
            "BodySm", parent=base["Normal"],
            fontName="Helvetica", fontSize=8, leading=11, textColor=_NAVY
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"],
            fontName="Helvetica-Oblique", fontSize=8, leading=11,
            textColor=_SLATE, alignment=TA_CENTER
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=8, leading=11,
            textColor=_ALERT_RED, alignment=TA_CENTER
        ),
        "disclaimer_body": ParagraphStyle(
            "DisclaimerBody", parent=base["Normal"],
            fontName="Helvetica", fontSize=8, leading=11,
            textColor=_ALERT_RED, alignment=TA_JUSTIFY
        ),
        "limitation": ParagraphStyle(
            "LimitationItem", parent=base["Normal"],
            fontName="Helvetica", fontSize=8, leading=12,
            textColor=_NAVY, leftIndent=12, spaceAfter=2
        ),
    }


def _hr(color=_TEAL, thickness=1.2) -> HRFlowable:
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceBefore=2, spaceAfter=6)


def _section_header(title: str, s: Dict) -> List:
    return [
        _hr(color=_INDIGO, thickness=0.8),
        Paragraph(title, s["section_num"]),
    ]


def _kv_table(rows: List, col_widths=None) -> Table:
    if col_widths is None:
        col_widths = [170, PAGE_W - 170]
    t = Table(rows, colWidths=list(col_widths))
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), _LIGHT_BG),
        ("GRID",        (0, 0), (-1, -1), 0.4, _BORDER),
        ("PADDING",     (0, 0), (-1, -1), 5),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _header_table(headers: List, data: List, col_widths: List) -> Table:
    all_rows = [headers] + data
    t = Table(all_rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  _HEADER_BG),
        ("GRID",        (0, 0), (-1, -1), 0.4, _BORDER),
        ("PADDING",     (0, 0), (-1, -1), 4),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
    ]))
    return t


# ---------------------------------------------------------------------------
# Chart embedding
# ---------------------------------------------------------------------------

def _try_embed_figure(fig, width_inch: float, caption: str, s: Dict) -> List:
    """Export a Plotly figure to PNG via kaleido and embed it in the PDF."""
    elements = []
    try:
        import plotly.io as pio
        png_bytes = pio.to_image(
            fig, format="png",
            width=int(width_inch * 96),
            height=int(width_inch * 96 * 0.55),
            scale=2
        )
        img_buf = io.BytesIO(png_bytes)
        img = Image(img_buf, width=width_inch * inch, height=width_inch * inch * 0.55)
        elements.append(img)
        elements.append(Paragraph(caption, s["caption"]))
        elements.append(Spacer(1, 6))
    except Exception:
        elements.append(Paragraph(
            f"[Chart unavailable - install kaleido to embed: {caption}]",
            s["body_sm"]
        ))
        elements.append(Spacer(1, 4))
    return elements


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_pdf_report(
    qc_results: Dict[str, Any],
    pred_results: Dict[str, Any],
    ref_results: Optional[Dict[str, Any]] = None,
    output_dir: str = "reports",
    filename: Optional[str] = None,
    sample_id: str = "sample_query",
    figures: Optional[Dict[str, Any]] = None
) -> str:
    """
    Builds the complete 11-section research report PDF.

    Args:
        qc_results:   Dict from src.qc.calculate_qc_metrics
        pred_results: Dict from src.predict.predict_pathogen_class
        ref_results:  Dict from src.reference_analysis.compare_with_references
        output_dir:   Directory to write the PDF (default: 'reports')
        filename:     Custom filename for the PDF (default: 'analysis_report.pdf')
        sample_id:    Human-readable sample / sequence identifier
        figures:      Optional dict of named Plotly Figure objects to embed.
                      Keys: 'base_composition', 'gc_indicator', 'genome_length',
                            'model_performance', 'confusion_matrix',
                            'prediction_probs', 'ref_similarity'

    Returns:
        Absolute path to generated PDF report
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_filename = filename if filename else "analysis_report.pdf"
    if not pdf_filename.lower().endswith(".pdf"):
        pdf_filename += ".pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)
    figures = figures or {}

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40,   bottomMargin=40
    )

    s  = _build_styles()
    el = []
    now_str = datetime.now().strftime("%B %d, %Y  %H:%M:%S")

    # Convenience references to common data fields
    header_val = qc_results.get("header", "N/A")
    seq_len    = qc_results.get("sequence_length", qc_results.get("length", 0))
    qc_status  = qc_results.get("status", qc_results.get("qc_status", "N/A"))
    gc_pct     = qc_results.get("gc_percentage", 0.0)
    ambiguous  = qc_results.get("ambiguous_base_count", 0)
    qc_warnings= qc_results.get("warnings", [])
    base_counts= qc_results.get("base_counts", {})
    base_pcts  = qc_results.get("base_percentages", {})

    pred_class   = pred_results.get("predicted_class", "N/A")
    confidence   = pred_results.get("confidence_percent", "N/A")
    model_name   = pred_results.get("best_model_name", "N/A")
    demo_banner  = pred_results.get("demo_banner", "DEMONSTRATION MODEL - NOT CLINICALLY VALIDATED")
    evidence_lbl = pred_results.get("evidence_label", "COMPUTATIONAL EVIDENCE - NOT CLINICAL DIAGNOSIS")
    class_probs  = pred_results.get("prediction_probabilities",
                   pred_results.get("class_probabilities", {}))
    val_comparison = pred_results.get("validation_comparison", {})
    test_metrics   = pred_results.get("test_metrics", {})
    split_sizes    = pred_results.get("split_sizes", {})
    k_list         = pred_results.get("k_list", [3, 4])
    num_features   = pred_results.get("num_features", "N/A")
    dataset_meta   = pred_results.get("dataset_metadata", {})

    # =========================================================================
    # SECTION 1 — Project Title and Cover
    # =========================================================================
    el.append(Spacer(1, 16))
    el.append(Paragraph("AI-Powered Pathogen Genome Intelligence", s["doc_title"]))
    el.append(Paragraph("Computational Bioinformatics Research Report", s["doc_subtitle"]))
    el.append(Paragraph(f"Generated: {now_str}", s["doc_subtitle"]))
    el.append(Spacer(1, 10))
    el.append(_hr(color=_TEAL, thickness=2.0))
    el.append(Spacer(1, 6))

    # Cover disclaimer banner
    cover_disclaimer = (
        "<b>WARNING: DEMONSTRATION MODEL - NOT CLINICALLY VALIDATED</b><br/>"
        "FOR COMPUTATIONAL RESEARCH AND SOFTWARE VALIDATION USE ONLY.<br/>"
        "This system operates on demonstration sequence profiles. Outputs do NOT constitute "
        "verified pathogen identification, clinical diagnosis, or medical evaluation. "
        "Sequence synthesis and design functions are strictly excluded."
    )
    cover_disc_table = Table([[Paragraph(cover_disclaimer, s["disclaimer"])]], colWidths=[PAGE_W])
    cover_disc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _RED_BG),
        ("BOX",        (0, 0), (-1, -1), 1.2, _ALERT_RED),
        ("PADDING",    (0, 0), (-1, -1), 8),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
    ]))
    el.append(cover_disc_table)
    el.append(Spacer(1, 12))

    # =========================================================================
    # SECTION 2 — Sample Information
    # =========================================================================
    el += _section_header("2.  Sample Information", s)
    sample_rows = [
        [Paragraph("<b>Sample ID:</b>",          s["body"]), Paragraph(str(sample_id), s["body"])],
        [Paragraph("<b>FASTA Header:</b>",        s["body"]), Paragraph(str(header_val), s["body"])],
        [Paragraph("<b>Sequence Length:</b>",     s["body"]), Paragraph(f"{seq_len:,} bp", s["body"])],
        [Paragraph("<b>QC Status:</b>",           s["body"]), Paragraph(f"<b>{qc_status}</b>", s["body"])],
        [Paragraph("<b>Report Timestamp:</b>",    s["body"]), Paragraph(now_str, s["body"])],
        [Paragraph("<b>Analysis System:</b>",     s["body"]), Paragraph(
            "AI-Powered Pathogen Genome Intelligence v1.0 - Demonstration Prototype", s["body"])],
    ]
    el.append(_kv_table(sample_rows))
    el.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 3 — Genome Quality Control
    # =========================================================================
    el += _section_header("3.  Genome Quality Control", s)
    el.append(Paragraph(
        f"Genome length: <b>{seq_len:,} bp</b> | GC content: <b>{gc_pct:.2f}%</b> | "
        f"Ambiguous bases: <b>{ambiguous}</b> | Status: <b>{qc_status}</b>",
        s["body"]
    ))
    el.append(Spacer(1, 6))

    qc_headers = [
        Paragraph("<b>Base</b>",             s["body"]),
        Paragraph("<b>Count (bp)</b>",       s["body"]),
        Paragraph("<b>Composition (%)</b>",  s["body"]),
    ]
    qc_data = []
    for base in ["A", "T/U", "G", "C", "N", "Other Ambiguous"]:
        qc_data.append([
            Paragraph(base, s["body"]),
            Paragraph(f"{base_counts.get(base, 0):,}", s["body"]),
            Paragraph(f"{base_pcts.get(base, 0.0):.2f}%", s["body"]),
        ])
    el.append(_header_table(qc_headers, qc_data, [PAGE_W * r for r in (0.25, 0.38, 0.37)]))
    el.append(Spacer(1, 6))

    if qc_warnings:
        el.append(Paragraph("<b>QC Warnings / Flags:</b>", s["body"]))
        for w in qc_warnings:
            el.append(Paragraph(f"  - {w}", s["limitation"]))
    el.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 4 — AI Prediction Results
    # =========================================================================
    el += _section_header("4.  AI Prediction Results", s)
    pred_rows = [
        [Paragraph("<b>Predicted Pathogen Class:</b>",  s["body"]), Paragraph(f"<b>{pred_class}</b>", s["body"])],
        [Paragraph("<b>Confidence Score:</b>",          s["body"]), Paragraph(str(confidence), s["body"])],
        [Paragraph("<b>Classification Model:</b>",      s["body"]), Paragraph(str(model_name), s["body"])],
        [Paragraph("<b>Evidence Label:</b>",            s["body"]), Paragraph(str(evidence_lbl), s["body"])],
        [Paragraph("<b>Model Status:</b>",              s["body"]), Paragraph(f"<b>{demo_banner}</b>", s["body"])],
    ]
    el.append(_kv_table(pred_rows))
    el.append(Spacer(1, 6))

    if class_probs:
        el.append(Paragraph("<b>Full Class Probability Distribution:</b>", s["body"]))
        el.append(Spacer(1, 3))
        prob_headers = [
            Paragraph("<b>Pathogen Class</b>",      s["body"]),
            Paragraph("<b>Probability (%)</b>",     s["body"]),
        ]
        prob_data = [
            [Paragraph(cls, s["body"]), Paragraph(f"{prob * 100:.2f}%", s["body"])]
            for cls, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        ]
        el.append(_header_table(prob_headers, prob_data, [PAGE_W * 0.72, PAGE_W * 0.28]))
    el.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 5 — Model and Evaluation Information
    # =========================================================================
    el += _section_header("5.  Model and Evaluation Information", s)
    is_synthetic = dataset_meta.get("is_synthetic", True)
    model_info_rows = [
        [Paragraph("<b>Best Selected Model:</b>",  s["body"]), Paragraph(str(model_name), s["body"])],
        [Paragraph("<b>k-mer Scales Used:</b>",    s["body"]), Paragraph(str(k_list), s["body"])],
        [Paragraph("<b>Feature Vector Size:</b>",  s["body"]), Paragraph(str(num_features), s["body"])],
        [Paragraph("<b>Train Samples:</b>",        s["body"]), Paragraph(str(split_sizes.get("train", "N/A")), s["body"])],
        [Paragraph("<b>Validation Samples:</b>",   s["body"]), Paragraph(str(split_sizes.get("validation", "N/A")), s["body"])],
        [Paragraph("<b>Test Samples:</b>",         s["body"]), Paragraph(str(split_sizes.get("test", "N/A")), s["body"])],
        [Paragraph("<b>Dataset Source:</b>",       s["body"]), Paragraph(
            "Synthetic Demonstration Profiles" if is_synthetic
            else dataset_meta.get("source", "Verified External Dataset"), s["body"])],
    ]
    el.append(_kv_table(model_info_rows))
    el.append(Spacer(1, 6))

    if test_metrics:
        el.append(Paragraph("<b>Final Held-Out Test Set Metrics (Best Model):</b>", s["body"]))
        el.append(Spacer(1, 3))
        metric_data = [
            [Paragraph("Accuracy",  s["body"]), Paragraph(f"{test_metrics.get('accuracy',  0)*100:.2f}%", s["body"])],
            [Paragraph("Precision", s["body"]), Paragraph(f"{test_metrics.get('precision', 0)*100:.2f}%", s["body"])],
            [Paragraph("Recall",    s["body"]), Paragraph(f"{test_metrics.get('recall',    0)*100:.2f}%", s["body"])],
            [Paragraph("F1-Score",  s["body"]), Paragraph(f"{test_metrics.get('f1_score',  0)*100:.2f}%", s["body"])],
        ]
        metric_headers = [Paragraph("<b>Metric</b>", s["body"]), Paragraph("<b>Score</b>", s["body"])]
        el.append(_header_table(metric_headers, metric_data, [PAGE_W * 0.55, PAGE_W * 0.45]))
    el.append(Spacer(1, 6))

    if val_comparison:
        el.append(Paragraph("<b>Candidate Model Validation Comparison:</b>", s["body"]))
        el.append(Spacer(1, 3))
        cmp_headers = [
            Paragraph("<b>Model</b>",       s["body_sm"]),
            Paragraph("<b>Val Acc</b>",     s["body_sm"]),
            Paragraph("<b>Val Prec</b>",    s["body_sm"]),
            Paragraph("<b>Val Recall</b>",  s["body_sm"]),
            Paragraph("<b>Val F1</b>",      s["body_sm"]),
        ]
        cmp_data = []
        for m_name, m_metrics in val_comparison.items():
            marker = " (selected)" if m_name == model_name else ""
            cmp_data.append([
                Paragraph(f"<b>{m_name}{marker}</b>" if marker else m_name, s["body_sm"]),
                Paragraph(f"{m_metrics.get('accuracy',  0)*100:.1f}%", s["body_sm"]),
                Paragraph(f"{m_metrics.get('precision', 0)*100:.1f}%", s["body_sm"]),
                Paragraph(f"{m_metrics.get('recall',    0)*100:.1f}%", s["body_sm"]),
                Paragraph(f"{m_metrics.get('f1_score',  0)*100:.1f}%", s["body_sm"]),
            ])
        el.append(_header_table(cmp_headers, cmp_data, [PAGE_W * r for r in (0.34, 0.165, 0.165, 0.165, 0.165)]))
    el.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 6 — Reference Comparison
    # =========================================================================
    el += _section_header("6.  Reference Sequence Comparison", s)
    ref_notice = (
        "<b>Reference Comparison Notice:</b> Similarity scores are statistical computational "
        "estimates based on k-mer and Jaccard metrics on demonstration sequences. "
        "They do NOT represent verified biological alignment or clinical identification."
    )
    ref_notice_table = Table([[Paragraph(ref_notice, s["disclaimer_body"])]], colWidths=[PAGE_W])
    ref_notice_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _AMBER_BG),
        ("BOX",        (0, 0), (-1, -1), 0.8, _AMBER),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    el.append(ref_notice_table)
    el.append(Spacer(1, 6))

    if ref_results and ref_results.get("rankings"):
        rankings = ref_results["rankings"]
        ref_headers = [
            Paragraph("<b>Rank</b>",            s["body_sm"]),
            Paragraph("<b>Reference Name</b>",  s["body_sm"]),
            Paragraph("<b>Length (bp)</b>",     s["body_sm"]),
            Paragraph("<b>Cosine Sim</b>",      s["body_sm"]),
            Paragraph("<b>Jaccard Sim</b>",     s["body_sm"]),
            Paragraph("<b>GC Delta (%)</b>",    s["body_sm"]),
            Paragraph("<b>Composite (%)</b>",   s["body_sm"]),
        ]
        ref_data = []
        for r in rankings:
            ref_data.append([
                Paragraph(str(r.get("Rank", "-")),                           s["body_sm"]),
                Paragraph(str(r.get("Reference Name", "N/A")),               s["body_sm"]),
                Paragraph(f"{r.get('Reference Length (bp)', 0):,}",          s["body_sm"]),
                Paragraph(f"{r.get('k-mer Cosine Sim', 0):.4f}",             s["body_sm"]),
                Paragraph(f"{r.get('Jaccard Sim', 0):.4f}",                  s["body_sm"]),
                Paragraph(f"{r.get('GC Delta (%)', 0):.2f}%",                s["body_sm"]),
                Paragraph(f"<b>{r.get('Composite Score (%)', 0):.2f}%</b>",  s["body_sm"]),
            ])
        el.append(_header_table(ref_headers, ref_data, [PAGE_W * r for r in (0.07, 0.25, 0.13, 0.12, 0.12, 0.11, 0.20)]))
    else:
        el.append(Paragraph("No reference comparison data available.", s["body"]))
    el.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 7 — Observed Differences vs Top Reference
    # =========================================================================
    el += _section_header("7.  Observed Differences vs Top Reference", s)

    if ref_results and ref_results.get("rankings"):
        top_ref   = ref_results["rankings"][0]
        ref_len   = top_ref.get("Reference Length (bp)", 0)
        ref_gc_raw   = top_ref.get("Reference GC (%)", 0.0)
        ref_gc_float = float(str(ref_gc_raw).rstrip("%")) if ref_gc_raw else 0.0
        len_diff  = seq_len - ref_len
        gc_diff   = top_ref.get("GC Delta (%)", gc_pct - ref_gc_float)
        cosine_s  = top_ref.get("k-mer Cosine Sim", 0.0)
        jaccard_s = top_ref.get("Jaccard Sim", 0.0)
        composite = top_ref.get("Composite Score (%)", 0.0)
        sign_len  = "+" if len_diff >= 0 else ""
        sign_gc   = "+" if gc_diff  >= 0 else ""

        diff_rows = [
            [Paragraph("<b>Top Reference:</b>",         s["body"]),
             Paragraph(str(top_ref.get("Reference Name", "N/A")), s["body"])],
            [Paragraph("<b>Reference ID:</b>",          s["body"]),
             Paragraph(str(top_ref.get("Reference ID", "N/A")), s["body"])],
            [Paragraph("<b>Length Delta (bp):</b>",     s["body"]),
             Paragraph(f"{sign_len}{len_diff:,} bp  (Query: {seq_len:,} | Reference: {ref_len:,})", s["body"])],
            [Paragraph("<b>GC Content Delta (%):</b>",  s["body"]),
             Paragraph(f"{sign_gc}{gc_diff:.2f}%  (Query: {gc_pct:.2f}% | Reference: {ref_gc_raw})", s["body"])],
            [Paragraph("<b>k-mer Cosine Similarity:</b>", s["body"]),
             Paragraph(f"{cosine_s:.4f}", s["body"])],
            [Paragraph("<b>Jaccard Similarity:</b>",    s["body"]),
             Paragraph(f"{jaccard_s:.4f}", s["body"])],
            [Paragraph("<b>Composite Match Score:</b>", s["body"]),
             Paragraph(f"<b>{composite:.2f}%</b>", s["body"])],
        ]
        el.append(_kv_table(diff_rows))
    else:
        el.append(Paragraph("No reference data available to compute differences.", s["body"]))
    el.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 8 — Charts
    # =========================================================================
    el.append(PageBreak())
    el += _section_header("8.  Analytical Charts", s)
    el.append(Paragraph(
        "Interactive charts from the Streamlit dashboard, embedded as static PNG images "
        "(2x resolution). Requires the kaleido package to render.",
        s["body"]
    ))
    el.append(Spacer(1, 6))

    chart_defs = [
        ("base_composition",  "Chart 1 - Base Composition (nucleotide counts and percentages)"),
        ("gc_indicator",      "Chart 2 - GC Percentage Indicator (biological plausibility gauge)"),
        ("genome_length",     "Chart 3 - Genome Length Display (length bullet metric)"),
        ("model_performance", "Chart 4 - Candidate Model Performance Comparison"),
        ("confusion_matrix",  "Chart 5 - Multi-Class Confusion Matrix Heatmap"),
        ("prediction_probs",  "Chart 6 - Prediction Probability Ranking"),
        ("ref_similarity",    "Chart 7 - Reference Similarity Ranking"),
    ]
    for fig_key, caption in chart_defs:
        fig = figures.get(fig_key)
        if fig is not None:
            el += _try_embed_figure(fig, width_inch=6.5, caption=caption, s=s)
        else:
            el.append(Paragraph(f"[{caption} - not provided to report generator]", s["body_sm"]))
            el.append(Spacer(1, 4))
    el.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 9 — Data Provenance
    # =========================================================================
    el += _section_header("9.  Data Provenance", s)
    prov_rows = [
        [Paragraph("<b>Dataset Type:</b>",       s["body"]),
         Paragraph(
             "Synthetic Demonstration Profiles (auto-generated for software validation)"
             if is_synthetic else "Verified External Dataset (user-supplied)", s["body"])],
        [Paragraph("<b>Dataset Source:</b>",     s["body"]),
         Paragraph(str(dataset_meta.get("source", "Internal synthetic generator - src/dataset_loader.py")), s["body"])],
        [Paragraph("<b>Pathogen Classes:</b>",   s["body"]),
         Paragraph(", ".join(pred_results.get("classes", [])) or "N/A", s["body"])],
        [Paragraph("<b>k-mer Config:</b>",       s["body"]),
         Paragraph(f"k = {k_list}  ->  {num_features} features", s["body"])],
        [Paragraph("<b>Feature Extractor:</b>",  s["body"]),
         Paragraph("src/kmer_features.py - normalized overlapping k-mer frequency vectors", s["body"])],
        [Paragraph("<b>QC Module:</b>",          s["body"]),
         Paragraph("src/qc.py - GC%, base counts, ambiguous detection, PASS/REVIEW rules", s["body"])],
        [Paragraph("<b>Reference Dir:</b>",      s["body"]),
         Paragraph(str(ref_results.get("reference_dir", "data/reference")) if ref_results else "N/A", s["body"])],
        [Paragraph("<b>Report Generator:</b>",   s["body"]),
         Paragraph("src/report_generator.py - ReportLab PDF engine", s["body"])],
        [Paragraph("<b>Report Timestamp:</b>",   s["body"]),
         Paragraph(now_str, s["body"])],
    ]
    el.append(_kv_table(prov_rows))
    el.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 10 — Limitations
    # =========================================================================
    el += _section_header("10. Limitations", s)
    limitations = [
        "Designed, trained, and tested exclusively on synthetic demonstration sequence profiles. "
        "<b>Not validated on real pathogen genomes.</b>",
        "k-mer frequency features capture statistical sequence composition but do <b>not perform "
        "true biological sequence alignment</b> or phylogenetic analysis.",
        "Trained on a small demonstration dataset with limited class diversity. "
        "Generalisation to novel or rare pathogens is <b>not guaranteed</b>.",
        "Reference similarity scores are based on k-mer cosine similarity and Jaccard overlap, "
        "<b>not BLAST or pairwise nucleotide alignment</b>. Scores are not equivalent to sequence identity.",
        "GC% and base composition thresholds for QC PASS/REVIEW decisions are generic heuristics "
        "and <b>may not apply to all organism types</b>.",
        "Does not provide mutation analysis, variant calling, insertion/deletion detection, "
        "or phylogenetic tree construction.",
        "Performance metrics in Section 5 reflect demonstration-mode accuracy on synthetic data only "
        "and <b>must not be extrapolated to real-world performance</b>.",
        "Chart embedding in the PDF requires the <b>kaleido</b> Python package. "
        "Install with: pip install kaleido",
    ]
    for lim in limitations:
        el.append(Paragraph(f"* {lim}", s["limitation"]))
    el.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 11 — Safety Disclaimer
    # =========================================================================
    el.append(PageBreak())
    el += _section_header("11. Safety and Regulatory Disclaimer", s)

    disclaimer_full = (
        "<b>WARNING: DEMONSTRATION MODEL - NOT CLINICALLY VALIDATED - NOT FOR MEDICAL USE</b><br/><br/>"
        "This software is a <b>computational research prototype</b> intended solely for "
        "academic research, software engineering validation, and bioinformatics pipeline "
        "demonstration purposes.<br/><br/>"
        "<b>NO CLINICAL USE:</b> Outputs produced by this system - including but not limited to "
        "pathogen classification predictions, probability scores, reference similarity rankings, "
        "and quality control metrics - do NOT constitute clinical diagnosis, medical evaluation, "
        "epidemiological surveillance findings, or therapeutic guidance of any kind.<br/><br/>"
        "<b>NO REAL PATHOGEN IDENTIFICATION:</b> All prediction results are statistical "
        "computational estimates derived from synthetic demonstration data. They do not represent "
        "verified biological identification of pathogens, organisms, or genetic material.<br/><br/>"
        "<b>NO SEQUENCE SYNTHESIS:</b> This system does not include, enable, or support the "
        "design, optimisation, or synthesis of pathogen sequences, virulence factors, or any "
        "biologically active genetic material.<br/><br/>"
        "<b>REGULATORY STATUS:</b> This software has not been reviewed, cleared, or approved by "
        "any regulatory authority (including but not limited to the FDA, CE/IVDR, or equivalent "
        "bodies) for diagnostic or clinical use.<br/><br/>"
        "Any use of this software for clinical, diagnostic, therapeutic, or public-health "
        "decision-making is strictly prohibited and is the sole responsibility of the user.<br/><br/>"
        "By using this report you acknowledge that it is generated by a DEMONSTRATION PROTOTYPE "
        "and that all outputs are FOR RESEARCH PURPOSES ONLY."
    )
    safety_table = Table([[Paragraph(disclaimer_full, s["disclaimer_body"])]], colWidths=[PAGE_W])
    safety_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _RED_BG),
        ("BOX",        (0, 0), (-1, -1), 1.5, _ALERT_RED),
        ("PADDING",    (0, 0), (-1, -1), 10),
    ]))
    el.append(safety_table)
    el.append(Spacer(1, 12))

    # Footer
    el.append(_hr(color=_SLATE, thickness=0.5))
    el.append(Paragraph(
        f"AI-Powered Pathogen Genome Intelligence  |  Demonstration Prototype  |  {now_str}  |  NOT FOR CLINICAL USE",
        s["doc_subtitle"]
    ))

    # Build PDF
    doc.build(el)
    return os.path.abspath(pdf_path)
