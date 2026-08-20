# 🧬 AI-Powered Pathogen Genome Intelligence

An analytical computational bioinformatics system for genomic sequence Quality Control (QC), multi-scale k-mer signature profiling, supervised machine learning pathogen classification, reference similarity matching, pluggable dataset integration, and automated PDF reporting.

---

## ⚠️ DEMONSTRATION MODEL — NOT CLINICALLY VALIDATED

> **Important Research & Safety Boundary:**  
> - **Demonstration Software Only:** This prototype uses clearly labeled synthetic/demonstration sequence profiles to validate the machine learning and bioinformatics pipeline.  
> - **No Real Pathogen Identification:** Predictions produced during demonstration mode do **NOT** constitute real biological pathogen identification.  
> - **Strictly Non-Clinical:** All outputs represent statistical computational research estimates and are **NOT FOR CLINICAL DIAGNOSIS, THERAPEUTIC DECISION-MAKING, OR MEDICAL USE**.  
> - **No Sequence Synthesis:** This system performs analysis on existing sequences and **cannot design, optimize, or synthesize** pathogens.

---

## 🔌 Pluggable Dataset Architecture

The software is decoupled so that verified, properly sourced labeled datasets can be plugged in at any time without modifying core application code:

### 1. NCBI Datasets Package Ingestion (e.g. Monkeypox)
Download directly from NCBI and pass the unzipped package directory:
```bash
datasets download virus genome taxon monkeypox --filename monkeypox.zip
unzip monkeypox.zip -d monkeypox
```
The loader automatically parses `data_report.jsonl`, `dataset_catalog.json`, and `genomic.fna`:
```bash
python -m src.train_model --dataset-dir monkeypox/
```

### 2. Directory-Based Labeled Dataset
Organize your verified FASTA files into class folders:
```text
my_verified_dataset/
├── Influenza_A/
│   ├── isolate_001.fasta
│   └── isolate_002.fa
├── SARS_CoV_2/
│   └── genome_001.fasta
└── Escherichia_coli/
    └── strain_k12.fasta
```
Train the model with one command:
```bash
python -m src.train_model --dataset-dir my_verified_dataset/
```

### 3. Metadata CSV Ingestion
Create a metadata CSV/TSV table mapping file paths to verified labels:
```csv
filepath,label
data/verified/flu_sample.fasta,Influenza A virus
data/verified/sars_sample.fasta,SARS-CoV-2
```
Train the model:
```bash
python -m src.train_model --metadata-csv data/verified/metadata.csv
```

---

## 🌟 Key Features

1. **Pluggable Dataset Loader (`src/dataset_loader.py`)**
   - Seamlessly ingests external directory structures or metadata CSV files.
   - Falls back to clearly-labeled synthetic demonstration data for pipeline testing.

2. **Robust FASTA Ingestion & Validation (`src/fasta_reader.py`)**
   - Biopython-powered parsing with uppercase standardization and IUPAC nucleotide validation.
   - Comprehensive edge-case handling (empty files, malformed headers, missing sequences).

3. **Bioinformatics Quality Control (`src/qc.py`)**
   - Computes sequence length, GC%, individual base percentages (A, T, G, C), and ambiguous counts.
   - Transparent rules-based `PASS` / `REVIEW` quality evaluation with JSON persistence.

4. **Multi-Scale k-Mer Profiling (`src/kmer_features.py`)**
   - Normalized overlapping k-mer frequency vectors ($k=[3, 4] \implies 320$ features).
   - Guarantees deterministic, immutable feature ordering across training and inference.

5. **Supervised ML Model Selection (`src/train_model.py`)**
   - Evaluates Logistic Regression, Random Forest, and SVM across Train (60%), Validation (20%), and Test (20%) splits.
   - Computes Accuracy, Precision, Recall, F1-score, and confusion matrices.
   - Model selection performed strictly on the validation set.

6. **Interactive Streamlit Web Dashboard (`app/app.py`)**
   - Dark glassmorphic interface with interactive Plotly visual charts.
   - Displays prominent demonstration warnings, sequence overviews, and pluggable dataset instructions.

7. **Automated PDF Report Generation (`src/report_generator.py`)**
   - Exports professional PDF research summaries using ReportLab with disclaimers and QC tables.

---

## 📁 Project Structure

```
AI-powered-pathogen-genome/
├── app/
│   └── app.py                      # Streamlit interactive web dashboard
├── data/
│   ├── raw/                        # Demo query samples (.fasta)
│   ├── reference/                  # Reference pathogen genome files (.fasta)
│   └── processed/                  # Processed feature caches
├── models/
│   ├── classifier.joblib           # Trained scikit-learn ML pipeline & metadata
│   ├── model_metrics.json          # Multi-model comparison metrics & confusion matrices
│   └── kmer_config.json            # k-mer feature extractor schema
├── reports/                        # Exported PDF analysis reports
├── src/
│   ├── __init__.py
│   ├── dataset_loader.py           # Pluggable dataset loader (dir/CSV) & synthetic demo generator
│   ├── fasta_reader.py             # Biopython FASTA parsing & validation
│   ├── kmer_features.py            # k-mer counting, normalization & feature vectors
│   ├── predict.py                  # ML inference & probability distribution
│   ├── qc.py                       # Quality control & base metrics (PASS/REVIEW)
│   ├── reference_analysis.py       # Reference similarity & distance scoring
│   ├── report_generator.py         # ReportLab PDF report generation
│   └── train_model.py              # Model training, multi-model evaluation & selection
├── tests/
│   ├── test_dataset_loader.py      # Unit tests for pluggable dataset loading
│   ├── test_fasta_reader.py        # Unit tests for FASTA reader
│   ├── test_kmer.py                # Unit tests for k-mer extractor & predict
│   ├── test_qc.py                  # Unit tests for QC module
│   ├── test_reference_and_report.py# Unit tests for reference comparison & PDF report
│   └── test_train_model.py         # Unit tests for model training & evaluation
└── requirements.txt                # Project dependencies
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
pytest -v
```

### 3. Run Command-Line Prediction (Demo)
```bash
python -m src.predict --fasta data/raw/sample_sars_cov_2.fasta
```

### 4. Launch Interactive Web Dashboard
```bash
streamlit run app/app.py
```
