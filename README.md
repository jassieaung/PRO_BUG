# PRO_BUG -- Prompt-Based Synthetic Augmentation

Minimal experimental pipeline for prompt-only synthetic data generation
(PRO_BUG).\
Configuration-driven and designed for reproducible experiments.

------------------------------------------------------------------------

## 📁 Structure

    PRO_BUG/
    ├── src/        # Core scripts
    ├── configs/    # YAML experiment configs
    ├── data/       # Download separately (Zenodo)
    └── README.md

------------------------------------------------------------------------

## 🚀 Quick Start

### 1) Clone

    git clone https://github.com/jassieaung/PRO_BUG.git
    cd PRO_BUG

### 2) Install

    python -m venv .venv
    .venv\Scripts\activate   # Windows
    pip install -U pip
    pip install torch transformers datasets pyyaml scikit-learn

### 3) Generate Synthetic Data (PRO_BUG)

    python src/02_generate_aug.py   --config configs/mozilla.yaml   --workdir workdir   --mode m2

### 4) Train & Evaluate

    python src/03_train_eval.py   --config configs/mozilla.yaml   --workdir workdir   --mode m2

------------------------------------------------------------------------

## 📊 Outputs

Saved under:

    workdir/outputs/<dataset>/

Includes: - Synthetic JSONL files\
- Logs\
- Metrics JSON

Example:

    workdir/outputs/mozilla/metrics/m2_r10_ALL.json

------------------------------------------------------------------------

## 📦 Dataset Availability

The dataset used in this study is hosted externally on Zenodo due to GitHub file size limitations.

🔗 Download Link:
https://zenodo.org/records/18737822
------------------------------------------------------------------------

## 📜 License

For academic and research use.
