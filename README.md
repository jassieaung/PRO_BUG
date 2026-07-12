# TriageTailor -- Adaptive Synthetic Augmentation  for Long-Tailed Bug Triage

Minimal experimental pipeline for prompt-only synthetic data generation
(PRO_BUG).\
Configuration-driven and designed for reproducible experiments.

------------------------------------------------------------------------

## 📁 Structure

    TriageTailor/
    ├── src/        # Core scripts
    ├── configs/    # YAML experiment configs
    ├── data/       # Download separately (Zenodo)
    └── README.md

------------------------------------------------------------------------

## 🚀 Quick Start

### 1) Clone

    git clone https://github.com/jassieaung/TriageTailor.git
    cd TriageTailor

### 2) Install
    pip install -r requirements.txt
    pip install -U -q transformers accelerate bitsandbytes

### 3) Pre-Proprocessing 
   python src/02_prepare_dataset.py \
  --config configs/mozilla.yaml \
  --outdir workdir/outputs/mozilla

### 4) Generate Synthetic Data (TriageTailor
)

    python src/02_generate_aug.py   --config configs/mozilla.yaml   --workdir workdir   --mode m2
    %%bash
    set +e
    set -x
    
    for r in r25 r50; do
      echo "=============================="
      echo "RUNNING: ratio=$r (m2)"
      echo "=============================="
    
      file="workdir/outputs/mozilla/augmented/m2_${r}.jsonl"
    
      if [ ! -s "$file" ]; then
        echo "SKIPPING ratio=$r because $file is missing or empty"
        continue
      fi
    
      python -u src/03_train_eval.py \
        --config configs/mozilla.yaml \
        --workdir workdir/outputs/mozilla \
        --mode m2 \
        --ratio "$r" \
        --models LinearSVC ClassWeightedSVC LogisticRegression CodeBERT BiLSTM CNN \
        --synthetic_weight 0.3 \
        --resume 0
    
      code=$?
      echo "EXIT CODE for ratio=$r: $code"
    
      if [ $code -ne 0 ]; then
        echo "FAILED at ratio=$r"
        break
      fi
      

### 5) Train & Evaluate

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
