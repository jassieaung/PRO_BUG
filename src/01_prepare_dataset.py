"""Prepare dataset: clean + frozen splits.

Run:
  python src/02_prepare_dataset.py --config configs/mozilla.yaml --outdir outputs/mozilla
"""

import argparse
import os

import numpy as np
import pandas as pd

from utils import ensure_dir, load_config, norm_text, read_json, write_json, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["dataset"]["split"]["seed"])
    seed_everything(seed)

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "splits"))

    data = read_json(cfg["dataset"]["input_json"], json_lines=bool(cfg["dataset"].get("json_lines", False)))
    df = pd.DataFrame(data)

    cols = cfg["dataset"]["columns"]
    for k in ["bug_id", "summary", "comments", "label", "status"]:
        if cols[k] not in df.columns:
            df[cols[k]] = ""

    # Status filter
    keep = set(cfg["dataset"].get("status_keep", []))
    if keep:
        df = df[df[cols["status"]].isin(list(keep))].copy()

    # Basic NA handling
    df[cols["bug_id"]] = df[cols["bug_id"]].astype(str)
    df[cols["summary"]] = df[cols["summary"]].fillna("").astype(str)
    df[cols["comments"]] = df[cols["comments"]].fillna("").astype(str)
    df[cols["label"]] = df[cols["label"]].fillna("UNKNOWN").astype(str)

    # Optional dedup
    dedup_cfg = cfg.get("dedup", {})
    if bool(dedup_cfg.get("enabled", False)):
        key_fields = dedup_cfg.get("key_fields", [cols["summary"]])
        # Create a stable key from normalized text
        keys = []
        for _, row in df.iterrows():
            parts = [norm_text(str(row.get(f, ""))) for f in key_fields]
            keys.append("||".join(parts))
        df["__dedup_key"] = keys
        df = df.drop_duplicates(subset=["__dedup_key"], keep="first").drop(columns=["__dedup_key"])

    df = df.reset_index(drop=True)

    # Save cleaned dataset
    clean_path = os.path.join(args.outdir, "clean.json")
    write_json(clean_path, df.to_dict("records"))

    # Create splits (by row index, not bug id, to keep stable for downstream)
    n = len(df)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    tr = float(cfg["dataset"]["split"]["train"])
    va = float(cfg["dataset"]["split"]["val"])
    te = float(cfg["dataset"]["split"]["test"])
    if abs(tr + va + te - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1")

    n_train = int(n * tr)
    n_val = int(n * va)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    split_dir = os.path.join(args.outdir, "splits")
    write_json(os.path.join(split_dir, "train_ids.json"), train_idx.tolist())
    write_json(os.path.join(split_dir, "val_ids.json"), val_idx.tolist())
    write_json(os.path.join(split_dir, "test_ids.json"), test_idx.tolist())

    print(f"Saved clean dataset: {clean_path}")
    print(f"Rows: total={n}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


if __name__ == "__main__":
    main()
