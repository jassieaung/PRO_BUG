#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Clean two-mode experimental setup:
- M1 is the baseline and does not require this script.
- M2 is TriageTailor synthetic augmentation, renamed from the previous improved M4.

This script generates only M2 files:
    <workdir>/augmented/m2_r05.jsonl
    <workdir>/augmented/m2_r10.jsonl
    <workdir>/augmented/m2_r25.jsonl
    <workdir>/augmented/m2_r50.jsonl

Expected config keys:
    dataset.columns.summary
    dataset.columns.comments
    dataset.columns.label
    augment.syn_ratios
    augment.max_total_syn_per_ratio optional
    augment.max_syn_per_label optional
    generator.* optional
    retriever.emb_model optional
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

try:
    import yaml
except Exception as exc:
    yaml = None
    YAML_IMPORT_ERROR = exc

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import psutil
except Exception:
    psutil = None

MODEL_NAME_DEFAULT = "mistralai/Mistral-7B-Instruct-v0.2"
EMB_MODEL_DEFAULT = "BAAI/bge-base-en-v1.5"
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logger(workdir: str) -> logging.Logger:
    log_dir = Path(workdir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"generate_m2_triagetailor_{ts}.log"

    logger = logging.getLogger("generate_m2_triagetailor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Logging to: %s", log_file)
    return logger


def log_memory(logger: logging.Logger, tag: str) -> None:
    if psutil:
        gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        logger.info("Memory %s: %.2f GB", tag, gb)
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserv = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info("GPU %s: allocated=%.2f GB reserved=%.2f GB", tag, alloc, reserv)


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return data
    except json.JSONDecodeError:
        rows = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"Bad JSONL at {path}:{i}: {exc}") from exc
        return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_tmp_to_drive(logger: logging.Logger, tmp_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp_path, out_path)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------
# Ratio and allocation
# ---------------------------------------------------------------------
def ratio_tag(r: float) -> str:
    return f"r{int(round(r * 100)):02d}"


def parse_ratio_token(token: str) -> float:
    s = str(token).strip().lower()
    if s.startswith("r"):
        s = s[1:]
    if s.endswith("%"):
        s = s[:-1]
    val = float(s)
    if val >= 1.0:
        val /= 100.0
    if not (0.0 < val < 1.0):
        raise ValueError(f"Invalid ratio: {token}")
    return float(f"{val:.2f}")


def filter_ratios(ratios_cfg: List[Any], ratios_arg: Optional[str]) -> List[float]:
    ratios_all = [float(r) for r in ratios_cfg]
    if not ratios_arg:
        return ratios_all
    wanted = {parse_ratio_token(tok) for tok in ratios_arg.replace(",", " ").split() if tok.strip()}
    selected = [r for r in ratios_all if float(f"{r:.2f}") in wanted]
    if not selected:
        raise ValueError(f"No requested ratios matched config ratios. requested={wanted}, config={ratios_all}")
    return selected


def total_syn_needed(n_train: int, ratio: float) -> int:
    # Makes synthetic/(real+synthetic) approximately equal to ratio.
    return int(np.ceil((ratio * n_train) / (1.0 - ratio)))


def pick_minority_labels(counts: Counter, min_real_examples: int = 1) -> List[str]:
    """
    Select minority labels safely for highly long-tailed bug-triage datasets.

    Important Mozilla fix:
    If the median label count is 1, the old rule `count < median` selects
    zero labels because no class has count < 1. This version uses
    `count <= threshold`, where threshold is the larger of the median count
    and the requested minimum real examples.
    """
    if not counts:
        return []

    values = [int(v) for v in counts.values()]
    median_count = int(np.median(values))
    threshold = max(median_count, int(min_real_examples))

    return [
        str(label)
        for label, count in counts.items()
        if int(count) <= threshold and int(count) >= int(min_real_examples)
    ]


def dynamic_cap_for_label(
    real_count: int,
    synthetic_total_fraction: float = 0.20,
    global_cap: int = 50,
    min_syn_per_label: int = 3,
) -> int:
    """
    Compute a per-label synthetic cap using the 80/20 idea, but avoid
    generating too few samples for very small minority labels.

    Base rule:
        synthetic / (real + synthetic) ~= synthetic_total_fraction

    If synthetic_total_fraction = 0.20:
        synthetic ~= 0.25 * real

    Option-2 fix:
        For eligible labels, keep at least min_syn_per_label synthetic samples,
        while still respecting the global cap.

    Examples with synthetic_total_fraction=0.20 and min_syn_per_label=3:
        1 real  -> 0 synthetic  (not enough evidence)
        2 real  -> 3 synthetic
        4 real  -> 3 synthetic
        8 real  -> 3 synthetic
        20 real -> 5 synthetic
        40 real -> 10 synthetic, if global_cap >= 10
    """
    real_count = int(real_count)
    global_cap = int(global_cap)
    min_syn_per_label = int(min_syn_per_label)

    if real_count < 2 or global_cap <= 0:
        return 0

    synthetic_total_fraction = float(synthetic_total_fraction)
    if synthetic_total_fraction <= 0:
        return 0
    if synthetic_total_fraction >= 1:
        raise ValueError("synthetic_total_fraction must be between 0 and 1, e.g. 0.20")

    syn_ratio_of_real = synthetic_total_fraction / (1.0 - synthetic_total_fraction)
    label_cap = int(round(real_count * syn_ratio_of_real))

    # Option 2: keep 80/20 scaling, but avoid tiny allocations such as 1 sample.
    label_cap = max(min_syn_per_label, label_cap)

    return max(1, min(label_cap, global_cap))

def allocate_adaptive(
    total_syn: int,
    counts: Counter,
    labels: List[str],
    cap: int,
    target_percentile: float,
    synthetic_total_fraction: float = 0.20,
    min_syn_per_label: int = 3,
) -> Dict[str, int]:
    if total_syn <= 0 or not labels:
        return {}
    target = max(1, int(np.percentile(list(counts.values()), target_percentile)))
    deficits = {label: max(0, target - int(counts[label])) for label in labels}
    if sum(deficits.values()) == 0:
        weights = {label: 1.0 / max(1, int(counts[label])) for label in labels}
    else:
        weights = {label: float(deficits[label]) for label in labels}
    total_w = sum(weights.values())

    # Dynamic per-label cap: keep approximately 80% real / 20% synthetic
    # for each developer/label, while still respecting the global cap.
    label_caps = {
        label: dynamic_cap_for_label(
            int(counts[label]),
            synthetic_total_fraction=synthetic_total_fraction,
            global_cap=cap,
            min_syn_per_label=min_syn_per_label,
        )
        for label in labels
    }

    alloc = {
        label: min(
            label_caps[label],
            int(np.floor(total_syn * weights[label] / total_w)),
        )
        for label in labels
    }

    remaining = total_syn - sum(alloc.values())
    rare_first = sorted(labels, key=lambda x: (counts[x], x))
    while remaining > 0:
        changed = False
        for label in rare_first:
            if remaining <= 0:
                break
            if alloc.get(label, 0) < label_caps.get(label, 0):
                alloc[label] = alloc.get(label, 0) + 1
                remaining -= 1
                changed = True
        if not changed:
            break
    return {label: n for label, n in alloc.items() if n > 0}


# ---------------------------------------------------------------------
# Text cleaning, profiles, retrieval, filtering
# ---------------------------------------------------------------------
def tokenize_terms(text: str) -> List[str]:
    text = (text or "").lower()
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text)
    stop = {
        "the", "and", "for", "with", "this", "that", "from", "have", "has", "was", "were", "are",
        "but", "not", "bug", "issue", "error", "exception", "please", "thanks", "when", "then",
        "there", "their", "will", "would", "should", "could", "into", "about", "using", "use", "used",
        "file", "line", "also", "only", "more", "less", "can", "cannot", "does", "did", "get", "got",
    }
    return [t for t in toks if t not in stop and not t.isdigit()]


def sanitize_no_leak(text: str, label_set: Optional[set] = None) -> str:
    t = str(text or "")
    t = EMAIL_RE.sub("[redacted_email]", t)
    t = re.sub(r"(?im)^\s*(assignee|assigned\s+to)\s*[:=].*$", "", t)
    t = re.sub(r"(?im)\b(assignee|assigned\s+to)\s*[:=]\s*[^\n\r]{0,120}", "", t)
    if label_set:
        for label in label_set:
            lab = str(label).strip()
            if lab:
                t = re.sub(rf"(?i)\b{re.escape(lab)}\b", "[redacted_label]", t)
    return re.sub(r"\s+", " ", t).strip()


def has_leak(text: str, label_set: Optional[set] = None) -> bool:
    if EMAIL_RE.search(text or ""):
        return True
    low = (text or "").lower()
    if "assigned to" in low or "assignee" in low:
        return True
    if label_set:
        for label in label_set:
            lab = str(label).strip().lower()
            if lab and re.search(rf"\b{re.escape(lab)}\b", low):
                return True
    return False


def build_label_profiles(train_raw: pd.DataFrame, label_col: str, summary_col: str, comments_col: str,
                         component_col: Optional[str] = None,
                         max_terms: int = 12, max_examples: int = 3, example_chars: int = 300) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for label, grp in train_raw.groupby(train_raw[label_col].astype(str)):
        summaries = grp.get(summary_col, pd.Series([""] * len(grp))).fillna("").astype(str).tolist()
        comments = grp.get(comments_col, pd.Series([""] * len(grp))).fillna("").astype(str).tolist()

        components: List[str] = []
        if component_col and component_col in grp.columns:
            components = grp[component_col].fillna("").astype(str).tolist()

        texts = summaries + comments + components
        terms = [term for text in texts for term in tokenize_terms(text)]
        top_components = [c for c, _ in Counter([c for c in components if c.strip()]).most_common(5)]

        examples = []
        for i, (s, c) in enumerate(zip(summaries, comments)):
            comp = components[i] if i < len(components) else ""
            candidate = re.sub(r"\s+", " ", f"{comp} {s} {c}".strip())
            if len(candidate) >= 40:
                examples.append(candidate[:example_chars])
            if len(examples) >= max_examples:
                break

        profile_texts = []
        for i, (s, c) in enumerate(zip(summaries, comments)):
            comp = components[i] if i < len(components) else ""
            joined = re.sub(r"\s+", " ", f"{comp}\n{s}\n{c}".strip())
            if joined:
                profile_texts.append(joined)

        profiles[str(label)] = {
            "terms": [term for term, _ in Counter(terms).most_common(max_terms)],
            "components": top_components,
            "examples": examples,
            "summaries": [s for s in summaries if s.strip()],
            "texts": profile_texts,
        }
    return profiles


def choose_summary(profile: Dict[str, Any], rng: np.random.Generator) -> str:
    summaries = profile.get("summaries", [])
    if summaries:
        return str(summaries[int(rng.integers(0, len(summaries)))]).strip()
    examples = profile.get("examples", [])
    if examples:
        return str(examples[int(rng.integers(0, len(examples)))])[:180]
    return ""


def load_embedder(model_name: str, device: str, logger: logging.Logger):
    if SentenceTransformer is None:
        logger.warning("sentence-transformers is not available. Semantic retrieval/filtering disabled.")
        return None
    logger.info("Loading embedding model: %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


def semantic_search_same_label(embedder, query: str, refs: List[str], k: int) -> List[str]:
    if embedder is None or not query or not refs:
        return refs[:k]
    try:
        q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        r = embedder.encode(refs, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores = r @ q[0]
        idx = np.argsort(-scores)[:k]
        return [refs[int(i)] for i in idx]
    except Exception:
        return refs[:k]


def max_semantic_similarity(embedder, text: str, refs: List[str]) -> Optional[float]:
    if embedder is None or not text or not refs:
        return None
    try:
        q = embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        r = embedder.encode(refs, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        return float(np.max(r @ q[0]))
    except Exception:
        return None


def term_overlap_score(text: str, terms: List[str]) -> float:
    ref = {t.lower() for t in terms if str(t).strip()}
    if not ref:
        return 0.0
    toks = set(tokenize_terms(text))
    return len(toks & ref) / max(1, min(len(ref), 12))


def build_prompt(summary_seed: str, profile: Dict[str, Any], retrieved_context: List[str]) -> str:
    terms = profile.get("terms", [])[:12]
    components = profile.get("components", [])[:5]
    examples = [sanitize_no_leak(e) for e in profile.get("examples", [])[:3]]
    ctx = [sanitize_no_leak(c) for c in retrieved_context[:5]]
    return (
        "You are generating high-quality synthetic training data for software bug triage.\n"
        "Generate ONE realistic Consolidated Comments section for a bug report.\n\n"
        f"Bug Summary:\n{summary_seed if summary_seed else '[no summary available]'}\n\n"
        f"Likely component/product context for this target label:\n{', '.join(components) if components else 'not explicitly available'}\n\n"
        f"Target expertise keywords inferred from historical reports:\n{', '.join(terms) if terms else 'debugging, reproduction, component, patch'}\n\n"
        "Same-label historical writing examples, sanitized:\n"
        + ("\n".join(f"Example {i+1}: {e}" for i, e in enumerate(examples)) if examples else "No examples available.")
        + "\n\nSame-label retrieved context, sanitized:\n"
        + ("\n---\n".join(ctx) if ctx else "No retrieved context available.")
        + "\n\nRequirements:\n"
        "- Write only the bug discussion/comment text.\n"
        "- Make the text consistent with the component/product context above.\n"
        "- Include reproduction context, observed behavior, likely component/API, and debugging clue when possible.\n"
        "- Keep it technically plausible and relevant to the bug summary.\n"
        "- Do not mention developer names, emails, labels, assignees, or 'Assigned to'.\n"
        "- Do not copy the examples verbatim; create a new but realistic instance.\n"
        "- Avoid URLs, private information, fabricated dates, and generic filler.\n"
        "Return ONLY the generated comments text."
    )


def quality_filter(texts: List[str], label_set: set, profile: Dict[str, Any], summary_seed: str, embedder,
                   min_words: int, max_words: int, min_term_overlap: float,
                   dup_threshold: float, min_relevance: float,
                   target_label: Optional[str] = None, confidence_vectorizer=None, confidence_clf=None,
                   confidence_threshold: float = 0.0) -> Tuple[List[str], Dict[str, int]]:
    kept: List[str] = []
    stats = Counter()
    refs = [sanitize_no_leak(x, label_set) for x in profile.get("texts", [])[:40]]
    positive_refs = [summary_seed] + refs[:10]
    seen = set()
    generic_patterns = ["please fix", "does not work", "there is an issue", "kindly check", "needs to be resolved"]

    for raw in texts:
        txt = sanitize_no_leak(raw, label_set)
        words = txt.split()
        if len(words) < min_words:
            stats["too_short"] += 1
            continue
        if len(words) > max_words:
            txt = " ".join(words[:max_words])
        if has_leak(txt, label_set):
            stats["leakage"] += 1
            continue
        if sum(1 for p in generic_patterns if p in txt.lower()) >= 2:
            stats["low_information"] += 1
            continue
        if min_term_overlap > 0 and term_overlap_score(txt, profile.get("terms", [])) < min_term_overlap:
            stats["low_term_overlap"] += 1
            continue
        h = hashlib.md5(txt.lower().encode("utf-8")).hexdigest()
        if h in seen:
            stats["duplicate_generated"] += 1
            continue
        seen.add(h)
        sim_dup = max_semantic_similarity(embedder, txt, refs)
        if sim_dup is not None and sim_dup >= dup_threshold:
            stats["semantic_duplicate"] += 1
            continue
        sim_rel = max_semantic_similarity(embedder, txt, positive_refs)
        if sim_rel is not None and min_relevance > 0 and sim_rel < min_relevance:
            stats["low_semantic_relevance"] += 1
            continue
        if target_label is not None and confidence_threshold > 0:
            # Use summary + generated comments for confidence checking so the
            # verifier sees the same kind of text as the final training sample.
            confidence_text = (str(summary_seed) + "\n" + str(txt)) if summary_seed else txt
            keep, prob, pred_label = confidence_keep(
                confidence_vectorizer, confidence_clf, confidence_text, target_label, confidence_threshold
            )
            if not keep:
                stats["low_confidence_or_wrong_label"] += 1
                continue
        kept.append(txt)

    stats["input"] = len(texts)
    stats["kept"] = len(kept)
    return kept, dict(stats)



# ---------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------
def make_train_texts(train_raw: pd.DataFrame, summary_col: str, comments_col: str, component_col: Optional[str] = None) -> pd.Series:
    summary = train_raw.get(summary_col, pd.Series([""] * len(train_raw))).fillna("").astype(str)
    comments = train_raw.get(comments_col, pd.Series([""] * len(train_raw))).fillna("").astype(str)
    if component_col and component_col in train_raw.columns:
        component = train_raw[component_col].fillna("").astype(str)
        return (component + "\n" + summary + "\n" + comments).astype(str)
    return (summary + "\n" + comments).astype(str)


def fit_confidence_filter(train_raw: pd.DataFrame, summary_col: str, comments_col: str, label_col: str,
                          component_col: Optional[str], logger: logging.Logger):
    """
    Fit a lightweight TF-IDF + calibrated LinearSVC model on real training data only.

    Important fix for long-tailed bug-triage data:
    CalibratedClassifierCV(cv=3) fails when a label has fewer than 3 examples.
    Therefore, this function trains the confidence filter only on labels with
    enough real examples and automatically lowers cv when needed.
    """
    try:
        labels_all = train_raw[label_col].fillna("UNKNOWN").astype(str)
        counts = labels_all.value_counts()
        eligible_labels = set(counts[counts >= 2].index.astype(str))
        filtered = train_raw[labels_all.isin(eligible_labels)].copy()

        if filtered.empty or filtered[label_col].astype(str).nunique() < 2:
            logger.warning("Confidence filter disabled: fewer than 2 eligible labels with >=2 examples.")
            return None, None

        labels = filtered[label_col].fillna("UNKNOWN").astype(str)
        min_count = int(labels.value_counts().min())
        cv = max(2, min(3, min_count))

        texts = make_train_texts(filtered, summary_col, comments_col, component_col)
        vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=1)
        X = vectorizer.fit_transform(texts)
        base = LinearSVC()
        clf = CalibratedClassifierCV(base, cv=cv)
        clf.fit(X, labels)
        logger.info(
            "Confidence filter fitted: TF-IDF + Calibrated LinearSVC | rows=%d labels=%d cv=%d",
            X.shape[0], labels.nunique(), cv,
        )
        return vectorizer, clf
    except Exception as exc:
        logger.warning("Confidence filter disabled because fitting failed: %s", exc)
        return None, None


def confidence_keep(vectorizer, clf, text: str, target_label: str, threshold: float) -> Tuple[bool, Optional[float], Optional[str]]:
    if vectorizer is None or clf is None:
        return True, None, None
    try:
        X = vectorizer.transform([text])
        probs = clf.predict_proba(X)[0]
        classes = list(clf.classes_)
        pred_idx = int(np.argmax(probs))
        pred_label = str(classes[pred_idx])
        target_prob = float(probs[classes.index(str(target_label))]) if str(target_label) in classes else 0.0
        return (pred_label == str(target_label) and target_prob >= threshold), target_prob, pred_label
    except Exception:
        return True, None, None


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------
def wrap_mistral_inst(prompt: str) -> str:
    return f"<s>[INST]\n{prompt.strip()}\n[/INST]\n"


def autocast_context():
    if torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_generator(model_name: str, load_in_4bit: bool, logger: logging.Logger):
    device_is_cuda = torch.cuda.is_available()
    logger.info("Loading generator: %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16 if device_is_cuda else torch.float32}
    if device_is_cuda:
        kwargs["device_map"] = "auto"
    if device_is_cuda and load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("Using 4-bit quantization.")
        except Exception as exc:
            logger.warning("4-bit unavailable; falling back. Reason: %s", exc)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tok


def generate_batch(model, tok, prompt: str, num_samples: int, max_prompt_tokens: int, max_new_tokens: int,
                   temperature: float, top_p: float, repetition_penalty: float) -> List[str]:
    full_prompt = wrap_mistral_inst(prompt)
    x = tok(full_prompt, return_tensors="pt", truncation=False)
    device = next(model.parameters()).device
    input_ids = x["input_ids"].to(device)
    attention_mask = x["attention_mask"].to(device)
    if input_ids.shape[1] > max_prompt_tokens:
        input_ids = input_ids[:, -max_prompt_tokens:]
        attention_mask = attention_mask[:, -max_prompt_tokens:]
    input_ids = input_ids.repeat(num_samples, 1)
    attention_mask = attention_mask.repeat(num_samples, 1)

    with torch.inference_mode(), autocast_context():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    generated = outputs[:, input_ids.shape[1]:]
    return [t.strip() for t in tok.batch_decode(generated, skip_special_tokens=True)]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--ratios", default=None, help="e.g., r05,r10 or 0.05 0.10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_total_syn_per_ratio", type=int, default=None)
    ap.add_argument("--max_syn_per_label", type=int, default=None)
    ap.add_argument("--synthetic_label_fraction", type=float, default=0.20,
                    help="Target synthetic fraction per label after augmentation. 0.20 means each label is approximately 80% real and 20% synthetic.")
    ap.add_argument("--min_syn_per_label", type=int, default=3,
                    help="Option-2 cap fix: minimum synthetic samples per eligible label after applying the 80/20 dynamic cap.")
    ap.add_argument("--min_real_examples_per_label", type=int, default=3,
                    help="Only generate for minority labels with at least this many real training examples.")
    ap.add_argument("--adaptive_target_percentile", type=float, default=50.0)
    ap.add_argument("--candidate_multiplier", type=float, default=1.8)
    ap.add_argument("--max_retry_rounds", type=int, default=2)
    ap.add_argument("--min_words", type=int, default=12)
    ap.add_argument("--max_words", type=int, default=180)
    ap.add_argument("--min_term_overlap", type=float, default=0.08)
    ap.add_argument("--semantic_dup_threshold", type=float, default=0.92)
    ap.add_argument("--semantic_min_relevance", type=float, default=0.20)
    ap.add_argument("--retrieval_k", type=int, default=5)
    ap.add_argument("--embedder_device", default="cpu")
    ap.add_argument("--embedder_model", default=None)
    ap.add_argument("--disable_semantic", type=int, default=0)
    ap.add_argument("--component_col", default=None, help="Optional component/product column. If omitted, uses dataset.columns.component/product if available.")
    ap.add_argument("--use_confidence_filter", type=int, default=1)
    ap.add_argument("--confidence_threshold", type=float, default=0.60)
    ap.add_argument("--synthetic_weight", type=float, default=0.30)
    ap.add_argument("--gen_model", default=MODEL_NAME_DEFAULT)
    ap.add_argument("--load_in_4bit", type=int, default=1)
    ap.add_argument("--tmp_root", default="/content/rag_bug_tmp")
    args = ap.parse_args()

    if yaml is None:
        raise RuntimeError(f"PyYAML is required: {YAML_IMPORT_ERROR}")

    logger = setup_logger(args.workdir)
    rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    cols = cfg["dataset"]["columns"]
    summary_col = cols["summary"]
    comments_col = cols["comments"]
    label_col = cols["label"]
    component_col = args.component_col or cols.get("component") or cols.get("product") or cols.get("component_col")
    if component_col and component_col not in pd.DataFrame(load_json_or_jsonl(Path(args.workdir) / "clean.json")).columns:
        component_col = None

    ratios = filter_ratios(cfg["augment"]["syn_ratios"], args.ratios)
    max_total_syn = int(args.max_total_syn_per_ratio or cfg.get("augment", {}).get("max_total_syn_per_ratio", 20000))
    max_syn_per_label = int(args.max_syn_per_label or cfg.get("augment", {}).get("max_syn_per_label", 50))

    gen_cfg = cfg.get("generator", {})
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 128))
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    max_prompt_tokens = int(gen_cfg.get("max_prompt_tokens", 768))
    repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.1))

    clean_path = Path(args.workdir) / "clean.json"
    train_ids_path = Path(args.workdir) / "splits" / "train_ids.json"
    df = pd.DataFrame(load_json_or_jsonl(clean_path))
    train_ids = pd.Series(load_json_or_jsonl(train_ids_path)).astype(int).to_numpy()
    train_raw = df.iloc[train_ids].reset_index(drop=True)
    train_raw[label_col] = train_raw[label_col].fillna("UNKNOWN").astype(str)

    counts = Counter(train_raw[label_col].astype(str))
    logger.info(
        "Label distribution: min=%d median=%d max=%d",
        int(min(counts.values())),
        int(np.median(list(counts.values()))),
        int(max(counts.values())),
    )
    minority_labels = pick_minority_labels(counts, min_real_examples=args.min_real_examples_per_label)
    logger.info("Selected minority labels=%d", len(minority_labels))
    label_set = set(counts.keys())
    profiles = build_label_profiles(train_raw, label_col, summary_col, comments_col, component_col=component_col)

    logger.info(
        "Train rows=%d labels=%d minority_labels=%d min_real_examples_per_label=%d",
        len(train_raw), len(counts), len(minority_labels), args.min_real_examples_per_label,
    )
    logger.info("Component-aware column: %s", component_col if component_col else "not available")
    logger.info("Ratios to generate: %s", [ratio_tag(r) for r in ratios])
    logger.info(
        "Caps: max_total_syn_per_ratio=%d max_syn_per_label=%d synthetic_label_fraction=%.2f min_syn_per_label=%d",
        max_total_syn, max_syn_per_label, args.synthetic_label_fraction, args.min_syn_per_label,
    )

    embedder = None
    if not args.disable_semantic:
        emb_model = args.embedder_model or cfg.get("retriever", {}).get("emb_model", EMB_MODEL_DEFAULT)
        embedder = load_embedder(emb_model, args.embedder_device, logger)

    conf_vectorizer, conf_clf = (None, None)
    if args.use_confidence_filter:
        conf_vectorizer, conf_clf = fit_confidence_filter(
            train_raw, summary_col, comments_col, label_col, component_col, logger
        )

    model, tok = load_generator(args.gen_model, bool(args.load_in_4bit), logger)
    log_memory(logger, "after_generator_load")

    tmp_dir = Path(args.tmp_root) / "augmented"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for ratio in ratios:
        tag = ratio_tag(ratio)
        target_total = min(total_syn_needed(len(train_raw), ratio), max_total_syn)
        alloc = allocate_adaptive(
            target_total,
            counts,
            minority_labels,
            max_syn_per_label,
            args.adaptive_target_percentile,
            synthetic_total_fraction=args.synthetic_label_fraction,
            min_syn_per_label=args.min_syn_per_label,
        )
        out_path = Path(args.workdir) / "augmented" / f"m2_{tag}.jsonl"
        tmp_path = tmp_dir / f"m2_{tag}.jsonl"
        if tmp_path.exists():
            tmp_path.unlink()

        logger.info("[%s] target_total=%d allocated=%d labels=%d", tag, target_total, sum(alloc.values()), len(alloc))
        if alloc:
            preview = [
                f"{label}:real={int(counts[label])},syn={int(n)}"
                for label, n in sorted(alloc.items(), key=lambda kv: (counts[kv[0]], kv[0]))[:10]
            ]
            logger.info("[%s] dynamic per-label allocation preview: %s", tag, "; ".join(preview))
        all_stats = Counter()
        written = 0
        started = time.time()

        for lab_i, (label, need) in enumerate(sorted(alloc.items(), key=lambda kv: (counts[kv[0]], kv[0])), start=1):
            profile = profiles.get(label, {"terms": [], "examples": [], "summaries": [], "texts": []})
            kept_for_label: List[str] = []
            rounds = 0
            while len(kept_for_label) < need and rounds <= args.max_retry_rounds:
                rounds += 1
                remaining = need - len(kept_for_label)
                n_candidates = max(1, int(np.ceil(remaining * args.candidate_multiplier)))
                summary_seed = choose_summary(profile, rng)
                same_label_refs = [sanitize_no_leak(x, label_set) for x in profile.get("texts", []) if str(x).strip()]
                retrieved = semantic_search_same_label(embedder, summary_seed, same_label_refs, args.retrieval_k)
                prompt = build_prompt(summary_seed, profile, retrieved)
                candidates = generate_batch(
                    model, tok, prompt, n_candidates,
                    max_prompt_tokens=max_prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                kept, stats = quality_filter(
                    candidates, label_set, profile, summary_seed, embedder,
                    min_words=args.min_words,
                    max_words=args.max_words,
                    min_term_overlap=args.min_term_overlap,
                    dup_threshold=args.semantic_dup_threshold,
                    min_relevance=args.semantic_min_relevance,
                    target_label=label,
                    confidence_vectorizer=conf_vectorizer,
                    confidence_clf=conf_clf,
                    confidence_threshold=args.confidence_threshold if args.use_confidence_filter else 0.0,
                )
                all_stats.update(stats)
                kept_for_label.extend(kept[:remaining])

            rows = []
            summaries = profile.get("summaries", [])
            for j, text in enumerate(kept_for_label[:need]):
                seed_summary = summaries[j % len(summaries)] if summaries else "Synthetic bug report"
                row = {
                    summary_col: seed_summary,
                    comments_col: text,
                    label_col: label,
                    "_synthetic": True,
                    "_mode": "m2",
                    "_ratio": tag,
                    "_sample_weight": float(args.synthetic_weight),
                }
                if component_col and profile.get("components"):
                    row[component_col] = profile["components"][j % len(profile["components"])]
                rows.append(row)
            if rows:
                write_jsonl(tmp_path, rows, append=True)
                written += len(rows)
            if lab_i % 25 == 0:
                logger.info("[%s] labels_done=%d/%d written=%d", tag, lab_i, len(alloc), written)

        if not tmp_path.exists():
            logger.warning("[%s] no synthetic rows were kept; writing an empty JSONL file.", tag)
            tmp_path.touch()
        copy_tmp_to_drive(logger, tmp_path, out_path)
        logger.info("[%s] done written=%d elapsed=%.1fs filter_stats=%s", tag, written, time.time() - started, dict(all_stats))


if __name__ == "__main__":
    main()
