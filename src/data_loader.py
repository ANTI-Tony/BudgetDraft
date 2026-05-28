"""
Dataset loader for the BudgetDraft evaluation.

Supports the three datasets used in the paper:
    gs                       — PG-19 long-form books
    longbench_packed_qmsum   — LongBench / QMSum meeting QA
    lwm                      — NarrativeQA book summaries (20 fixed indices)

Each loader prefers a local jsonl shipped in ../data/ over a HuggingFace
download, so evaluation runs offline once the bundle is in place.

Prompts are truncated to max_length tokens.
"""

import json
import os

from datasets import load_dataset
from tqdm import tqdm


def load_prompts(dataset_name, tokenizer, max_length=4096, max_samples=20):
    """Load prompts from a dataset, tokenize and truncate to max_length."""
    if dataset_name == 'gs':
        return _load_pg19(tokenizer, max_length, max_samples)
    elif dataset_name == 'longbench_packed_qmsum':
        return _load_longbench_qmsum(tokenizer, max_length, max_samples)
    elif dataset_name == 'lwm':
        return _load_narrativeqa(tokenizer, max_length, max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _find_local(name):
    """Look for `name` in common local layouts (../data/<name> or ../../data/<name>)."""
    here = os.path.dirname(__file__)
    for rel in ("../data", "../../data"):
        p = os.path.abspath(os.path.join(here, rel, name))
        if os.path.exists(p):
            return p
    return None


def _load_pg19(tokenizer, max_length, max_samples):
    """PG-19 test set (first max_samples books)."""
    local_path = _find_local("gs.jsonl") or _find_local("pg19_test.jsonl")
    if local_path:
        texts = []
        with open(local_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                data = json.loads(line)
                texts.append(data['text'])
    else:
        dataset = load_dataset("pg19", split="test")
        texts = [dataset[i]['text'] for i in range(min(max_samples, len(dataset)))]

    prompts = []
    for text in tqdm(texts, desc="Tokenizing PG-19"):
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        if len(tokens) >= max_length:
            prompts.append({'text': text[:10000], 'tokens': tokens[:max_length]})

    if not prompts:
        for text in texts:
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            prompts.append({'text': text[:10000], 'tokens': tokens})

    print(f"[gs] {len(prompts)} prompts loaded (max_length={max_length})")
    return prompts


def _load_longbench_qmsum(tokenizer, max_length, max_samples):
    """LongBench QMSum - meeting transcripts."""
    local_qmsum = _find_local("longbench.jsonl") or _find_local("qmsum.jsonl")
    if local_qmsum:
        with open(local_qmsum, 'r') as f:
            dataset = [json.loads(line) for line in f]
    else:
        try:
            dataset = load_dataset("THUDM/LongBench", "qmsum", split="test")
        except (RuntimeError, ValueError, FileNotFoundError):
            from huggingface_hub import hf_hub_download
            import zipfile
            zip_path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset")
            extract_dir = os.path.join(os.path.dirname(zip_path), "longbench_extracted")
            if not os.path.exists(os.path.join(extract_dir, "qmsum.jsonl")):
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(extract_dir)
            qmsum_path = None
            for root, dirs, files in os.walk(extract_dir):
                for f in files:
                    if f == "qmsum.jsonl":
                        qmsum_path = os.path.join(root, f)
                        break
            if qmsum_path is None:
                raise FileNotFoundError(f"qmsum.jsonl not found in {extract_dir}")
            data = []
            with open(qmsum_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            dataset = data

    prompts = []
    for item in tqdm(dataset, desc="Tokenizing QMSum"):
        if len(prompts) >= max_samples:
            break
        context = item.get('context', '') or item.get('input', '')
        query = item.get('input', '') if 'context' in item else ''
        text = context + "\n" + query if query else context
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        if len(tokens) >= max_length // 2:
            prompts.append({'text': text[:10000], 'tokens': tokens[:max_length]})

    print(f"[longbench_packed_qmsum] {len(prompts)} prompts loaded (max_length={max_length})")
    return prompts


def _load_narrativeqa(tokenizer, max_length, max_samples):
    """NarrativeQA - book summarization. Prefers a local pre-extracted jsonl
    of just the 20 sample indices used; falls back to HF streaming."""
    idx_set = {0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750,
               3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550}
    max_idx = max(idx_set)

    local_lwm = _find_local("lwm.jsonl") or _find_local("narrativeqa_samples.jsonl")
    if local_lwm:
        collected = {}
        with open(local_lwm, 'r') as f:
            for line in f:
                row = json.loads(line)
                collected[row["idx"]] = {"document": {"text": row["document_text"]}}
    else:
        try:
            stream = load_dataset("deepmind/narrativeqa", split="train", streaming=True)
        except Exception:
            stream = load_dataset("narrativeqa", split="train", streaming=True)
        collected = {}
        for i, item in enumerate(tqdm(stream, desc="Streaming NarrativeQA", total=max_idx + 1)):
            if i in idx_set:
                collected[i] = item
            if i > max_idx:
                break

    prompts = []
    for idx in sorted(idx_set):
        if idx not in collected or len(prompts) >= max_samples:
            continue
        item = collected[idx]
        if isinstance(item.get('document'), dict):
            doc_text = item['document'].get('text', '')
        else:
            doc_text = item.get('document', '') or item.get('text', '')

        if not doc_text:
            continue

        book_tokens = tokenizer.encode(doc_text)[:max_length - 100]
        prompt = (
            "You are a helpful assistant. USER: Please read a part of the book below, "
            "and then give me the summary.\n[start of the book]\n"
            + tokenizer.decode(book_tokens, skip_special_tokens=True)
            + "\n[end of the book]\n\nNow you have read it. Please summarize it for me.\n\nASSISTANT: "
        )
        tokens = tokenizer.encode(prompt, truncation=True, max_length=max_length)
        prompts.append({'text': prompt[:10000], 'tokens': tokens[:max_length]})

    print(f"[lwm] {len(prompts)} prompts loaded (max_length={max_length})")
    return prompts
