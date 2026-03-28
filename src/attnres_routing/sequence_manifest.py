from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import load_dataset

from .data import _resolve_dataset_spec, _text_column, load_tokenizer
from .utils import resolve_hf_token


_TOP_LEVEL_HEADING_RE = re.compile(r"^=\s+[^=].*?[^=]\s+=$")


def _wikitext_documents(lines: Iterable[str]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_title = "untitled"

    def flush() -> None:
        nonlocal current_lines, current_title
        text = "".join(current_lines).strip()
        if text:
            documents.append(
                {
                    "title": current_title,
                    "text": text,
                }
            )
        current_lines = []
        current_title = "untitled"

    for raw_line in lines:
        line = raw_line or ""
        stripped = line.strip()
        if _TOP_LEVEL_HEADING_RE.fullmatch(stripped):
            flush()
            current_title = stripped.strip("= ").strip() or "untitled"
            current_lines = [line]
            continue
        if current_lines or stripped:
            current_lines.append(line)
    flush()
    return documents


def _split_documents(
    docs: list[dict[str, Any]],
    source_split: str,
    seed: int,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
) -> list[dict[str, Any]]:
    if source_split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported synthetic split: {source_split!r}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    num_docs = len(docs)
    val_count = max(1, int(round(num_docs * val_ratio)))
    test_count = max(1, int(round(num_docs * test_ratio)))
    train_count = max(num_docs - val_count - test_count, 1)
    train_ids = set(order[:train_count].tolist())
    val_ids = set(order[train_count : train_count + val_count].tolist())
    test_ids = set(order[train_count + val_count :].tolist())
    target_ids = {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
    }[source_split]
    return [doc for idx, doc in enumerate(docs) if idx in target_ids]


def load_documents(dataset_name: str, source_split: str, seed: int = 42) -> list[dict[str, Any]]:
    spec = _resolve_dataset_spec(dataset_name)
    token = resolve_hf_token()
    dataset = load_dataset(spec["path"], spec.get("name"), token=token)
    if source_split in dataset:
        split_dataset = dataset[source_split]
        text_col = _text_column(split_dataset)

        if dataset_name == "wikitext103":
            docs = _wikitext_documents(split_dataset[text_col])
        else:
            docs = []
            for row in split_dataset:
                text = row[text_col]
                if text is None or not text.strip():
                    continue
                docs.append({"title": "", "text": text})
    elif "train" in dataset and source_split in {"train", "validation", "test"}:
        split_dataset = dataset["train"]
        text_col = _text_column(split_dataset)
        docs = []
        for row in split_dataset:
            text = row[text_col]
            if text is None or not text.strip():
                continue
            docs.append({"title": "", "text": text})
        docs = _split_documents(docs, source_split=source_split, seed=seed)
    else:
        raise ValueError(f"Split {source_split!r} not found in {dataset_name}. Available: {sorted(dataset.keys())}")

    for doc_idx, doc in enumerate(docs):
        doc["document_idx"] = doc_idx
        doc["source_split"] = source_split
    return docs


def load_all_documents(dataset_name: str, seed: int = 42) -> list[dict[str, Any]]:
    spec = _resolve_dataset_spec(dataset_name)
    token = resolve_hf_token()
    if "data_files" in spec:
        dataset = load_dataset(spec["path"], data_files=spec["data_files"], token=token)
        combined_docs: list[dict[str, Any]] = []
        for split_name in ("train", "validation", "test"):
            if split_name not in dataset:
                continue
            split_dataset = dataset[split_name]
            text_col = _text_column(split_dataset)
            local_idx = 0
            for row in split_dataset:
                text = row[text_col]
                if text is None or not str(text).strip():
                    continue
                combined_docs.append(
                    {
                        "title": "",
                        "text": str(text),
                        "document_idx": local_idx,
                        "source_split": split_name,
                    }
                )
                local_idx += 1
        return combined_docs

    if spec.get("streaming", False):
        combined_docs: list[dict[str, Any]] = []
        source_split = str(spec.get("stream_source_split", "train"))
        offsets = spec.get("stream_offsets", {})
        limits = spec.get("stream_limits", {})
        for split_name in ("train", "validation", "test"):
            offset = int(offsets.get(split_name, 0))
            limit = int(limits.get(split_name, 0))
            if limit <= 0:
                continue
            stream_ds = load_dataset(
                spec["path"],
                spec.get("name"),
                split=source_split,
                streaming=True,
                token=token,
            )
            local_idx = 0
            for idx, row in enumerate(stream_ds):
                if idx < offset:
                    continue
                text = None
                for column in ("text", "content", "document"):
                    if column in row and row[column] is not None:
                        text = row[column]
                        break
                if text is None or not str(text).strip():
                    continue
                combined_docs.append(
                    {
                        "title": "",
                        "text": str(text),
                        "document_idx": local_idx,
                        "source_split": split_name,
                    }
                )
                local_idx += 1
                if local_idx >= limit:
                    break
        return combined_docs

    if any(key in spec for key in ("train_split", "validation_split", "test_split")):
        combined_docs: list[dict[str, Any]] = []
        split_specs = {
            split_name: spec.get(f"{split_name}_split")
            for split_name in ("train", "validation", "test")
            if spec.get(f"{split_name}_split") is not None
        }
        for split_name, split_spec in split_specs.items():
            split_dataset = load_dataset(spec["path"], spec.get("name"), split=split_spec, token=token)
            text_col = _text_column(split_dataset)
            local_idx = 0
            for row in split_dataset:
                text = row[text_col]
                if text is None or not text.strip():
                    continue
                combined_docs.append(
                    {
                        "title": "",
                        "text": text,
                        "document_idx": local_idx,
                        "source_split": split_name,
                    }
                )
                local_idx += 1
        return combined_docs

    dataset = load_dataset(spec["path"], spec.get("name"), token=token)

    combined_docs: list[dict[str, Any]] = []
    if dataset_name == "wikitext103":
        split_names = [split for split in ("train", "validation", "test") if split in dataset]
        for split_name in split_names:
            split_dataset = dataset[split_name]
            text_col = _text_column(split_dataset)
            docs = _wikitext_documents(split_dataset[text_col])
            for doc_idx, doc in enumerate(docs):
                combined_docs.append(
                    {
                        "title": doc["title"],
                        "text": doc["text"],
                        "document_idx": doc_idx,
                        "source_split": split_name,
                    }
                )
        return combined_docs

    if len(dataset.keys()) == 1 and "train" in dataset:
        split_names = ["train"]
    else:
        split_names = list(dataset.keys())

    for split_name in split_names:
        split_dataset = dataset[split_name]
        text_col = _text_column(split_dataset)
        local_docs: list[dict[str, Any]] = []
        for row in split_dataset:
            text = row[text_col]
            if text is None or not text.strip():
                continue
            local_docs.append({"title": "", "text": text})

        if len(dataset.keys()) == 1 and split_name == "train":
            # Keep access to deterministic synthetic sub-splits possible elsewhere via `load_documents`,
            # but for lockbox partitioning we want to start from the raw document pool once.
            pass

        for doc_idx, doc in enumerate(local_docs):
            combined_docs.append(
                {
                    "title": doc["title"],
                    "text": doc["text"],
                    "document_idx": doc_idx,
                    "source_split": split_name,
                }
            )
    return combined_docs


def document_uid(document: dict[str, Any]) -> str:
    text_hash = hashlib.sha1(document["text"].encode("utf-8", errors="ignore")).hexdigest()
    return f"{document['source_split']}::{document['document_idx']}::{text_hash}"


def build_document_window_records(
    tokenizer: Any,
    document: dict[str, Any],
    prompt_len: int,
    decode_len: int,
    max_windows_per_doc: int = 8,
    stride: int | None = None,
) -> list[dict[str, Any]]:
    total_tokens = prompt_len + decode_len
    if total_tokens <= 0:
        raise ValueError("prompt_len + decode_len must be positive")
    if stride is None:
        stride = total_tokens

    token_ids = tokenizer(document["text"], add_special_tokens=False, truncation=False)["input_ids"]
    max_start = len(token_ids) - total_tokens
    if max_start < 0:
        return []

    records: list[dict[str, Any]] = []
    window_count = 0
    for start in range(0, max_start + 1, stride):
        if window_count >= max_windows_per_doc:
            break
        records.append(
            {
                "source_split": str(document["source_split"]),
                "document_idx": int(document["document_idx"]),
                "document_title": str(document["title"]),
                "document_uid": document_uid(document),
                "window_idx": int(window_count),
                "window_start": int(start),
                "window_end": int(start + total_tokens),
                "prompt_len": int(prompt_len),
                "decode_len": int(decode_len),
                "input_ids": token_ids[start : start + total_tokens],
            }
        )
        window_count += 1
    return records


def build_window_records(
    dataset_name: str,
    tokenizer_name: str,
    source_split: str,
    prompt_len: int,
    decode_len: int,
    target_count: int | None,
    seed: int = 42,
    max_windows_per_doc: int = 8,
    stride: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = load_tokenizer(tokenizer_name)
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0)), 1_000_000_000)
    total_tokens = prompt_len + decode_len
    if total_tokens <= 0:
        raise ValueError("prompt_len + decode_len must be positive")
    if stride is None:
        stride = total_tokens

    documents = load_documents(dataset_name, source_split, seed=seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(documents)).tolist()

    records: list[dict[str, Any]] = []
    used_documents: set[int] = set()
    total_candidate_windows = 0
    sequence_idx = 0

    for order_rank, doc_pos in enumerate(order):
        document = documents[doc_pos]
        token_ids = tokenizer(document["text"], add_special_tokens=False, truncation=False)["input_ids"]
        window_count = 0
        max_start = len(token_ids) - total_tokens
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            total_candidate_windows += 1
            if window_count >= max_windows_per_doc:
                break
            records.append(
                {
                    "sequence_idx": sequence_idx,
                    "source_split": source_split,
                    "document_idx": int(document["document_idx"]),
                    "document_title": str(document["title"]),
                    "document_order": int(order_rank),
                    "window_idx": int(window_count),
                    "window_start": int(start),
                    "window_end": int(start + total_tokens),
                    "prompt_len": int(prompt_len),
                    "decode_len": int(decode_len),
                    "input_ids": token_ids[start : start + total_tokens],
                }
            )
            used_documents.add(int(document["document_idx"]))
            sequence_idx += 1
            window_count += 1
            if target_count is not None and len(records) >= target_count:
                summary = {
                    "dataset_name": dataset_name,
                    "tokenizer_name": tokenizer_name,
                    "source_split": source_split,
                    "prompt_len": prompt_len,
                    "decode_len": decode_len,
                    "total_tokens": total_tokens,
                    "stride": stride,
                    "target_count": target_count,
                    "created_count": len(records),
                    "documents_available": len(documents),
                    "documents_used": len(used_documents),
                    "total_candidate_windows_seen": total_candidate_windows,
                    "max_windows_per_doc": max_windows_per_doc,
                    "seed": seed,
                }
                return records, summary

    summary = {
        "dataset_name": dataset_name,
        "tokenizer_name": tokenizer_name,
        "source_split": source_split,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        "total_tokens": total_tokens,
        "stride": stride,
        "target_count": target_count,
        "created_count": len(records),
        "documents_available": len(documents),
        "documents_used": len(used_documents),
        "total_candidate_windows_seen": total_candidate_windows,
        "max_windows_per_doc": max_windows_per_doc,
        "seed": seed,
    }
    return records, summary


def save_manifest_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_manifest_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
