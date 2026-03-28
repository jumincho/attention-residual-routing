from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from .utils import resolve_hf_token


DATASET_ALIASES: dict[str, dict[str, Any]] = {
    "tinystories": {"path": "roneneldan/TinyStories"},
    "wikitext103": {"path": "wikitext", "name": "wikitext-103-raw-v1"},
    "openwebtext10k": {"path": "stas/openwebtext-10k"},
    "c4_en": {"path": "c4", "name": "en"},
    "cc_news": {"path": "cc_news"},
    "fineweb_edu_sample10bt": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "train_split": "train[:12000]",
        "validation_split": "train[12000:14000]",
        "test_split": "train[14000:16000]",
    },
    "fineweb_edu_sample10bt_stream": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "streaming": True,
        "stream_source_split": "train",
        "stream_offsets": {"train": 0, "validation": 12000, "test": 14000},
        "stream_limits": {"train": 10000, "validation": 2000, "test": 2000},
    },
    "fineweb_edu_sample10bt_local_v7": {
        "path": "json",
        "data_files": {
            "train": "data/third_corpus_v7/fineweb_edu_sample10bt_train.jsonl",
            "validation": "data/third_corpus_v7/fineweb_edu_sample10bt_validation.jsonl",
            "test": "data/third_corpus_v7/fineweb_edu_sample10bt_test.jsonl",
        },
    },
}


@dataclass
class DataConfig:
    dataset_name: str
    tokenizer_name: str = "openai-community/gpt2"
    seq_len: int = 1024
    max_train_texts: Optional[int] = None
    max_val_texts: Optional[int] = None
    max_test_texts: Optional[int] = None
    val_split_ratio: float = 0.01
    num_proc: int = 1
    trust_remote_code: bool = False


def load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    token = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_dataset_spec(dataset_name: str) -> dict[str, Any]:
    if dataset_name not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    return DATASET_ALIASES[dataset_name]


def _text_column(dataset: Dataset) -> str:
    for column in ("text", "content", "document"):
        if column in dataset.column_names:
            return column
    raise ValueError(f"No supported text column found in {dataset.column_names}")


def _limit_split(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _build_dataset_from_stream_rows(rows: list[dict[str, Any]]) -> Dataset:
    if not rows:
        return Dataset.from_dict({"text": []})
    keys = rows[0].keys()
    columns: dict[str, list[Any]] = {key: [] for key in keys}
    for row in rows:
        for key in keys:
            columns[key].append(row.get(key))
    return Dataset.from_dict(columns)


def _collect_stream_rows(
    spec: dict[str, Any],
    split_name: str,
    limit_override: Optional[int],
    token: Optional[str],
    trust_remote_code: bool,
) -> Dataset:
    offset = int(spec.get("stream_offsets", {}).get(split_name, 0))
    default_limit = spec.get("stream_limits", {}).get(split_name)
    limit = int(limit_override if limit_override is not None else default_limit)
    if limit <= 0:
        return Dataset.from_dict({"text": []})
    source_split = str(spec.get("stream_source_split", split_name))
    stream_ds = load_dataset(
        spec["path"],
        spec.get("name"),
        split=source_split,
        streaming=True,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(stream_ds):
        if idx < offset:
            continue
        rows.append(dict(row))
        if len(rows) >= limit:
            break
    return _build_dataset_from_stream_rows(rows)


def _ensure_splits(dataset: DatasetDict) -> tuple[Dataset, Dataset]:
    if "train" in dataset and "validation" in dataset:
        return dataset["train"], dataset["validation"]
    if "train" in dataset and "test" in dataset:
        split = dataset["train"].train_test_split(test_size=0.01, seed=42)
        return split["train"], dataset["test"]
    if "train" in dataset:
        split = dataset["train"].train_test_split(test_size=0.01, seed=42)
        return split["train"], split["test"]
    first_key = next(iter(dataset.keys()))
    split = dataset[first_key].train_test_split(test_size=0.01, seed=42)
    return split["train"], split["test"]


def _tokenize_and_group(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    seq_len: int,
    num_proc: int,
) -> Dataset:
    text_col = _text_column(dataset)

    def nonempty(example: dict[str, Any]) -> bool:
        text = example[text_col]
        return text is not None and len(text.strip()) > 0

    dataset = dataset.filter(nonempty)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch[text_col], add_special_tokens=False, truncation=False)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated: dict[str, list[int]] = {}
        for key in batch:
            concatenated[key] = sum(batch[key], [])
        total_length = (len(concatenated["input_ids"]) // seq_len) * seq_len
        output = {
            key: [values[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for key, values in concatenated.items()
        }
        return output

    grouped = tokenized.map(group_texts, batched=True, num_proc=num_proc, desc="Grouping")
    grouped.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return grouped


class LanguageModelCollator:
    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def prepare_lm_datasets(config: DataConfig) -> tuple[Dataset, Dataset, AutoTokenizer]:
    splits, tokenizer = prepare_lm_dataset_splits(config)
    return splits["train"], splits["validation"], tokenizer


def prepare_lm_dataset_splits(config: DataConfig) -> tuple[dict[str, Dataset], AutoTokenizer]:
    spec = _resolve_dataset_spec(config.dataset_name)
    token = resolve_hf_token()
    tokenizer = load_tokenizer(config.tokenizer_name)

    if "data_files" in spec:
        dataset = load_dataset(
            spec["path"],
            data_files=spec["data_files"],
            token=token,
            trust_remote_code=config.trust_remote_code,
        )
        splits: dict[str, Dataset] = {}
        for split_name, limit in [
            ("train", config.max_train_texts),
            ("validation", config.max_val_texts),
            ("test", config.max_test_texts),
        ]:
            if split_name in dataset:
                splits[split_name] = _limit_split(dataset[split_name], limit)
        tokenized_splits = {
            name: _tokenize_and_group(split, tokenizer, config.seq_len, config.num_proc)
            for name, split in splits.items()
        }
        return tokenized_splits, tokenizer

    if spec.get("streaming", False):
        splits: dict[str, Dataset] = {}
        for split_name, limit in [
            ("train", config.max_train_texts),
            ("validation", config.max_val_texts),
            ("test", config.max_test_texts),
        ]:
            stream_ds = _collect_stream_rows(
                spec=spec,
                split_name=split_name,
                limit_override=limit,
                token=token,
                trust_remote_code=config.trust_remote_code,
            )
            if len(stream_ds) == 0:
                continue
            splits[split_name] = stream_ds
        tokenized_splits = {
            name: _tokenize_and_group(split, tokenizer, config.seq_len, config.num_proc)
            for name, split in splits.items()
        }
        return tokenized_splits, tokenizer

    if any(key in spec for key in ("train_split", "validation_split", "test_split")):
        splits: dict[str, Dataset] = {}
        for split_name, limit in [
            ("train", config.max_train_texts),
            ("validation", config.max_val_texts),
            ("test", config.max_test_texts),
        ]:
            split_spec = spec.get(f"{split_name}_split")
            if split_spec is None:
                continue
            split_dataset = load_dataset(
                spec["path"],
                spec.get("name"),
                split=split_spec,
                token=token,
                trust_remote_code=config.trust_remote_code,
            )
            splits[split_name] = _limit_split(split_dataset, limit)
        tokenized_splits = {
            name: _tokenize_and_group(split, tokenizer, config.seq_len, config.num_proc)
            for name, split in splits.items()
        }
        return tokenized_splits, tokenizer

    dataset = load_dataset(
        spec["path"],
        spec.get("name"),
        token=token,
        trust_remote_code=config.trust_remote_code,
    )

    train_split, val_split = _ensure_splits(dataset)
    splits: dict[str, Dataset] = {
        "train": _limit_split(train_split, config.max_train_texts),
        "validation": _limit_split(val_split, config.max_val_texts),
    }
    if "test" in dataset:
        splits["test"] = _limit_split(dataset["test"], config.max_test_texts)

    tokenized_splits = {
        name: _tokenize_and_group(split, tokenizer, config.seq_len, config.num_proc)
        for name, split in splits.items()
    }
    return tokenized_splits, tokenizer
