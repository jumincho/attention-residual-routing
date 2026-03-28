#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attnres_routing.routing import (  # noqa: E402
    continuation_loss_from_decode_logits,
    teacher_forced_decode_logits,
    teacher_forced_decode_timing,
)
from attnres_routing.sequence_manifest import load_manifest_jsonl  # noqa: E402


def load_ranker_module():
    script_path = ROOT / "scripts" / "train_candidate_conditioned_ranker_v5.py"
    spec = importlib.util.spec_from_file_location("ranker_v5_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ranker helper from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model_from_checkpoint(path: Path, device: torch.device, precision: str):
    payload = torch.load(path, map_location=device)
    sys.path.insert(0, str(ROOT / "src"))
    from attnres_routing.model import AttnResConfig, DecoderLM  # noqa: E402

    config = payload["config"]
    model = DecoderLM(AttnResConfig.from_dict(config["model"])).to(device)
    model.load_state_dict(payload["model_state"])
    if precision == "fp16" and device.type == "cuda":
        model = model.half()
    model.eval()
    return model, config


def fit_selector(
    ranker_module,
    bank_tag: str,
    bank_size: int,
    skip_count: int,
    feature_mode: str,
    selected_model: str,
    train_tags: list[str],
    eval_tags: list[str],
    hidden_train_tags: list[str] | None,
    hidden_eval_tags: list[str] | None,
    seed: int,
    train_doc_frac: float,
) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    bank_dir = ROOT / "results" / "bank_hygiene"
    oracle_dir = ROOT / "results" / "oracles"
    rich_dir = ROOT / "results" / "rich_features"

    bank_df = pd.read_csv(bank_dir / f"{bank_tag}_candidate_bank.csv")
    bank_eval_df = pd.read_csv(bank_dir / f"{bank_tag}_per_sequence.csv")
    train_features = ranker_module.concat_frames(oracle_dir, train_tags, "sequence_features")
    train_oracle = ranker_module.concat_frames(oracle_dir, train_tags, "oracle_mask_alignment")
    train_masks = ranker_module.concat_frames(oracle_dir, train_tags, "exhaustive_mask_losses")
    eval_features = ranker_module.concat_frames(oracle_dir, eval_tags, "sequence_features")
    eval_oracle = ranker_module.concat_frames(oracle_dir, eval_tags, "oracle_mask_alignment")
    eval_masks = ranker_module.concat_frames(oracle_dir, eval_tags, "exhaustive_mask_losses")
    train_hidden = ranker_module.concat_frames(rich_dir, hidden_train_tags, "hidden_prompt_features") if hidden_train_tags else None
    eval_hidden = ranker_module.concat_frames(rich_dir, hidden_eval_tags, "hidden_prompt_features") if hidden_eval_tags else None

    train_feature_table = ranker_module.build_feature_table(train_features, train_oracle, train_hidden)
    eval_feature_table = ranker_module.build_feature_table(eval_features, eval_oracle, eval_hidden)
    train_ds = ranker_module.build_ranker_dataset(train_feature_table, train_masks, bank_df, bank_size, skip_count, feature_mode)
    eval_ds = ranker_module.build_ranker_dataset(eval_feature_table, eval_masks, bank_df, bank_size, skip_count, feature_mode)

    rng = np.random.default_rng(seed)
    train_docs = np.asarray(sorted(train_ds.rows["document_idx"].unique().tolist()), dtype=int)
    rng.shuffle(train_docs)
    split_at = max(1, int(round(len(train_docs) * train_doc_frac)))
    fit_docs = set(train_docs[:split_at].tolist())
    train_rows = train_ds.rows[train_ds.rows["document_idx"].isin(fit_docs)].copy()
    eval_rows_df = eval_ds.rows.copy()

    x_train_pair = np.stack(train_rows["pair_features"].to_list(), axis=0)
    x_eval_pair = np.stack(eval_rows_df["pair_features"].to_list(), axis=0)
    x_train_prompt = np.stack([train_ds.prompt_vectors[int(seq)] for seq in train_rows["sequence_idx"].tolist()], axis=0)
    x_eval_prompt = np.stack([eval_ds.prompt_vectors[int(seq)] for seq in eval_rows_df["sequence_idx"].tolist()], axis=0)
    x_train_cand = np.stack(train_rows["candidate_features"].to_list(), axis=0)
    x_eval_cand = np.stack(eval_rows_df["candidate_features"].to_list(), axis=0)
    y_train = train_rows["delta_to_global_static"].to_numpy(dtype=np.float64)

    fit_obj: Any = None
    t0 = time.perf_counter()
    if selected_model == "hgb_pair":
        fit_obj = ranker_module.HistGradientBoostingRegressor(max_depth=6, max_iter=300, learning_rate=0.05, random_state=seed)
        fit_obj.fit(x_train_pair, y_train)
        preds = fit_obj.predict(x_eval_pair)
    elif selected_model == "rf_pair":
        fit_obj = ranker_module.RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        fit_obj.fit(x_train_pair, y_train)
        preds = fit_obj.predict(x_eval_pair)
    elif selected_model == "knn_prompt":
        preds = ranker_module.knn_predict(
            x_train_prompt,
            train_rows["mask_id"].to_numpy(dtype=object),
            y_train,
            x_eval_prompt,
            eval_rows_df["mask_id"].to_numpy(dtype=object),
        )
    elif selected_model == "mlp_late_fusion":
        fit_obj = ranker_module.train_mlp(x_train_prompt, x_train_cand, y_train, seed=seed)
        preds = ranker_module.predict_mlp(fit_obj[0], fit_obj[1], fit_obj[2], x_eval_prompt, x_eval_cand)
    else:
        raise ValueError(f"Unsupported selected_model for deployment measurement: {selected_model}")
    selector_seconds = time.perf_counter() - t0

    eval_rows_df[selected_model] = preds
    bank_upper = (
        bank_eval_df[(bank_eval_df["bank_size"] == bank_size) & (bank_eval_df["skip_count"] == skip_count)][
            ["sequence_idx", "bank_upper_bound_loss", "bank_best_mask_id"]
        ]
        .drop_duplicates(subset=["sequence_idx"])
        .set_index("sequence_idx")
    )
    selected_df = ranker_module.evaluate_selected_masks(
        rows_df=eval_rows_df,
        score_col=selected_model,
        global_static=eval_ds.global_static,
        bank_upper_df=bank_upper,
        output_tag="deployment_v6",
        feature_mode=feature_mode,
        skip_count=skip_count,
        bank_size=bank_size,
        model_name=selected_model,
    )
    global_static_lookup = (
        eval_ds.rows[eval_ds.rows["mask_id"] == eval_ds.global_static][["sequence_idx", "actual_loss"]]
        .drop_duplicates(subset=["sequence_idx"])
        .set_index("sequence_idx")["actual_loss"]
    )
    return selected_df, eval_ds, {
        "selector_seconds_total": selector_seconds,
        "global_static_mask_id": eval_ds.global_static,
        "global_static_loss_lookup": global_static_lookup,
    }


def manifest_subset(manifest_path: Path, sequence_ids: list[int], prompt_len: int, decode_len: int) -> list[dict[str, Any]]:
    manifest_rows = load_manifest_jsonl(manifest_path)
    lookup = {int(row["sequence_idx"]): row for row in manifest_rows}
    subset = []
    for seq_id in sequence_ids:
        row = lookup.get(int(seq_id))
        if row is None:
            continue
        input_ids = row["input_ids"]
        subset.append(
            {
                "sequence_idx": int(seq_id),
                "prompt_ids": torch.tensor(input_ids[:prompt_len], dtype=torch.long).unsqueeze(0),
                "continuation_ids": torch.tensor(input_ids[prompt_len : prompt_len + decode_len], dtype=torch.long).unsqueeze(0),
            }
        )
    return subset


def batch_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def evaluate_method_timing(
    model,
    method_name: str,
    grouped_rows: dict[str, list[dict[str, Any]]],
    mask_lookup: dict[str, np.ndarray | None],
    device: torch.device,
    batch_size: int,
    repeats: int,
    warmup: bool,
) -> dict[str, float]:
    repeat_rows = []
    for repeat_idx in range(repeats + (1 if warmup else 0)):
        prefill_total = 0.0
        decode_total = 0.0
        route_total = 0.0
        decode_tokens_total = 0
        seq_total = 0
        for mask_id, rows in grouped_rows.items():
            mask_array = mask_lookup[mask_id]
            mask_tensor = None if mask_array is None else torch.tensor(mask_array, device=device, dtype=torch.bool)
            for batch in batch_rows(rows, batch_size):
                prompt_ids = torch.cat([row["prompt_ids"].to(device) for row in batch], dim=0)
                continuation_ids = torch.cat([row["continuation_ids"].to(device) for row in batch], dim=0)
                timing = teacher_forced_decode_timing(
                    model,
                    prompt_ids=prompt_ids,
                    continuation_ids=continuation_ids,
                    active_block_mask=mask_tensor,
                )
                prefill_total += timing.prefill_seconds
                decode_total += timing.decode_seconds
                route_total += timing.routing_overhead_seconds
                decode_tokens_total += continuation_ids.size(0) * continuation_ids.size(1)
                seq_total += continuation_ids.size(0)
        if warmup and repeat_idx == 0:
            continue
        repeat_rows.append(
            {
                "method": method_name,
                "prefill_seconds_per_sequence": prefill_total / max(seq_total, 1),
                "decode_seconds_per_sequence": decode_total / max(seq_total, 1),
                "end_to_end_seconds_per_sequence": (prefill_total + decode_total + route_total) / max(seq_total, 1),
                "decode_tokens_per_sec": decode_tokens_total / max(decode_total, 1e-6),
                "routing_overhead_seconds_total": route_total,
                "num_sequences": seq_total,
            }
        )
    repeat_df = pd.DataFrame(repeat_rows)
    med = repeat_df.median(numeric_only=True).to_dict()
    med["method"] = method_name
    med["num_repeats"] = repeats
    return med


def evaluate_full_loss(
    model,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> tuple[float, dict[int, float]]:
    losses = {}
    for batch in batch_rows(rows, batch_size):
        prompt_ids = torch.cat([row["prompt_ids"].to(device) for row in batch], dim=0)
        continuation_ids = torch.cat([row["continuation_ids"].to(device) for row in batch], dim=0)
        _, decode_logits = teacher_forced_decode_logits(
            model,
            prompt_ids=prompt_ids,
            continuation_ids=continuation_ids,
        )
        token_losses = []
        for row, logits, cont in zip(batch, decode_logits, continuation_ids):
            loss = continuation_loss_from_decode_logits(logits.unsqueeze(0), cont.unsqueeze(0))
            losses[int(row["sequence_idx"])] = float(loss)
            token_losses.append(loss)
    return float(np.mean(list(losses.values()))), losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--bank-tag", type=str, required=True)
    parser.add_argument("--bank-size", type=int, default=32)
    parser.add_argument("--skip-count", type=int, required=True)
    parser.add_argument("--train-tags", type=str, nargs="+", required=True)
    parser.add_argument("--eval-tags", type=str, nargs="+", required=True)
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--feature-mode", type=str, default="attnres")
    parser.add_argument("--selected-model", type=str, default="rf_pair")
    parser.add_argument("--hidden-train-tags", type=str, nargs="*", default=None)
    parser.add_argument("--hidden-eval-tags", type=str, nargs="*", default=None)
    parser.add_argument("--output-tag", type=str, required=True)
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-doc-frac", type=float, default=0.8)
    args = parser.parse_args()

    ranker_module = load_ranker_module()
    device = torch.device(args.device)
    model, config = load_model_from_checkpoint(Path(args.checkpoint), device=device, precision=args.precision)
    prompt_len = int(config["data"]["seq_len"]) // 2
    prompt_len = int(config["data"].get("seq_len", prompt_len)) if "prompt_len" not in config["data"] else int(config["data"]["prompt_len"])
    if prompt_len == int(config["data"].get("seq_len", 0)):
        prompt_len = None
    if prompt_len is None:
        manifest_rows_tmp = load_manifest_jsonl(Path(args.manifest_path))
        prompt_len = int(manifest_rows_tmp[0]["prompt_len"])
        decode_len = int(manifest_rows_tmp[0]["decode_len"])
    else:
        manifest_rows_tmp = load_manifest_jsonl(Path(args.manifest_path))
        decode_len = int(manifest_rows_tmp[0]["decode_len"])

    selected_df, eval_ds, selector_meta = fit_selector(
        ranker_module=ranker_module,
        bank_tag=args.bank_tag,
        bank_size=args.bank_size,
        skip_count=args.skip_count,
        feature_mode=args.feature_mode,
        selected_model=args.selected_model,
        train_tags=args.train_tags,
        eval_tags=args.eval_tags,
        hidden_train_tags=args.hidden_train_tags,
        hidden_eval_tags=args.hidden_eval_tags,
        seed=args.seed,
        train_doc_frac=args.train_doc_frac,
    )

    chosen_seq_ids = sorted(selected_df["sequence_idx"].astype(int).tolist())[: args.num_sequences]
    subset_rows = manifest_subset(Path(args.manifest_path), chosen_seq_ids, prompt_len=prompt_len, decode_len=decode_len)
    chosen_seq_ids = [int(row["sequence_idx"]) for row in subset_rows]
    selected_subset = selected_df[selected_df["sequence_idx"].isin(chosen_seq_ids)].copy()

    global_static_mask_id = str(selector_meta["global_static_mask_id"])
    global_static_lookup = selector_meta["global_static_loss_lookup"]
    selected_subset["selected_loss"] = selected_subset.apply(
        lambda row: float(global_static_lookup.loc[int(row["sequence_idx"])]) + float(row["actual_delta_to_static"]),
        axis=1,
    )
    static_loss_mean = float(np.mean([float(global_static_lookup.loc[int(seq_id)]) for seq_id in chosen_seq_ids]))

    all_on_mask = np.ones(eval_ds.num_blocks, dtype=np.float32)
    global_static_mask = ranker_module.parse_mask_id(global_static_mask_id, eval_ds.num_blocks)
    dynamic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset_rows:
        mask_id = str(selected_subset[selected_subset["sequence_idx"] == int(row["sequence_idx"])].iloc[0]["selected_mask_id"])
        dynamic_groups[mask_id].append(row)

    full_groups = {"full": subset_rows}
    static_groups = {global_static_mask_id: subset_rows}
    mask_lookup = {"full": None, global_static_mask_id: global_static_mask}
    for mask_id in dynamic_groups:
        mask_lookup[mask_id] = ranker_module.parse_mask_id(mask_id, eval_ds.num_blocks)

    full_loss_mean, full_loss_lookup = evaluate_full_loss(
        model=model,
        rows=subset_rows,
        device=device,
        batch_size=args.batch_size,
    )
    dynamic_loss_mean = float(selected_subset["selected_loss"].mean())

    timing_rows = [
        evaluate_method_timing(
            model=model,
            method_name="full_model",
            grouped_rows=full_groups,
            mask_lookup=mask_lookup,
            device=device,
            batch_size=args.batch_size,
            repeats=args.timing_repeats,
            warmup=not args.no_warmup,
        ),
        evaluate_method_timing(
            model=model,
            method_name="global_static",
            grouped_rows=static_groups,
            mask_lookup=mask_lookup,
            device=device,
            batch_size=args.batch_size,
            repeats=args.timing_repeats,
            warmup=not args.no_warmup,
        ),
        evaluate_method_timing(
            model=model,
            method_name="dynamic_selector",
            grouped_rows=dynamic_groups,
            mask_lookup=mask_lookup,
            device=device,
            batch_size=args.batch_size,
            repeats=args.timing_repeats,
            warmup=not args.no_warmup,
        ),
    ]

    selector_total = float(selector_meta["selector_seconds_total"]) * (len(chosen_seq_ids) / max(len(selected_df), 1))
    for row in timing_rows:
        method = row["method"]
        if method == "full_model":
            mean_loss = full_loss_mean
            delta_to_static = full_loss_mean - static_loss_mean
        elif method == "global_static":
            mean_loss = static_loss_mean
            delta_to_static = 0.0
        else:
            mean_loss = dynamic_loss_mean
            delta_to_static = dynamic_loss_mean - static_loss_mean
        row["mean_continuation_loss"] = mean_loss
        row["delta_to_full_model"] = mean_loss - full_loss_mean
        row["delta_to_global_static"] = delta_to_static
        row["selector_overhead_seconds_total"] = selector_total if method == "dynamic_selector" else 0.0
        row["route_count"] = len(dynamic_groups) if method == "dynamic_selector" else 1

    out_dir = ROOT / "results" / "deployment_measurement_v6"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(timing_rows)
    summary_df.insert(0, "output_tag", args.output_tag)
    summary_df.insert(1, "feature_mode", args.feature_mode)
    summary_df.insert(2, "selected_model", args.selected_model)
    summary_df.insert(3, "skip_count", args.skip_count)
    summary_df.insert(4, "bank_size", args.bank_size)
    summary_df.to_csv(out_dir / f"{args.output_tag}_summary.csv", index=False)

    per_seq = pd.DataFrame(
        {
            "sequence_idx": chosen_seq_ids,
            "full_model_loss": [float(full_loss_lookup[int(seq_id)]) for seq_id in chosen_seq_ids],
            "global_static_loss": [float(global_static_lookup.loc[int(seq_id)]) for seq_id in chosen_seq_ids],
        }
    )
    dynamic_lookup = selected_subset.set_index("sequence_idx")
    per_seq["dynamic_mask_id"] = [str(dynamic_lookup.loc[int(seq_id), "selected_mask_id"]) for seq_id in chosen_seq_ids]
    per_seq["dynamic_loss"] = [float(dynamic_lookup.loc[int(seq_id), "selected_loss"]) for seq_id in chosen_seq_ids]
    per_seq.to_csv(out_dir / f"{args.output_tag}_per_sequence.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.2, 4.2))
        plt.scatter(
            summary_df["decode_tokens_per_sec"],
            summary_df["delta_to_full_model"],
            s=60,
        )
        for _, row in summary_df.iterrows():
            plt.text(float(row["decode_tokens_per_sec"]), float(row["delta_to_full_model"]), str(row["method"]))
        plt.xlabel("decode tokens/sec")
        plt.ylabel("delta to full model loss")
        plt.tight_layout()
        plt.savefig(plot_dir / f"deployment_measurement_v6_{args.output_tag}_quality_speed.png", dpi=160)
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
