#!/usr/bin/env python3
"""Run GPT-OSS SpecPrefill with plain HF Transformers prefill only.

This bypasses SGLang, vLLM, TRT-LLM, and OpenAI-compatible HTTP servers.  The
target model is called directly with ``use_cache=True`` and no decode/generate
step, so the measured target time is the prefill forward pass that builds the
KV cache for the prompt.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from specprefill_core import (
        build_target_chunks,
        compress_prompt,
        mean,
        rouge_l_f1,
    )
    from run_gpt_oss_specprefill import (
        LogitSurpriseScorer,
        fit_prompt_to_token_budget,
        generate_diverse_nonce_prompt,
        load_quality_prompts,
        load_tokenizer,
        parse_csv_numbers,
        score_and_compress,
    )
except ImportError:
    from .specprefill_core import (
        build_target_chunks,
        compress_prompt,
        mean,
        rouge_l_f1,
    )
    from .run_gpt_oss_specprefill import (
        LogitSurpriseScorer,
        fit_prompt_to_token_budget,
        generate_diverse_nonce_prompt,
        load_quality_prompts,
        load_tokenizer,
        parse_csv_numbers,
        score_and_compress,
    )


@dataclass
class PrefillResult:
    ok: bool
    prefill_s: float
    prompt_tokens: int
    device: str
    error: str | None = None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_max_memory(text: str) -> dict[int | str, str] | None:
    if not text:
        return None
    result: dict[int | str, str] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        result[int(key) if key.isdigit() else key] = value.strip()
    return result


def dtype_from_name(torch, name: str):
    table = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"unknown dtype: {name}")
    return table[name]


def cuda_sync_all(torch) -> None:
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(idx)


class VanillaPrefillTarget:
    def __init__(self, args) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        dtype = dtype_from_name(torch, args.target_dtype)
        max_memory = parse_max_memory(args.target_max_memory)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": args.trust_remote_code,
            "device_map": args.target_device_map,
            "low_cpu_mem_usage": True,
        }
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = "auto"
        if max_memory:
            model_kwargs["max_memory"] = max_memory

        print(
            f"Loading target for vanilla prefill: {args.target_model} "
            f"(device_map={args.target_device_map})",
            flush=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(args.target_model, **model_kwargs)
        self.model.eval()
        self.device = self._first_parameter_device()
        print(f"Target input device: {self.device}", flush=True)
        self.include_lm_head = args.include_lm_head
        self.prefill_module = self.model if args.include_lm_head else getattr(self.model, "model", self.model)

    def _first_parameter_device(self) -> str:
        for param in self.model.parameters():
            return str(param.device)
        return "cpu"

    def prefill(self, prompt: str, tokenizer, empty_cache: bool = True) -> PrefillResult:
        torch = self.torch
        try:
            encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            token_count = int(input_ids.shape[-1])
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            cuda_sync_all(torch)
            t0 = time.perf_counter()
            with torch.inference_mode():
                output = self.prefill_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                # Touch one scalar so lazy host-side wrappers cannot skip materialization.
                if self.include_lm_head and hasattr(output, "logits"):
                    touched = output.logits[:, -1, :1]
                elif hasattr(output, "last_hidden_state"):
                    touched = output.last_hidden_state[:, -1, :1]
                else:
                    touched = output[0][:, -1, :1]
                _ = float(touched.float().detach().cpu().item())
            cuda_sync_all(torch)
            elapsed = time.perf_counter() - t0

            del output, input_ids, attention_mask, encoded
            if empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return PrefillResult(
                ok=True,
                prefill_s=elapsed,
                prompt_tokens=token_count,
                device=self.device,
            )
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return PrefillResult(
                ok=False,
                prefill_s=0.0,
                prompt_tokens=0,
                device=self.device,
                error=f"{type(exc).__name__}: {exc}",
            )


def fmt_seconds(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}s"


def fmt_status(result: PrefillResult) -> str:
    return "ok" if result.ok else (result.error or "failed").replace("\n", " ")[:180]


def skipped_prefill(device: str, reason: str) -> PrefillResult:
    return PrefillResult(
        ok=False,
        prefill_s=0.0,
        prompt_tokens=0,
        device=device,
        error=f"skipped: {reason}",
    )


def run_prefill_rows(args, tokenizer, scorer: LogitSurpriseScorer, target: VanillaPrefillTarget):
    prompt_targets = parse_csv_numbers(args.prompt_tokens, int)
    keep_rates = parse_csv_numbers(args.keep_rates, float)
    if args.prefill_side not in {"both", "full", "compressed"}:
        raise ValueError("--prefill-side must be one of: both, full, compressed")
    records = []
    for target_tokens in prompt_targets:
        for keep_rate in keep_rates:
            total_runs = args.warmups + args.measurements
            for run_idx in range(total_runs):
                phase = "warmup" if run_idx < args.warmups else "measure"
                nonce = f"vanilla-prefill-{target_tokens}-{keep_rate}-{run_idx}-{utc_stamp()}"
                prompt = generate_diverse_nonce_prompt(target_tokens, tokenizer, nonce)
                compression, scorer_meta, scorer_s = score_and_compress(
                    prompt, tokenizer, scorer, args, keep_rate
                )
                full = (
                    target.prefill(prompt, tokenizer, empty_cache=args.empty_cache)
                    if args.prefill_side in {"both", "full"}
                    else skipped_prefill(target.device, "prefill_side=compressed")
                )
                compressed = (
                    target.prefill(
                        compression.compressed_prompt,
                        tokenizer,
                        empty_cache=args.empty_cache,
                    )
                    if args.prefill_side in {"both", "compressed"}
                    else skipped_prefill(target.device, "prefill_side=full")
                )
                record = {
                    "mode": "vanilla_prefill",
                    "phase": phase,
                    "target_prompt_tokens": target_tokens,
                    "keep_rate": keep_rate,
                    "run_index": run_idx,
                    "nonce": nonce,
                    "full": asdict(full),
                    "compressed": asdict(compressed),
                    "scorer_s": scorer_s,
                    "compressed_e2e_s": (
                        scorer_s + compressed.prefill_s if compressed.ok else None
                    ),
                    "prefill_speedup": (
                        full.prefill_s / compressed.prefill_s
                        if full.ok and compressed.ok and compressed.prefill_s > 0
                        else None
                    ),
                    "e2e_speedup": (
                        full.prefill_s / (scorer_s + compressed.prefill_s)
                        if full.ok and compressed.ok and scorer_s + compressed.prefill_s > 0
                        else None
                    ),
                    "compression": {
                        "original_target_tokens": compression.original_target_tokens,
                        "selected_target_tokens": compression.selected_target_tokens,
                        "compressed_target_tokens": compression.compressed_target_tokens,
                        "effective_keep_rate": compression.effective_keep_rate,
                        "selected_chunks": len(compression.selected_indices),
                        "total_chunks": len(compression.chunks),
                    },
                    "scorer": scorer_meta,
                }
                records.append(record)
                print(
                    f"[{phase}] L={target_tokens} keep={keep_rate:.2f} "
                    f"full_prefill={fmt_seconds(full.prefill_s if full.ok else None)} "
                    f"comp_prefill={fmt_seconds(compressed.prefill_s if compressed.ok else None)} "
                    f"scorer={fmt_seconds(scorer_s)} "
                    f"e2e={fmt_seconds(record['compressed_e2e_s'])} "
                    f"tokens={compression.original_target_tokens}->{compression.compressed_target_tokens} "
                    f"full_status={fmt_status(full)} comp_status={fmt_status(compressed)}",
                    flush=True,
                )
    measured = [r for r in records if r["phase"] == "measure"]
    summary = []
    for target_tokens in prompt_targets:
        for keep_rate in keep_rates:
            group = [
                r
                for r in measured
                if r["target_prompt_tokens"] == target_tokens
                and r["keep_rate"] == keep_rate
                and (args.prefill_side == "compressed" or r["full"]["ok"])
                and (args.prefill_side == "full" or r["compressed"]["ok"])
            ]
            if not group:
                summary.append(
                    {
                        "target_prompt_tokens": target_tokens,
                        "keep_rate": keep_rate,
                        "error": "no valid runs",
                    }
                )
                continue
            full_times = [r["full"]["prefill_s"] for r in group if r["full"]["ok"]]
            comp_times = [
                r["compressed"]["prefill_s"] for r in group if r["compressed"]["ok"]
            ]
            scorer_times = [r["scorer_s"] for r in group]
            e2e_times = [r["compressed_e2e_s"] for r in group if r["compressed_e2e_s"]]
            item = {
                "target_prompt_tokens": target_tokens,
                "keep_rate": keep_rate,
                "n": len(group),
                "prefill_side": args.prefill_side,
                "scorer_s_mean": statistics.mean(scorer_times),
                "avg_original_target_tokens": mean(
                    r["compression"]["original_target_tokens"] for r in group
                ),
                "avg_compressed_target_tokens": mean(
                    r["compression"]["compressed_target_tokens"] for r in group
                ),
            }
            if full_times:
                item["full_prefill_s_mean"] = statistics.mean(full_times)
                item["full_prefill_s_std"] = (
                    statistics.stdev(full_times) if len(full_times) > 1 else 0.0
                )
            if comp_times:
                item["compressed_prefill_s_mean"] = statistics.mean(comp_times)
                item["compressed_prefill_s_std"] = (
                    statistics.stdev(comp_times) if len(comp_times) > 1 else 0.0
                )
            if full_times and comp_times:
                item["prefill_speedup"] = statistics.mean(full_times) / statistics.mean(comp_times)
            if full_times and e2e_times:
                item["compressed_e2e_s_mean"] = statistics.mean(e2e_times)
                item["e2e_speedup"] = statistics.mean(full_times) / statistics.mean(e2e_times)
            elif e2e_times:
                item["compressed_e2e_s_mean"] = statistics.mean(e2e_times)
            summary.append(item)
    return {
        "kind": "vanilla_prefill",
        "created_at": utc_stamp(),
        "config": vars(args),
        "records": records,
        "summary": summary,
    }


def run_quality_proxy_rows(
    args,
    tokenizer,
    scorer: LogitSurpriseScorer,
    target: VanillaPrefillTarget | None,
):
    keep_rates = parse_csv_numbers(args.quality_keep_rates, float)
    prompts = load_quality_prompts(args, tokenizer)
    records = []
    for idx, prompt in enumerate(prompts):
        prompt = fit_prompt_to_token_budget(prompt, tokenizer, args.quality_prompt_tokens)
        chunks = build_target_chunks(prompt, tokenizer, args.chunk_tokens)
        score_t0 = time.perf_counter()
        scorer_meta = scorer.score_prompt(prompt, chunks)
        scorer_s = time.perf_counter() - score_t0
        full_prefill = target.prefill(prompt, tokenizer, args.empty_cache) if target else None
        for keep_rate in keep_rates:
            compression = compress_prompt(
                prompt,
                tokenizer,
                chunks,
                keep_rate=keep_rate,
                anchor_tokens=args.anchor_tokens,
                delimiter=args.delimiter,
            )
            compressed_prefill = (
                target.prefill(compression.compressed_prompt, tokenizer, args.empty_cache)
                if target
                else None
            )
            prompt_rouge_l = rouge_l_f1(compression.compressed_prompt, prompt)
            record = {
                "mode": "vanilla_quality_proxy",
                "prompt_index": idx,
                "keep_rate": keep_rate,
                "rouge_l_f1_compressed_prompt_vs_full_prompt": prompt_rouge_l,
                "full_prefill": asdict(full_prefill) if full_prefill else None,
                "compressed_prefill": asdict(compressed_prefill)
                if compressed_prefill
                else None,
                "scorer_s": scorer_s,
                "compression": {
                    "original_target_tokens": compression.original_target_tokens,
                    "selected_target_tokens": compression.selected_target_tokens,
                    "compressed_target_tokens": compression.compressed_target_tokens,
                    "effective_keep_rate": compression.effective_keep_rate,
                    "selected_chunks": len(compression.selected_indices),
                    "total_chunks": len(compression.chunks),
                },
                "scorer": scorer_meta,
                "note": (
                    "No generation was run. This is a prompt-compression proxy, "
                    "not QA answer ROUGE-L."
                ),
            }
            records.append(record)
            comp_s = compressed_prefill.prefill_s if compressed_prefill and compressed_prefill.ok else None
            print(
                f"[quality-proxy] i={idx} keep={keep_rate:.2f} "
                f"prompt_rougeL={prompt_rouge_l:.3f} "
                f"comp_prefill={fmt_seconds(comp_s)} "
                f"tokens={compression.original_target_tokens}->{compression.compressed_target_tokens}",
                flush=True,
            )
    summary = []
    for keep_rate in keep_rates:
        group = [r for r in records if r["keep_rate"] == keep_rate]
        if not group:
            summary.append({"keep_rate": keep_rate, "error": "no valid prompts"})
            continue
        row: dict[str, Any] = {
            "keep_rate": keep_rate,
            "n": len(group),
            "rouge_l_f1_compressed_prompt_vs_full_prompt_mean": statistics.mean(
                r["rouge_l_f1_compressed_prompt_vs_full_prompt"] for r in group
            ),
            "scorer_s_mean": statistics.mean(r["scorer_s"] for r in group),
            "avg_original_target_tokens": mean(
                r["compression"]["original_target_tokens"] for r in group
            ),
            "avg_compressed_target_tokens": mean(
                r["compression"]["compressed_target_tokens"] for r in group
            ),
            "note": "No generation was run; this is not the original QA quality row.",
        }
        valid_prefills = [
            r for r in group if r["compressed_prefill"] and r["compressed_prefill"]["ok"]
        ]
        if valid_prefills:
            row["compressed_prefill_s_mean"] = statistics.mean(
                r["compressed_prefill"]["prefill_s"] for r in valid_prefills
            )
        summary.append(row)
    return {
        "kind": "vanilla_quality_proxy",
        "created_at": utc_stamp(),
        "config": vars(args),
        "records": records,
        "summary": summary,
    }


def write_result(output_dir: str, prefix: str, payload: dict[str, Any]) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{prefix}_{utc_stamp()}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prefill", "quality-proxy", "all"], default="prefill")
    parser.add_argument("--target-model", default=os.getenv("TARGET_MODEL_ID", "openai/gpt-oss-120b"))
    parser.add_argument(
        "--target-tokenizer",
        default=os.getenv("TARGET_TOKENIZER_ID", os.getenv("TARGET_MODEL_ID", "openai/gpt-oss-120b")),
    )
    parser.add_argument("--target-device-map", default=os.getenv("TARGET_DEVICE_MAP", "auto"))
    parser.add_argument("--target-max-memory", default=os.getenv("TARGET_MAX_MEMORY", ""))
    parser.add_argument("--target-dtype", default=os.getenv("TARGET_DTYPE", "auto"))
    parser.add_argument("--attn-implementation", default=os.getenv("ATTN_IMPLEMENTATION", "eager"))
    parser.add_argument("--scorer-model", default=os.getenv("SCORER_MODEL_ID", "openai/gpt-oss-20b"))
    parser.add_argument("--scorer-device", default=os.getenv("SCORER_DEVICE", "cuda:0"))
    parser.add_argument("--scorer-dtype", default=os.getenv("SCORER_DTYPE", "auto"))
    parser.add_argument("--score-window", type=int, default=int(os.getenv("SCORE_WINDOW", "2048")))
    parser.add_argument("--chunk-tokens", type=int, default=int(os.getenv("CHUNK_TOKENS", "128")))
    parser.add_argument("--anchor-tokens", type=int, default=int(os.getenv("ANCHOR_TOKENS", "512")))
    parser.add_argument("--delimiter", default=os.getenv("SPEC_DELIMITER", ""))
    parser.add_argument("--keep-rates", default=os.getenv("KEEP_RATES", "0.3"))
    parser.add_argument("--prompt-tokens", default=os.getenv("PROMPT_TOKENS", "16000,32000"))
    parser.add_argument(
        "--prefill-side",
        choices=["both", "full", "compressed"],
        default=os.getenv("PREFILL_SIDE", "both"),
        help="Measure both prompts, only the full prompt, or only the compressed prompt.",
    )
    parser.add_argument("--measurements", type=int, default=int(os.getenv("MEASUREMENTS", "1")))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "0")))
    parser.add_argument("--empty-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-lm-head",
        action="store_true",
        help="Also run the CausalLM lm_head/logits path. Default is pure transformer prefill.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "results/vanilla_prefill"))

    parser.add_argument("--quality-source", choices=["longbench", "synthetic"], default="longbench")
    parser.add_argument("--quality-jsonl", default=os.getenv("QUALITY_JSONL", ""))
    parser.add_argument("--dataset-name", default=os.getenv("DATASET_NAME", "THUDM/LongBench-v2"))
    parser.add_argument("--dataset-split", default=os.getenv("DATASET_SPLIT", "test"))
    parser.add_argument("--quality-n", type=int, default=int(os.getenv("QUALITY_N", "20")))
    parser.add_argument(
        "--quality-prompt-tokens",
        type=int,
        default=int(os.getenv("QUALITY_PROMPT_TOKENS", "16000")),
    )
    parser.add_argument("--quality-keep-rates", default=os.getenv("QUALITY_KEEP_RATES", "0.3,0.5"))
    parser.add_argument(
        "--skip-quality-target-prefill",
        action="store_true",
        help="For quality-proxy rows, only score/compress prompts and skip target forward passes.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"Loading target tokenizer: {args.target_tokenizer}", flush=True)
    tokenizer = load_tokenizer(args.target_tokenizer, args.trust_remote_code)
    print(
        f"Loading scorer: {args.scorer_model} on {args.scorer_device} "
        f"(window={args.score_window})",
        flush=True,
    )
    scorer = LogitSurpriseScorer(
        args.scorer_model,
        args.scorer_device,
        args.scorer_dtype,
        args.score_window,
        args.trust_remote_code,
    )
    target = VanillaPrefillTarget(args)

    paths = []
    if args.mode in ("prefill", "all"):
        payload = run_prefill_rows(args, tokenizer, scorer, target)
        paths.append(write_result(args.output_dir, "vanilla_prefill_gpt_oss_120b", payload))
        print(json.dumps(payload["summary"], indent=2), flush=True)

    if args.mode in ("quality-proxy", "all"):
        quality_target = None if args.skip_quality_target_prefill else target
        payload = run_quality_proxy_rows(args, tokenizer, scorer, quality_target)
        paths.append(write_result(args.output_dir, "vanilla_quality_proxy_gpt_oss_120b", payload))
        print(json.dumps(payload["summary"], indent=2), flush=True)

    for path in paths:
        print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
