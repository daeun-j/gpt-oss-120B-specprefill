#!/usr/bin/env python3
"""Reproduce PAG-style SpecPrefill rows on gpt-oss-120B.

Expected stack:
  - Target: gpt-oss-120B, OpenAI-compatible API
  - Scorer/draft: openai/gpt-oss-20b or an override supplied by SCORER_MODEL_ID

The script intentionally sends full and compressed prompts to the same target
server. This isolates the prefill-length effect without needing two target
servers.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

try:
    from specprefill_core import (
        PromptChunk,
        assign_scores_by_chunk_text,
        assign_scores_from_offsets,
        build_target_chunks,
        compress_prompt,
        mean,
        rouge_l_f1,
    )
except ImportError:
    from .specprefill_core import (
        PromptChunk,
        assign_scores_by_chunk_text,
        assign_scores_from_offsets,
        build_target_chunks,
        compress_prompt,
        mean,
        rouge_l_f1,
    )


BASE_PASSAGES = [
    """Distributed inference systems split large language model serving into
prefill, decode, scheduling, and memory management stages. Time to first token
is dominated by the prefill stage when prompts reach tens of thousands of
tokens, because every token must be embedded, positioned, attended over, and
written into the key-value cache before the first decode step can begin.""",
    """Speculative prefill compresses a prompt before it reaches the target
model. A smaller scorer estimates which chunks carry the most useful
information. The target model then receives only a subset of the original
prompt, reducing the amount of attention work and KV-cache allocation needed
for prefill.""",
    """The quality risk is that retrieval or question-answering prompts often
depend on small details. If a compressor removes the paragraph containing the
answer, the target model can produce fluent but unsupported output. Therefore
speed measurements must be paired with full-prompt answer agreement or
task-level accuracy.""",
    """Modern serving engines such as SGLang and vLLM include chunked prefill,
continuous batching, prefix caching, and optimized CUDA kernels. Benchmarking
must avoid accidental prefix-cache reuse by adding unique nonces or disabling
cache features when the goal is fresh prefill latency.""",
]


def tokenizer_candidates(model_id: str) -> list[str]:
    candidates = [model_id]
    fallback = os.getenv("TOKENIZER_FALLBACK_ID")
    if fallback:
        candidates.append(fallback)
    model_text = str(model_id).rstrip("/")
    if "openai__gpt-oss-120b" in model_text:
        candidates.append("openai/gpt-oss-120b")
    if "openai__gpt-oss-20b" in model_text:
        candidates.append("openai/gpt-oss-20b")
    # Keep order while removing duplicates.
    return list(dict.fromkeys(candidates))


def auto_tokenizer_from_pretrained(model_id: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    errors = []
    for candidate in tokenizer_candidates(model_id):
        try:
            return AutoTokenizer.from_pretrained(
                candidate,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
        except Exception as fast_exc:
            errors.append(f"{candidate} fast: {type(fast_exc).__name__}: {fast_exc}")
            try:
                return AutoTokenizer.from_pretrained(
                    candidate,
                    trust_remote_code=trust_remote_code,
                    use_fast=False,
                )
            except Exception as slow_exc:
                errors.append(f"{candidate} slow: {type(slow_exc).__name__}: {slow_exc}")
    raise RuntimeError(
        "Could not load tokenizer from any candidate. Install tokenizer extras "
        "with `python3 -m pip install -U tiktoken sentencepiece`, or set "
        "TOKENIZER_FALLBACK_ID=openai/gpt-oss-120b. Errors: "
        + " | ".join(error.replace("\n", " ")[:240] for error in errors)
    )


@dataclass
class StreamResult:
    ok: bool
    ttft_s: float
    total_s: float
    text: str
    usage: dict[str, Any] | None = None
    first_event_s: float | None = None
    error: str | None = None


class LogitSurpriseScorer:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        score_window: int,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = device
        self.score_window = score_window
        self.tokenizer = auto_tokenizer_from_pretrained(model_id, trust_remote_code)

        torch_dtype = {
            "auto": torch.bfloat16 if device.startswith("cuda") else torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[dtype]
        if device.startswith("cuda") and importlib.util.find_spec("accelerate") is None:
            raise RuntimeError(
                "GPU scorer needs accelerate for Transformers device_map. "
                "Run: python3 -m pip install -U accelerate"
            )
        device_map = {"": device} if device.startswith("cuda") else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=os.getenv(
                "SCORER_ATTN_IMPLEMENTATION",
                os.getenv("ATTN_IMPLEMENTATION", "eager"),
            ),
        )
        if not device.startswith("cuda"):
            self.model.to(device)
        self.model.eval()

    def _encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded["input_ids"])
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        return input_ids, offsets

    def _score_token_ids(self, input_ids: list[int]) -> list[float]:
        torch = self.torch
        scores = [0.0] * len(input_ids)
        if len(input_ids) < 2:
            return scores

        with torch.inference_mode():
            for start in range(0, len(input_ids), self.score_window):
                end = min(start + self.score_window, len(input_ids))
                if end - start < 2:
                    continue
                window_ids = torch.tensor(
                    [input_ids[start:end]],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = self.model(input_ids=window_ids).logits[:, :-1, :]
                labels = window_ids[:, 1:]
                losses = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                )
                for local_idx, loss in enumerate(losses.detach().cpu().tolist(), start=1):
                    scores[start + local_idx] = float(loss)
                del logits, labels, losses, window_ids
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        scores[0] = scores[1]
        return scores

    def score_prompt(self, prompt: str, chunks: list[PromptChunk]) -> dict[str, Any]:
        try:
            input_ids, offsets = self._encode_with_offsets(prompt)
            token_scores = self._score_token_ids(input_ids)
            assign_scores_from_offsets(chunks, offsets, token_scores)
            return {
                "method": "logit_surprise_offsets",
                "scorer_tokens": len(input_ids),
                "fallback": False,
            }
        except Exception as exc:
            chunk_scores = []
            scorer_tokens = 0
            for chunk in chunks:
                ids = self.tokenizer.encode(chunk.text, add_special_tokens=False)
                scorer_tokens += len(ids)
                losses = self._score_token_ids(list(ids))
                chunk_scores.append(mean(losses) if losses else 0.0)
            assign_scores_by_chunk_text(chunks, chunk_scores)
            return {
                "method": "logit_surprise_chunk_text",
                "scorer_tokens": scorer_tokens,
                "fallback": True,
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            }


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_csv_numbers(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def load_tokenizer(model_id: str, trust_remote_code: bool):
    return auto_tokenizer_from_pretrained(model_id, trust_remote_code)


def encode_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def decode_ids(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def generate_diverse_nonce_prompt(target_tokens: int, tokenizer, nonce: str) -> str:
    prefix = (
        "You are evaluating a long-context prefill benchmark. The document below "
        "contains technical notes, numbered observations, and unique nonce markers. "
        f"Run nonce: {nonce}.\n\n--- BEGIN DOCUMENT ---\n"
    )
    suffix = (
        "\n--- END DOCUMENT ---\n\n"
        "Using only the document above, summarize the three most important "
        "implementation risks for speculative prefill. Keep the answer concise."
    )
    reserved = encode_len(tokenizer, prefix + suffix)
    body_budget = max(256, target_tokens - reserved)

    rng = random.Random(nonce)
    blocks = []
    idx = 0
    while encode_len(tokenizer, "\n".join(blocks)) < body_budget + 512:
        passage = BASE_PASSAGES[idx % len(BASE_PASSAGES)]
        salt = rng.getrandbits(64)
        blocks.append(
            f"[section={idx:05d} nonce={nonce}-{salt:016x}]\n"
            f"{passage}\n"
            "Observation: preserve exact measurements for scorer time, target "
            "prefill time, compressed token count, and full prompt token count.\n"
        )
        idx += 1

    body_ids = tokenizer.encode("\n".join(blocks), add_special_tokens=False)
    body = decode_ids(tokenizer, body_ids[:body_budget])
    return prefix + body + suffix


def build_quality_prompt(row: dict[str, Any]) -> str | None:
    def first(names: list[str]):
        for name in names:
            if name in row and row[name] not in (None, ""):
                value = row[name]
                if isinstance(value, list):
                    return "\n\n".join(str(v) for v in value)
                if isinstance(value, dict):
                    return "\n".join(f"{k}: {v}" for k, v in value.items())
                return str(value)
        return None

    context = first(["context", "contexts", "document", "documents", "article", "passage", "input"])
    question = first(["question", "query", "prompt", "instruction"])
    choices = first(["choices", "options"])
    if not context and not question:
        return None
    if context and question:
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    else:
        prompt = context or question or ""
    if choices:
        prompt += f"\n\nChoices:\n{choices}"
    prompt += "\n\nAnswer the question using the context. Be concise."
    return prompt


def fit_prompt_to_token_budget(prompt: str, tokenizer, target_tokens: int) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= target_tokens:
        return prompt

    marker = "\n\nQuestion:"
    split_at = prompt.rfind(marker)
    if split_at > 0:
        context_part = prompt[:split_at]
        suffix = prompt[split_at:]
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        context_budget = max(64, target_tokens - len(suffix_ids))
        context_ids = tokenizer.encode(context_part, add_special_tokens=False)
        return decode_ids(tokenizer, context_ids[:context_budget]) + suffix

    return decode_ids(tokenizer, list(ids)[:target_tokens])


def synthetic_quality_prompts(n: int, target_tokens: int, tokenizer) -> list[str]:
    prompts = []
    for i in range(n):
        nonce = f"quality-{i:04d}-{utc_stamp()}"
        base = generate_diverse_nonce_prompt(target_tokens, tokenizer, nonce)
        prompts.append(
            base
            + "\n\nAdditional question: Which nonce-indexed sections discuss "
            "quality risk, and what failure mode do they warn about?"
        )
    return prompts


def load_quality_prompts(args, tokenizer) -> list[str]:
    if args.quality_jsonl:
        prompts = []
        with open(args.quality_jsonl, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt") if isinstance(row, dict) else None
                if not prompt and isinstance(row, dict):
                    prompt = build_quality_prompt(row)
                if prompt:
                    prompts.append(fit_prompt_to_token_budget(prompt, tokenizer, args.quality_prompt_tokens))
                if len(prompts) >= args.quality_n:
                    break
        return prompts

    if args.quality_source == "synthetic":
        return synthetic_quality_prompts(args.quality_n, args.quality_prompt_tokens, tokenizer)

    from datasets import load_dataset

    split_candidates = [args.dataset_split, "test", "validation", "dev", "train"]
    last_error = None
    dataset = None
    for split in dict.fromkeys(split_candidates):
        try:
            dataset = load_dataset(args.dataset_name, split=split)
            break
        except Exception as exc:
            last_error = exc
    if dataset is None:
        raise RuntimeError(f"Could not load {args.dataset_name}: {last_error}")

    prompts = []
    for row in dataset:
        prompt = build_quality_prompt(dict(row))
        if not prompt:
            continue
        prompts.append(fit_prompt_to_token_budget(prompt, tokenizer, args.quality_prompt_tokens))
        if len(prompts) >= args.quality_n:
            break
    if not prompts:
        raise RuntimeError(f"No usable prompts found in {args.dataset_name}")
    return prompts


async def check_endpoint(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    status: dict[str, Any] = {"base_url": base_url, "ok": False}
    try:
        resp = await client.get(f"{base_url}/v1/models", timeout=20)
        status["models_status"] = resp.status_code
        if resp.status_code == 200:
            status["ok"] = True
            status["models"] = resp.json()
            return status
    except Exception as exc:
        status["models_error"] = f"{type(exc).__name__}: {exc}"
    try:
        resp = await client.get(f"{base_url}/health", timeout=20)
        status["health_status"] = resp.status_code
        status["health_text"] = resp.text[:200]
        status["ok"] = resp.status_code == 200
    except Exception as exc:
        status["health_error"] = f"{type(exc).__name__}: {exc}"
    return status


async def call_openai_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_timeout: float,
) -> StreamResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first_event = None
    first_token = None
    text_parts: list[str] = []
    usage = None
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=request_timeout,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return StreamResult(
                    ok=False,
                    ttft_s=0.0,
                    total_s=time.perf_counter() - t0,
                    text="",
                    error=f"HTTP {resp.status_code}: {body[:500]!r}",
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                if first_event is None:
                    first_event = time.perf_counter()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if data.get("usage"):
                    usage = data["usage"]
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("text")
                    or ""
                )
                if piece:
                    if first_token is None:
                        first_token = time.perf_counter()
                    text_parts.append(piece)
        t1 = time.perf_counter()
        ttft_anchor = first_token or first_event or t1
        return StreamResult(
            ok=True,
            ttft_s=ttft_anchor - t0,
            total_s=t1 - t0,
            first_event_s=(first_event - t0) if first_event else None,
            text="".join(text_parts),
            usage=usage,
        )
    except Exception as exc:
        return StreamResult(
            ok=False,
            ttft_s=0.0,
            total_s=time.perf_counter() - t0,
            text="",
            error=f"{type(exc).__name__}: {exc}",
        )


def score_and_compress(prompt: str, tokenizer, scorer: LogitSurpriseScorer, args, keep_rate: float):
    t0 = time.perf_counter()
    chunks = build_target_chunks(prompt, tokenizer, args.chunk_tokens)
    scorer_meta = scorer.score_prompt(prompt, chunks)
    compression = compress_prompt(
        prompt,
        tokenizer,
        chunks,
        keep_rate=keep_rate,
        anchor_tokens=args.anchor_tokens,
        delimiter=args.delimiter,
    )
    scorer_s = time.perf_counter() - t0
    return compression, scorer_meta, scorer_s


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}s"


def fmt_stream_error(result: StreamResult) -> str:
    if result.ok:
        return "ok"
    return (result.error or "failed").replace("\n", " ")[:180]


async def run_ttft(args, tokenizer, scorer: LogitSurpriseScorer) -> dict[str, Any]:
    prompt_targets = parse_csv_numbers(args.prompt_tokens, int)
    keep_rates = parse_csv_numbers(args.keep_rates, float)
    records = []
    endpoint = None
    async with httpx.AsyncClient() as client:
        endpoint = await check_endpoint(client, args.target_url)
        if not endpoint.get("ok"):
            raise RuntimeError(f"Target endpoint is not ready: {endpoint}")

        for target_tokens in prompt_targets:
            for keep_rate in keep_rates:
                total_runs = args.warmups + args.measurements
                for run_idx in range(total_runs):
                    phase = "warmup" if run_idx < args.warmups else "measure"
                    nonce = f"ttft-{target_tokens}-{keep_rate}-{run_idx}-{utc_stamp()}"
                    prompt = generate_diverse_nonce_prompt(target_tokens, tokenizer, nonce)
                    compression, scorer_meta, scorer_s = score_and_compress(
                        prompt, tokenizer, scorer, args, keep_rate
                    )
                    full = await call_openai_chat(
                        client,
                        args.target_url,
                        args.target_model,
                        prompt,
                        args.max_tokens,
                        args.temperature,
                        args.request_timeout,
                    )
                    compressed = await call_openai_chat(
                        client,
                        args.target_url,
                        args.target_model,
                        compression.compressed_prompt,
                        args.max_tokens,
                        args.temperature,
                        args.request_timeout,
                    )
                    record = {
                        "mode": "ttft",
                        "phase": phase,
                        "target_prompt_tokens": target_tokens,
                        "keep_rate": keep_rate,
                        "run_index": run_idx,
                        "nonce": nonce,
                        "full": asdict(full),
                        "compressed": asdict(compressed),
                        "scorer_s": scorer_s,
                        "compressed_e2e_s": scorer_s + compressed.ttft_s if compressed.ok else None,
                        "prefill_speedup": (
                            full.ttft_s / compressed.ttft_s
                            if full.ok and compressed.ok and compressed.ttft_s > 0
                            else None
                        ),
                        "e2e_speedup": (
                            full.ttft_s / (scorer_s + compressed.ttft_s)
                            if full.ok and compressed.ok and scorer_s + compressed.ttft_s > 0
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
                        f"full={fmt_seconds(full.ttft_s if full.ok else None)} "
                        f"comp_prefill={fmt_seconds(compressed.ttft_s if compressed.ok else None)} "
                        f"scorer={fmt_seconds(scorer_s)} "
                        f"e2e={fmt_seconds(record['compressed_e2e_s'])} "
                        f"tokens={compression.original_target_tokens}->{compression.compressed_target_tokens} "
                        f"full_status={fmt_stream_error(full)} "
                        f"comp_status={fmt_stream_error(compressed)}",
                        flush=True,
                    )
                    await asyncio.sleep(args.sleep_s)

    measured = [r for r in records if r["phase"] == "measure"]
    summary = []
    for target_tokens in prompt_targets:
        for keep_rate in keep_rates:
            group = [
                r
                for r in measured
                if r["target_prompt_tokens"] == target_tokens and r["keep_rate"] == keep_rate
                and r["full"]["ok"] and r["compressed"]["ok"]
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
            full_ttft = [r["full"]["ttft_s"] for r in group]
            comp_ttft = [r["compressed"]["ttft_s"] for r in group]
            scorer_times = [r["scorer_s"] for r in group]
            e2e_times = [r["compressed_e2e_s"] for r in group if r["compressed_e2e_s"] is not None]
            summary.append(
                {
                    "target_prompt_tokens": target_tokens,
                    "keep_rate": keep_rate,
                    "n": len(group),
                    "full_ttft_s_mean": statistics.mean(full_ttft),
                    "full_ttft_s_std": statistics.stdev(full_ttft) if len(full_ttft) > 1 else 0.0,
                    "compressed_prefill_s_mean": statistics.mean(comp_ttft),
                    "compressed_prefill_s_std": statistics.stdev(comp_ttft) if len(comp_ttft) > 1 else 0.0,
                    "scorer_s_mean": statistics.mean(scorer_times),
                    "compressed_e2e_s_mean": statistics.mean(e2e_times),
                    "prefill_speedup": statistics.mean(full_ttft) / statistics.mean(comp_ttft),
                    "e2e_speedup": statistics.mean(full_ttft) / statistics.mean(e2e_times),
                    "avg_original_target_tokens": mean(
                        r["compression"]["original_target_tokens"] for r in group
                    ),
                    "avg_compressed_target_tokens": mean(
                        r["compression"]["compressed_target_tokens"] for r in group
                    ),
                }
            )
    return {
        "kind": "ttft",
        "created_at": utc_stamp(),
        "endpoint": endpoint,
        "config": vars(args),
        "records": records,
        "summary": summary,
    }


async def run_quality(args, tokenizer, scorer: LogitSurpriseScorer) -> dict[str, Any]:
    keep_rates = parse_csv_numbers(args.keep_rates, float)
    prompts = load_quality_prompts(args, tokenizer)
    records = []
    endpoint = None
    async with httpx.AsyncClient() as client:
        endpoint = await check_endpoint(client, args.target_url)
        if not endpoint.get("ok"):
            raise RuntimeError(f"Target endpoint is not ready: {endpoint}")

        for idx, prompt in enumerate(prompts):
            prompt = fit_prompt_to_token_budget(prompt, tokenizer, args.quality_prompt_tokens)
            full = await call_openai_chat(
                client,
                args.target_url,
                args.target_model,
                prompt,
                args.quality_max_tokens,
                args.temperature,
                args.request_timeout,
            )
            chunks = build_target_chunks(prompt, tokenizer, args.chunk_tokens)
            score_t0 = time.perf_counter()
            scorer_meta = scorer.score_prompt(prompt, chunks)
            scorer_s = time.perf_counter() - score_t0
            for keep_rate in keep_rates:
                compression = compress_prompt(
                    prompt,
                    tokenizer,
                    chunks,
                    keep_rate=keep_rate,
                    anchor_tokens=args.anchor_tokens,
                    delimiter=args.delimiter,
                )
                compressed = await call_openai_chat(
                    client,
                    args.target_url,
                    args.target_model,
                    compression.compressed_prompt,
                    args.quality_max_tokens,
                    args.temperature,
                    args.request_timeout,
                )
                rouge_l = rouge_l_f1(compressed.text, full.text) if full.ok and compressed.ok else 0.0
                record = {
                    "mode": "quality",
                    "prompt_index": idx,
                    "keep_rate": keep_rate,
                    "full": asdict(full),
                    "compressed": asdict(compressed),
                    "rouge_l_f1_vs_full": rouge_l,
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
                }
                records.append(record)
                print(
                    f"[quality] i={idx} keep={keep_rate:.2f} "
                    f"rougeL={rouge_l:.3f} "
                    f"full_ttft={fmt_seconds(full.ttft_s if full.ok else None)} "
                    f"comp_ttft={fmt_seconds(compressed.ttft_s if compressed.ok else None)} "
                    f"tokens={compression.original_target_tokens}->{compression.compressed_target_tokens} "
                    f"full_status={fmt_stream_error(full)} "
                    f"comp_status={fmt_stream_error(compressed)}",
                    flush=True,
                )
                await asyncio.sleep(args.sleep_s)

    summary = []
    for keep_rate in keep_rates:
        group = [
            r
            for r in records
            if r["keep_rate"] == keep_rate and r["full"]["ok"] and r["compressed"]["ok"]
        ]
        if not group:
            summary.append({"keep_rate": keep_rate, "error": "no valid runs"})
            continue
        summary.append(
            {
                "keep_rate": keep_rate,
                "n": len(group),
                "rouge_l_f1_mean": statistics.mean(r["rouge_l_f1_vs_full"] for r in group),
                "rouge_l_f1_std": (
                    statistics.stdev(r["rouge_l_f1_vs_full"] for r in group)
                    if len(group) > 1
                    else 0.0
                ),
                "full_ttft_s_mean": statistics.mean(r["full"]["ttft_s"] for r in group),
                "compressed_prefill_s_mean": statistics.mean(
                    r["compressed"]["ttft_s"] for r in group
                ),
                "scorer_s_mean": statistics.mean(r["scorer_s"] for r in group),
                "avg_original_target_tokens": mean(
                    r["compression"]["original_target_tokens"] for r in group
                ),
                "avg_compressed_target_tokens": mean(
                    r["compression"]["compressed_target_tokens"] for r in group
                ),
            }
        )
    return {
        "kind": "quality",
        "created_at": utc_stamp(),
        "endpoint": endpoint,
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
    parser.add_argument("--mode", choices=["ttft", "quality"], required=True)
    parser.add_argument(
        "--target-url",
        "--sglang-url",
        dest="target_url",
        default=os.getenv("TARGET_URL", os.getenv("SGLANG_URL", "http://localhost:30000")),
        help="OpenAI-compatible target endpoint base URL.",
    )
    parser.add_argument("--target-model", default=os.getenv("SERVED_MODEL_NAME", "gpt-oss-120b-fp8"))
    parser.add_argument(
        "--target-tokenizer",
        default=os.getenv("TARGET_TOKENIZER_ID", "openai/gpt-oss-120b"),
    )
    parser.add_argument("--scorer-model", default=os.getenv("SCORER_MODEL_ID", "openai/gpt-oss-20b"))
    parser.add_argument("--scorer-device", default=os.getenv("SCORER_DEVICE", "cuda:0"))
    parser.add_argument("--scorer-dtype", default=os.getenv("SCORER_DTYPE", "auto"))
    parser.add_argument("--score-window", type=int, default=int(os.getenv("SCORE_WINDOW", "2048")))
    parser.add_argument("--chunk-tokens", type=int, default=int(os.getenv("CHUNK_TOKENS", "128")))
    parser.add_argument("--anchor-tokens", type=int, default=int(os.getenv("ANCHOR_TOKENS", "512")))
    parser.add_argument("--delimiter", default=os.getenv("SPEC_DELIMITER", ""))
    parser.add_argument("--keep-rates", default=os.getenv("KEEP_RATES", "0.3"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.0")))
    parser.add_argument("--request-timeout", type=float, default=float(os.getenv("REQUEST_TIMEOUT", "900")))
    parser.add_argument("--sleep-s", type=float, default=float(os.getenv("SLEEP_S", "0.2")))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "results/gpt_oss_specprefill"))
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--prompt-tokens", default=os.getenv("PROMPT_TOKENS", "16000,32000"))
    parser.add_argument("--measurements", type=int, default=int(os.getenv("MEASUREMENTS", "5")))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "1")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "1")))

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
    parser.add_argument(
        "--quality-max-tokens",
        type=int,
        default=int(os.getenv("QUALITY_MAX_TOKENS", "80")),
    )
    return parser


async def async_main(args) -> Path:
    async with httpx.AsyncClient() as client:
        endpoint = await check_endpoint(client, args.target_url)
        if not endpoint.get("ok"):
            raise RuntimeError(f"Target endpoint is not ready: {endpoint}")

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

    if args.mode == "ttft":
        payload = await run_ttft(args, tokenizer, scorer)
        path = write_result(args.output_dir, "ttft_gpt_oss_120b", payload)
    else:
        payload = await run_quality(args, tokenizer, scorer)
        path = write_result(args.output_dir, "quality_gpt_oss_120b", payload)

    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"Saved: {path}", flush=True)
    return path


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
