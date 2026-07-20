#!/usr/bin/env python3
"""Save vanilla prefill results in the stack-note format."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def latest(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt_s(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}s"


def fmt_x(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def short_error(value: str | None) -> str:
    if not value:
        return "ok"
    value = value.replace("\n", " ")
    for marker in [
        "CUDA out of memory.",
        "Tensor on device meta",
        "Pointer argument",
        "skipped:",
    ]:
        if marker in value:
            start = value.find(marker)
            return value[start : start + 160]
    return value[:160]


def record_status(record: dict[str, Any]) -> str:
    full = record.get("full") or {}
    comp = record.get("compressed") or {}
    full_ok = bool(full.get("ok"))
    comp_ok = bool(comp.get("ok"))
    if full_ok and comp_ok:
        return "full+compressed measured"
    if comp_ok and not full_ok:
        return "compressed prefill measured; full baseline unavailable"
    if full_ok and not comp_ok:
        return "full measured; compressed failed"
    return "failed"


def normalized_record(record: dict[str, Any]) -> dict[str, Any]:
    full = record.get("full") or {}
    comp = record.get("compressed") or {}
    compression = record.get("compression") or {}
    full_s = full.get("prefill_s") if full.get("ok") else None
    comp_s = comp.get("prefill_s") if comp.get("ok") else None
    e2e_s = record.get("compressed_e2e_s") if comp.get("ok") else None
    speedup = None
    if full_s and e2e_s:
        speedup = full_s / e2e_s
    return {
        "phase": record.get("phase"),
        "target_prompt_tokens": record.get("target_prompt_tokens"),
        "keep_rate": record.get("keep_rate"),
        "status": record_status(record),
        "full_prefill_s": full_s,
        "compressed_prefill_s": comp_s,
        "scorer_s": record.get("scorer_s"),
        "compressed_e2e_s": e2e_s,
        "e2e_speedup": speedup,
        "original_target_tokens": compression.get("original_target_tokens"),
        "compressed_target_tokens": compression.get("compressed_target_tokens"),
        "full_error": None if full.get("ok") else short_error(full.get("error")),
        "compressed_error": None if comp.get("ok") else short_error(comp.get("error")),
    }


def stack_block(payload: dict[str, Any], source: Path) -> str:
    config = payload.get("config") or {}
    target = config.get("target_model") or config.get("target_tokenizer") or "gpt-oss-120b"
    scorer = config.get("scorer_model") or "unknown"
    scorer_device = config.get("scorer_device") or "unknown"
    attn = config.get("attn_implementation") or "unknown"
    target_memory = config.get("target_max_memory") or ""
    records = [
        normalized_record(r)
        for r in payload.get("records", [])
        if r.get("phase") == "measure"
    ]

    lines: list[str] = []
    lines.extend(
        [
            "STACK",
            "-----",
            f"Target:  GPT-OSS-120B official MXFP4 via HF Transformers vanilla prefill ({attn})",
            "Hardware: 4x RTX PRO 6000 Blackwell WS (SM120), Vast.ai",
            f"Target checkpoint: {target}",
            f"Scorer: {scorer} on {scorer_device}",
            "Method: logit-surprise chunk top-k + head/tail anchor -> re-tokenize kept spans",
            "        with GPT-OSS tokenizer -> direct HF `use_cache=True` prefill forward.",
            f"Target max memory: {target_memory or 'not set'}",
            f"Source JSON: {source}",
            f"Saved: {utc_stamp()}",
            "",
            "RESULTS SO FAR",
            "==============",
            "",
            "1. VANILLA PREFILL SPEED - PARTIAL",
            "----------------------------------",
            "No serving runtime was used. This is not TTFT from SGLang/vLLM/TRT-LLM;",
            "it is direct Transformers prefill forward timing. Full eager attention may OOM.",
            "",
        ]
    )

    if not records:
        lines.extend(["No measured records found.", ""])
    for rec in records:
        l_value = int(rec["target_prompt_tokens"] or 0)
        keep = float(rec["keep_rate"] or 0.0)
        full_text = (
            f"{fmt_s(rec['full_prefill_s'])}"
            if rec["full_prefill_s"] is not None
            else f"FAIL ({rec['full_error']})"
        )
        comp_text = (
            f"{fmt_s(rec['compressed_e2e_s'])} e2e "
            f"(scorer {fmt_s(rec['scorer_s'])} + prefill {fmt_s(rec['compressed_prefill_s'])})"
            if rec["compressed_prefill_s"] is not None
            else f"FAIL ({rec['compressed_error']})"
        )
        tok_text = f"{rec['original_target_tokens']} -> {rec['compressed_target_tokens']} tokens"
        lines.extend(
            [
                f"L~{round(l_value / 1000):.0f}K, keep={keep:.2f}:",
                f"  full prefill:       {full_text}",
                f"  compressed prefill: {comp_text}",
                f"  tokens:             {tok_text}",
                f"  e2e speedup:        {fmt_x(rec['e2e_speedup'])}",
                f"  status:             {rec['status']}",
                "",
            ]
        )

    compressed_ok = [r for r in records if r["compressed_prefill_s"] is not None]
    full_ok = [r for r in records if r["full_prefill_s"] is not None]
    full_fail = [r for r in records if r["full_prefill_s"] is None]
    lines.extend(
        [
            "2. CURRENT INTERPRETATION",
            "-------------------------",
        ]
    )
    if compressed_ok:
        lines.append(
            f"Compressed target prefill is proven on {len(compressed_ok)} measured row(s)."
        )
    if full_ok:
        lines.append(f"Full baseline prefill is available on {len(full_ok)} row(s).")
    if full_fail:
        lines.append(
            "Full baseline prefill failed on at least one row, so speedup cannot be "
            "reported for those rows."
        )
    lines.extend(
        [
            "For GPT-OSS vanilla HF, `attn_implementation=eager` is required, but it",
            "materializes large attention tensors. This makes long-context full prefill",
            "memory-bound and is not equivalent to optimized serving-engine TTFT.",
            "",
            "3. BLOCKERS / NOTES",
            "-------------------",
            "- GPT-OSS 20B scorer cannot run on CPU in this path because the official",
            "  checkpoint uses MXFP4/Triton kernels that expect CUDA tensors.",
            "- 32K full prefill is not expected to fit with vanilla eager attention on",
            "  a single 96GB RTX PRO 6000 GPU allocation.",
            "- Quality rows need generation; prefill-only mode cannot reproduce QA",
            "  ROUGE-L against full-prompt answers.",
            "",
            "4. REPRODUCE",
            "------------",
            "cd /workspace/gpt_oss_specprefill",
            "source /workspace/venvs/specprefill/bin/activate",
            "bash scripts/run_pag_vanilla_prefill.sh",
            "python3 benchmark/save_vanilla_results_stack.py \\",
            "  --results-dir results/vanilla_prefill \\",
            "  --output results/vanilla_prefill/RESULTS.md \\",
            "  --json-output results/vanilla_prefill/RESULTS.json",
            "",
        ]
    )
    return "\n".join(lines)


def build_json(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    return {
        "created_at": utc_stamp(),
        "source_json": str(source),
        "kind": payload.get("kind"),
        "config": payload.get("config") or {},
        "records": [
            normalized_record(r)
            for r in payload.get("records", [])
            if r.get("phase") == "measure"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/vanilla_prefill")
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    source = Path(args.input) if args.input else latest(
        results_dir, "vanilla_prefill_gpt_oss_120b_*.json"
    )
    if source is None or not source.exists():
        raise SystemExit(f"No vanilla prefill result JSON found under {results_dir}")

    payload = load_json(source)
    markdown = stack_block(payload, source)
    json_payload = build_json(payload, source)

    output = Path(args.output) if args.output else results_dir / "RESULTS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(output)

    if args.json_output:
        json_output = Path(args.json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json_output)


if __name__ == "__main__":
    main()
