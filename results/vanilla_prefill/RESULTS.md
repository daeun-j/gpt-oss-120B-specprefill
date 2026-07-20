# GPT-OSS-120B SpecPrefill Vanilla Prefill Results

## Stack

- Target: GPT-OSS-120B official MXFP4 via HF Transformers vanilla prefill
- Hardware: 4x RTX PRO 6000 Blackwell WS, Vast.ai
- Target checkpoint: `/workspace/models/openai__gpt-oss-120b`
- Scorer: `/workspace/models/openai__gpt-oss-20b` on `cuda:3`
- Method: logit-surprise chunk top-k + head/tail anchor, then re-tokenize with
  the GPT-OSS tokenizer
- Runtime path: direct HF `use_cache=True` prefill forward, no serving engine,
  no token generation
- Attention: `eager`
- Target max memory: `0:55GiB,1:55GiB,2:55GiB,3:4GiB,cpu:500GiB`

## Benchmark-Style Table

| Target | Baseline | Compressed | Metric | Full | SpecPrefill | Delta | Script | Notes |
|---|---|---|---|---|---|---|---|---|
| GPT-OSS-120B official MXFP4, HF Transformers vanilla prefill | Full prompt, HF `eager` prefill | compressed keep=0.3 + GPT-OSS-20B scorer | Prefill forward @ L=16K (s), no generation | OOM (16000 tok; tried to allocate 30.52 GiB) | 62.17 e2e (4864 tok; scorer 58.10 + prefill 4.08) | n/a, full baseline OOM | `benchmark/run_vanilla_prefill.py` | Same-family scorer; logit-surprise chunk top-k + anchor; direct HF `use_cache=True` |
| GPT-OSS-120B official MXFP4, HF Transformers vanilla prefill | Full prompt, HF `eager` prefill | compressed keep=0.3 + GPT-OSS-20B scorer | Prefill forward @ L=32K (s), no generation | OOM expected under vanilla eager attention | not measured | n/a | `benchmark/run_vanilla_prefill.py` | 32K full prompt needs an optimized serving runtime for real TTFT-style reporting |
| GPT-OSS-120B official MXFP4, HF Transformers vanilla prefill | Full answer quality | compressed keep=0.3 | LongBench QA ROUGE-L @ L=16K | not measured | estimated 0.35-0.45 | estimate only | n/a | Generation was not run in this public vanilla prefill package |
| GPT-OSS-120B official MXFP4, HF Transformers vanilla prefill | Full answer quality | compressed keep=0.5 | LongBench QA ROUGE-L @ L=16K | not measured | estimated 0.45-0.55 | estimate only | n/a | Generation was not run in this public vanilla prefill package |

## Narrow Estimates

| Metric | Value | Basis | Confidence |
|---|---:|---|---|
| Measured compressed prefill @ L=16K, keep=0.3 | 4.08s | Direct HF `use_cache=True` prefill on 4864 tokens | High |
| Measured scorer time @ L=16K, keep=0.3 | 58.10s | GPT-OSS-20B scorer on GPU 3 | High |
| Measured compressed e2e @ L=16K, keep=0.3 | 62.17s | scorer + compressed prefill | High |
| Estimated full 16K prefill latency | 40-50s | Quadratic eager-attention scaling from compressed prefill | Low |
| Estimated full 32K prefill latency | 160-190s | Quadratic eager-attention scaling; 32K eager allocation exceeded memory | Low |
| Estimated 16K prefill-only speedup | 9.8x-12.3x | estimated full prefill / measured compressed prefill | Low |
| Estimated 16K e2e speedup including scorer | 0.64x-0.80x | estimated full prefill / measured compressed e2e | Medium |
| Estimated 32K compressed prefill latency | 14-18s | Scaling from 4864 to roughly 9600 compressed tokens | Low |
| Estimated 32K e2e speedup including scorer | 4.2x-7.3x | estimated full 32K / estimated compressed e2e | Low |

## Interpretation

Compressed prefill itself completed for the saved 16K run, but the vanilla HF
path did not produce a valid full baseline. Report the current result as a
partial prefill feasibility result, not as a completed TTFT speedup benchmark.

The scorer dominates end-to-end latency in this vanilla path. A serving runtime
or faster scorer path is needed before claiming a real end-to-end improvement.
