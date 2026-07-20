# GPT-OSS-120B vLLM Quality Results

## Stack

- Target: GPT-OSS-120B official MXFP4 served through a vLLM OpenAI-compatible endpoint
- Endpoint: `http://localhost:8000`
- Endpoint model: `gpt-oss-120b`
- Scorer: `Qwen/Qwen3-0.6B` on `cuda:0`
- Compression: logit-surprise chunk top-k + head/tail anchor
- Dataset: `THUDM/LongBench-v2`, split `test`, n=20
- Prompt budget: 16000 target tokens
- Answer budget: 256 generated tokens
- Metric: ROUGE-L F1 of compressed-prompt answer against full-prompt answer
- Source file: `quality_gpt_oss_120b_20260709T214611Z.json`
- Saved: 2026-07-09 21:49:37 UTC

## Benchmark-Style Table

| Target | Baseline | Compressed | Metric | Full | SpecPrefill | Delta | Script | Notes |
|---|---|---|---|---|---|---|---|---|
| vLLM quality, GPT-OSS-120B MXFP4 target + Qwen scorer | Full prompt answer | compressed keep=0.3 | Quality ROUGE-L vs full answer @ L=16K | 1.00 full-ref | 0.215 +/- 0.200 | -78.5% vs full-ref | `benchmark/run_gpt_oss_specprefill.py --mode quality` | n=20; tokens 16000->4863; full TTFT 0.067s; comp TTFT 0.034s; scorer 0.263s; text source full=content:6, reasoning:14; comp=content:7, reasoning:13 |
| vLLM quality, GPT-OSS-120B MXFP4 target + Qwen scorer | Full prompt answer | compressed keep=0.5 | Quality ROUGE-L vs full answer @ L=16K | 1.00 full-ref | 0.339 +/- 0.218 | -66.1% vs full-ref | `benchmark/run_gpt_oss_specprefill.py --mode quality` | n=20; tokens 16000->8063; full TTFT 0.067s; comp TTFT 0.047s; scorer 0.263s; text source full=content:6, reasoning:14; comp=content:7, reasoning:13 |

## Interpretation

This measures answer agreement with the full-prompt output, not external
task accuracy. A higher ROUGE-L means the compressed prompt preserves more
of the full-prompt answer behavior.
