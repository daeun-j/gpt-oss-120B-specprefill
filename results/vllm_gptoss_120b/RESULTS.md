# GPT-OSS-120B vLLM TTFT Results

## Stack

- Target: GPT-OSS-120B official MXFP4 served through a vLLM OpenAI-compatible endpoint
- Endpoint model: `gpt-oss-120b`
- Target checkpoint: `/workspace/models/openai__gpt-oss-120b`
- Max model length: 32768
- Scorer: `Qwen/Qwen3-0.6B` on CPU
- Compression: logit-surprise chunk top-k + head/tail anchor, keep=0.3
- Measurement: TTFT with `max_tokens=1`, `temperature=0`
- Source file: `ttft_gpt_oss_120b_20260709T202047Z.json`

## Benchmark-Style Table

| Target | Prompt | Keep | n | Full TTFT (s) | Compressed TTFT (s) | Prefill Speedup | Scorer (s) | Compressed E2E (s) | E2E Speedup | Compressed Tokens | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vLLM TTFT, GPT-OSS-120B MXFP4 | 16K | 0.3 | 5 | 0.908 ± 0.004 | 0.244 ± 0.003 | 3.72x | 58.798 | 59.042 | 0.015x | 4864 | measured |
| vLLM TTFT, GPT-OSS-120B MXFP4 | 32K | 0.3 | 5 | 1.985 ± 0.011 | 0.506 ± 0.006 | 3.93x | 120.888 | 121.394 | 0.016x | 9600 | measured |

## Interpretation

The vLLM path gives valid full and compressed TTFT measurements at both 16K and 32K. Compressed prefill/TTFT is roughly 3.7-3.9x faster than full-prompt TTFT, but the current CPU scorer dominates end-to-end latency, so this is not yet an end-to-end speedup result.
