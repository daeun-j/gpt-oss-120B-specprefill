# GPT-OSS-120B All-GPU vLLM TTFT Results with GPU Qwen Scorer

## Stack

- Target: GPT-OSS-120B official MXFP4 served through a vLLM OpenAI-compatible endpoint on the GPU machine
- Endpoint model: `gpt-oss-120b`
- Target checkpoint: `/workspace/models/openai__gpt-oss-120b`
- Scorer: `Qwen/Qwen3-0.6B` on `cuda:3` (bf16)
- Compression: logit-surprise chunk top-k + head/tail anchor, keep=0.3
- Measurement: TTFT with `max_tokens=1`, `temperature=0`
- Source file: `ttft_gpt_oss_120b_20260709T204301Z.json`

## Benchmark-Style Table

| Target | Prompt | Keep | n | Full TTFT (s) | Compressed TTFT (s) | Prefill Speedup | Scorer (s) | Compressed E2E (s) | E2E Speedup | Compressed Tokens | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vLLM TTFT, GPT-OSS-120B MXFP4 + Qwen GPU scorer | 16K | 0.3 | 5 | 0.912 ± 0.007 | 0.247 ± 0.002 | 3.70x | 0.671 | 0.918 | 0.994x | 4864 | measured |
| vLLM TTFT, GPT-OSS-120B MXFP4 + Qwen GPU scorer | 32K | 0.3 | 5 | 1.994 ± 0.002 | 0.514 ± 0.002 | 3.88x | 1.334 | 1.848 | 1.079x | 9600 | measured |

## Interpretation

With the target served by vLLM on the GPU machine and the Qwen scorer on `cuda:3`, the dominant CPU-scorer bottleneck is removed. Compressed TTFT is still roughly 3.7-3.9x faster than full-prompt TTFT. End-to-end latency is approximately parity at 16K and slightly faster at 32K in this measurement, but generation quality was not measured.
