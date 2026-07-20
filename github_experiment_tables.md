# GitHub-Ready Experiment Tables

## Saved Results

The primary result is the all-GPU vLLM path: GPT-OSS-120B is served through a local vLLM endpoint on the GPU machine, and the Qwen scorer runs on `cuda:3` (bf16). CPU-scorer and vanilla HF results are included only as supplementary diagnostics.

### Primary: All-GPU vLLM TTFT with GPU Qwen Scorer

Source: `results/vllm_gptoss_120b_gpu_qwen/RESULTS.md`

| Target | Prompt | Keep | n | Full TTFT (s) | Compressed TTFT (s) | Prefill Speedup | Scorer (s) | Compressed E2E (s) | E2E Speedup | Compressed Tokens | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vLLM TTFT, GPT-OSS-120B MXFP4 target on GPU + Qwen GPU scorer | 16K | 0.3 | 5 | 0.912 ± 0.007 | 0.247 ± 0.002 | 3.70x | 0.671 | 0.918 | 0.994x | 4864 | measured |
| vLLM TTFT, GPT-OSS-120B MXFP4 target on GPU + Qwen GPU scorer | 32K | 0.3 | 5 | 1.994 ± 0.002 | 0.514 ± 0.002 | 3.88x | 1.334 | 1.848 | 1.079x | 9600 | measured |

This is the result to cite for the current all-GPU draft. Compressed TTFT is 3.70x faster at 16K and 3.88x faster at 32K. Including GPU scorer time, the compressed path is approximately parity at 16K and slightly faster at 32K.

### Supplement: vLLM TTFT with CPU Qwen Scorer

Source: `results/vllm_gptoss_120b/RESULTS.md`

| Target | Prompt | Keep | n | Full TTFT (s) | Compressed TTFT (s) | Prefill Speedup | Scorer (s) | Compressed E2E (s) | E2E Speedup | Compressed Tokens | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vLLM TTFT, GPT-OSS-120B MXFP4 + Qwen CPU scorer | 16K | 0.3 | 5 | 0.908 ± 0.004 | 0.244 ± 0.003 | 3.72x | 58.798 | 59.042 | 0.015x | 4864 | measured |
| vLLM TTFT, GPT-OSS-120B MXFP4 + Qwen CPU scorer | 32K | 0.3 | 5 | 1.985 ± 0.011 | 0.506 ± 0.006 | 3.93x | 120.888 | 121.394 | 0.016x | 9600 | measured |

### Supplement: Vanilla HF Prefill

Source: `results/vanilla_prefill/RESULTS.md`

| Target | Baseline | Compressed | Metric | Full | SpecPrefill | Delta | Script | Notes |
|---|---|---|---|---|---|---|---|---|
| GPT-OSS-120B official MXFP4, HF Transformers vanilla prefill | Full prompt, HF `eager` prefill | compressed keep=0.3 + GPT-OSS-20B scorer | Prefill forward @ L=16K (s), no generation | OOM (16000 tok; tried to allocate 30.52 GiB) | 62.17 e2e (4864 tok; scorer 58.10 + prefill 4.08) | n/a, full baseline OOM | `benchmark/run_vanilla_prefill.py` | Same-family scorer; logit-surprise chunk top-k + anchor; direct HF `use_cache=True` |
| GPT-OSS-120B official MXFP4, HF Transformers vanilla prefill | Full prompt, HF `eager` prefill | compressed keep=0.3 + GPT-OSS-20B scorer | Prefill forward @ L=32K (s), no generation | OOM expected under vanilla eager attention | not measured | n/a | `benchmark/run_vanilla_prefill.py` | 32K full prompt needs an optimized serving runtime for real TTFT-style reporting |

### Quality ROUGE-L Rows 65-66

Source: `results/vllm_gptoss_120b_gpu_qwen_quality/RESULTS.md`

Run:

```bash
CUDA_VISIBLE_DEVICES=3 \
TARGET_URL=http://localhost:8000 \
TARGET_MODEL=gpt-oss-120b \
TARGET_TOKENIZER_ID=/workspace/models/openai__gpt-oss-120b \
SCORER_MODEL_ID=Qwen/Qwen3-0.6B \
SCORER_DEVICE=cuda:0 \
SCORER_DTYPE=bf16 \
QUALITY_N=20 \
QUALITY_PROMPT_TOKENS=16000 \
QUALITY_MAX_TOKENS=256 \
  bash scripts/run_pag_vllm_quality.sh
```

| Target | Baseline | Compressed | Metric | Full | SpecPrefill | Delta | Script | Notes |
|---|---|---|---|---|---|---|---|---|
| vLLM quality, GPT-OSS-120B MXFP4 target + Qwen scorer | Full prompt answer | compressed keep=0.3 | LongBench QA ROUGE-L vs full answer @ L=16K | 1.00 full-ref | 0.215 +/- 0.200 | -78.5% vs full-ref | `benchmark/run_gpt_oss_specprefill.py --mode quality` | n=20; tokens 16000->4863; full TTFT 0.067s; comp TTFT 0.034s; scorer 0.263s; text source full=content:6, reasoning:14; comp=content:7, reasoning:13 |
| vLLM quality, GPT-OSS-120B MXFP4 target + Qwen scorer | Full prompt answer | compressed keep=0.5 | LongBench QA ROUGE-L vs full answer @ L=16K | 1.00 full-ref | 0.339 +/- 0.218 | -66.1% vs full-ref | `benchmark/run_gpt_oss_specprefill.py --mode quality` | n=20; tokens 16000->8063; full TTFT 0.067s; comp TTFT 0.047s; scorer 0.263s; text source full=content:6, reasoning:14; comp=content:7, reasoning:13 |

Important: this quality metric is ROUGE-L against the full-prompt answer, not
human gold-answer accuracy. The GPU-scorer path uses `Qwen/Qwen3-0.6B`, not a
same-family GPT-OSS scorer.

Quality interpretation: rows 65-66 were rerun with the patched GPT-OSS/vLLM
response parser and `QUALITY_MAX_TOKENS=256`. The measurement is now valid as a
full-answer agreement diagnostic, but quality preservation is weak: keep=0.3
reaches ROUGE-L 0.215, and keep=0.5 improves to 0.339 while still losing most
of the full-prompt answer surface.

## Suggested Interpretation

The all-GPU vLLM path shows that compressed TTFT is roughly 3.7-3.9x faster than full-prompt TTFT at 16K and 32K. With the scorer also on GPU, the compressed path reaches near parity end-to-end at 16K and a small end-to-end speedup at 32K. Quality rows 65-66 are now measured, but they do not show answer-quality preservation: keep=0.3 reaches ROUGE-L 0.215 and keep=0.5 reaches 0.339 against full-prompt answers.
