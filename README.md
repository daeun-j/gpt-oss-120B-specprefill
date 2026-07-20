# GPT-OSS-120B SpecPrefill Benchmark Summary

Minimal reproduction package for PAG-style SpecPrefill experiments on
`gpt-oss-120b`.

This public version centers the all-GPU vLLM TTFT measurement: GPT-OSS-120B is
served through a local vLLM endpoint on the GPU machine, and the Qwen scorer is
loaded on `cuda:3`. CPU-scorer and vanilla Hugging Face Transformers results
are kept as supplementary diagnostics.
Experimental Vast/SGLang/TRT-LLM/NVFP4/GPTQ bring-up scripts were moved out of
the repo before GitHub upload.

## Scope

- Target: `openai/gpt-oss-120b`, served as `gpt-oss-120b` through a local vLLM
  OpenAI-compatible endpoint on the GPU machine
- Primary scorer: `Qwen/Qwen3-0.6B` on `cuda:3` (bf16)
- Supplementary scorers: `Qwen/Qwen3-0.6B` on CPU, and
  `openai/gpt-oss-20b` on `cuda:3` in the vanilla HF path
- Method: logit-surprise chunk scoring, top-k chunk selection, head/tail anchor,
  then re-tokenize the compressed prompt with the target tokenizer
- Measurement paths:
  - direct HF `use_cache=True` prefill forward pass
  - vLLM OpenAI-compatible endpoint TTFT with `max_tokens=1`
  - vLLM OpenAI-compatible endpoint quality generation for rows 65-66
- Not included here: human/gold-answer grading; quality is ROUGE-L against the
  full-prompt answer, matching the original compression-vs-full style.

## Files

| Path | Purpose |
|---|---|
| `benchmark/run_vanilla_prefill.py` | Vanilla HF prefill measurement entrypoint |
| `benchmark/run_gpt_oss_specprefill.py` | Shared prompt, tokenizer, scorer, and compression helpers |
| `benchmark/specprefill_core.py` | Standalone compression utilities |
| `benchmark/save_vanilla_results_stack.py` | Converts raw vanilla runs to `RESULTS.md` / `RESULTS.json` |
| `benchmark/save_quality_results_stack.py` | Converts raw quality runs to `RESULTS.md` / `RESULTS.json` |
| `scripts/run_pag_vanilla_prefill.sh` | Vast/server runnable wrapper |
| `scripts/run_pag_vllm_quality.sh` | Runs quality rows 65-66 against a running vLLM endpoint |
| `results/vanilla_prefill/RESULTS.md` | Saved public result summary |
| `results/vllm_gptoss_120b/RESULTS.md` | Saved vLLM TTFT result summary with CPU Qwen scorer |
| `results/vllm_gptoss_120b_gpu_qwen/RESULTS.md` | Primary all-GPU vLLM TTFT result summary with GPU Qwen scorer |
| `results/vllm_gptoss_120b_gpu_qwen_quality/RESULTS.md` | Quality row 65-66 diagnostic result summary |
| `tests/test_specprefill_core.py` | Lightweight unit coverage for the compression core |

## Setup

```bash
cd /workspace/gpt_oss_specprefill
python3 -m venv /workspace/venvs/specprefill
source /workspace/venvs/specprefill/bin/activate
python3 -m pip install -r benchmark/requirements.txt
```

Models should already exist on the machine or be available from Hugging Face:

```text
/workspace/models/openai__gpt-oss-120b
/workspace/models/openai__gpt-oss-20b
```

## Run

Recommended 4x RTX PRO 6000 WS configuration with GPU 3 reserved for the
20B scorer:

```bash
cd /workspace/gpt_oss_specprefill
source /workspace/venvs/specprefill/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3 \
TARGET_MODEL_ID=/workspace/models/openai__gpt-oss-120b \
TARGET_TOKENIZER_ID=/workspace/models/openai__gpt-oss-120b \
SCORER_MODEL_ID=/workspace/models/openai__gpt-oss-20b \
SCORER_DEVICE=cuda:3 \
ATTN_IMPLEMENTATION=eager \
SCORER_ATTN_IMPLEMENTATION=eager \
TARGET_MAX_MEMORY=0:55GiB,1:55GiB,2:55GiB,3:4GiB,cpu:500GiB \
PROMPT_TOKENS=16000 \
MEASUREMENTS=1 \
WARMUPS=0 \
  bash scripts/run_pag_vanilla_prefill.sh
```

For a CPU-friendly scorer, use Qwen instead:

```bash
SCORER_MODEL_ID=Qwen/Qwen3-0.6B \
SCORER_DEVICE=cpu \
TARGET_MAX_MEMORY=0:55GiB,1:55GiB,2:55GiB,3:55GiB,cpu:500GiB \
  bash scripts/run_pag_vanilla_prefill.sh
```

The wrapper saves:

```text
results/vanilla_prefill/RESULTS.md
results/vanilla_prefill/RESULTS.json
```

For quality rows 65-66, start the vLLM target first, then run:

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

That saves:

```text
results/vllm_gptoss_120b_gpu_qwen_quality/RESULTS.md
results/vllm_gptoss_120b_gpu_qwen_quality/RESULTS.json
```

## Saved Results

The primary result is the all-GPU vLLM path: the GPT-OSS-120B target is served
by vLLM on the GPU machine, while the Qwen scorer runs on `cuda:3` (bf16). CPU
scoring is included only to show the scorer bottleneck, and the vanilla HF path
is a partial feasibility result because full-prompt eager prefill OOMed.

### Primary: All-GPU vLLM TTFT with GPU Qwen Scorer

Source: `results/vllm_gptoss_120b_gpu_qwen/RESULTS.md`

| Target | Prompt | Keep | n | Full TTFT (s) | Compressed TTFT (s) | Prefill Speedup | Scorer (s) | Compressed E2E (s) | E2E Speedup | Compressed Tokens | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vLLM TTFT, GPT-OSS-120B MXFP4 target on GPU + Qwen GPU scorer | 16K | 0.3 | 5 | 0.912 ± 0.007 | 0.247 ± 0.002 | 3.70x | 0.671 | 0.918 | 0.994x | 4864 | measured |
| vLLM TTFT, GPT-OSS-120B MXFP4 target on GPU + Qwen GPU scorer | 32K | 0.3 | 5 | 1.994 ± 0.002 | 0.514 ± 0.002 | 3.88x | 1.334 | 1.848 | 1.079x | 9600 | measured |

This is the result to cite for the current all-GPU draft. Compressed TTFT is
3.70x faster at 16K and 3.88x faster at 32K. Including GPU scorer time, the
compressed path is approximately parity at 16K and slightly faster at 32K.

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

| Target | Baseline | Compressed | Metric | Full | SpecPrefill | Delta | Script | Notes |
|---|---|---|---|---|---|---|---|---|
| vLLM quality, GPT-OSS-120B MXFP4 target + Qwen scorer | Full prompt answer | compressed keep=0.3 | Quality ROUGE-L vs full answer @ L=16K | 1.00 full-ref | 0.215 +/- 0.200 | -78.5% vs full-ref | `benchmark/run_gpt_oss_specprefill.py --mode quality` | n=20; tokens 16000->4863; full TTFT 0.067s; comp TTFT 0.034s; scorer 0.263s; text source full=content:6, reasoning:14; comp=content:7, reasoning:13 |
| vLLM quality, GPT-OSS-120B MXFP4 target + Qwen scorer | Full prompt answer | compressed keep=0.5 | Quality ROUGE-L vs full answer @ L=16K | 1.00 full-ref | 0.339 +/- 0.218 | -66.1% vs full-ref | `benchmark/run_gpt_oss_specprefill.py --mode quality` | n=20; tokens 16000->8063; full TTFT 0.067s; comp TTFT 0.047s; scorer 0.263s; text source full=content:6, reasoning:14; comp=content:7, reasoning:13 |

Important: the primary quality runner uses `Qwen/Qwen3-0.6B`, not a
same-family GPT-OSS scorer. The metric is answer agreement with the full prompt,
not external task accuracy.

Quality interpretation: rows 65-66 were rerun with the patched GPT-OSS/vLLM
response parser and `QUALITY_MAX_TOKENS=256`. The measurement is now valid as a
full-answer agreement diagnostic, but quality preservation is weak: keep=0.3
reaches ROUGE-L 0.215, and keep=0.5 improves to 0.339 while still losing most
of the full-prompt answer surface.

## Narrow Estimates

These are directional estimates from the saved vanilla run, not benchmark
claims.

| Metric | Value | Basis | Confidence |
|---|---:|---|---|
| Measured compressed prefill @ L=16K, keep=0.3 | 4.08s | Direct HF `use_cache=True` prefill on 4864 tokens | High |
| Measured scorer time @ L=16K, keep=0.3 | 58.10s | GPT-OSS-20B scorer on GPU 3 | High |
| Measured compressed e2e @ L=16K, keep=0.3 | 62.17s | scorer + compressed prefill | High |


## Limitations

- Full 16K and 32K baselines did not produce valid direct measurements in this
  vanilla HF path because eager attention was memory-bound.
- The all-GPU vLLM TTFT path measured valid full and compressed TTFT. GPU Qwen
  scoring reaches near-parity at 16K and a small end-to-end speedup at 32K.
- The CPU-scorer vLLM result is a bottleneck diagnostic, not the primary result.
- The GPU-scorer result uses `Qwen/Qwen3-0.6B`, not a same-family GPT-OSS-20B
  scorer.
- Quality ROUGE-L is measured against the full-prompt output, not human gold
  answers. GPT-OSS/vLLM often returned text under `reasoning` rather than final
  `content`, so the quality table reports both text-source counts.
- This repo intentionally avoids shipping failed serving-runtime bring-up
  scripts as public runnable instructions; saved vLLM JSON files are included
  as measurement artifacts.

## Archived Cleanup Files

Files that were useful during debugging but should not be part of the GitHub
release were moved to the sibling archive:

```text
../gpt_oss_specprefill_archive_github_cleanup_20260709T132836
```

That archive includes the removed SGLang, TRT-LLM, NVFP4/GPTQ, Vast tunnel,
notebook, third-party, cache files, and the invalid earlier quality raw run.
