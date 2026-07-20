"""Core helpers for the gpt-oss SpecPrefill reproduction.

The implementation mirrors the PAG text-compression path:
target-token chunks -> cross-family logit-surprise scores -> chunk top-k
selection with head/tail anchors -> re-tokenized compressed text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class PromptChunk:
    index: int
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    text: str
    token_count: int
    score: float = 0.0
    selected: bool = False
    selected_reason: str = ""


@dataclass
class CompressionResult:
    compressed_prompt: str
    chunks: list[PromptChunk]
    selected_indices: list[int]
    original_target_tokens: int
    selected_target_tokens: int
    compressed_target_tokens: int
    keep_rate: float
    effective_keep_rate: float
    anchor_tokens: int
    delimiter: str


def _encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "tolist"):
        return ids.tolist()
    return list(ids)


def _safe_offsets(tokenizer, text: str):
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return None
    offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
    ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if offsets is None or ids is None or len(offsets) != len(ids):
        return None
    return list(ids), [(int(a), int(b)) for a, b in offsets]


def build_target_chunks(prompt: str, target_tokenizer, chunk_tokens: int) -> list[PromptChunk]:
    """Split a prompt into fixed-size target-token chunks.

    Fast tokenizers provide character offsets, which preserves exact spans in
    the original prompt. If offsets are unavailable, we fall back to decoding
    token chunks; token counts remain correct, but char spans are approximate.
    """

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")

    encoded_offsets = _safe_offsets(target_tokenizer, prompt)
    if encoded_offsets is not None:
        token_ids, offsets = encoded_offsets
        chunks: list[PromptChunk] = []
        for idx, start in enumerate(range(0, len(token_ids), chunk_tokens)):
            end = min(start + chunk_tokens, len(token_ids))
            chunk_offsets = [(a, b) for a, b in offsets[start:end] if b > a]
            if chunk_offsets:
                start_char = min(a for a, _ in chunk_offsets)
                end_char = max(b for _, b in chunk_offsets)
                text = prompt[start_char:end_char]
            else:
                start_char = 0
                end_char = 0
                text = target_tokenizer.decode(token_ids[start:end], skip_special_tokens=False)
            chunks.append(
                PromptChunk(
                    index=idx,
                    start_token=start,
                    end_token=end,
                    start_char=start_char,
                    end_char=end_char,
                    text=text,
                    token_count=end - start,
                )
            )
        return chunks

    token_ids = _encode(target_tokenizer, prompt)
    chunks = []
    cursor = 0
    for idx, start in enumerate(range(0, len(token_ids), chunk_tokens)):
        end = min(start + chunk_tokens, len(token_ids))
        text = target_tokenizer.decode(token_ids[start:end], skip_special_tokens=False)
        start_char = cursor
        cursor += len(text)
        chunks.append(
            PromptChunk(
                index=idx,
                start_token=start,
                end_token=end,
                start_char=start_char,
                end_char=cursor,
                text=text,
                token_count=end - start,
            )
        )
    return chunks


def assign_scores_from_offsets(
    chunks: Sequence[PromptChunk],
    scorer_offsets: Sequence[tuple[int, int]],
    scorer_scores: Sequence[float],
) -> None:
    """Assign mean scorer-token surprise to each target-token chunk."""

    if len(scorer_offsets) != len(scorer_scores):
        raise ValueError("scorer_offsets and scorer_scores must have the same length")

    scored = []
    for (start, end), score in zip(scorer_offsets, scorer_scores):
        if end <= start:
            continue
        scored.append(((start + end) / 2.0, float(score)))

    ptr = 0
    for chunk in chunks:
        values = []
        while ptr < len(scored) and scored[ptr][0] < chunk.start_char:
            ptr += 1
        scan = ptr
        while scan < len(scored) and scored[scan][0] < chunk.end_char:
            values.append(scored[scan][1])
            scan += 1
        chunk.score = sum(values) / len(values) if values else 0.0


def assign_scores_by_chunk_text(chunks: Sequence[PromptChunk], chunk_scores: Sequence[float]) -> None:
    if len(chunks) != len(chunk_scores):
        raise ValueError("chunks and chunk_scores must have the same length")
    for chunk, score in zip(chunks, chunk_scores):
        chunk.score = float(score)


def select_chunks(
    chunks: Sequence[PromptChunk],
    keep_rate: float,
    anchor_tokens: int,
) -> tuple[list[PromptChunk], int]:
    """Select head/tail anchors and highest-scoring middle chunks."""

    if not 0.0 < keep_rate <= 1.0:
        raise ValueError("keep_rate must be in (0, 1]")
    if anchor_tokens < 0:
        raise ValueError("anchor_tokens must be non-negative")

    total_tokens = sum(c.token_count for c in chunks)
    target_keep = max(1, int(total_tokens * keep_rate))
    tail_start = max(0, total_tokens - anchor_tokens)

    for chunk in chunks:
        chunk.selected = False
        chunk.selected_reason = ""

    selected: set[int] = set()
    for chunk in chunks:
        if anchor_tokens and chunk.start_token < anchor_tokens:
            selected.add(chunk.index)
            chunk.selected_reason = "head_anchor"
        if anchor_tokens and chunk.end_token > tail_start:
            selected.add(chunk.index)
            chunk.selected_reason = "tail_anchor"

    selected_tokens = sum(chunks[i].token_count for i in selected)
    middle = [c for c in chunks if c.index not in selected]
    middle.sort(key=lambda c: (c.score, -c.index), reverse=True)

    for chunk in middle:
        if selected_tokens >= target_keep:
            break
        selected.add(chunk.index)
        selected_tokens += chunk.token_count
        chunk.selected_reason = "topk"

    ordered = []
    for chunk in chunks:
        chunk.selected = chunk.index in selected
        if chunk.selected:
            ordered.append(chunk)
    return ordered, selected_tokens


def compress_prompt(
    prompt: str,
    target_tokenizer,
    chunks: list[PromptChunk],
    keep_rate: float,
    anchor_tokens: int = 512,
    delimiter: str = "",
) -> CompressionResult:
    selected, selected_target_tokens = select_chunks(chunks, keep_rate, anchor_tokens)
    compressed_prompt = delimiter.join(chunk.text for chunk in selected)
    original_target_tokens = sum(chunk.token_count for chunk in chunks)
    compressed_target_tokens = len(_encode(target_tokenizer, compressed_prompt))
    effective_keep_rate = (
        compressed_target_tokens / original_target_tokens if original_target_tokens else 0.0
    )
    return CompressionResult(
        compressed_prompt=compressed_prompt,
        chunks=chunks,
        selected_indices=[chunk.index for chunk in selected],
        original_target_tokens=original_target_tokens,
        selected_target_tokens=selected_target_tokens,
        compressed_target_tokens=compressed_target_tokens,
        keep_rate=keep_rate,
        effective_keep_rate=effective_keep_rate,
        anchor_tokens=anchor_tokens,
        delimiter=delimiter,
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for token in long:
        curr = [0]
        for j, other in enumerate(short, start=1):
            if token == other:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
