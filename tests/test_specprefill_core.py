from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

from specprefill_core import (  # noqa: E402
    assign_scores_by_chunk_text,
    build_target_chunks,
    compress_prompt,
    rouge_l_f1,
)


class ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


class SpecPrefillCoreTests(unittest.TestCase):
    def test_chunk_topk_keeps_anchors_and_high_score_middle(self):
        tokenizer = ToyTokenizer()
        prompt = "abcdefghij" * 10
        chunks = build_target_chunks(prompt, tokenizer, chunk_tokens=10)
        scores = [0.0] * len(chunks)
        scores[5] = 99.0
        assign_scores_by_chunk_text(chunks, scores)

        result = compress_prompt(
            prompt,
            tokenizer,
            chunks,
            keep_rate=0.3,
            anchor_tokens=10,
        )

        self.assertIn(0, result.selected_indices)
        self.assertIn(5, result.selected_indices)
        self.assertIn(len(chunks) - 1, result.selected_indices)
        self.assertLessEqual(result.compressed_target_tokens, result.original_target_tokens)

    def test_rouge_l_identical_is_one(self):
        self.assertEqual(rouge_l_f1("alpha beta gamma", "alpha beta gamma"), 1.0)

    def test_rouge_l_empty_is_zero(self):
        self.assertEqual(rouge_l_f1("", "alpha beta"), 0.0)


if __name__ == "__main__":
    unittest.main()
