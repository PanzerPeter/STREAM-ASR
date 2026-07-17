# src/slices/ExtractFeatures/FrameBucketSampler.py
import json
import random

from torch.utils.data import Sampler

from src.shared_kernel.Config_Adapter import get_config


class FrameBucketSampler(Sampler):
    """Groups utterances of similar length so each batch fills a frame budget,
    keeping padding low and VRAM near-constant across batches.

    With ``shuffle`` the length-bucketed batches are re-ordered every epoch (a fresh seed per
    epoch), so batch *order* varies while intra-batch length grouping — and thus padding
    efficiency — is preserved. Off (default) the batch stream is fully deterministic, which is
    what dev/val evaluation needs for comparable WER across steps.
    """

    def __init__(
        self,
        manifest: str,
        max_frames_per_batch: int,
        shuffle: bool = False,
        seed: int = 0,
        max_tokens_per_batch: int | None = None,
    ) -> None:
        hop_length = get_config().audio.hop_length
        rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
        self._frames = [r["num_samples"] // hop_length for r in rows]
        # Transcript char count is a cheap upper bound on subword tokens (BPE never expands past
        # chars), so it conservatively caps the RNN-T joiner lattice B*T*(U+1) without a tokenizer.
        self._tokens = [len(r["text"]) for r in rows]
        self._order = sorted(range(len(rows)), key=lambda i: self._frames[i])
        self._max_frames = max_frames_per_batch
        self._max_tokens = max_tokens_per_batch
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def _build_batches(self) -> list[list[int]]:
        batches: list[list[int]] = []
        batch: list[int] = []
        frame_budget = 0
        token_budget = 0
        for idx in self._order:
            over_frames = frame_budget + self._frames[idx] > self._max_frames
            over_tokens = (
                self._max_tokens is not None and token_budget + self._tokens[idx] > self._max_tokens
            )
            if batch and (over_frames or over_tokens):
                batches.append(batch)
                batch, frame_budget, token_budget = [], 0, 0
            batch.append(idx)
            frame_budget += self._frames[idx]
            token_budget += self._tokens[idx]
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._build_batches()
        if self._shuffle:
            # Per-epoch seed: reproducible, but a different batch order each pass so the optimizer
            # never sees the same length-sorted sequence twice (matters most for Stage A, whose
            # SpecAugment was off — batches were otherwise fully deterministic).
            random.Random(self._seed + self._epoch).shuffle(batches)
            self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        total = sum(self._frames)
        return max(1, total // self._max_frames)
