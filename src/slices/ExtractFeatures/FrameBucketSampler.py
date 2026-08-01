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
        max_lattice_per_batch: int | None = None,
        token_sort_window: int = 1,
    ) -> None:
        hop_length = get_config().audio.hop_length
        rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
        self._frames = [r["num_samples"] // hop_length for r in rows]
        # Transcript char count is a cheap upper bound on subword tokens (BPE never expands past
        # chars), so it conservatively caps the RNN-T joiner lattice B*T*(U+1) without a tokenizer.
        self._tokens = [len(r["text"]) for r in rows]
        self._max_frames = max_frames_per_batch
        self._max_tokens = max_tokens_per_batch
        self._max_lattice = max_lattice_per_batch
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._order = self._sorted_order(token_sort_window)
        self._num_batches: int | None = None

    def _sorted_order(self, window: int) -> list[int]:
        # Duration-sorted, then re-sorted by transcript length inside a sliding window of that
        # order. Frame bucketing alone leaves the RNN-T lattice padded in U: two utterances of the
        # same duration can differ 2x in token count (speaking rate), and the lattice is charged at
        # the batch's *max* U, not its mean. Measured over the 281k utterances of train.jsonl, this
        # drops lattice padding waste from 21.6% to 13.9% -- 8.8% fewer cells for the same audio, on
        # the step's dominant cost, worth 138 -> 126 ms/step -- while keeping frame padding at 0.2%,
        # because the window is narrow enough that every utterance in it has a near-equal duration.
        #
        # A window is a compromise in both directions: too narrow and there is no token spread to
        # exploit, too wide and it starts trading frame padding for token padding. 256/1024/4096
        # measure 15.0/13.9/15.3%.
        #
        # Set from training.transducer.token_sort_window; see that key for why it defaults to off.
        order = sorted(range(len(self._frames)), key=lambda i: self._frames[i])
        if window <= 1 or self._max_tokens is None:
            return order
        return [
            i
            for start in range(0, len(order), window)
            for i in sorted(order[start : start + window], key=lambda j: self._tokens[j])
        ]

    def _build_batches(self) -> list[list[int]]:
        # Three budgets. The two *sum* budgets (frames, tokens) set the average batch size. The
        # *product* budget bounds its worst case: peak VRAM is driven by the RNN-T lattice, which
        # costs B x max(frames) x max(tokens) -- a quantity two independent sum caps never bound.
        # Left uncapped on train_sp, the densest batch of an epoch is 1.7x the p99.9 one, so a few
        # batches per thousand blow the VRAM budget that every other batch sits comfortably inside
        # (observed as periodic "CUDA OOM - batch dropped" during training). Capping costs 14 extra
        # batches in 60,347 and leaves the mean batch size unchanged at 13.98.
        batches: list[list[int]] = []
        batch: list[int] = []
        frame_budget = 0
        token_budget = 0
        frame_max = 0
        token_max = 0
        for idx in self._order:
            frames, tokens = self._frames[idx], self._tokens[idx]
            over_frames = frame_budget + frames > self._max_frames
            over_tokens = self._max_tokens is not None and token_budget + tokens > self._max_tokens
            over_lattice = (
                self._max_lattice is not None
                and (len(batch) + 1) * max(frame_max, frames) * max(token_max, tokens)
                > self._max_lattice
            )
            if batch and (over_frames or over_tokens or over_lattice):
                batches.append(batch)
                batch, frame_budget, token_budget, frame_max, token_max = [], 0, 0, 0, 0
            batch.append(idx)
            frame_budget += frames
            token_budget += tokens
            frame_max = max(frame_max, frames)
            token_max = max(token_max, tokens)
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._build_batches()
        if self._shuffle:
            # Per-epoch seed: reproducible, but a different batch order each pass so the optimizer
            # never sees the same length-sorted sequence twice (without it, and with SpecAugment
            # off, batches would be fully deterministic).
            random.Random(self._seed + self._epoch).shuffle(batches)
            self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        # Count the batches the greedy fill actually produces. Estimating from the frame budget
        # alone ignores the token and lattice budgets and under-reports by ~3x on train_sp; the
        # batch list depends only on the (fixed) sort order, so building it once and caching is
        # exact and costs one pass.
        if self._num_batches is None:
            self._num_batches = len(self._build_batches())
        return self._num_batches
