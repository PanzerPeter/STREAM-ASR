import json
import random

from torch.utils.data import Sampler

from src.shared_kernel.Config_Adapter import get_config


class FrameBucketSampler(Sampler):
    """Groups utterances of similar length so each batch fills a frame budget,
    keeping padding low and VRAM near-constant across batches.

    With ``shuffle`` the length-bucketed batches are re-ordered every epoch (a fresh seed per
    epoch), so batch *order* varies while intra-batch length grouping (and thus padding
    efficiency) is preserved. Off (default) the batch stream is fully deterministic, which is
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
        self._batches: list[list[int]] | None = None

    def _sorted_order(self, window: int) -> list[int]:
        # Duration-sorted, then re-sorted by transcript length inside a sliding window of that
        # order. Frame bucketing alone leaves the RNN-T lattice padded in U: two utterances of the
        # same duration can differ 2x in token count (speaking rate), and the full [B,T,U+1,V]
        # lattice is charged at the batch's *max* U, not its mean. Measured over the 281k utterances
        # of train.jsonl: lattice padding waste 21.6% -> 13.9% (8.8% fewer cells for the same audio,
        # on a term worth ~26% of the step), frame padding still 0.2%, 138 -> 126 ms/step.
        #
        # A window is a compromise in both directions: too narrow and there is no token spread to
        # exploit, too wide and it starts trading frame padding for token padding. 256/1024/4096
        # measure 15.0/13.9/15.3% waste.
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

    def _batch_list(self) -> list[list[int]]:
        # The greedy fill reads only the (fixed) sort order and the three budgets, so it returns the
        # identical list every epoch. Building it once turns a 562k-row pass per epoch -- run in
        # the main process, with the loader stalled behind it -- into one at first use, and stops
        # __len__ and __iter__ from each paying for their own.
        if self._batches is None:
            self._batches = self._build_batches()
        return self._batches

    def _build_batches(self) -> list[list[int]]:
        # Three budgets. The two *sum* budgets (frames, tokens) set the average batch size. The
        # *product* budget bounds its worst case: B x max(frames) x max(tokens) is a quantity two
        # independent sum caps never bound, and the densest batch of an epoch runs 1.7x the p99.9
        # one. Under the full objective that product was peak VRAM itself (the [B,T,U+1,V] lattice)
        # and leaving it uncapped showed up as periodic "CUDA OOM - batch dropped". Under the pruned
        # objective the real lattice no longer scales with U, but the simple loss still streams the
        # same product through its frame-slab loop, so the cap now bounds bandwidth rather than
        # VRAM -- a looser constraint, but the tail it clips is the same tail.
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
        batches = self._batch_list()
        if self._shuffle:
            # Per-epoch seed: reproducible, but a different batch order each pass so the optimizer
            # never sees the same length-sorted sequence twice (without it, and with SpecAugment
            # off, batches would be fully deterministic). Shuffle a copy, so the cached list stays
            # in its canonical order and epoch N's permutation depends only on the seed, not on
            # which permutations preceded it.
            batches = list(batches)
            random.Random(self._seed + self._epoch).shuffle(batches)
            self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        # Count the batches the greedy fill actually produces. Estimating from the frame budget
        # alone ignores the token and lattice budgets and under-reports by ~3x on train_sp.
        return len(self._batch_list())
