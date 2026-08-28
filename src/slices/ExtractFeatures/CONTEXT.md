# ExtractFeatures

## Purpose
Turn manifest rows into padded log-mel/token training batches.

## Entry Point
- Type: PyTorch Dataset + collate + sampler (`LibriSpeechDataset`, `FeatureCollator`,
  `FrameBucketSampler`), plus the cache builder CLI `scripts/precompute_features.py`
- Input: manifest path + `SentencePieceTokenizer`; `PrecomputeFeaturesCommand` for the cache
- Output: `FeatureBatch`; side effect: the fp16 mel cache

## Data Ownership
- Consumes artifacts: `data/manifests/*.jsonl`, `data/tokenizer/bpe500.model`
- Produces artifact: `data/features/mel/<split>.{f16,index.npy,header.json}`, the fp16 log-mel
  cache, read back through `FeatureCacheReader` as a mmap so the training epoch loop is GPU-bound.

## Shared Kernel
- AudioIO_Adapter, LogMel_Transform: audio → features
- Config_Adapter.get_config(): feature/augmentation tunables (`cfg.audio.*`, `cfg.augment.*`)

## Notes
`FrameBucketSampler` batches by budget, not by a fixed count: `max_frames_per_batch` and
`max_tokens_per_batch` are sums (they bound the average batch), while `max_lattice_per_batch` caps
`B × max(frames) × max(chars)` on the *worst* batch. Under `rnnt_loss: full` that product was peak
VRAM itself; under `pruned` the real joiner sees `[B,T,s_range,V]` and the product bounds the simple
loss's frame-slab bandwidth instead.

Order is a duration sort, optionally re-sorted by transcript length inside a sliding window of that
order (`token_sort_window`, default 1 = off). The full lattice is charged at the batch's **max** `U`,
and two utterances of equal duration differ up to 2× in token count, so a mixed batch pays the spread
on every cell. Measured on `train_sp2` at the shipped budgets (`scripts/measure_lattice_waste.py`):
22.9 % of lattice cells are `U` padding at window 1, 5.6 % at 256, 2.0 % at 1024, which is 21 % less
lattice work per epoch. The window must be far wider than a batch (mean `B` ≈ 22) or it is inert,
since the saving needs a whole batch to fall inside one homogeneous run. It buys nothing under
`pruned`, whose lattice does not depend on `U`. Off by default: batches homogeneous in speaking rate
are a training-semantics change no run has isolated.

The batch list is built once and cached. It depends only on the fixed sort order and the three
budgets, so it is identical every epoch; `__iter__` shuffles a copy, leaving the cached order
canonical so epoch *N*'s permutation depends only on the seed and *N*.

SpecAugment runs as a GPU batch op in the trainer, so the dataset yields log-mel without any
CPU-side *spectral* augmentation. The one audio-domain augmentation is speed perturbation: when a
manifest row carries `speed != 1.0` (from `SpeedPerturbManifest`), the waveform is resampled via
`AudioIO_Adapter.speed_perturb` before the log-mel. That happens at **extraction** time, so the
cache is built PER MANIFEST and already carries the perturbation, so `LibriSpeechDataset`'s cached
path is a plain mmap slice and must never re-apply the factor. The cache-free path still resamples
per row, for a run without a cache.

Because the cache is a flat memmap indexed by manifest ROW ORDER, the manifest is half the data
structure, not metadata: pairing a cache with the wrong manifest reads the right row count and the
right shapes and trains on another utterance's audio. `manifest_fingerprint` (sha1 over each row's
`(uttid, speed)`) is written into the cache header and checked by both `FeatureCacheReader` and
`LibriSpeechDataset.__init__`, turning that into a load-time error. The header is also the
completion marker (it is written last, after the whole flat file), so `cached_utt_count` (header
present, fingerprint matching, index rows and `.f16` byte count agreeing with it) lets
`scripts/precompute_features.py` skip splits it already finished and be restarted after an
interruption. Resume granularity is one split; there is no mid-split resume. Bucketing is by pre-subsampling
frame count (num_samples // cfg.audio.hop_length), and perturbed rows carry the corrected
`num_samples` so the budget stays accurate.
