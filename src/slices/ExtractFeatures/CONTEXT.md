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
`B × max(frames) × max(chars)` on the *worst* batch, which is what peak VRAM tracks.
`token_sort_window` optionally re-sorts by transcript length inside a window of the duration sort so
a batch is homogeneous in `U` as well as `T`; it defaults to off (see the key's comment in
`config/training.yaml`).

SpecAugment runs as a GPU batch op in the trainer, so the dataset yields log-mel without any
CPU-side *spectral* augmentation. The one audio-domain augmentation is 3-way speed perturbation:
when a manifest row carries `speed != 1.0` (from `SpeedPerturbManifest`), `LibriSpeechDataset`
resamples the waveform via `AudioIO_Adapter.speed_perturb` before the log-mel. It requires
resampling raw audio, so it is rejected when a feature cache is attached, because the cache holds
clean, un-perturbed mel. Bucketing is by pre-subsampling frame count
(num_samples // cfg.audio.hop_length), and perturbed rows carry the corrected `num_samples` so the
budget stays accurate.
