# ExtractFeatures

## Purpose
Turn manifest rows into padded log-mel/token training batches.

## Entry Point
- Type: PyTorch Dataset + collate + sampler
- Input: manifest path + `SentencePieceTokenizer`
- Output: `FeatureBatch`

## Data Ownership
- Consumes artifacts: `data/manifests/*.jsonl`, `data/tokenizer/bpe500.model`

## Shared Kernel
- AudioIO_Adapter, LogMel_Transform — audio → features
- Config_Adapter.get_config() — feature/augmentation tunables (`cfg.audio.*`, `cfg.augment.*`)

## Notes
SpecAugment runs as a GPU batch op in the trainer; speed-perturb was dropped (SP1). The dataset
yields clean log-mel.
Bucketing is by pre-subsampling frame count (num_samples // cfg.audio.hop_length).
