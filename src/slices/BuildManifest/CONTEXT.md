# BuildManifest

## Purpose
Index a LibriSpeech split into a JSONL manifest of utterances, and train the BPE tokenizer over its
transcripts.

## Entry Point
- Type: CLI (`scripts/build_manifests.py`, `scripts/train_tokenizer.py`) / function call
- Manifest: `BuildManifestCommand` → `build_manifest` → `int` (row count); side effect:
  `manifest_out` JSONL
- Tokenizer: `TrainTokenizerCommand` → `train_tokenizer` → `str` (model path); side effect:
  `data/tokenizer/bpe500.{model,vocab}`
- Speed-perturb: `SpeedPerturbManifestCommand` → `build_speed_perturb_manifest` → `int` (row
  count); side effect: `manifest_out` JSONL with a `speed` field per row (3-way 0.9/1.0/1.1)

## Data Ownership
- Produces artifacts: `data/manifests/*.jsonl` (incl. `train_sp.jsonl`, the 3-way speed-perturbed
  train set) and `data/tokenizer/bpe500.{model,vocab}`.

## Notes
VSA `FN-001` specifies `[Feature].[Role]` naming; files here spell that separator as an underscore
(`BuildManifest_Handler.py`) because a dotted module name is not importable in Python. The role
vocabulary is otherwise unchanged.

Frame counts come from `soundfile.info` rather than `torchaudio.info`, because torchaudio 2.11
removed its metadata/decode backend.

`SpeedPerturbManifest` is a pure manifest transform with no audio decode: it emits each utterance
once per speed factor with a corrected `num_samples` (`round(orig / speed)`), so
`FrameBucketSampler`'s frame budget stays accurate. The waveform itself is resampled at load time by
`LibriSpeechDataset` (ExtractFeatures), keyed off the row's `speed`, which is incompatible with the
precomputed mel cache.

The tokenizer is upstream of everything: changing `vocab_size` invalidates the CMVN statistics, the
packed LM data and every existing checkpoint.
