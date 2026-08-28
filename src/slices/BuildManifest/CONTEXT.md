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
  count); side effect: `manifest_out` JSONL with a `speed` field per row (paired 2x: the
  original plus one factor drawn per utterance from `speeds`, default 0.9/1.1)

## Data Ownership
- Produces artifacts: `data/manifests/*.jsonl` (incl. `train_sp2.jsonl`, the paired 2x
  speed-perturbed train set) and `data/tokenizer/bpe500.{model,vocab}`.

## Notes
VSA `FN-001` specifies `[Feature].[Role]` naming; files here spell that separator as an underscore
(`BuildManifest_Handler.py`) because a dotted module name is not importable in Python. The role
vocabulary is otherwise unchanged.

Frame counts come from `soundfile.info` rather than `torchaudio.info`, because torchaudio 2.11
removed its metadata/decode backend.

`SpeedPerturbManifest` is a pure manifest transform with no audio decode: it emits each utterance
twice -- untouched, then at one factor drawn from `speeds` -- with a corrected `num_samples`
(`round(orig / speed)`), so `FrameBucketSampler`'s frame budget stays accurate. 2x rather than
icefall's 3x cross product because the binding constraint is disk: the fp16 mel cache is ~53 GB
per copy of the corpus. The factor is `sha1(uttid, seed)`, never the builtin `hash()` (Python
randomises string hashing per process), so a rebuild is byte-identical -- the feature cache is
indexed by manifest row order, so a reshuffle would silently mispair every mel. The waveform is
resampled at feature-extraction time and baked into the cache, which is therefore built per
manifest and bound to it by a fingerprint (ExtractFeatures).

The tokenizer is upstream of everything: changing `vocab_size` invalidates the CMVN statistics, the
packed LM data and every existing checkpoint.
