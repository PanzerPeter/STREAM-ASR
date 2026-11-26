# ComputeCmvn

## Purpose
Compute global cepstral mean/variance statistics over the training log-mels, once, so the encoder can
normalize its input to zero-mean / unit-variance per mel bin.

## Entry Point
- Type: CLI (`scripts/compute_cmvn.py`) → `compute_cmvn`
- Input: `ComputeCmvnCommand`
- Output: `dict {"mean": [80], "std": [80]}`; side effect: `data/features/cmvn.pt`

## Data Ownership
- Consumes artifacts: `data/manifests/train_sp2.jsonl`, `data/manifests/cv-train.jsonl` (the
  `manifests` tuple of `ComputeCmvnCommand`)
- Produces artifact: `data/features/cmvn.pt`

## Shared Kernel
- `AudioIO_Adapter.load_row_audio`: a row's audio as training reads it (trim span, then speed),
  `soundfile`-backed, not `torchaudio.load`.
- `LogMel_Transform`: the same 80-bin front end training uses, so the statistics match.
- `Config_Adapter.get_config().audio`: n_mels, hop, CMVN epsilon.

## Notes
Accumulates in float64 for numerical stability, stores float32. The rows of all manifests are
pooled before sampling (5 % by default), so each source weighs in by its share of training frames;
a spawn pool with one torch thread per worker decodes them, and `imap` keeps the order (and so
the float sums) fixed for a given seed.

The result is baked into every checkpoint's normalisation, so recomputing it against a different
tokenizer, split or front-end configuration invalidates existing checkpoints.
