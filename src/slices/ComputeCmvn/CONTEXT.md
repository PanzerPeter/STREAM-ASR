# ComputeCmvn

## Purpose
Compute global cepstral mean/variance statistics over the training log-mels, once, so the encoder can
normalize its input to zero-mean / unit-variance per mel bin.

## Entry Point
- Type: CLI (`scripts/compute_cmvn.py`) → `compute_cmvn`
- Input: `ComputeCmvnCommand`
- Output: `dict {"mean": [80], "std": [80]}`; side effect: `data/features/cmvn.pt`

## Data Ownership
- Consumes artifact: `data/manifests/train.jsonl`
- Produces artifact: `data/features/cmvn.pt`

## Shared Kernel
- `AudioIO_Adapter.load_audio`: `soundfile`-backed FLAC decode, not `torchaudio.load`.
- `LogMel_Transform`: the same 80-bin front end training uses, so the statistics match.
- `Config_Adapter.get_config().audio`: n_mels, hop, CMVN epsilon.

## Notes
Accumulates in float64 for numerical stability, stores float32. A 15 % sample of the train split is
enough, since the statistics converge well before the full set.

The result is baked into every checkpoint's normalisation, so recomputing it against a different
tokenizer, split or front-end configuration invalidates existing checkpoints.
