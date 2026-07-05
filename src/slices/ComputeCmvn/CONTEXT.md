# ComputeCmvn

## Purpose
Compute global cepstral mean/variance stats over the training log-mels, once,
so the encoder can normalize its input to zero-mean/unit-var per mel bin.

## Entry Point
- Input: `ComputeCmvnCommand`
- Output: dict {"mean":[80],"std":[80]}; side effect: `data/features/cmvn.pt`

## Data Ownership
- Artifact: `data/features/cmvn.pt`

## Notes
Accumulates in float64 for numerical stability, stores float32.
Uses `soundfile`-backed `load_audio` (not torchaudio.load).
