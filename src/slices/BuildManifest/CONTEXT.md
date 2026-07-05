# BuildManifest

## Purpose
Index a LibriSpeech split into a JSONL manifest of utterances.

## Entry Point
- Type: CLI / function call
- Input: `BuildManifestCommand`
- Output: `int` (row count); side effect: `manifest_out` JSONL file

## Data Ownership
- Artifacts: `data/manifests/*.jsonl`

## Notes
On-disk file names use VSA `FN-001` display names (`BuildManifest.Handler`);
Python imports use the underscore alias (`BuildManifest_Handler`) because
dotted module names are not importable.
Frame counts come from `soundfile.info` (not `torchaudio.info`) because
torchaudio 2.11 removed its metadata/decode backend.
