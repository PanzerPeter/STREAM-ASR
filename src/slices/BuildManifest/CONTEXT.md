# BuildManifest

## Purpose
Index a speech corpus (LibriSpeech splits, Common Voice) into JSONL manifests of utterances, and
train the BPE tokenizer over their transcripts.

## Entry Point
- Type: CLI (`scripts/build_manifests.py`, `scripts/build_cv_manifests.py`,
  `scripts/train_tokenizer.py`, `scripts/write_trim_samples.py`) / function call
- Manifest: `BuildManifestCommand` → `build_manifest` → `int` (row count); side effect:
  `manifest_out` JSONL
- Common Voice: `BuildCommonVoiceManifestCommand` → `build_common_voice_manifest` →
  `BuildCommonVoiceManifestResponse` (row counts, per-filter drops, hours before/after trimming);
  side effect: `cv-train`, `cv-dev`, `cv-test`, `cv-dev2k` JSONL
- Tokenizer: `TrainTokenizerCommand(manifests, ...)` → `train_tokenizer` → `str` (model path);
  side effect: `data/tokenizer/bpe500.{model,vocab}`
- Speed-perturb: `SpeedPerturbManifestCommand` → `build_speed_perturb_manifest` → `int` (row
  count); side effect: `manifest_out` JSONL with a `speed` field per row (paired 2x: the
  original plus one factor drawn per utterance from `speeds`, default 0.9/1.1)

## Data Ownership
- Consumes: `data/{Train,Val,Test}` (LibriSpeech FLAC), `data/CommonVoice/` (an extracted
  `cv-corpus-*/en`: `validated.tsv`, `dev.tsv`, `test.tsv`, `clips/*.mp3`).
- Produces artifacts: `data/manifests/*.jsonl` (incl. `train_sp2.jsonl`, the paired 2x
  speed-perturbed train set, and `cv-{train,dev,test,dev2k}.jsonl`) and
  `data/tokenizer/bpe500.{model,vocab}`.

## Shared Kernel
- `Transcript_Transform.normalise_transcript`: Common Voice sentences into the output alphabet,
  the same function the LM corpus uses.
- `AudioIO_Adapter.load_audio`: MP3 decode + resample to 16 kHz for exact sample counts.
- `Config_Adapter.get_config().trim`: the four silence-trim parameters.

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

**Common Voice.** `cv-train` is `validated.tsv` minus every clip, speaker (`client_id`) and
normalised sentence of `dev.tsv` / `test.tsv`, so no Common Voice score measures a clip, voice or
sentence the model trained on; each filter's count is logged and returned. A sentence the output
alphabet cannot spell (a digit, `&`, `%`, ...) is dropped rather than transcribed wrong, as is an
undecodable clip and one whose transcript runs faster than 25 chars/s of kept audio (a truncated
recording). cv-dev / cv-test stay as published and untrimmed; `cv-dev2k` is an evenly strided
2,000-clip slice of cv-dev (trainer validation, evaluation tuning). Every clip is decoded once,
in a spawn pool: MP3 headers do not give exact 16 kHz lengths.

**Silence trim** (`SilenceTrim.speech_span`, cv-train only). dB frame energy over 25 ms / 10 ms;
speech is above `floor + threshold_fraction * (peak - floor)` with floor / peak the 10th / 99th
percentile; the kept span widens by `margin_s` each side. No trim when `peak - floor <
min_dynamic_range_db` or the span is under `min_span_s`. A trimmed row stores `start_sample` with
`num_samples` as the span length (16 kHz, after resampling); an untrimmed row has no
`start_sample`. Every reader goes through `AudioIO_Adapter.load_row_audio`.

The tokenizer is upstream of everything: changing `vocab_size` invalidates the CMVN statistics, the
packed LM data and every existing checkpoint.
