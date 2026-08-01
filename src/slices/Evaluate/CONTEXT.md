# Evaluate

## Purpose
Produce the project's final quality and latency numbers: corpus WER/CER plus RTF, first-partial
latency and finalize cost, across an ablation of the single-pass transducer decoder.

## Entry Point
- Type: CLI (`evaluate.py` → `EvaluateCorpus_Handler`; GPU, user-run)
- Input: `EvaluateCorpus_Command` (`--clean` / `--other` and the caps/grid/weight overrides)
- Output: `EvaluateCorpus_Response`; side effect: `runs/eval/report-{split}.json`

## Data Ownership
- Consumes artifacts: `data/manifests/*.jsonl`, `data/checkpoints/transducer_avg.pt`,
  `data/checkpoints/lm_best.pt`, `data/tokenizer/bpe500.model`.
- Produces artifact: `runs/eval/report-{split}.json` (`EvalConfig.report_path`), one row per
  stage × mode, each carrying the raw `num_word_errors`/`num_ref_words` counts behind its rate so two
  runs can be compared for significance without re-decoding, plus full provenance (split,
  checkpoint, beam size, per-mode weights, the whole dev sweep, and how timing was measured).

## Shared Kernel
- `Config_Adapter.get_config().eval`: ablation stages, report path, workers, RTF probe size.
- `Tokenizer_Adapter`, `AudioIO_Adapter`: manifest reading and waveform prefetch.

## Notes
The slice reuses the Decode slice's `StreamingDecoder_Handler` per utterance. The model definitions
are the artifact contract, and no trainer internals are imported. `Metrics.word_errors/char_errors`
return raw counts (the summed-counts identity with `corpus_wer`/`corpus_cer` is test-locked), and
`EvaluateCorpus_Handler` aggregates over the manifest and drives the heartbeat in `Progress.py`.

**Split selection.** `--clean` / `--other`, exactly one, with no default. A split is an acoustic
condition rather than a file, so each one binds all three paths together:

| Flag | Scored | Tuned on | Report |
|---|---|---|---|
| `--clean` | `data/manifests/test.jsonl` (test-clean, n=2,620) | `dev.jsonl` (dev-clean) | `runs/eval/report-clean.json` |
| `--other` | `test-other.jsonl` (n=2,939) | `dev-other.jsonl` | `runs/eval/report-other.json` |

Binding them removes three ways to publish a wrong number: weights tuned on the wrong acoustic
condition, a test-other table overwriting the test-clean report, and a WER whose split you have to
reconstruct from a filename.

**Ablation.** `evaluate.py` centralizes the stage → decode-feature mapping and runs each stage ×
{offline, streaming}. Stages are cumulative:

| Stage | Decode features |
|---|---|
| `greedy_transducer` | beam_size 1, no LM |
| `beam` | full beam search, no LM |
| `beam_lm` | + LM n-best rescoring (`fuse_lm=True`): re-rank the acoustic beam by `acoustic + alpha·lm_seq - beta·ilm_seq` |

**Two passes: quality, then timing.** *Quality* decodes every stage × mode concurrently over the
full manifest, in worker processes (`eval.workers`, default 4). A single decode leaves the GPU
~70 % idle, but the idle is Python: the beam's recombination and per-symbol host sync are what it
waits on, and the GIL serialises exactly that. Measured over 6 passes × 40 utterances on the 5070:

| Workers | Threads | Processes |
|---|---|---|
| 1 | 100.7 s | 96.0 s |
| 6 | 89.5 s (1.1×, GPU ~30 %) | 53.5 s (1.8×, GPU 85 %, 5.3 GB) |

Threads buy nothing here, processes do, and the gain flattens past 3 or 4 workers because the GPU is
already at ~85 %. Each worker rebuilds its own model from the checkpoint and shares nothing but the
read-only file, which is why the ceiling is VRAM rather than cores. The parent process holds no
model at all until the timing pass, and CUDA-graph capture stops being a hazard since each process
has one thread.

*Timing* then re-runs the same configurations alone and serially over `eval.rtf_probe_utts` (200)
evenly strided utterances. RTF, first-partial latency and finalize cost mean nothing under
contention, so they are measured only here and are `null` when the probe is off (`--rtf-probe 0`).

Reported timing is deliberately literal:

- `rtf` is `sum(decode_s) / sum(audio_s)` rather than the mean of per-utterance ratios, because
  LibriSpeech utterances run 1.3 s to 35 s and a mean of ratios lets the short ones outvote the long
  ones.
- `latency_s` is `null` offline: an offline pass emits no partials, and 0.0 would read as "instant".
- `finalize_s` is everything after the encoder consumed the last chunk (search + n-best rescoring),
  meaning what a live session still owes once the audio stops. The LM's rescoring cost shows up here.
- Audio IO is excluded from all of it (the harness prefetches waveforms on a separate thread), so
  the numbers are the model's rather than libsndfile's.

Caps (`--limit`, `--tune-limit`, `--rtf-probe`) take an evenly strided subsample, never a head
slice: manifests are sorted by uttid, so `rows[:n]` is a couple of speakers reading a couple of
chapters.

**Rescoring weights.** Tuning runs automatically whenever a stage uses the LM, unless `--no-tune` or
an explicit `--lm-weight`/`--ilm-weight` is given. Dev is decoded once per mode acoustic-only. Each
utterance's n-best is cached with its `(acoustic, LM-sequence, internal-LM-sequence)` terms kept
apart, and the whole `--lm-grid` × `--ilm-grid` is then swept over that cache for free, since
neither weight moves the acoustic beam and so no grid point costs another decode. Alignment is done
once per hypothesis too, so a grid point is an argmax plus a table lookup.

Offline and streaming get their own (α, β): streaming's acoustic scores are weaker, and the weight
that best offsets them is not the offline one. The ranking expression must match the live decoder's
(`StreamingDecoder_Handler._search_rescore`) so the weights chosen here are the weights used there.

Tuning also prints the n-best oracle WER: the corpus WER you would get by picking, per utterance,
the lowest-error hypothesis the beam produced. It is the floor any rescoring can reach. A tuned WER
close to it means the acoustic beam's coverage is the bottleneck (widen `beam_size`), while a large
gap means the rescoring terms still have room.
