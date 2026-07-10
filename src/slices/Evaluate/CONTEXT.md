# Evaluate Slice

This slice produces the project's final quality/latency numbers: corpus WER/CER plus mean RTF and
first-partial latency, across an ablation of the two-pass decoder. It consumes a LibriSpeech
manifest (`data/manifests/{dev,test}.jsonl`) and the same trained artifacts the Decode slice runs
on: `data/checkpoints/stage_b_best.pt` and `data/tokenizer/bpe500.model` (and, for the LM stages,
`data/checkpoints/lm_best.pt` via `decode.lm_weight > 0`). It reuses the Decode slice's
`StreamingDecoder_Handler` per utterance — the model definitions are the artifact contract; no
trainer internals are imported.

## Ablation

`evaluate.py` centralizes the stage → decode-feature mapping and runs each stage × {offline,
streaming}. Stages are cumulative:

| Stage | Decode features |
|---|---|
| `ctc_greedy` | beam_size 1, no rescore, no LM |
| `prefix_beam` | full prefix beam, no rescore, no LM |
| `attn_rescore` | + bidirectional attention rescore |
| `lm_rescore` | + LM `sequence_score` added in the rescorer |
| `lm_fusion` | + LM shallow fusion in the first-pass beam |

The `lm_*` stages require a fusion weight `alpha > 0` (so the LM checkpoint is loaded); `alpha = 0`
reproduces the pre-LM decoder exactly and the script warns that those stages are inactive. Because
the LM contributes nothing at `alpha = 0`, the honest way to evaluate it is `--tune DEV`: the script
sweeps `--lm-grid` on a dev manifest, picks the `alpha` minimising the tune-stage WER, then freezes
it for the test table — so the headline number is never tuned on the test set.

## Artifacts

- Consumes: `data/manifests/*.jsonl`, `data/checkpoints/stage_b_best.pt`,
  `data/checkpoints/lm_best.pt`, `data/tokenizer/bpe500.model`.
- Produces: `runs/eval/report.json` (`EvalConfig.report_path`) — one row per stage × mode.

## Entry point

`PYTHONPATH=. .venv/bin/python -m src.slices.Evaluate.evaluate data/manifests/test.jsonl`
(`--limit N` caps utterances for a smoke run; GPU; user-run). To evaluate the LM, tune then run in
one command: `... evaluate data/manifests/test.jsonl --tune data/manifests/dev.jsonl` (optional
`--lm-grid`, `--tune-limit`, `--tune-stage`). `Metrics.corpus_wer/corpus_cer` wrap `jiwer`;
`EvaluateCorpus_Handler` aggregates over the manifest; config is `config/eval.yaml`. The report
JSON is `{"lm_weight": <alpha>, "rows": [...]}`.
