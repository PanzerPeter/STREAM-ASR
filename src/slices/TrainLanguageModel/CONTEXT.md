# TrainLanguageModel

## Purpose
STREAM-LM: a from-scratch deep-narrow causal Transformer language model with its own tokenizer, used
by the Decode slice to rescore the acoustic decoder's n-best list by hypothesis text.

## Entry Points
- Corpus: `BuildLmCorpus_Command` -> `BuildLmCorpus_Handler(fetch).run` ->
  `BuildLmCorpus_Response` (lines/words per source, held-out lines dropped); side effect:
  `data/lm_text/lm-corpus.txt`. CLI `scripts/build_lm_corpus.py` (fetch = urllib download).
- Tokenizer (`scripts/train_lm_tokenizer.py`): `TrainLmTokenizer_Command` ->
  `train_lm_tokenizer(cmd) -> str` (the `.model` path). An `lm.vocab_size` SentencePiece unigram
  trained on a sampled 2M lines of the corpus, `lm.eos_id` reserved as a control symbol.
- Data prep (`scripts/prepare_lm_data.py`): `PrepareLmData_Command` -> `PrepareLmData_Handler.run`
  -> `train.bin` / `val.bin` (packed `uint16` LM-tokenizer ids, one line per sentence + EOS). The
  handler rejects a tokenizer whose size is not `lm.vocab_size`.
- Training: `TrainLm_Command` -> `TrainLm_Handler.run(cmd) -> float` (best val perplexity),
  writing `lm_best.pt` / `lm_last.pt` / `lm_branch{N}.pt` via `Checkpoint_Adapter`. `train_lm.py`
  is the GPU-run CLI entry point (`python -m src.slices.TrainLanguageModel.train_lm`).
- Model: `StreamLmModel(tokens [B,T], segments=[B,T] | None) -> logits [B,T,lm.vocab_size]`
  (`segments` drives document masking, see below); also exposes `sequence_logprob` and
  `sequence_logprob_batch`, the full-sequence scorers the n-best rescorer calls once per utterance.

## Shared Kernel
- `Config_Adapter.get_config().lm`: the tokenizer path, vocab size and EOS id, and every model and
  schedule hyperparameter (`extra="forbid"`, so a stale key is an error).
- `Transcript_Transform.normalise_transcript`: every corpus line into the acoustic output
  alphabet, the function the Common Voice manifests use, so LM text and transcripts match.
- `AudioIO_Adapter.load_manifest`: cv-train sentences and the held-out dev/test transcripts.
- `Tokenizer_Adapter.SentencePieceTokenizer`: loads `lm.tokenizer` for packing (and, in Decode,
  for scoring).
- `RoPE_Transform`: rotary tables (inverse frequencies built once, held as a buffer).
- `Optimizer_Adapter.partition_params` / `lr_groups`, `Muon_Optimizer`: the Muon + AdamW split, no
  weight decay on vectors or scalars.
- `GradientClipping`: matrices clipped as one group, vectors and scalars per tensor.
- `LrSchedule.lr_at`: the WSD shape the acoustic trainer uses.
- `Checkpoint_Adapter`, `SignalGuard`: atomic checkpointing plus interrupt-safe resume.

## Model
MODEL_ARCHITECTURE.md §8.4. Pre-norm blocks with weightless RMSNorm (`F.rms_norm`):
`CausalGqaAttention` (GQA, QK-norm, RoPE, separate K and V matrices, and a learnable value residual
`v + λ_l · v_0` on every layer below 0, init `value_residual_lambda`), then a bias-free `SwiGluFfn`.
Untied input/output embeddings with a zero-initialised readout, so step 0 starts at loss ln V.

## Training loop
`TrainLm_Handler.run`: Muon on the block matrices; fused AdamW (`betas=(0.9, 0.95)`) on the
embedding, the readout and the `λ_l` scalars, with weight decay on matrices only. WSD (warmup,
constant, 1-sqrt anneal over the last `lr_decay_frac`) to `TrainLm_Command.total_steps`, applied as
a shape on each group's own peak LR. bf16 autocast on CUDA with each block compiled
(`compile_blocks`, static shapes); the loss is fp32 with one log-normaliser serving both CE and the
`lm.z_loss` term. Val perplexity is token-weighted over every window of `val.bin`, run eager, every
`eval_interval` steps and on the last step; `lm_best.pt` is written whenever it improves.

Each step listed in `lm.branch_points` also writes `lm_branch{N}.pt`, never overwritten: the trunk
at the decay start of a shorter budget. Copied into a fresh `--out-dir` as `lm_last.pt` and resumed
with that `--total-steps`, it anneals into the shorter budget's model (COMMANDS.md Step 5).

Resumable and interrupt-safe on the same harness as the transducer trainer: `lm_last.pt` is written
every `ckpt_every` steps, `resume_if_available` restores model, both optimizers, RNG and step on the
next launch (`--fresh` opts out), and `SignalGuard` turns Ctrl-C into a checkpoint-then-exit at the
next step boundary. The best-so-far val perplexity travels in the checkpoint's `extra` so a resumed
run cannot overwrite `lm_best.pt` with a worse model.

## Data Ownership
- Consumes: `data/lm_text/librispeech-lm-norm.txt.gz`, FineWeb-Edu `sample/10BT` parquet shards
  (fetched, read, deleted one at a time), `data/manifests/cv-train.jsonl` and every dev/test
  manifest (held out)
- Produces: `data/lm_text/lm-corpus.txt`, `data/tokenizer/lm8k.{model,vocab}`,
  `data/lm_data/train.bin`, `data/lm_data/val.bin`, `data/checkpoints/lm_best.pt`,
  `data/checkpoints/lm_last.pt`, `data/checkpoints/lm_branch{N}.pt`

Download, tokenizer, packing and training are one user-run sequence; see COMMANDS.md §2, Step 5.

## Notes
**Corpus.** LibriSpeech-LM (book English), FineWeb-Edu split into sentences at `.?!` and line
breaks (fragments under 3 words dropped) until `fineweb_words` survive, and the unique cv-train
sentences; any line equal to a normalised dev/test transcript (LibriSpeech or Common Voice) is
dropped. Each source is written to its own file, then they are interleaved in 64-line blocks by
always advancing the source furthest behind its share, so every prefix is a proportional sample:
`val.bin` (the first `val_words`) monitors all three and a `subset_words` cap trims them evenly.

**Own tokenizer.** Rescoring reads text, so the LM's vocab is independent of the acoustic BPE-500.
Measured tokens per word on the corpus: BPE-500 2.24, 4k 1.38, 8k 1.23, 16k 1.14. 8k halves the
sequence length; 16k would buy another 8 % for twice the tables.

**Windows.** `LmDataset` memory-maps the packed bin and serves non-overlapping windows (window `i`
starts at `i * context_len`); the trainer shuffles them without replacement, so a pass reads every
token once. A bin needs more than `context_len` tokens for one window or `LmDataset` raises.

**EOS is BOS, and document masking.** Windows straddle several corpus lines, so each carries
per-token segment ids and `StreamLmModel.forward(tokens, segments=...)` restricts attention to
earlier tokens of the same line. Segment ids advance ON each EOS: the EOS that ends a line opens
the next, so every line trains as `[EOS] + line`, which is exactly the row `sequence_logprob_batch`
scores a hypothesis as (inputs `[EOS] + ids`, targets `ids + [EOS]`). `segments=None` keeps the
plain causal path that the single-sequence scorers use.
