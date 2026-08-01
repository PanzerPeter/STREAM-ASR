# TrainLanguageModel

## Purpose
STREAM-LM: a from-scratch deep-narrow causal Transformer language model trained on LibriSpeech-LM
text, used by the Decode slice to rescore the acoustic decoder's n-best list.

## Entry Points
- Data prep: `PrepareLmData_Command` -> `PrepareLmData_Handler.run` -> `train.bin` / `val.bin`
  (packed `uint16` BPE-500 token ids, one line per utterance + EOS).
- Training: `TrainLm_Command` -> `TrainLm_Handler.run(cmd) -> float` (best val perplexity),
  writing `lm_best.pt` / `lm_last.pt` via `Checkpoint_Adapter`. `train_lm.py` is the GPU-run CLI
  entry point (`python -m src.slices.TrainLanguageModel.train_lm`).
- Model: `StreamLmModel(tokens [B,T], segments=[B,T] | None) -> logits [B,T,vocab]` (`segments`
  drives document masking, see below); also exposes `sequence_logprob` and
  `sequence_logprob_batch`, the full-sequence scorers the n-best rescorer calls once per utterance.

## Shared Kernel
- `Config_Adapter.get_config().lm`: every model and schedule hyperparameter.
- `Tokenizer_Adapter.SentencePieceTokenizer`: the acoustic model's BPE-500 vocab, shared verbatim.
- `BiasNorm`, `SwiGluFfn`, `RoPE_Transform`: blocks shared with the acoustic encoder.
- `Optimizer_Adapter.partition_params` / `Muon_Optimizer`: the same Muon + AdamW split.
- `Checkpoint_Adapter`, `SignalGuard`: atomic checkpointing plus interrupt-safe resume.

## Model
Deep-narrow causal Transformer: `BiasNorm` + `SwiGluFfn` (shared with the acoustic encoder),
`CausalGqaAttention` (grouped-query attention with RoPE and QK-norm), tied input/output
embeddings, and value-residual (layer-0 attention values injected into every deeper layer,
`value_residual_lambda`). All hyperparameters come from `config/lm.yaml` via `get_config().lm`
(`d_model`, `layers`, `heads`, `kv_groups`, `context_len`, `lr_peak`, `warmup_steps`,
`total_steps`, etc.), with no hardcoded constants.

## Training loop
`TrainLm_Handler.run`: Muon on the block weight matrices plus AdamW (`betas=(0.9, 0.95)`,
`weight_decay`) on the tied embedding/readout and the norms. That is the same split the acoustic
stack uses, via the shared `Optimizer_Adapter.partition_params` (`lm.optimizer: adamw` falls back to
a single AdamW over everything). Linear warmup -> cosine decay applied as a shape multiplier on each
group's own peak LR, bf16 autocast on CUDA (fp32 on CPU), gradient clipping, and an optional
`lm.z_loss` term squaring the softmax log-normaliser to keep the logits from drifting. Evaluates
val perplexity (bounded to `_VAL_WINDOWS = 1280` windows, so the bound is independent of
`grad_accum`) every `eval_interval` steps and on the final step, checkpointing whenever perplexity
improves. `TrainLm_Command.max_steps` caps the run below `lm.total_steps` for smoke/overfit tests;
production training uses the full `total_steps`.

Resumable and interrupt-safe on the same harness as the acoustic trainers: `lm_last.pt` is written
every `ckpt_every` steps, `resume_if_available` restores model, both optimizers, RNG and step on the
next launch (`--fresh` opts out), and `SignalGuard` turns Ctrl-C into a checkpoint-then-exit at the
next step boundary. The best-so-far val perplexity travels in the checkpoint's `extra` so a resumed
run cannot overwrite `lm_best.pt` with a worse model. The train sampler's generator is seeded
`lm.seed + resume_count`, so each resume draws a fresh stretch of the window stream rather than
replaying what the interrupted run already saw.

## Data Ownership
- Consumes: `data/lm_text/*.txt` (downloaded corpus), `data/tokenizer/bpe500.model`
  (`SentencePieceTokenizer`, shared with the acoustic model's vocab)
- Produces: `data/lm_data/train.bin`, `data/lm_data/val.bin`, `data/checkpoints/lm_best.pt`,
  `data/checkpoints/lm_last.pt`

Download, packing and training are one user-run GPU sequence; see COMMANDS.md, Step 6.

## Notes
`LmDataset` memory-maps the packed `uint16` bin files for cheap random access (nanoGPT-style).
Both train and val bins must contain more tokens than `context_len` or `LmDataset.__len__` goes
negative. Vocab size and `bos_id`/`eos_id` are shared with the acoustic model
(`get_config().model`), which keeps the LM and ASR tokenizers in lockstep.

**BOS is EOS.** `PrepareLmData` packs the corpus as `line tokens + eos_id` and writes no separate
start symbol, so the only sentence-start context that ever gets trained is "previous line's EOS".
Scoring a hypothesis from a distinct start id would read an embedding row the model never saw as an
input, which would make every hypothesis' first-token score noise. Hence
`ModelConfig.bos_id == eos_id`.

**Document masking.** `LmDataset` cuts windows at arbitrary offsets, so nearly every one straddles
several corpus lines. Each window therefore also carries per-token segment ids, and
`StreamLmModel.forward(tokens, segments=...)` restricts attention to earlier tokens of the same
line. Training context then matches decode context exactly: a rescored ASR hypothesis is always a
single sentence scored from BOS with nothing before it. `segments=None` keeps the plain causal path
that the single-sequence scorers use.
