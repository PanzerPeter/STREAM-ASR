# TrainLanguageModel

## Purpose
STREAM-LM: a from-scratch deep-narrow causal Transformer language model trained on
LibriSpeech-LM text, used by Plan 4's two-pass decode for shallow-fusion / n-best rescoring
alongside the acoustic model's CTC+attention scores.

## Entry Points
- Data prep: `PrepareLmData_Command` -> `PrepareLmData_Handler.run` -> `train.bin` / `val.bin`
  (packed `uint16` BPE-500 token ids, one line per utterance + EOS).
- Training: `TrainLm_Command` -> `TrainLm_Handler.run(cmd) -> float` (best val perplexity),
  writing `lm_best.pt` / `lm_last.pt` via `Checkpoint_Adapter`. `train_lm.py` is the GPU-run CLI
  entry point (`python -m src.slices.TrainLanguageModel.train_lm`).
- Model: `StreamLmModel(tokens [B,T]) -> logits [B,T,vocab]`; also exposes `sequence_logprob`
  (full-sequence scorer, n-best rescoring) and `step_logprob` (incremental scorer with KV cache,
  shallow fusion during streaming decode).

## Model
Deep-narrow causal Transformer: `BiasNorm` + `SwiGluFfn` (shared with the acoustic encoder),
`CausalGqaAttention` (grouped-query attention with RoPE and QK-norm), tied input/output
embeddings, and value-residual (layer-0 attention values injected into every deeper layer,
`value_residual_lambda`). All hyperparameters come from `config/lm.yaml` via `get_config().lm`
(`d_model`, `layers`, `heads`, `kv_groups`, `context_len`, `lr_peak`, `warmup_steps`,
`total_steps`, etc.) — no hardcoded constants.

## Training loop
`TrainLm_Handler.run`: AdamW (`betas=(0.9, 0.95)`, `weight_decay`), linear warmup -> cosine decay
LR schedule, bf16 autocast on CUDA (fp32 on CPU), gradient clipping. Evaluates val perplexity
(bounded to 20 batches) every `eval_interval` steps and on the final step, checkpointing whenever
perplexity improves. `TrainLm_Command.max_steps` caps the run below `lm.total_steps` for
smoke/overfit tests; production training uses the full `total_steps`.

## Data Ownership
- Consumes: `data/lm_text/*.txt` (downloaded corpus), `data/tokenizer/bpe500.model`
  (`SentencePieceTokenizer`, shared with the acoustic model's vocab)
- Produces: `data/lm_data/train.bin`, `data/lm_data/val.bin`, `data/checkpoints/lm_best.pt`,
  `data/checkpoints/lm_last.pt`

## Command sequence (user, download + GPU)
```
.venv/bin/python scripts/download_lm_text.py                       # fetch librispeech-lm-norm.txt.gz
gunzip data/lm_text/librispeech-lm-norm.txt.gz
# one-off: PrepareLmData_Handler(SentencePieceTokenizer(...)).run(PrepareLmData_Command(...))
#   -> data/lm_data/train.bin, data/lm_data/val.bin
PYTHONPATH=. .venv/bin/python -m src.slices.TrainLanguageModel.train_lm   # ~2-3 GPU-h
```

## Notes
`LmDataset` memory-maps the packed `uint16` bin files for cheap random access (nanoGPT-style);
both train and val bins must contain more tokens than `context_len` or `LmDataset.__len__` goes
negative. Vocab size and `sos_id`/`eos_id` are shared with the acoustic model
(`get_config().model`), keeping the LM and ASR tokenizers in lockstep.
