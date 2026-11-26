# Decode

## Purpose
Transcribe audio with the trained transducer, streaming or offline, via a single-pass RNN-T search
(label-looping greedy or a time-synchronous beam).

## Entry Point
- Type: CLI (`streaming_decode.py`) / function call
- Input: `StreamingDecode_Command`; also `StreamingDecoder_Handler.decode_waveform` for an
  in-memory waveform and `StreamingSession` for incremental live audio
- Output: `StreamingDecode_Response`

## Data Ownership
- Consumes artifacts: `data/checkpoints/transducer_avg.pt` (the averaged tail of
  `TrainAcousticModel`'s snapshots), `data/tokenizer/bpe500.model`, `data/features/cmvn.pt` (loaded
  inside the encoder), and `data/checkpoints/lm_best.pt` + `data/tokenizer/lm8k.model` (`lm.tokenizer`)
  when `decode.lm_weight > 0`. The LM pair can be overridden per handler (`lm_checkpoint`,
  `lm_tokenizer`), which is how a released `.safetensors` LM is used.
- Produces: nothing on disk.

## Shared Kernel
- `Config_Adapter.get_config().decode`: chunk size, beam size, rescoring weights.
- `Checkpoint_Adapter`, `Tokenizer_Adapter`, `AudioIO_Adapter`: model load, detokenisation, audio.

## Notes
The tie to `TrainAcousticModel` is limited to the model classes it must run: it instantiates
`TransducerModel` (reaching `model.encoder`, `model.predictor`, `model.joiner`, `greedy_decode`)
and reads `FRONTEND_STRIDE`, the mel frames per 25 Hz encoder frame. The model
definitions are the artifact contract, so no trainer handlers, collators, or training-specific
utilities are imported.

**Single pass.** `TransducerBeamSearch` runs a pure-acoustic time-synchronous beam search over the
encoder memory (predictor + joiner), evaluating the whole live beam in one batched predictor+joiner
call per symbol step (batch dim = beam width), so a frame costs a few GPU launches and one host sync
rather than one per hypothesis. When an LM is attached (`fuse_lm`, `decode.lm_weight > 0`),
`StreamingDecoder_Handler._search_rescore` re-ranks the n-best by
`acoustic + alpha·lm_seq - beta·ilm_seq + length_bonus·len`. That is n-best rescoring, not
per-emission shallow fusion, which costs far more GPU time for a corpus decode. Both LM terms are
one batched forward over the whole n-best rather than one call per hypothesis. The external LM has
its own tokenizer, so `LmScorer` takes hypothesis TEXT (decoded with the acoustic tokenizer) and
re-encodes it with `lm.tokenizer`; ILME stays on the acoustic token ids, the path the transducer
actually scored.

**ILME.** `beta` (`decode.ilm_weight`) is internal-LM estimation (`InternalLmScorer`, after
arXiv:2011.01991): the predictor+joiner already carry a language prior learned from the 960 h
transcripts, so adding an external LM on top double-counts it. The internal prior is estimated by
running the joiner with the encoder memory zeroed and renormalising over the non-blank labels, then
subtracted. With this repo's stateless predictor that prior is inherently low-order, which is the
regime LODR argues is the right thing to subtract. `beta = 0` reproduces plain fusion exactly.

**Streaming** feeds the encoder `4·decode.chunk_size` mel frames per step through
`S2Encoder.streaming_forward` with a cache from `init_stream(1, decode.left_context)`: frontend
tails, each block's conv tail, and a ring of the last `left_context` frames of pre-RoPE keys and
values per attention block, so the state is constant-size (1.73 M values, 6.9 MB in fp32, at
5.12 s) for any stream length. Offline runs one full-context `forward`. Both funnel into the same search.

**Tail.** The final partial chunk is completed and `decode.tail_chunks` (1) whole chunks follow;
the encoder mean-fills and key-masks everything past `n_valid`, and EVERY output frame is kept.
Tokens the transducer emits a little late need those frames: dropping them costs ~0.5 dev-other
WER (`MODEL_ARCHITECTURE.md` §7.3). Streaming equals the batched `TransducerModel.decode_window`
to fp32 round-off (`test_streaming_forward_equivalence`, `test_streaming_decode`).

**Greedy.** `beam_size == 1` runs the model's label-looping greedy (`TransducerModel.greedy_decode`),
exact for greedy with one joiner call per emitted token; the beam is the Graves A/B search.

**Emission times.** The response carries the best hypothesis' `token_ids` and `token_emit_s`: per
token, the audio time it could first be shown (end of its chunk streaming, end of its frame
offline). Evaluate turns them into per-word emission latency.
