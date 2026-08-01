# Decode

## Purpose
Transcribe audio with the trained transducer, streaming or offline, via a single-pass RNN-T beam
search.

## Entry Point
- Type: CLI (`streaming_decode.py`) / function call
- Input: `StreamingDecode_Command`; also `StreamingDecoder_Handler.decode_waveform` for an
  in-memory waveform and `StreamingSession` for incremental live audio
- Output: `StreamingDecode_Response`

## Data Ownership
- Consumes artifacts: `data/checkpoints/transducer_avg.pt` (the averaged tail of
  `TrainAcousticModel`'s snapshots), `data/tokenizer/bpe500.model`, `data/features/cmvn.pt` (loaded
  inside the encoder), and `data/checkpoints/lm_best.pt` when `decode.lm_weight > 0`.
- Produces: nothing on disk.

## Shared Kernel
- `Config_Adapter.get_config().decode`: chunk size, beam size, rescoring weights, `cuda_graph`.
- `Checkpoint_Adapter`, `Tokenizer_Adapter`, `AudioIO_Adapter`: model load, detokenisation, audio.

## Notes
The tie to `TrainAcousticModel` is limited to the model classes it must run: it instantiates
`TransducerModel` (reaching `model.encoder`, `model.predictor`, `model.joiner`) and imports
`StreamCache`, the companion type of the encoder's public `streaming_forward` API. The model
definitions are the artifact contract, so no trainer handlers, collators, or training-specific
utilities are imported.

**Single pass.** `TransducerBeamSearch` runs a pure-acoustic time-synchronous beam search over the
encoder memory (predictor + joiner), evaluating the whole live beam in one batched predictor+joiner
call per symbol step (batch dim = beam width), so a frame costs a few GPU launches and one host sync
rather than one per hypothesis. When an LM is attached (`fuse_lm`, `decode.lm_weight > 0`),
`StreamingDecoder_Handler._search_rescore` re-ranks the n-best by
`acoustic + alpha·lm_seq - beta·ilm_seq + length_bonus·len`. That is n-best rescoring, not
per-emission shallow fusion, which costs far more GPU time for a corpus decode. Both LM terms are
one batched forward over the whole n-best rather than one call per hypothesis.

**ILME.** `beta` (`decode.ilm_weight`) is internal-LM estimation (`InternalLmScorer`, after
arXiv:2011.01991): the predictor+joiner already carry a language prior learned from the 960 h
transcripts, so adding an external LM on top double-counts it. The internal prior is estimated by
running the joiner with the encoder memory zeroed and renormalising over the non-blank labels, then
subtracted. With this repo's stateless predictor that prior is inherently low-order, which is the
regime LODR argues is the right thing to subtract. `beta = 0` reproduces plain fusion exactly.

**Streaming** feeds the encoder feature-rate chunks of `2·decode.chunk_size` through
`streaming_forward` with a carried `StreamCache`; offline runs one full-context `forward`. Both
funnel into the same beam search.

**CUDA-graph step** (`decode.cuda_graph`, opt-in, default off, CUDA-only). When set, the handler
hands `TransducerBeamSearch` a `CudaGraphedTransducerStep` that captures the batched
`predictor.step` + `joiner.step` + `log_softmax` into a CUDA graph at a fixed batch = `beam_size`.
Each symbol step is then one graph replay plus a few host copies instead of a fresh kernel-launch
chain. The searcher pads its live hyps up to `beam_size` and reads back the valid rows, so the
n-best is numerically identical to the eager path (`test_cuda_graph_decode`, GPU-gated). Off keeps
the eager launch-per-step path. RTF is already ≪ 1, so this is latency polish rather than a
correctness dependency.

**Known tail approximation.** The streaming path pads the final feature chunk to an aligned size and
trims the padding-derived output frames. The encoder is bit-exact vs `forward(chunk_size=B)` for
every aligned frame (see `test_streaming_forward_equivalence`), but for utterances whose
post-frontend length is odd, the last 1 to 2 output frames differ from the offline reference: the
padded frames leak into the boundary chunk through same-chunk attention, and the ×8 downsampling
stack cannot separate fewer than 8 real base frames from padding at its rate. The effect is confined
to the utterance tail with negligible WER impact. An exact fix would require threading a
valid-length mask through every streaming module (attention/conv/downsample) and is deferred.
