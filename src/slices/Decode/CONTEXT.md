# Decode Slice

This slice implements streaming and offline single-pass RNN-T decode for the trained transducer.
It consumes three artifacts from prior slices: `data/checkpoints/transducer_best.pt` (trained
weights), `data/tokenizer/bpe500.model` (SentencePiece BPE tokenizer), and `data/features/cmvn.pt`
(CMVN statistics, loaded inside the encoder). Its tie to `TrainAcousticModel` is limited to the
model classes it must run: it instantiates `TransducerModel` (reaching `model.encoder`,
`model.predictor`, `model.joiner`) and imports `StreamCache`, the companion type of the encoder's
public `streaming_forward` API. The model definitions are the artifact contract — the slice
imports no trainer handlers, collators, or training-specific utilities.

Decode is a single pass: `TransducerBeamSearch` runs **pure-acoustic** time-synchronous beam search
over the encoder memory (predictor + joiner), evaluating the whole live beam in one batched
predictor+joiner call per symbol step (batch dim = beam width) so a frame costs a few GPU launches +
one host sync, not one per hypothesis. When an LM is attached (`fuse_lm`, `decode.lm_weight > 0`)
`StreamingDecoder_Handler._search_rescore` re-ranks the n-best by
`acoustic + alpha·lm_seq − beta·ilm_seq + length_bonus·len` — n-best rescoring, **not** per-emission
shallow fusion (that was dropped to keep corpus decode within its GPU budget). Both LM terms are one
batched forward over the whole n-best, not one call per hypothesis.

`beta` (`decode.ilm_weight`) is **ILME** (`InternalLmScorer`, after arXiv:2011.01991): the
transducer's predictor+joiner already carry a language prior learned from the 960 h transcripts, so
adding an external LM on top double-counts it. The internal prior is estimated by running the joiner
with the encoder memory zeroed and renormalising over the non-blank labels, then subtracted. With
this repo's stateless predictor that prior is inherently low-order, which is the regime LODR argues
is the right thing to subtract. `beta = 0` reproduces plain fusion exactly.

Streaming feeds the encoder
feature-rate chunks of `2·decode.chunk_size` through `streaming_forward` with a carried
`StreamCache`; offline runs one full-context `forward`. Both funnel into the same beam search.

**Known tail approximation.** The streaming path pads the final feature chunk to an aligned size and trims the padding-derived output frames. The encoder is bit-exact vs `forward(chunk_size=B)` for every aligned frame (see `test_streaming_forward_equivalence`), but for utterances whose post-frontend length is odd, the **last 1–2 output frames** differ from the offline reference: the padded frames leak into the boundary chunk through same-chunk attention, and the ×8 downsampling stack cannot separate fewer than 8 real base frames from padding at its rate. The effect is confined to the utterance tail with negligible WER impact; an exact fix would require threading a valid-length mask through every streaming module (attention/conv/downsample) and is deferred.
