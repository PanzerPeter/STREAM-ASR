# STREAM ASR v2.0: model card

English speech recognition trained from scratch on LibriSpeech 960 h: a 36.7 M-parameter streaming
transducer (S2 encoder, banded RNN-T) that decodes offline and streaming (320 ms chunks) from one set
of weights, plus an optional 44.7 M-parameter STREAM-LM for n-best rescoring. No pretrained weights
or external data went into the acoustic model.

## Files

| File | Contents | Where |
|---|---|---|
| `model.safetensors` | transducer (encoder, CTC head, predictor, joiner, CMVN buffers), 36.73 M params, fp32 | release asset |
| `lm.safetensors` | STREAM-LM, 44.69 M params, fp32 | release asset |
| `bpe500.model` | SentencePiece BPE-500 tokenizer shared by both models | this folder |
| `SHA256SUMS` | checksums of the three files above | this folder |

`model.safetensors` is the average of the last five transducer snapshots (steps 577.8k to 599.4k of
a 600k-batch run), bit-exact with the `transducer_avg.pt` the reported numbers were measured on.
Each `.safetensors` header records its source checkpoint's sha256 and the git commit it was
exported from.

## Usage

The weights need the code at tag `v2.0`.

```bash
git clone --branch v2.0 https://github.com/PanzerPeter/STREAM-ASR.git && cd STREAM-ASR
uv venv .venv --python 3.12 && uv pip install -r requirements.txt

V=release/stream-asr-v2.0
gh release download v2.0 -D $V -p '*.safetensors'
(cd $V && sha256sum -c SHA256SUMS)

.venv/bin/python -m src.slices.Decode.streaming_decode AUDIO.flac --offline \
    --checkpoint $V/model.safetensors --tokenizer $V/bpe500.model
```

Drop `--offline` for streaming. Input is 16 kHz mono; other rates must be resampled first.

**LM rescoring.** Set `lm_checkpoint: release/stream-asr-v2.0/lm.safetensors` in
`config/decode.yaml`, then pass rescoring weights, for example to the local web demo:

```bash
.venv/bin/python -m src.slices.Demo.serve_demo --checkpoint $V/model.safetensors \
    --tokenizer $V/bpe500.model --lm-weight 0.4 --ilm-weight 0.2
```

or let `evaluate.py` tune them on dev. In plain PyTorch the files are ordinary state_dicts:
`TransducerModel().load_state_dict(safetensors.torch.load_file(...))`.

## Results

LibriSpeech test sets, from `runs/eval/report-{clean,other}.json`. Rescoring weights (α, β, length
bonus) were tuned on the matching dev split, never on test. Streaming uses 320 ms chunks, 5.12 s of
left context and one kept tail chunk.

| Decoding | test-clean offline | test-clean streaming | test-other offline | test-other streaming |
|---|---|---|---|---|
| greedy | 3.17 % | 4.24 % | 7.91 % | 10.92 % |
| beam 10 | 3.15 % | 4.24 % | 7.76 % | 10.48 % |
| beam 10 + LM + ILME | **2.64 %** | **3.18 %** | **6.51 %** | **8.87 %** |

Tuned (α, β, length bonus): clean (0.4, 0.2, 0.5) offline, (0.6, 0.1, 1.0) streaming; other
(0.7, 0.2, 0.75) offline, (0.6, 0.3, 0.25) streaming. On one RTX 5070 with the LM: RTF 0.019
offline and 0.049 streaming, 13 ms to first partial, 0.12 s to finalize. Streaming emission
latency per word: median 80 ms, p90 240 ms. The streaming state is constant at 5.3 MB for any
stream length.

## Training

- Data: LibriSpeech `train-clean-100 + train-clean-360 + train-other-500`, each utterance paired
  with one copy at 0.9× or 1.1× speed (1,933 h). STREAM-LM: the LibriSpeech LM corpus.
- Transducer: banded RNN-T + CTC (0.5), from scratch, Muon + AdamW, WSD learning rate, bf16,
  dynamic chunk training over {full, 320, 640 ms} × {2.56 s, 5.12 s, unlimited} left context,
  600k batches (24 passes over the data). 15.1 h on one RTX 5070.
- Full recipe: [COMMANDS.md](../../COMMANDS.md). Architecture, invariants and design notes:
  [MODEL_ARCHITECTURE.md](../../MODEL_ARCHITECTURE.md).

## Limitations

- Read audiobook English at 16 kHz only. Expect a large WER increase on conversational, far-field,
  noisy or accented speech, none of which LibriSpeech covers.
- Output is upper-case and unpunctuated, as in the LibriSpeech transcripts.
- The STREAM-LM is trained on public-domain books and biases decoding toward that register.

## License

Apache-2.0, weights included. Distributions built on these files must carry the
[NOTICE](../../NOTICE) attribution. LibriSpeech is CC BY 4.0 and its audio is not redistributed
here.
