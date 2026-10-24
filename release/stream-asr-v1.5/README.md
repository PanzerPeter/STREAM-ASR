# STREAM ASR v1.5: model card

English speech recognition trained from scratch on LibriSpeech 960 h: a 55.3 M-parameter Zipformer
transducer that decodes both offline and streaming (320 ms chunks) from one set of weights, plus an
optional 44.7 M-parameter STREAM-LM for n-best rescoring. No pretrained weights or external data
went into the acoustic model.

## Files

| File | Contents | Where |
|---|---|---|
| `model.safetensors` | transducer (encoder, CTC heads, predictor, joiner, CMVN buffers), 55.27 M params, fp32 | release asset |
| `lm.safetensors` | STREAM-LM, 44.69 M params, fp32 (byte-identical to the v1.0 file) | release asset |
| `bpe500.model` | SentencePiece BPE-500 tokenizer shared by both models | this folder |
| `SHA256SUMS` | checksums of the three files above | this folder |

`model.safetensors` is the average of the last five transducer snapshots (steps 577.8k to 599.4k of
a 600k-batch run), bit-exact with the `transducer_avg.pt` the reported numbers were measured on.
Each `.safetensors` header records its source checkpoint's sha256 and the git commit it was
exported from.

## Usage

The weights need the code at tag `v1.5`. They do **not** load into v1.0 code, and v1.0's weights do
not load into v1.5 code, because v1.5 adds a normaliser on every inter-stack trunk (`trunk_norm`).

```bash
git clone --branch v1.5 https://github.com/PanzerPeter/STREAM-ASR.git && cd STREAM-ASR
uv venv .venv --python 3.12 && uv pip install -r requirements.txt

V=release/stream-asr-v1.5
gh release download v1.5 -D $V -p '*.safetensors'
(cd $V && sha256sum -c SHA256SUMS)

.venv/bin/python -m src.slices.Decode.streaming_decode AUDIO.flac --offline \
    --checkpoint $V/model.safetensors --tokenizer $V/bpe500.model
```

Drop `--offline` for streaming. To rescore with the LM, set `lm_checkpoint: $V/lm.safetensors` in
`config/decode.yaml`, then pass the tuned weights to the demo
(`python -m src.slices.Demo.serve_demo --checkpoint ... --tokenizer ... --lm-weight 0.5 --ilm-weight 0.3`)
or let `evaluate.py` tune them on dev. In plain PyTorch, the files are ordinary state_dicts:
`TransducerModel().load_state_dict(safetensors.torch.load_file(...))`.

## Results

LibriSpeech test sets, from `runs/eval/report-{clean,other}.json`. Rescoring weights (α, β, length
bonus) were tuned on the matching dev split, never on test.

| Decoding | test-clean offline | test-clean streaming | test-other offline | test-other streaming |
|---|---|---|---|---|
| greedy | 3.23 % | 4.60 % | 7.88 % | 11.12 % |
| beam 10 | 3.13 % | 4.49 % | 7.58 % | 10.83 % |
| beam 10 + LM + ILME | **2.57 %** | **3.43 %** | **6.57 %** | **9.01 %** |

Tuned (α, β, length bonus): clean (0.5, 0.3, 0.25) offline, (0.7, 0.2, 0.5) streaming; other
(0.7, 0.4, 0.25) offline, (1.1, 0.5, 1.0) streaming. On one RTX 5070: RTF 0.037 offline and 0.072
streaming with the LM, 17 ms to first partial, 0.21 to 0.25 s to finalize.

Against v1.0 (3.43 / 4.45 % clean, 9.17 / 12.24 % other with the same LM) the whole gain is
acoustic: the transducer trained 600k batches over the 2× speed-perturbed corpus (24.1 passes over
1,933 h) instead of 175k over the clean one.

## Training

- Data: LibriSpeech `train-clean-100 + train-clean-360 + train-other-500`, each utterance paired
  with one copy at 0.9× or 1.1× speed. STREAM-LM: the LibriSpeech LM corpus.
- Encoder warm start: BEST-RQ self-supervised pretrain on the same audio, 180k steps.
- Transducer: RNN-T + CTC (0.2) + InterCTC (2 × 0.15), Muon + AdamW, WSD learning rate, bf16,
  dynamic chunk training over {full, 16, 32} base-rate frames. ~27 h on one RTX 5070.
- Full recipe: [COMMANDS.md](../../COMMANDS.md). Architecture and every amplitude bound:
  [MODEL_ARCHITECTURE.md](../../MODEL_ARCHITECTURE.md).

## Limitations

- Read audiobook English at 16 kHz only. Expect a large WER increase on conversational,
  far-field, noisy or accented speech, none of which LibriSpeech covers.
- Output is upper-case and unpunctuated, as in the LibriSpeech transcripts.
- The STREAM-LM is trained on public-domain books and biases decoding toward that register.

## License

Apache-2.0, weights included. Distributions built on these files must carry the
[NOTICE](../../NOTICE) attribution. LibriSpeech is CC BY 4.0 and its audio is not redistributed
here.
