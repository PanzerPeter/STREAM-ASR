# STREAM ASR v2.5: model card

English speech recognition trained from scratch on LibriSpeech 960 h and Common Voice 27.0 English:
a 63.1 M-parameter streaming transducer (S2 encoder, banded RNN-T) that decodes offline and
streaming (320 ms chunks) from one set of weights, plus an optional 52.5 M-parameter STREAM-LM for
n-best rescoring. No pretrained weights went into either model.

## Files

| File | Contents | Where |
|---|---|---|
| `model.safetensors` | transducer (encoder, CTC head, predictor, joiner, CMVN buffers), 63.08 M params, fp32 | release asset |
| `lm.safetensors` | STREAM-LM, 52.49 M params, fp32 | release asset |
| `bpe500.model` | SentencePiece BPE-500 tokenizer of the transducer | this folder |
| `lm8k.model` | SentencePiece 8,192-piece tokenizer of the LM (it scores hypothesis text) | this folder |
| `SHA256SUMS` | checksums of the four files above | this folder |

`model.safetensors` is the average of the last five transducer snapshots (steps 1,177.2k to
1,198.8k of a 1.2M-batch run), bit-exact with the `transducer_avg.pt` the reported numbers were
measured on. Each `.safetensors` header records its source checkpoint's sha256 and the git commit it
was exported from.

## Usage

The weights need the code at tag `v2.5`.

```bash
git clone --branch v2.5 https://github.com/PanzerPeter/STREAM-ASR.git && cd STREAM-ASR
uv venv .venv --python 3.12 && uv pip install -r requirements.txt

V=release/stream-asr-v2.5
gh release download v2.5 -D $V -p '*.safetensors'
(cd $V && sha256sum -c SHA256SUMS)

.venv/bin/python -m src.slices.Decode.streaming_decode AUDIO.flac --offline \
    --checkpoint $V/model.safetensors --tokenizer $V/bpe500.model
```

Drop `--offline` for streaming. Input is 16 kHz mono; other rates must be resampled first.

**LM rescoring** is off until `--lm-weight` is above zero. It re-ranks the beam's n-best with the
LM and subtracts an internal-LM estimate, so it takes the three weights and the LM's own files:

```bash
.venv/bin/python -m src.slices.Decode.streaming_decode AUDIO.flac --offline \
    --checkpoint $V/model.safetensors --tokenizer $V/bpe500.model \
    --lm-checkpoint $V/lm.safetensors --lm-tokenizer $V/lm8k.model \
    --lm-weight 0.8 --ilm-weight 0.2 --length-bonus 0.5
```

`0.8 / 0.2 / 0.5` is the setting tuned on dev-other and the recommended default: on all three test
sets it stays within 0.05 WER points of weights tuned per split (below). `evaluate.py` tunes them on
dev for any split. The same flags work on the local web demo
(`python -m src.slices.Demo.serve_demo`). In plain PyTorch the files are ordinary state_dicts:
`TransducerModel().load_state_dict(safetensors.torch.load_file(...))`.

## Results

Test sets, from `runs/eval/report-{clean,other,cv}.json`. Rescoring weights (α, β, length bonus)
were tuned on the matching dev split (Common Voice: a 2,000-clip slice of its dev set), per mode,
never on test. The last row runs one setting (0.8, 0.2, 0.5) on all three, from
`report-{clean,other,cv}-shipped.json`. Streaming uses 320 ms chunks, 5.12 s of left context and one
kept tail chunk. WER %, offline / streaming:

| Decoding | test-clean | test-other | Common Voice test |
|---|---|---|---|
| greedy | 2.83 / 3.80 | 6.55 / 9.07 | 15.69 / 20.07 |
| beam 10 | 2.89 / 3.76 | 6.50 / 8.75 | 15.43 / 19.31 |
| beam 10 + LM + ILME | **2.29 / 2.88** | **5.46 / 7.19** | **13.42 / 16.80** |
| beam 10 + LM + ILME, one setting | 2.31 / 2.83 | 5.46 / 7.19 | 13.40 / 16.75 |

Tuned (α, β, length bonus), offline / streaming: clean (0.6, 0.1, 1.0) / (0.7, 0.0, 0.75); other
(0.8, 0.2, 0.5) / (0.8, 0.2, 0.5); Common Voice (0.9, 0.3, 1.0) / (1.0, 0.3, 0.5). A length bonus of
1.0 was the top of the grid searched for clean and Common Voice offline, so those two are bounds.
Common Voice test is the hard condition (many accents and microphones), and its 16,402 clips make
the Common Voice column the tightest estimate of the three.

On one RTX 5070 with the LM: RTF 0.017 to 0.018 offline and 0.046 to 0.048 streaming, 11 to 14 ms to
first partial, about 0.1 s to finalize. The streaming state is constant at 6.9 MB for any stream
length. Batch-1 decoding is bound by kernel launches, so the wider encoder does not show in the RTF.

## Training

- Data: LibriSpeech `train-clean-100 + train-clean-360 + train-other-500`, each utterance paired
  with one copy at 0.9× or 1.1× speed (1,933 h), plus Common Voice 27.0 English (CC0), 1.34 M clips
  of 1,934 h after the dev and test clips, speakers and sentences were removed and edge silence
  trimmed. STREAM-LM: LibriSpeech LM text, FineWeb-Edu `sample/10BT` and the unique Common Voice
  training sentences, with every dev and test transcript held out.
- Transducer: banded RNN-T + CTC (0.5), from scratch, Muon + AdamW, WSD learning rate, bf16,
  dynamic chunk training over {full, 320, 640 ms} × {2.56 s, 5.12 s, unlimited} left context,
  1.2M batches in 35.7 h on one RTX 5070. Stack widths `[320, 512, 640, 512]`, chosen over
  `[256, 384, 512, 384]` by a 60k-batch screen.
- STREAM-LM: 16-layer causal Transformer, 140k steps of 64 × 256 tokens in 7.2 h.
- Full recipe: [COMMANDS.md](../../COMMANDS.md). Architecture, invariants and design notes:
  [MODEL_ARCHITECTURE.md](../../MODEL_ARCHITECTURE.md).

## Limitations

- English at 16 kHz only. LibriSpeech is read audiobooks and Common Voice is read sentences from
  volunteers' own microphones: spontaneous, overlapping, far-field or heavily noisy speech is not
  covered, and Common Voice's 13 to 17 % WER shows how much accent and channel variety costs.
- Output is upper-case letters, apostrophes and spaces: no punctuation, and no digits (training
  transcripts containing digits were dropped), so numbers come out only as whatever words the
  model recognised.
- The STREAM-LM is trained on books, Common Voice's sentences and educational web text. It biases
  decoding toward those registers and inherits the biases of web text.

## License

Apache-2.0, weights included. Distributions built on these files must carry the
[NOTICE](../../NOTICE) attribution. LibriSpeech is CC BY 4.0, Common Voice is CC0 1.0 and
FineWeb-Edu is ODC-By 1.0; none of their audio or text is redistributed here.
