# STREAM ASR command reference

Operational companion to [README.md](README.md) and [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md):
every command, its flags, and the failure modes worth knowing before you hit them.

> **Conventions.** Commands assume the repository root as the working directory with the venv
> activated (`source .venv/bin/activate`), so `python` is the 3.12 venv interpreter. Otherwise spell
> it `.venv/bin/python`. `python -m …` already has the repo root on `sys.path`; `scripts/*.py`,
> `pytest` and `mypy` need the `PYTHONPATH=.` prefix.

## 1. Pipeline at a glance

```bash
# Step 1: environment
uv venv .venv --python 3.12
uv pip install -r requirements.txt
python scripts/verify_env.py

# Step 2: data foundation (960 h)
PYTHONPATH=. python scripts/build_manifests.py
PYTHONPATH=. python scripts/train_tokenizer.py
PYTHONPATH=. python scripts/compute_cmvn.py
PYTHONPATH=. python scripts/build_speed_perturb_manifest.py
PYTHONPATH=. python scripts/precompute_features.py

# Step 3: transducer, from scratch, with branched cooldowns
python -m src.slices.TrainAcousticModel.train_transducer
tensorboard --logdir runs/transducer

# Step 4: average the newest rolling snapshots into the decode checkpoint (required)
PYTHONPATH=. python scripts/average_checkpoints.py --last-n 5

# Step 5: STREAM-LM for n-best rescoring (data prep in §2)
python -m src.slices.TrainLanguageModel.train_lm

# Step 6: decode a single file
python -m src.slices.Decode.streaming_decode data/Val/dev-clean/1272/128104/1272-128104-0000.flac --offline
python -m src.slices.Decode.streaming_decode data/Val/dev-clean/1272/128104/1272-128104-0000.flac

# Step 7: corpus evaluation (tunes on the split's dev, reports on its test)
python -m src.slices.Evaluate.evaluate --clean
python -m src.slices.Evaluate.evaluate --other

# Step 8: local web demo at http://127.0.0.1:8000
python -m src.slices.Demo.serve_demo --lm-weight 0.4 --ilm-weight 0.2

# Step 9: export the release weights (§2, Step 9)
```

Measured wall clock on one RTX 5070: transducer 15.1 h (600k batches), STREAM-LM ~5.7 h, each
corpus evaluation ~1.2 h.

---

## 2. Step details

### Step 1: environment
`verify_env.py` checks for CUDA, compute capability `(12, 0)` (RTX 5070, Blackwell sm_120) and a
working `torch.compile`, and prints `OK: ... cap=(12, 0)`. On other hardware, edit its
`EXPECTED_CAPABILITY`. CUDA 12.8 wheels come from the extra index in `requirements.txt`.

### Step 2: data foundation
* 5-split JSONL manifests over `train-clean-100 + train-clean-360 + train-other-500`: train 281,241
  / dev-clean 2,703 / dev-other 2,864 / test-clean 2,620 / test-other 2,939 utterances. The corpus
  is expected under `data/{Train,Val,Test}`.
* BPE-500 SentencePiece tokenizer (`data/tokenizer/bpe500.model`), CMVN over a 15 % train sample
  (`data/features/cmvn.pt`), the paired 2× speed-perturbed `train_sp2.jsonl` (562,482 rows,
  1,933 h), and the fp16 mel cache `data/features/mel/<split>.{f16,index.npy,header.json}`
  (157 GiB). Re-running `precompute_features.py` skips finished splits; delete a
  `<split>.header.json` to force one to rebuild.
* A mel cache is bound to the manifest it was built from by a fingerprint in its header, because
  row order is the cache index: `--train-manifest` and `--train-cache-split` always move together.
* Changing the tokenizer invalidates the LM, the caches' token budgets and every checkpoint.

### Step 3: transducer training

```bash
python -m src.slices.TrainAcousticModel.train_transducer            # resumes transducer_last.pt
python -m src.slices.TrainAcousticModel.train_transducer --fresh    # ignore it, start at step 0
#   flags: --train-manifest/--train-cache-split (move together), --cache-dir, --total-steps,
#          --log-dir, --ckpt-dir, --device
#   where a step's time goes, with the real loader and budgets (free GPU, ~3 min):
PYTHONPATH=. python scripts/profile_transducer_step.py
```

* **Recipe** (MODEL_ARCHITECTURE.md §9): S2 encoder `[4, 8, 0, 8]`, CTC head (weight 0.5) +
  predictor + banded-RNN-T joiner, Muon 1e-2 / AdamW 7.5e-4, from scratch, 600,000 loader batches
  (WSD: 5,000 warmup, hold, 1-√ anneal over the last 25 %), 3 batches per update. The first 10,000
  batches run the full lattice, then the band.
* **`total_steps` counts loader batches.** 600k batches of 28k frames are 24 passes over
  `train_sp2`.
* **Wall clock**: 15.1 h for 600k (0.078 s per batch steady, 0.091 s averaged with validation).
  Ctrl-C and re-run resumes, so a run can span days.
* **Branched cooldowns** (MODEL_ARCHITECTURE.md §9.3). The run keeps `transducer_branch225000.pt`,
  `transducer_branch337500.pt` and `transducer_branch450000.pt`, the trunk at the WSD decay start of
  a 300k, 450k and 600k budget. Each other budget is its own anneal resumed from that file into its
  own directory:

  ```bash
  mkdir -p data/checkpoints/b300k && cp data/checkpoints/transducer_branch225000.pt data/checkpoints/b300k/transducer_last.pt
  python -m src.slices.TrainAcousticModel.train_transducer --total-steps 300000 \
    --ckpt-dir data/checkpoints/b300k --log-dir runs/transducer_b300k
  # a longer budget from the 600k run's decay start, e.g. 1.2M batches (~17 h more)
  mkdir -p data/checkpoints/b1200k && cp data/checkpoints/transducer_branch450000.pt data/checkpoints/b1200k/transducer_last.pt
  python -m src.slices.TrainAcousticModel.train_transducer --total-steps 1200000 \
    --ckpt-dir data/checkpoints/b1200k --log-dir runs/transducer_b1200k
  ```

  Never judge a WSD run by its constant phase: most of the gain arrives in the anneal.
* **What to watch** (MODEL_ARCHITECTURE.md §10): `dev/transducer_wer_stream_other` (the selection
  metric), `dev/band_loss_gap` (< 0.01 nats/token), `train/band_infeasible` (must be 0),
  `train/res_gate_floor_count`, `stack_rms/*_exit`, `train/grad_norm_guarded`.
* **Ctrl-C is safe**: the step finishes, `transducer_last.pt` is written, the process exits.
  Re-launch the same command to resume with a fresh epoch order (`seed + resume_count`).
* **OOM**: a batch that does not fit is dropped with a warning; persistent skips mean the three
  budgets in `config/training.yaml` are too high, and they move together.
* **`expandable_segments`**: the trainer and profiler set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Without it, peak VRAM reads allocator
  fragmentation, not footprint; anything sizing a batch budget must set it too.

### Step 4: checkpoint averaging (required)

```bash
PYTHONPATH=. python scripts/average_checkpoints.py --last-n 5            # -> data/checkpoints/transducer_avg.pt
PYTHONPATH=. python scripts/average_checkpoints.py --ckpt-dir data/checkpoints/b300k \
  --last-n 5 --out data/checkpoints/b300k/transducer_avg.pt              # a cooldown branch
```

Element-wise mean of the newest `transducer_step*.pt` snapshots: same architecture, same RTF, lower
WER. Decode, Evaluate and the demo load `transducer_avg.pt` by default.

### Step 5: STREAM-LM

`data/checkpoints/lm_best.pt` (512-d, 16 layers, 44.7 M parameters, val perplexity 36.75) scores
BPE-500 token ids. To train it:

```bash
python scripts/download_lm_text.py && gunzip -k data/lm_text/librispeech-lm-norm.txt.gz
python - <<'PY'
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainLanguageModel.PrepareLmData_Command import PrepareLmData_Command
from src.slices.TrainLanguageModel.PrepareLmData_Handler import PrepareLmData_Handler

lm = get_config().lm
PrepareLmData_Handler(SentencePieceTokenizer("data/tokenizer/bpe500.model")).run(
    PrepareLmData_Command(
        "data/lm_text/librispeech-lm-norm.txt", "data/lm_data", lm.subset_words, lm.val_words
    )
)
PY
python -m src.slices.TrainLanguageModel.train_lm   # auto-resumes from lm_last.pt; --fresh to restart
```

### Step 6: decode options
* Flags: `--checkpoint` (default `transducer_avg.pt`, or a `.safetensors` release file),
  `--tokenizer`, `--offline`. Everything else comes from `config/decode.yaml`: `chunk_size` 8
  (320 ms) and `left_context` 128 (5.12 s), both in 25 Hz frames and multiples of 4; `tail_chunks`
  1 (kept); `beam_size`; `max_symbols`; the rescoring weights (all 0.0 = the pure-acoustic
  regression lock).

### Step 7: evaluation

```bash
python -m src.slices.Evaluate.evaluate --clean     # test-clean, tuned on dev-clean
python -m src.slices.Evaluate.evaluate --other     # test-other, tuned on dev-other
python -m src.slices.Evaluate.evaluate --clean --stages greedy_transducer,beam --no-tune
python -m src.slices.Evaluate.evaluate --other --on-dev --stages greedy_transducer   # score dev-other itself
# liveness smoke, ~1 min (strided subsamples, never a head slice)
python -m src.slices.Evaluate.evaluate --clean --limit 30 --tune-limit 30 \
  --lm-grid 0.0,0.2,0.4 --ilm-grid 0.0,0.2 --lb-grid 0.0,0.5 --rtf-probe 10
```

* `--clean` / `--other`: exactly one; each binds its test manifest, its dev manifest and its report
  path (`runs/eval/report-{split}.json`). `--on-dev` scores the dev manifest instead, with fixed
  weights, and writes `report-{split}-dev.json`. `--checkpoint` and `--report` override the defaults.
* The weights (α, β, length bonus) are tuned on dev per mode from one cached acoustic decode; a
  weight selected at the top of its grid prints a warning: widen that grid.
* Every row also reports the first-2 / middle / last-2 word error rates with their deletion share,
  and streaming rows the per-word emission latency (median / p90) against a CTC forced alignment.
  RTF and first-partial latency come from a separate serial pass.

### Step 8: local demo
FastAPI on `127.0.0.1:8000`: file upload (offline beam) and live microphone (streaming partials,
then an offline re-decode at endpoint). `--checkpoint`, `--tokenizer`, `--lm-weight`,
`--ilm-weight`, `--beam-size` (1 = label-looping greedy), `--host`, `--port`. The weights default to
`config/decode.yaml`'s zeros; pass the tuned pair from `runs/eval/report-clean.json` (`weights`).

### Step 9: release export

Export from a clean tree after the code is committed: the header records the git commit, and a
dirty tree is recorded as `<commit>-dirty`.

```bash
V=release/stream-asr-v2.0
PYTHONPATH=. python scripts/export_safetensors.py --src data/checkpoints/transducer_avg.pt --dst $V/model.safetensors
PYTHONPATH=. python scripts/export_safetensors.py --src data/checkpoints/lm_best.pt --dst $V/lm.safetensors
cp data/tokenizer/bpe500.model $V/
(cd $V && sha256sum model.safetensors lm.safetensors bpe500.model > SHA256SUMS)
```

`export_safetensors.py` writes the bare `model` state_dict with the source checkpoint's sha256 in
the header. `.safetensors` files are gitignored and ship as GitHub release assets; the model card,
tokenizer and `SHA256SUMS` are tracked. Every loader accepts a `.safetensors` path directly.

---

## 3. Tests

```bash
PYTHONPATH=. python -m pytest -q --basetemp=$HOME/.cache/stream-asr-pytest   # 214 passed, 2 deselected
PYTHONPATH=. python -m pytest -m slow -s                                      # GPU gates: overfit + LM
```

Checkpoint-writing tests produce hundreds of MB; on a tmpfs `/tmp` give `--basetemp` a disk path
and delete it afterwards. The suite locks: streaming ≡ batched (tail chunk and bounded cache
included), label-looping ≡ frame-wise greedy, banded ≡ full RNN-T at full width with float64
gradchecks and the late-burst band repair, the amplitude bounds of MODEL_ARCHITECTURE.md §5, and an
end-to-end CPU training run with a branch point and a resume.

## 4. Lint, format, types

```bash
black src scripts tests && flake8 src scripts tests && PYTHONPATH=. mypy src
```

## 5. Configuration reference

Every tunable is loaded and pydantic-validated by `get_config()` from `config/<section>.yaml`, the
authoritative values. An unknown key in `model.yaml` is an error.

| File | Parameters |
| --- | --- |
| `audio.yaml` | sample rate, n_mels, FFT/window/hop, CMVN epsilon |
| `augment.yaml` | SpecAugment: 2 freq masks ≤ 27 bins, ≤ 25 time masks ≤ 5 % each and 37.5 % in total |
| `features.yaml` | mel cache directory |
| `model.yaml` | `frontend_channels`, `stack_dims`, `stack_blocks`, `head_dim`, `conv_kernel_size`, `gate_floor`, `gate_init`, `rope_base`, `vocab_size` |
| `transducer.yaml` | `predictor_dim`, `predictor_context`, `joiner_dim`, `ctc_loss_scale`, `band_width` |
| `training.yaml` | `transducer`: the three batch budgets, `grad_accum`, WSD shape, `branch_points`, `grad_clip`, guard, cadences, `keep_last_n`, `band_warmup_steps`, `compile_blocks`, `seed`, `chunk_sizes`, `left_contexts`, `spec_augment` |
| `optim.yaml` | `muon_lr`, `adamw_lr`, `muon_momentum`, `ns_steps`, `weight_decay` |
| `lm.yaml` | STREAM-LM architecture, schedule, data sizes, and the BiasNorm window `lm_best.pt` runs under |
| `decode.yaml` | `chunk_size`, `left_context`, `tail_chunks`, `beam_size`, `max_symbols`, `lm_checkpoint`, `lm_weight` (α), `ilm_weight` (β), `length_bonus` |
| `eval.yaml` | `ablation_stages`, `report_path`, `workers`, `rtf_probe_utts` |
