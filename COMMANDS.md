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

# Step 2: data foundation (LibriSpeech 960 h + Common Voice)
PYTHONPATH=. python scripts/build_manifests.py
PYTHONPATH=. python scripts/build_cv_manifests.py
PYTHONPATH=. python scripts/train_tokenizer.py
PYTHONPATH=. python scripts/build_speed_perturb_manifest.py
PYTHONPATH=. python scripts/compute_cmvn.py
PYTHONPATH=. python scripts/precompute_features.py

# Step 3: transducer, from scratch, with branched cooldowns
python -m src.slices.TrainAcousticModel.train_transducer
tensorboard --logdir runs/transducer

# Step 4: average the newest rolling snapshots into the decode checkpoint (required)
PYTHONPATH=. python scripts/average_checkpoints.py --last-n 5

# Step 5: STREAM-LM for n-best rescoring (corpus and packing in §2)
PYTHONPATH=. python scripts/build_lm_corpus.py
PYTHONPATH=. python scripts/train_lm_tokenizer.py
PYTHONPATH=. python scripts/prepare_lm_data.py
python -m src.slices.TrainLanguageModel.train_lm

# Step 6: decode a single file
python -m src.slices.Decode.streaming_decode data/Val/dev-clean/1272/128104/1272-128104-0000.flac --offline
python -m src.slices.Decode.streaming_decode data/Val/dev-clean/1272/128104/1272-128104-0000.flac

# Step 7: corpus evaluation (tunes on the split's dev, reports on its test)
python -m src.slices.Evaluate.evaluate --clean
python -m src.slices.Evaluate.evaluate --other
python -m src.slices.Evaluate.evaluate --cv

# Step 8: local web demo at http://127.0.0.1:8000
python -m src.slices.Demo.serve_demo --lm-weight 0.8 --ilm-weight 0.2 --length-bonus 0.5

# Step 9: export the release weights (§2, Step 9)
```

Measured wall clock on one RTX 5070: transducer 35.7 h for 1.2M batches (0.107 s per batch with
validation), STREAM-LM 7.2 h for 140k steps, a LibriSpeech corpus evaluation ~1 h and cv-test
~3.5 h.

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
* Common Voice Scripted Speech 27.0 English extracted into `data/CommonVoice/` (`validated.tsv`,
  `dev.tsv`, `test.tsv`, `clips/`; 2,591,456 clips). The archive is 95 GB and extracts
  single-threaded in ~1 h with a peak of ~185 GB of disk, then can be deleted:

  ```bash
  mkdir -p data/CommonVoice
  tar -xzf data/cv-corpus-27.0-2026-09-11-en.tar.gz -C data/CommonVoice \
    --strip-components=2 cv-corpus-27.0-2026-09-11/en
  ```

  `build_cv_manifests.py` (`--corpus-dir`, `--workers`) writes
  `cv-train` (validated minus every dev/test clip, speaker and sentence; silence-trimmed per
  `config/trim.yaml`), `cv-dev`, `cv-test` (as published, untrimmed) and `cv-dev2k` (a strided
  2,000-clip slice of cv-dev), and prints each filter's drop count and the hours before and after
  trimming. `write_trim_samples.py --out DIR [--n 20]` writes before/after WAV pairs to listen to.
* BPE-500 SentencePiece tokenizer over `train` + `cv-train` transcripts
  (`data/tokenizer/bpe500.model`), CMVN over a 5 % sample of `train_sp2` + `cv-train` as the
  trainer reads them (`data/features/cmvn.pt`), the paired 2× speed-perturbed `train_sp2.jsonl`
  (562,482 rows, 1,933 h), and the fp16 mel caches `data/features/mel/<split>.{f16,index.npy,
  header.json}` for every split the trainer reads (`train_sp2` 104 GiB, `cv-train` 104 GiB). Re-running `precompute_features.py` skips finished splits; delete a
  `<split>.header.json` to force one to rebuild.
* A mel cache is bound to the manifest it was built from by a fingerprint in its header, because
  row order is the cache index: `--train-manifest` and `--train-cache-split` always move together.
* Changing the tokenizer invalidates the LM, the caches' token budgets and every checkpoint.

### Step 3: transducer training

```bash
python -m src.slices.TrainAcousticModel.train_transducer            # resumes transducer_last.pt
python -m src.slices.TrainAcousticModel.train_transducer --fresh    # ignore it, start at step 0
#   flags: --train-manifest/--train-cache-split (repeatable, paired in order; giving them replaces
#          the default train_sp2 + cv-train sources), --cache-dir, --total-steps, --log-dir,
#          --ckpt-dir, --device
#   where a step's time goes, with the real loader and budgets (free GPU, ~3 min):
PYTHONPATH=. python scripts/profile_transducer_step.py
```

* **Recipe** (MODEL_ARCHITECTURE.md §9): S2 encoder `[4, 8, 0, 8]`, CTC head (weight 0.5) +
  predictor + banded-RNN-T joiner, Muon 1e-2 / AdamW 7.5e-4, from scratch, 1,200,000 loader
  batches over `train_sp2` + `cv-train` (WSD: 5,000 warmup, hold, 1-√ anneal over the last 25 %),
  3 batches per update. The first 10,000 batches run the full lattice, then the band.
* **`total_steps` counts loader batches.** 1.2M batches of at most 28k frames are about 24 passes
  over `train_sp2` + `cv-train`.
* **Wall clock**: the 1.2M run took 35.7 h (0.107 s per batch averaged with validation), the 600k
  branch anneal 4.9 h more. Ctrl-C and re-run resumes, so a run can span days.
* **Branched cooldowns** (MODEL_ARCHITECTURE.md §9.3). The run keeps `transducer_branch450000.pt`,
  the trunk at the WSD decay start of a 600k budget. That budget is its own anneal resumed from the
  file into its own directory (the same pattern serves any branch point added to `branch_points`):

  ```bash
  mkdir -p data/checkpoints/b600k && cp data/checkpoints/transducer_branch450000.pt data/checkpoints/b600k/transducer_last.pt
  python -m src.slices.TrainAcousticModel.train_transducer --total-steps 600000 \
    --ckpt-dir data/checkpoints/b600k --log-dir runs/transducer_b600k
  ```

  Never judge a WSD run by its constant phase: most of the gain arrives in the anneal.
* **What to watch** (MODEL_ARCHITECTURE.md §10): `dev/transducer_wer_stream_other` (the selection
  metric), `dev/transducer_wer_stream_cv` (Common Voice, cv-dev2k), `dev/band_loss_gap` (< 0.01 nats/token), `train/band_infeasible` (must be 0),
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
PYTHONPATH=. python scripts/average_checkpoints.py --ckpt-dir data/checkpoints/b600k \
  --last-n 5 --out data/checkpoints/b600k/transducer_avg.pt              # a cooldown branch
```

Element-wise mean of the newest `transducer_step*.pt` snapshots: same architecture, same RTF, lower
WER. Decode, Evaluate and the demo load `transducer_avg.pt` by default.

### Step 5: STREAM-LM

`data/checkpoints/lm_best.pt` (512-d, 16 layers, 52.5 M parameters, MODEL_ARCHITECTURE.md §8.4)
scores hypothesis text through its own tokenizer, `data/tokenizer/lm8k.model`. To train it:

```bash
python scripts/download_lm_text.py                  # LibriSpeech-LM, once (read as .gz)
PYTHONPATH=. python scripts/build_lm_corpus.py      # -> data/lm_text/lm-corpus.txt
PYTHONPATH=. python scripts/train_lm_tokenizer.py   # -> data/tokenizer/lm8k.{model,vocab}
PYTHONPATH=. python scripts/prepare_lm_data.py      # -> data/lm_data/{train,val}.bin
python -m src.slices.TrainLanguageModel.train_lm    # auto-resumes from lm_last.pt; --fresh to restart
```

* `build_lm_corpus.py` writes LibriSpeech-LM (~803M words), FineWeb-Edu `sample/10BT` (ODC-BY,
  sentences of 3+ words, shards fetched one at a time and deleted after reading, until 1B words
  survive) and the unique cv-train sentences, all through `normalise_transcript`, with every line
  equal to a LibriSpeech or Common Voice dev/test transcript held out. The sources are interleaved
  in 64-line blocks in proportion to their sizes, so `val.bin` (the first `lm.val_words`) samples
  all three and `lm.subset_words` (2B, above the corpus) packs all of it. Needs network access to
  huggingface.co.
* `train_lm_tokenizer.py` trains an 8,192-piece SentencePiece unigram on a 2M-line sample of the
  corpus, with `lm.eos_id` reserved as a control symbol. It is independent of the acoustic BPE-500;
  `prepare_lm_data.py` and the decode-time scorer both read `lm.tokenizer`.
* `prepare_lm_data.py` packs the corpus with the LM tokenizer: re-run it after every LM tokenizer
  change, then train with `--fresh`.
* **Budget and branch.** WSD over `lm.total_steps` (140k, one pass over the packed corpus). The run
  keeps `lm_branch52500.pt`, the trunk at the decay start of a 70k budget. Anneal it in its own
  directory to measure what the second half of the corpus buys, and rescore with whichever wins on
  dev WER:

  ```bash
  mkdir -p data/checkpoints/lm_b70k && cp data/checkpoints/lm_branch52500.pt data/checkpoints/lm_b70k/lm_last.pt
  python -m src.slices.TrainLanguageModel.train_lm --total-steps 70000 \
    --out-dir data/checkpoints/lm_b70k --log-dir runs/lm_b70k
  ```
* **What to watch**: `val/ppl` (every window of `val.bin`, token-weighted), `train/loss` (CE without
  the z-loss term, so it is comparable to `val/ppl`), `train/grad_norm` (matrices, pre-clip).

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

* `--clean` / `--other` / `--cv`: exactly one; each binds its test manifest, its dev manifest and
  its report path (`runs/eval/report-{split}.json`). `--cv` scores cv-test (16,402 clips) with
  weights tuned on cv-dev2k. `--on-dev` scores the dev manifest instead, with fixed
  weights, and writes `report-{split}-dev.json`. `--checkpoint` and `--report` override the defaults.
* The weights (α, β, length bonus) are tuned on dev per mode from one cached acoustic decode; a
  weight selected at the top of its grid prints a warning: widen that grid. `--lm-weight`,
  `--ilm-weight` and `--length-bonus` fix the weights instead (unset ones fall back to
  `decode.yaml`), for scoring one shipped setting on every split.
* The report records the transducer, both tokenizers and the LM by path and SHA-256: every run
  writes the same file names, so the path alone does not say which model scored.
* Every row also reports the first-2 / middle / last-2 word error rates with their deletion share,
  and streaming rows the per-word emission latency (median / p90) against a CTC forced alignment
  from the scored model's own CTC head: comparable across decode settings, not across models.
  RTF and first-partial latency come from a separate serial pass.

### Step 8: local demo
FastAPI on `127.0.0.1:8000`: file upload (offline beam) and live microphone (streaming partials,
then an offline re-decode at endpoint). `--checkpoint`, `--tokenizer`, `--lm-weight`,
`--ilm-weight`, `--length-bonus`, `--beam-size` (1 = label-looping greedy), `--host`, `--port`.
The weights default to `config/decode.yaml`'s zeros; pass a tuned triple from
`runs/eval/report-{split}.json` (`weights`).

### Step 9: release export

Export from a clean tree after the code is committed: the header records the git commit, and a
dirty tree is recorded as `<commit>-dirty`.

```bash
V=release/stream-asr-v2.5
PYTHONPATH=. python scripts/export_safetensors.py --src data/checkpoints/transducer_avg.pt --dst $V/model.safetensors
PYTHONPATH=. python scripts/export_safetensors.py --src data/checkpoints/lm_best.pt --dst $V/lm.safetensors
cp data/tokenizer/bpe500.model data/tokenizer/lm8k.model $V/
(cd $V && sha256sum model.safetensors lm.safetensors bpe500.model lm8k.model > SHA256SUMS)
sha256sum data/tokenizer/bpe500.model data/tokenizer/lm8k.model    # must match the two lines above
```

`export_safetensors.py` writes the bare `model` state_dict with the source checkpoint's sha256 in
the header. `.safetensors` files are gitignored and ship as GitHub release assets; the model card,
tokenizer and `SHA256SUMS` are tracked. Every loader accepts a `.safetensors` path directly.

---

## 3. Tests

```bash
PYTHONPATH=. python -m pytest -q --basetemp=$HOME/.cache/stream-asr-pytest   # 246 passed, 2 deselected
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
| `trim.yaml` | cv-train silence trim: `threshold_fraction`, `margin_s`, `min_dynamic_range_db`, `min_span_s` |
| `model.yaml` | `frontend_channels`, `stack_dims`, `stack_blocks`, `head_dim`, `conv_kernel_size`, `gate_floor`, `gate_init`, `rope_base`, `vocab_size` |
| `transducer.yaml` | `predictor_dim`, `predictor_context`, `joiner_dim`, `ctc_loss_scale`, `band_width` |
| `training.yaml` | `transducer`: the three batch budgets, `grad_accum`, WSD shape, `branch_points`, `grad_clip`, guard, cadences, `keep_last_n`, `band_warmup_steps`, `compile_blocks`, `seed`, `chunk_sizes`, `left_contexts`, `spec_augment` |
| `optim.yaml` | `muon_lr`, `adamw_lr`, `muon_momentum`, `ns_steps`, `weight_decay` |
| `lm.yaml` | STREAM-LM tokenizer (`tokenizer`, `vocab_size`, `eos_id`), architecture, WSD schedule and `branch_points`, `compile_blocks`, data sizes |
| `decode.yaml` | `chunk_size`, `left_context`, `tail_chunks`, `beam_size`, `max_symbols`, `lm_checkpoint`, `lm_weight` (α), `ilm_weight` (β), `length_bonus` |
| `eval.yaml` | `ablation_stages`, `report_path`, `workers`, `rtf_probe_utts` |
