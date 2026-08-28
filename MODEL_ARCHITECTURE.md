# MODEL_ARCHITECTURE.md: STREAM ASR, every operator and what bounds it

Ground truth for the acoustic model's shapes, math and amplitude budget. `CLAUDE.md` says what the
project is and what has gone wrong; `VSA.md` governs structure; **this file says what the model
computes**, operator by operator, so a failure can be located by reading rather than by loading
five checkpoints and diffing them.

Numbers below are the effective config (`get_config()`, authoritative) plus measurements from named
checkpoints. Parameter counts are regenerated from `TransducerModel()` and were last verified
2026-08-22; the config values are from `config/{audio,model,transducer,training,optim,pretrain,decode,augment,features,lm,eval}.yaml`.

**Read [§0 Debug playbook](#0-debug-playbook) and [§5 Amplitude ledger](#5-amplitude-ledger) first
when diagnosing.** Every failure this model has had was an amplitude failure, and four of them were
invisible in the metrics that existed at the time.

---

## 0. Debug playbook

Start from the symptom. Each row says what to read *first*, and what that reading means. The
right-hand column is the section with the mechanism.

| Symptom | Read, in this order | What it usually is | § |
|---|---|---|---|
| `train/grad_norm` climbing, loss fine | `train/grad_norm_guarded` → `stack_mix/*_trunk` → `train/trunk_gain_max` | Almost always ONE scalar gate; the global norm is ~95-99 % `stacks.*.bypass`. Two runs were aborted on this as false positives. If `grad_norm_guarded` is flat, nothing is wrong with the weights, but check the trunk anyway, because a rising gate gradient is what an inflating trunk *looks* like | §11, §5 |
| dev ctc-WER turns upward mid-run | `stack_mix/*_trunk` (all 6) → `stack_mix/*_residual` → `train/gains_at_ceiling` | A trunk RMS rising superlinearly, or a `bypass` approaching its 0.1 floor. Both collapses in 2026-08-2x turned dev WER at the exact step `bypass₂` hit its floor | §5.1, §3.1 |
| a stack's `1 − b` at 0.00 or 1.00 | `stack_mix/{i}_trunk`, then load the checkpoint and check `bypass` | `b → 0.1` means 90 % of that stack's output is bare unnormalised `in_proj`; below the floor it would be an absorbing state with 0 gradient into 10.6 M params. `b → 1` drops the skip: the upsample is `repeat_interleave`, so the stack's output becomes piecewise-constant over its own `factor` base-rate frames and the trunk is the only path carrying detail inside that window. Benign at `factor = 2` (v1.0 runs stack 5 at 1.00) and not at `factor = 8` (v1.0 runs stack 3 at 0.62, both 2026-08-2x collapses at 0.97-0.99). Read it as a **symptom** of an inflated trunk: a pre-normed branch loses relative influence as the residual grows, and `b` rises to compensate | §3.1 |
| `train/gains_at_ceiling` warns | `stack_mix/{i}_processed` per stack → dev ctc-WER | A branch is amplifying as hard as it is allowed. Led the 2026-08-09 divergence by ~25k steps. A bad regime, not yet a diverged one, so the trainer warns rather than aborts | §5.1, §11 |
| `train/trunk_gain_max` pinned at 10.0 | `train/trunk_stable_rank_min` → `stack_mix/*_trunk` | The projection is binding every step. σ₁ is a worst-case bound, so pinned ≠ broken (v1.0 sits at 9.73 with realized trunk RMS 1.5-2.7), but it never plateaued on its own in any diverged run, and while it is pinned **σ₁ is constant by construction and can tell you nothing more**. Read the stable rank instead | §5.2, §5.5, §8.5 |
| `train/trunk_stable_rank_min` falling | `stack_mix/*_trunk` → `train/trunk_gain_max` | The trunk is collapsing toward a rank-1 amplifier. Healthy: 64-95 at the BEST-RQ warm start, 100-120 for stacks that never bind. It halved every ~20k steps in 2026-08-24 while σ₁ read exactly 10.00 at every checkpoint | §5.5 |
| OOM mid-run, periodic | `oom_skips` count in the log → `max_lattice_per_batch` | The RNN-T lattice on the epoch's densest bucket. The trainer drops the batch and continues; persistent skips mean the budgets are too high for the free VRAM | §9.2, §10.2 |
| peak VRAM looks like a ceiling | whether `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set | Allocator fragmentation: 5.25 GiB allocated vs 9.71 reserved on the default allocator, 7.41 vs 7.51 with expandable segments | §13 |
| streaming WER ≫ offline WER | `dev/ctc_wer_stream_gap` through the anneal | The WSD anneal sharpens against whatever context it is given. A widening gap while `dev/ctc_wer` falls means the schedule is buying offline accuracy at streaming's expense | §11, §4 |
| dev/transducer_wer moved a lot | ignore it; read `dev/ctc_wer` | ~1k words, σ ≈ 0.008, and its sampling has changed between runs. Never compare it across runs | §11 |
| loss NaN / inf | `train/rnnt` vs `train/ctc` vs `train/interctc` separately | CTC uses `zero_infinity=True` so it cannot produce inf; RNN-T on an empty transcript returns `−Σ log p(blank)`, not 0 (unlike torchaudio's CUDA kernel) | §6.3 |
| checkpoint won't load | `rnnt_loss` value in `config/training.yaml`; `lm.d_model` for the LM | `pruned` adds `simple_am_proj`/`simple_lm_proj`, so a checkpoint saved under one objective does not load under the other. The LM's `d_model` is load-bearing at inference | §13 |
| step time regressed hours into a run | log for a dynamo recompile-limit warning | Exceeding the per-code-object recompile limit does not raise; dynamo silently demotes that function to eager **for the rest of the process**. `_RECOMPILE_LIMIT = 32` exists for this | §9.5 |
| a metric contradicts another | §11's "blind to" column | `stack_mix/*_residual` and `*_processed` both describe the PROCESSED half and read *healthier* at the 2026-08-22 collapse than 20k steps before it | §11 |

**The one-line rule this model keeps teaching: a parameter bound is not an activation bound.** Four
consecutive fixes bounded a parameter and the amplitude found the next unbounded path. The quantity
that actually failed every time is `stack_mix/*_trunk`, the realized RMS of what each `in_proj`
emits.

---

## 1. Shape and rate ledger

One utterance of `S` samples at 16 kHz, batch `B`, vocabulary `V = 501` (500 BPE + blank 500).

| # | Stage | Op | Shape out | Rate |
|---|---|---|---|---|
| 0 | audio | `soundfile` load, `AudioIO_Adapter` | `[S]` | 16 kHz |
| 1 | log-mel | `LogMel_Transform`, `n_fft=400 win=400 hop=160 n_mels=80` | `[T₀, 80]`, `T₀ = S/160` | 100 fps |
| 2 | CMVN | `(x - cmvn_mean) / cmvn_std`, per mel bin, buffers on the encoder | `[B, T₀, 80]` | 100 fps |
| 3 | frontend | `Conv2dSubsampling` ×2 in time, ×4 in freq | `[B, T₁, 192]`, `T₁ = (T₀-1)//2+1` | 50 fps = **base rate** |
| 4 | stacks 0-5 | `ZipformerStack`, each length-preserving | `[B, T₁, dᵢ]` | base rate in/out |
| 5 | final | `SimpleDownsample(2)` | `[B, T₂, 256]`, `T₂ = ⌈T₁/2⌉` | ~25 fps |
| 6 | out_norm | `BiasNorm(256)` | `[B, T₂, 256]` = **encoder memory** | ~25 fps |
| 7a | CTC head | `Linear(256 → 501)` | `[B, T₂, 501]` | ~25 fps |
| 7b | InterCTC | `Linear(dᵢ → 501)` off stacks 3 and 4 | `[B, T₁, 501]` | base rate |
| 8 | predictor | `StatelessPredictor` on blank-prefixed labels | `[B, U+1, 512]` | per label |
| 9 | joiner | `TransducerJoiner` | `[B, T₂, U+1, 501]` lattice | n/a |
| 10 | loss | `RnntLoss` + CTC + 2×InterCTC | scalar | n/a |

Total frame reduction audio → memory: 160 samples/frame × 2 × 2 = **640 samples ≈ 40 ms per memory
frame**.

Per-stack geometry (`encoder_dims / downsampling / layers / heads`):

| stack | dim | internal ÷ | blocks | heads | head_dim | SwiGLU hidden | params | `in_proj` |
|---|---|---|---|---|---|---|---|---|
| 0 | 192 | 1 | 2 | 4 | 48 | 512 | 1,711,759 | `Identity` (dim_in == dim) |
| 1 | 256 | 2 | 2 | 4 | 64 | 688 | 3,101,841 | `Linear(192→256)` |
| 2 | 384 | 4 | 3 | 6 | 64 | 1024 | 10,321,178 | `Linear(256→384)` |
| 3 | 512 | 8 | 4 | 8 | 64 | 1368 | 24,432,549 | `Linear(384→512)` |
| 4 | 384 | 4 | 3 | 6 | 64 | 1024 | 10,419,482 | `Linear(512→384)` |
| 5 | 256 | 2 | 2 | 4 | 64 | 688 | 3,150,993 | `Linear(384→256)` |

**55,267,106 parameters total** in **609 `state_dict` keys**: encoder 53,778,637 (frontend 640,576
+ stacks 53,137,802 + `final_downsample` 2 + `out_norm` 257), predictor 258,561, joiner 651,253,
CTC head 128,757, InterCTC heads 449,898. 98 `BiasNorm` modules (97 in the encoder + 1 in the
predictor). 16 blocks in the encoder, 96 `BiasNorm` calls per encoder forward plus `out_norm`.

Dropout is `model.encoder_dropout = 0.1`, and it appears in exactly two places per block:
`RotaryAttention` (on the attention weights in train mode, and on its output projection) and
`SwiGluFfn` (after `w_down`). `ConvModule` has none. `BiasNorm.eps` is `audio.cmvn_eps = 1e-5`,
shared with CMVN. It is a numerical floor, not an amplitude bound, and floors nothing at the
scales §5 is about.

---

## 2. Front end

### 2.1 `LogMel_Transform`
80-bin log-mel, 25 ms window / 10 ms hop. Computed on the fly from FLAC, or read from the fp16
mmap cache in `data/features/mel/`, whose row order **is** the manifest row order, so a
`manifest_fingerprint` in the cache header rejects a mismatched pair at dataset construction (§10.1).

### 2.2 CMVN
`x ← (x − mean) / std` with per-bin statistics from `data/features/cmvn.pt`, applied **inside**
`ZipformerEncoder.forward`, i.e. downstream of the mel cache. Raw log-mel is mean −5.65, std 4.06;
anything that fabricates values in mel space (BEST-RQ's mask fill, §7.2) must de-normalise through
these statistics or it writes a constant plateau the encoder never sees again. Missing `cmvn.pt`
falls back to mean 0 / std 1 inside `ZipformerEncoder`: correct for tests, and harmless at
inference, where the statistics are `state_dict` buffers a checkpoint restores. **Both trainers now
raise instead of reaching it** (`_require_cmvn`): a fresh run that did would train on raw log-mel
with nothing in the loop reporting it, and in the BEST-RQ stage the mask fill de-normalises through
the same statistics, so the fallback additionally rebuilds the +1.46 σ plateau §7.2 exists to
prevent. `cmvn_path=""` opts out deliberately.

### 2.3 `Conv2dSubsampling`
```
x:[B,T,80] → [B,1,T,80]
conv1: Conv2d(1→128, k=3, stride=(2,2), pad=(0,1)), left time-pad 2 → ReLU     80 → 40 freq bins
conv2: Conv2d(128→128, k=3, stride=(1,2), pad=(0,1)), left time-pad 2 → ReLU   40 → 20 freq bins
reshape [B, T₁, 128*20 = 2560] → linear: Linear(2560 → 192)
out_lengths = (lengths - 1)//2 + 1
```
Time padding is **left only**, so an output frame reads no future input; frequency padding stays
symmetric (frequency is not streamed). `streaming_forward` keeps a two-level cache (`in_tail` for
conv1's window, `mid_tail` for conv2's) and is exact from frame 0; chunks shorter than
`2*time_pad = 4` frames are rejected with a clear error, because below that the cache silently
under-fills and conv2 fails two chunks later.

**`frontend.linear` is a trunk operator with no normaliser after it and no projection on it.** It
is the one amplitude path `ZipformerStack.project` does not reach; measured 3.47 → 9.98 across the
two BEST-RQ pretrains. `stack_mix/0_trunk` reports it (stack 0's `in_proj` is an `Identity`, so its
logged trunk value *is* the frontend's output). See §5.

---

## 3. Encoder

### 3.1 `ZipformerStack`: the mix
```
residual = in_proj(input)                     # [B, T₁, d]   ← THE TRUNK
x        = downsample(residual)               # ÷ factor, learnable softmax pooling
x        = block_k(... block_0(x) ...)        # value residual: block 0's v feeds every deeper block
x        = upsample(x, out_len=T₁)            # repeat_interleave, trimmed
b        = clamp(bypass, stack_bypass_min, 1)
out      = residual + b * (x - residual)      # = (1-b)*residual + b*x
```
Two independent amplitude paths meet here and only one of them is normalised:

* the **processed** half `x` exits `blocks[-1].norm_out`, a `BiasNorm`, so it is bounded;
* the **residual** half is `in_proj(input)` and passes through **no normaliser at all**.

`forward` writes `self.trunk_rms` (an fp32 vector-norm of the detached residual, no extra copy),
which is what `stack_mix/{i}_trunk` logs. Padded frames carry `in_proj.bias` rather than 0 and are
counted; at ~0.2 % frame padding that is below the quantity's step-to-step noise.

`bypass` is a single learnable scalar per stack, initialised 0.5, clamped in `forward` (the value
used) and projected after every optimizer step (the value stored). Floor `0.1`, never 0: at `b = 0`
every block parameter is multiplied by zero and receives no gradient, an absorbing state measured
on `transducer_step81000.pt` as 0 of 105 block parameters with a gradient and `d(loss)/d(bypass)`
still pressing downward.

The chunk mask is built here, not in the block: `chunk_size` arrives in base-rate frames and is
rescaled to the stack's own rate as `max(1, chunk_size // downsample.factor)`, so all six stacks see
the same wall-clock chunk boundary. `chunk_size = 0` means full context and builds no mask at all.

### 3.2 `ZipformerBlock`: macaron
```
x = x + 0.5 * ffn1(norm_ffn1(x))
h, v = attn(norm_attn(x), pad_mask, attn_visible, value_residual)
x = x + h
x = x + conv(norm_conv(x), pad_mask)
x = x + 0.5 * ffn2(norm_ffn2(x))
return norm_out(x), v
```
Six `BiasNorm` per block (`ffn1, attn, conv, conv.norm, ffn2, out`). Every branch is **pre-normed**,
so each contributes an O(1) correction regardless of the residual it is added to. That is the
mechanism behind every collapse in this model: a branch's work falls off as 1/RMS² of its input.
MEASURED on a fresh block, `1 − cos(in, out)` against input RMS: 0.060 at 1, 0.009 at 2.72, 0.0004
at 12.18, 0.00007 at 30.

### 3.3 `BiasNorm`
```
sq      = mean(x², dim=-1)
centred = mean((x - bias)², dim=-1)
inv_rms = rsqrt(max(centred, sq / max_amplification²) + eps)
gain    = exp(clamp(log_scale, log_scale_min, log_scale_max))
out     = x * (inv_rms * gain)
```
`bias` is per channel, `log_scale` is a **single scalar per module**. Window `[-2.0, +1.0]`,
asymmetric on purpose: only amplification silences the next stack, and the healthy population runs
low (p05 = −1.54 over 97 encoder norms). The three config values are read once in `__init__` and
cached on the instance, because a `get_config()` call inside `forward` would break inductor's graph
(§9.5).

It divides by the RMS of `x − bias` but scales `x`, so without the floor its output RMS is
`exp(log_scale) · rms(x)/rms(x−bias)` and the second factor is a free parameter's distance from the
data, measured at **68.7×** with `log_scale` pinned inside its bound. `max_amplification = 4.0`
floors the normaliser at `rms(x)/4`, bounding output RMS at `4·exp(log_scale) ≤ 10.87`. Past the
cap the denominator no longer depends on `bias`, so the escape direction stops receiving gradient
while the gain itself keeps training, the same projected-gradient argument as the bound on
`log_scale`.

The whole module is one pass over `x`: the reciprocal and the gain fold into the `[.., 1]`
statistic rather than being applied to the activation. There are 96 of these per encoder forward and
all of them are bandwidth-bound, so the pass count *is* the cost.

### 3.4 `SwiGluFfn`
`hidden = round8(dim * 4 * 2/3)`; `w_down(silu(w_gate(x)) * w_up(x))` then dropout 0.1. The 2/3
keeps the SwiGLU parameter count near a dense FFN's; the round-to-8 is tensor-core alignment.

### 3.5 `RotaryAttention`
`qkv: Linear(d → 3d)` → `[B, H, T, d/H]`; RoPE applied to q and k
(`inv_freq = base^(-i/half)`, `base = 10000`, `pos_offset` carries chunk-local position in
streaming); `F.scaled_dot_product_attention` with `attn_mask = ~pad_mask & attn_visible`, scale
`1/√head_dim`, dropout on the weights in train mode; `out: Linear(d → d)` + dropout. SDPA is fused
(flash / mem-efficient under bf16 autocast), so the `[B,H,T,T]` score matrix is never materialised,
which is also why this module stays eager (§9.5).

Returns `(out, v)`, where `v` is block 0's value tensor, injected into deeper blocks as
`v + res_lambda * v₀`, **before** attention and before caching, so a cached value already carries
the residual. `res_lambda` initialises from `encoder_value_residual_lambda = 0.0`; that is an
**init, not a disable** (gates learn −1.40 to +0.47 across the 16 tensors). Block 0's own
`res_lambda` never receives gradient and exists only so `load_state_dict` matches.

### 3.6 `ConvModule`
```
x = glu(pointwise1(x))            # Conv1d k=1 run as F.linear, d → 2d → d
x = x.masked_fill(pad_mask, 0)    # padding must not enter the conv window
x = depthwise(left_pad(x, k-1))   # k = 15, groups = d, NO future frames
x = silu(norm(x))                 # BiasNorm, PER FRAME over channels
x = pointwise2(x)                 # Conv1d k=1 run as F.linear
```
The per-frame norm is load-bearing: a GroupNorm over time would tie the output to the whole
sequence and break streaming equivalence. `streaming_forward` carries the last `k-1` frames of the
**glu output** (not the input), which is the tensor the depthwise conv actually consumes.

Both kernel-1 convs are executed as `F.linear` through a squeezed view of the same parameter: a 1×1
Conv1d insists on `[B,C,T]` while the block carries `[B,T,C]`, so running it as a convolution costs
two transposes, a cuDNN copy of the non-contiguous result, and an implicit-GEMM path slower than
cuBLAS. The module, its `state_dict` keys and every checkpoint (including the BEST-RQ warm start)
are unchanged: the parameter stays canonical and only the call site reshapes.

**Their storage class is load-bearing for the optimizer, not just for cuDNN.** `partition_params`
routes `nn.Linear.weight` with `ndim == 2` to Muon; these are `nn.Conv1d` weights of shape
`[out, in, 1]`, so all **6,807,552** of them (12 % of the model) sit on **AdamW** and take weight
decay, on `ndim` alone. Re-declaring them as `nn.Linear`, the obvious cleanup once they are already
called through `F.linear`, would silently move an eighth of the model onto the optimizer §5.2 names
as the trunk-inflation mechanism. See §8.1 and invariant 13.

### 3.7 `SimpleDownsample` / `SimpleUpsample`
Downsample: softmax over a learnable `[factor]` weight vector, weighted sum over each window,
`out_lengths = ⌈lengths/factor⌉`, time zero-padded up to a multiple of `factor` first. Zero-init →
uniform pooling. Upsample: `repeat_interleave` trimmed to the pre-downsample length. Both are
per-frame linear, hence streaming-transparent. The 6 `weights` tensors are excluded from weight
decay: they are pre-softmax logits, so decaying them toward 0 forces uniform pooling.

---

## 4. Streaming contract

`ZipformerEncoder.streaming_forward(chunk)` is **exactly** equal to the batched `forward` with
`chunk_size = B`, no re-forward fallback, all frames covered, gated by
`test_streaming_forward_equivalence` at 100 frames with random init. It holds because the only three
time-coupling ops are made causal or per-frame:

1. `Conv2dSubsampling`: left time-pad only, two-level cache (`in_tail`, `mid_tail`).
2. `ConvModule.depthwise`: left pad `k-1`, `ConvCache.left` on the glu output.
3. `RotaryAttention`: KV cache per head per layer, `pos_offset = cache.seen`.

`BiasNorm` and `ConvModule.norm` are per frame; downsample/upsample/bypass are per-frame linear.
Training samples `chunk_size ∈ {0, 16, 32}` base-rate frames per batch (0 = full context), so one
set of weights serves offline and streaming decode. BEST-RQ pretraining samples the same pool
(§7.4), so the encoder meets limited right-context before the supervised stage rather than during it.

`chunk_lcm()` = lcm(1,2,4,8,4,2,2) = **8**. A decode `chunk_size` must be a multiple of it, and the
handler feeds the encoder `2 * chunk_size` **feature-rate** frames per step (the default 16 base-rate
→ 32 feature frames → 320 ms).

**The attention KV cache is never evicted.** `AttnCache.k/v` grow at every step at all 16 blocks,
but a block runs at **its stack's** rate, not the base rate, so a block in stack 3 caches
`chunk/8` frames. Per base-rate frame of audio the whole encoder caches
`Σᵢ blocksᵢ · 2 · dᵢ / fᵢ` = 768 + 512 + 576 + 512 + 576 + 512 = **3,456 elements** (≈ 346 KB per
second of audio at bf16, ~12 MB for a 35 s utterance); counting all 16 blocks at base rate
overstates it by 3.3×. The growth is still unbounded, and that is exactly what makes streaming
bit-equal to full-context `forward`: memory grows linearly and attention cost quadratically in
utterance length. Fine for LibriSpeech's ≤ 35 s utterances; it is the thing to bound first for
genuinely open-ended streaming.

---

## 5. Amplitude ledger

**The diagnostic table.** Every operator that can change the activation scale, what normalises it,
what bounds it, and the value it takes in a model that works.

| Operator | Normalised by | Bounded by | Healthy value (v1.0 `step174000`, 3.43 % WER) |
|---|---|---|---|
| `frontend.conv1/conv2/linear` | n/a | none (weight decay only) | frontend output RMS **4.08** |
| `stacks.i.in_proj` (the TRUNK) | n/a | `stack_in_proj_max_sigma = 10.0`, projected | σ₁ 5.91 / 9.73 / 6.65 / 5.23 / 5.12; realized trunk RMS **2.68 / 2.24 / 1.51 / 2.69 / 2.25** |
| `bypass` | n/a | `stack_bypass_min = 0.1`, projected | 0.96 / 0.92 / 0.88 / 0.62 / 0.83 / 1.00 |
| every `BiasNorm` (97 in encoder) | itself | `log_scale ∈ [-2, +1]` and `max_amplification = 4` | output RMS ≤ 10.87 by construction |
| block branches (ffn/attn/conv) | pre-norm | inherited from their `BiasNorm` | `1 − cos(in,out)` over 16 blocks: min 0.215, median 0.402, max 0.982 |
| `out_norm` | itself | same window | encoder memory RMS **0.359** |
| predictor `BiasNorm` | itself | same window, projected in both trainers | n/a |
| joiner `enc_proj`/`pred_proj`/`out` | n/a | none (tanh bounds the pre-readout activation to ±1) | n/a |

### 5.1 The failure family, in one sentence each

1. **2026-08-09**. `BiasNorm.log_scale` unbounded: one stack amplified to RMS 12 and silenced the
   next. Fixed by the `[-2, +1]` window. *A one-sided cap at 2.5 bought 25k steps and the same
   collapse happened.*
2. **2026-08-21**. bounding `log_scale` bounds nothing: the module divides by `rms(x − bias)` and
   scales `x`; that free ratio reached 68.7× with the gain pinned inside its window. Fixed by
   `biasnorm_max_amplification = 4.0`.
3. **2026-08-22**. every `BiasNorm` is on a **branch**; the trunk (`in_proj`) had no bound at all.
   Fixed by projecting `in_proj` and flooring `bypass`.
4. **2026-08-22 (second)**. that trunk bound was on the **wrong norm**. `‖W‖_F/√n_out` is the gain
   against isotropic input; the inflation is anisotropic. Fixed by projecting σ₁ instead.

The pattern to recognise: **a parameter bound is not an activation bound.** Each fix bounded a
parameter and the amplitude found the next unbounded path. What actually failed every time is the
realized activation RMS, now logged directly as `stack_mix/*_trunk`.

Three paths still carry no bound, by design or by omission: `frontend.linear` (no projection
reaches it), `in_proj.bias` (an offset, not a gain: measured RMS 0.21-0.70 against a trunk at
25-570), and the joiner's projections (the `tanh` bounds what reaches the readout).

### 5.2 Why the trunk inflates

Muon's Newton-Schulz sets every singular value of the update to 1, so a direction the loss cannot
feel takes the same size step as one it can. At step 81k `stacks.3.in_proj`'s raw gradient had
condition number 61,148 and a radial component of 2.5e-5 against a Muon step of 0.226. Weight
decay's equilibrium is `√min(m,n)·√max(1,m/n)/wd` = 1960 at `wd = 1e-2`, which is 175× the init it would
have to bind at. Nothing opposes the drift; only a projection does. Scaling `in_proj` ×2 at step 81k
moved dev CTC by −0.015 and ×0.5 by +0.081: flat upward, restoring downward, i.e. a diffusion that
only goes one way.

### 5.3 Reference measurements (same 4 dev utterances, CPU, `scripts`-free probe)

realized trunk RMS after `in_proj`, stacks 1-5:

| checkpoint | frontend | trunk 1-5 | encoder out |
|---|---|---|---|
| OLD `bestrq_encoder.pt` (v1.0 warm start) | 3.47 | 3.14 2.32 2.22 3.59 3.34 | 2.350 |
| **v1.0 `step174000`, 3.43 % test-clean** | **4.08** | **2.68 2.24 1.51 2.69 2.25** | **0.359** |
| NEW `bestrq_encoder.pt` (2026-08-20 overhaul) | 9.98 | 13.05 5.68 12.83 16.95 13.98 | 2.874 |
| 600k run, step 43,200 | 7.21 | 5.23 9.97 **43.19** 3.06 3.92 | 0.370 |
| 600k run, step 48,265 (stopped) | 7.17 | 7.33 35.58 **337.75** 3.76 3.91 | 0.387 |

`in_proj` spectral norm σ₁ over the same checkpoints:

| checkpoint | σ₁ stacks 1-5 | `bypass` 0-5 |
|---|---|---|
| OLD bestrq | 3.42 3.53 2.56 2.19 2.28 | 0.60 0.72 0.42 0.26 0.63 0.99 |
| v1.0 `step174000` | 5.91 9.73 6.65 5.23 5.12 | 0.96 0.92 0.88 0.62 0.83 1.00 |
| NEW bestrq | 5.63 4.09 3.65 3.52 3.54 | 0.54 0.84 0.20 0.60 0.77 0.94 |
| 600k step 43,200 | 9.85 11.36 9.79 5.01 6.04 | 0.98 0.90 0.35 0.97 0.77 0.98 |
| 600k step 48,265 | 10.25 **20.05** 14.46 5.63 6.18 | 0.98 0.87 **0.10** 0.99 0.78 0.98 |

At step 48,265 `bypass₂` is resting **exactly on its 0.1 floor**. Without the floor it would be at
0.0 and stack 2's 10.6 M block parameters would have stopped receiving gradient. The floor keeps
them trainable; it does not stop the 90 % of the output that is now bare `in_proj` at σ₁ = 20.

σ₁/isotropic ran 2.1-3.7 across the five stacks and rose at every checkpoint (fresh init 1.8-2.2),
which is why the Frobenius bound of 4.0 projected **zero** matrices in 43,200 steps.
`stack_in_proj_max_sigma = 10.0` is calibrated on that table: inert on both pretrains and on the
shipped model, clips 3 of 5 matrices on the diverged one. **The bound cannot be retro-fitted**:
projecting `step81000` takes the joint loss 2.40 → 7.13.

**σ₁ does not rank matrices by the quantity that fails, either.** Read the two tables above at the
same step 43,200: the trunk that had already inflated 7× past healthy is stack 3's, at RMS 43.19,
and its σ₁ is **9.79, inside the 10.0 ball**. The one matrix the bound would have projected at that
step is stack 2's, at σ₁ 11.36 with a trunk of 9.97. A worst-case gain caps how far a matrix can
eventually travel; it does not say which one is travelling. So `stack_in_proj_max_sigma` is the last
line and not the first: the first is `stack_mix/*_trunk` at `log_every`, with superlinear growth read
as the abort condition. Neither the σ₁ bound nor the `bypass` floor has yet survived a full GPU run.

### 5.4 The bound this ledger did not try until 2026-08-24: now taken

> **TAKEN.** `model.trunk_norm` is on. What follows is the argument for it, then the measured cost
> of adopting it mid-run. §5.5 and §5.6 are why it stopped being optional.

Every fix in §5.1 bounds a **parameter**, and §5.3 is the fourth measurement in a row showing that
the realized activation escaped anyway. The intervention none of them is: **normalise the trunk**.
One `BiasNorm` on `in_proj`'s output (257 params, one bandwidth-bound pass, 6 more against the 96
already run per forward) makes `stack_in_proj_max_sigma`, `stack_bypass_min`, `train/trunk_gain_max`
and all six `stack_mix/*_trunk` redundant, because the quantity that failed four times would be
normalised by construction rather than watched.

It is not free and it is not a mid-run patch:

* it stops being Zipformer's bypass: `b` currently interpolates against an *unnormalised* residual,
  and every downstream `bias` and gain in the model is calibrated against that scale;
* it invalidates both BEST-RQ warm starts and the shipped model's amplitude ledger, so adopting it
  costs a fresh pretrain (~6 h 20 m) before the transducer stage can start;
* `frontend.linear` would still be unbounded (§2.3), so it removes the failure family, not the last
  unnormalised operator.

Recorded here so the fifth amplitude failure does not re-derive it. Until it is taken, this model is
**monitored, not controlled**: the projections cap the tail, and noticing the trunk move is the job.

**What it actually cost.** A `BiasNorm` on each of the five `Linear` `in_proj` outputs, ~1,800
params, `model.trunk_norm_log_scale_max = 1.2` so the trunk may not exceed RMS `exp(1.2) = 3.32`.
Stack 0 keeps an `Identity` (no inter-stack operator, nothing upstream to compound from), so
`frontend.linear` remains unnormalised and `stack_mix/0_trunk` remains its output. Calibration and
insertion cost are measured in `config/model.yaml` and §5.6; the short version is that the ceiling
sits at the knee of the cost curve, and 82-89 % of the insertion cost falls on InterCTC, a linear
head calibrated to the old amplitude that re-fits in a few hundred steps.

Two of the three objections above survive and are simply accepted: `b` now interpolates against a
*normalised* residual, which is no longer Zipformer's bypass, and `frontend.linear` is still
unbounded. The third, that adopting it costs a fresh pretrain, did not hold: a mid-run insertion
is a checkpoint migration (`scripts/migrate_trunk_norm.py`), and a warm start from a pretrain that
predates the norms simply starts them at gain 1.0.

### 5.5 The fifth failure: the projection ate the matrix (2026-08-24)

§5.4's warning was written one run early. The fifth failure is not another escape past a bound. It
is **the bound's own enforcement**, and it needed no new mechanism beyond the one §5.2 already
describes.

`ZipformerStack.project` implemented `σ₁ ≤ c` as a uniform rescale, `W *= c/σ₁`. That trims every
singular direction by the same factor while the gradient re-inflates only the top one, so it is
harmless exactly as long as it fires rarely, which is what its own docstring assumed. MEASURED on
the 600k run warm-started 2026-08-24: `train/trunk_gain_max` reached 10.0 at step **71,000** and
bound on **99.2 %** of every logged step after it. What that did to `stacks.3.in_proj`:

| | warm start | 75.6k | 86.4k | 97.2k |
|---|---|---|---|---|
| σ₁ | 3.32 | **10.00** | **10.00** | **10.00** |
| ‖W‖_F/√n_out | 1.18 | 3.09 | 2.52 | 2.06 |
| σ₂/σ₁ | 0.988 | 0.843 | 0.793 | 0.719 |
| **stable rank** ‖W‖_F²/σ₁² | **64.4** | **48.9** | **32.6** | **21.7** |
| realized trunk RMS | n/a | ~5.0 | 6.3 | 9.6 |

σ₁ frozen on the ceiling, Frobenius norm *falling*, stable rank halving every ~20k steps out of 384
available directions. And the realized amplitude climbed on the same clock: 2.02 (10k) → 4.45 (70k)
→ **12.34** (102k) against **1.51** for the model that shipped 3.43 % test-clean, because the two
are the same event: an operator collapsing toward rank 1 pulls its own input into the one direction
it has left, so the gain the data gets rises toward the worst case while the worst case sits still.
Stack 2 began the same climb at exactly step 70k, as the pressure migrated. At 102k, stack 3
(24.4 M params, **45.4 %** of the encoder) sat on a residual stream at RMS 12.34, where the measured
1/RMS² law leaves its blocks ~0.7 % of their RMS-1 work, and stack 2 (19.2 %) at RMS 5.42 → 3.4 %.

Reproduced from the real `step81000` weights, 2,000 projections under one inflating gradient:

| projection | σ₁ | stable rank 0 → 2,000 steps |
|---|---|---|
| uniform rescale (old) | 10.000 throughout | 38.4 → 30.9 → 5.6 → **1.1** |
| top-direction deflation (new) | 10.000 throughout | 38.4 → **39.3**, flat |

The fix is the minimal projection onto the ball, `W −= (σ₁ − c)·u₁v₁ᵀ`, at the cost of one rank-1
update: the power iteration already computes `v₁`, and one extra matvec gives the matched `u₁`. See
§8.5. **The rule this adds to the family: a projection that binds continuously is not a bound, it is
an operator applied 100,000 times, and what it does to the directions it was not aimed at is part of
the model.** `train/trunk_stable_rank_min` exists because σ₁ resting on its ceiling is constant by
construction and therefore reports nothing. Every scalar the trainer logged was blind to this, in
the same way §5.3's four predecessors were blind to the realized activation.

### 5.6 The sixth failure: bounded in gain, so it inflated in RANK (2026-08-24)

§5.5's fix worked on exactly what it was aimed at and made the real problem faster, which is how
this ledger closed. Resumed from step 81,000 with the deflating projection:

| stack 3 `in_proj` | 81.0k | 86.4k | 91.8k | 97.2k |
|---|---|---|---|---|
| σ₁ | 10.000 | 10.000 | 10.054 | 10.164 |
| **stable rank** | **38.4** | 46.5 | 53.6 | **62.2** |
| ‖W‖_F | 61.94 | 68.22 | 73.63 | **80.14** |
| σ₂/σ₁ | 0.796 | 0.982 | 0.995 | 0.994 |

The rank collapse is gone (the spectrum re-expands toward the warm start's 64.4) and dev is
better at the same step (ctc-WER 0.1001 vs 0.1011, transducer probe 0.0792 vs 0.0801, stream gap
0.0276 vs 0.0304). **But the realized trunk went 5.85 → 21.94 over those same 19k steps, doubling
every 9.4k against ~20k under the old projection.** The rescale had been an accidental brake: it
crushed ‖W‖_F while it destroyed the matrix. With the matrix preserved, ‖W‖_F was free and grew
29 %, σ₂/σ₁ went to 0.994, and the operator started inflating in **rank** at gain 10 rather than in
gain, a direction a σ₁ bound cannot see, and one where σ₁ itself stops being enforceable one
direction at a time (it reads 10.164).

**The calibration that ended the argument.** v1.0's five `in_proj`, the model that shipped 3.43 %:

| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| σ₁ | 5.91 | 9.73 | 6.65 | 5.23 | 5.12 |
| **‖W‖_F/√n_out** | **2.77** | **2.84** | **2.76** | **2.70** | **3.11** |

σ₁ ranges 5.1-9.7; the isotropic gain is uniform at ~2.8. So §5.3's conclusion was half wrong:
the isotropic bound was not the wrong *norm*, it was the wrong *value*: 4.0 sits 45 % above the
level the working model runs at, which is exactly why it clipped nothing in 43k steps. Bound σ₁
alone and the inflation goes to Frobenius; bound Frobenius loosely and it goes anisotropic.

**And bounding both would still not have been the fix.** Realized trunk RMS factors as
`input RMS × isotropic gain × alignment`. At stack 3, this run at 97.2k against v1.0: the isotropic
gain is 3.54 vs 2.76, a factor of **1.28**, but `input × alignment` is 6.19 vs 0.55, a factor of
**11.3**. The per-matrix gain is a tenth of the problem. Five per-stack bounds of 10.0 multiply to
10⁵, and nothing in the parameter space reaches the compounding. That is §5.4, and it is why the
seventh member of this family is a normaliser instead of a bound.

---

## 6. Predictor, joiner, RNN-T

### 6.1 `StatelessPredictor`
```
padded = left_pad(labels, context-1, value=blank)      # context = 2
emb    = Embedding(501, 512)(padded)
out    = Conv1d(512, 512, k=context, groups=512)(emb)  # depthwise, causal, NO recurrence
return BiasNorm(512)(out)
```
The embedding table is `logits_width = 501` wide, so `blank_id = 500` is a valid id and doubles as
the sequence-start symbol. Padding happens in **token-id space** with `blank_id`, not as a zero-pad
of the embeddings, which is what makes `step()` bit-equal to this batched `forward`. Streaming state
is the last `context-1` token ids (`init_state` fills them with `blank_id`); `step()` evaluates one
position and returns the state **after** consuming the token, so reuse it verbatim and never re-derive it
by calling `step` again with the just-emitted token.

### 6.2 `TransducerJoiner`
```
forward: readout(tanh(enc_proj(enc)[:, :, None] + pred_proj(pred)[:, None]))   # [B,T,U+1,501]
band:    readout(tanh(enc_proj(enc)[:, :, None] + pred_proj(pred_band)))       # [B,T,s_range,501]
step:    out(tanh(enc_proj(enc_t) + pred_proj(pred_u)))                        # one (t,u) cell
```
`readout` pads the output width to a multiple of 8 with a **−inf bias** on the pad columns:
`exp(-inf) = 0`, so every log-softmax, gather and gradient downstream is the 501-wide result and the
pad columns' own gradient is exactly 0. Parameters stay 501-wide (checkpoints unaffected).
Reason: `N = 501` is not 16-byte aligned in bf16 and cuBLAS fell to an alignment-2 kernel at
29 TFLOPS against 57 at `N = 504`, worth 14 % of the training step. `step()` is left unpadded (one
cell wide, GEMM-bound by nothing).

`enc_proj` and `pred_proj` are hidden 2-D matrices and go to **Muon**; `joiner.out` is a vocab-width
readout and is held on **AdamW** by the head patterns (§8.1).

### 6.3 `RnntLoss`: own forward-backward
Graves recursion on the `[B, T, U+1]` alignment grid:
```
α[0,0] = 0
α[t,u] = logaddexp(α[t-1,u] + blank_lp[t-1,u],  α[t,u-1] + label_lp[t,u-1])
β[t,u] = logaddexp(blank_lp[t,u] + β[t+1,u],    label_lp[t,u] + β[t,u+1])
cost   = -(α[T-1,U] + blank_lp[T-1,U])
```
Scanned over **anti-diagonals** `t+u = d`: `T+U` vectorised steps instead of `T·U`. Reversed in both
axes β obeys α's recursion, so both stack along the batch dim and share every kernel launch; the
`U`-shift comes from a `-inf` sentinel column instead of a `cat`; the accepting state enters as a
`logaddexp` against a `-inf` plane. Two ops per diagonal for both variables instead of five each.

Locked against `torchaudio` to fp32 round-off plus a float64 `gradcheck`; 4.3× faster on the kernel
(191 → 45 ms, then 27.3 → 13.1 ms after the merged scan). Also fixes a torchaudio CUDA bug where an
empty transcript returns cost 0.0 instead of `−Σ log p(blank)`.

The lattice reaches the loss as **bf16** under autocast, and the loss promotes to fp32 inside its own
log-softmax, so the bf16 lattice is the only `[B,T,U+1,V]` tensor that has to survive until backward.
Materialising an fp32 copy would double the step's largest allocation.

Reduction is `"sum"` at the loss and divided by `token_lengths.sum()` in
`TransducerModel.rnnt_loss` is normalised **per token**, matching CTC. A per-utterance mean under-weighted the
aux CTC terms ~30× and was the first run's divergence.

`RnntLossPruned` implements icefall's two-stage objective and is **off** (`rnnt_loss: full`): it is
bit-identical to the reference at full band width in fp64 but 30 % slower end to end, because its
frame loop runs through plain autograd (82,937 launches/step at 5.4 µs against 20,166 at 17.1 µs).
Under `pruned` the model gains `simple_am_proj`/`simple_lm_proj` (both linear on purpose, because a *sum*
of two vocab-width projections keeps the log-normaliser separable, so the implied lattice is a
transient inside the loss rather than a stored activation), the loss weights ramp over
`prune_warmup_steps = 4000` (simple 1.0 → 0.5, pruned 0.1 → 1.0), and `train/rnnt_simple` becomes a
live metric. **This key is part of the architecture**: a checkpoint saved under one value will not
load under the other.

### 6.4 Total loss
```
total = rnnt + 0.2 * ctc + 0.15 * interctc[3] + 0.15 * interctc[4]
```
CTC over encoder memory at ~25 fps; InterCTC taps stacks 3 and 4 at base rate (~50 fps), both with
`blank = 500`, `zero_infinity=True`. CTC targets are passed in their padded `[B, S]` 2-D form,
concatenating the rows instead would cost `B` device→host syncs per CTC call, and this path runs
3× a step. `train/interctc` logs the **raw mean** over the taps, not the weighted sum, so a climb is
attributable to the encoder rather than to the weight.

SpecAugment (`apply_spec_augment_batch`) is a train-only GPU batch op applied to the features inside
`joint_loss`, gated on both `training.transducer.spec_augment` and `self.training`. Policy
(`config/augment.yaml`, the "LD" policy at designed strength): 2 frequency masks of width
`U{0..27}`, up to 10 time masks with count and span both scaled by the utterance's own valid length
at ratio 0.05. Fully vectorised as `[B,K]` geometry reduced into one boolean mask, so it costs a
fixed handful of kernels and never `.item()`s a length.

CR-CTC exists, is off, and requires SpecAugment (its signal is the disagreement between two masked
views; with masking off the KL is identically zero and the second encoder forward teaches nothing,
the constructor raises rather than let that happen silently). Under CR-CTC the RNN-T lattice is
computed on view 1 only.

---

## 7. BEST-RQ pretrain: where the warm start comes from

`data/checkpoints/bestrq_encoder.pt` is the transducer's default `warm_start`, and §5.3 shows the
handoff amplitude decides how the supervised run behaves. This stage is therefore part of the
acoustic model's ground truth, not a preamble to it.

### 7.1 `BestRqModel`
```
masked, mask = apply_span_mask(mel, lengths, p, span, noise_std, cmvn_mean, cmvn_std)
enc, out_len = encoder(masked, lengths, chunk_size)             # the SAME ZipformerEncoder
mel_n        = (mel - cmvn_mean) / cmvn_std                     # CLEAN, normalised
stacked      = stack_frames(mel_n, stack)                       # stack = 2 * final_downsample = 4
tgt_mask     = any(mask over each group of `stack` frames)
select       = tgt_mask & (position < out_len)
targets      = [q(stacked[select].float()) for q in quantizers] # [S, N], frozen, fp32
logits       = pred_head(enc[select]).view(-1, N, codebook_size)
loss         = cross_entropy(logits, targets)   # mean over the N codebooks
```
Encoder + 4 frozen quantizers + one `Linear(256 → 4·8192 = 32,768)` head. **62,200,013 trainable
parameters**: encoder 53,778,637 + head **8,421,376**, plus 544,768 frozen quantizer buffers,
62,744,941 elements over 602 `state_dict` keys. The head is 13.5 % of the stage and scales with
`num_codebooks` (one codebook is 2.1 M), which is the other reason §7.1 evaluates it on selected
positions only. It is discarded at warm start; only `encoder.*` is saved to `bestrq_encoder.pt`
(592 keys, `extra={"quantizer_seed": seed}`). The encoder's 592 keys carry 53,778,797 elements,
the 160 extra over the parameter count are the `cmvn_mean`/`cmvn_std` buffers, which is why a
checkpoint restores its own normalisation and only a *fresh* run can be hurt by a missing
`cmvn.pt` (§2.2).

Head and quantizers are evaluated **only on the selected positions** (~45 % of the grid), which is
what makes 4 codebooks affordable: both are 8192-wide readouts, and computing them on the full grid
threw away more than half of the stage's two largest GEMMs. The boolean index is the step's one
device→host sync, and it is cheaper than the `select.any()` probe it replaced.

Targets are computed in **fp32 regardless of autocast**: the quantizer is an argmax over 8192
near-ties, so a bf16 projection would resolve some of them differently step to step and feed the
encoder label noise. It is 0.3 GFLOP and frozen.

### 7.2 `apply_span_mask`
Span starts are drawn per frame at `mask_prob` (only where a span still fits, which also keeps
starts off the padding), expanded by a left-window max-pool of width `mask_span`. An utterance that
drew no start gets one forced, computed unconditionally because branching on `starts.any()` costs
a sync.

**The fill is specified in CMVN-normalized space** and de-normalized here as
`cmvn_mean + cmvn_std * N(0, noise_std)`. Drawing `N(0, 0.1)` in raw log-mel, which is what this did through
v1.0, puts masked frames at +1.46 σ of the data with 0.024 σ of spread, i.e. a constant
high-energy plateau in a region of input space the encoder never sees again at fine-tune time.

`mask_prob × mask_span` sets a **task**, not just a coverage number. Current `0.02 × 30` (300 ms,
the paper's streaming recipe) masks 0.455 of frames with ~17 % of masked frames touching an unmasked
neighbour; the previous `0.05 × 10` masked a similar 0.478 with 39.9 % touching, i.e. solvable by
local interpolation.

### 7.3 `RandomProjectionQuantizer`
Frozen `xavier_uniform` projection `[stack*n_mels → 16]` and a frozen L2-normed random codebook
`[8192, 16]`, both seeded (`seed + i` per codebook). Forward is normalise → cosine similarity →
argmax; both operands are unit-norm so max cosine ≡ nearest L2. Nothing trains.

`num_codebooks = 4` is USM's multi-softmax: one codebook makes the whole pretraining signal a single
frozen random draw, and across 20 seeds the target code entropy spans 7.64-8.90 bits (this repo's
seed 42 draws 7.93, below the mean). Averaging N independent draws replaces one sample of that
distribution with its mean.

### 7.4 The stage's loop, and how it differs from the transducer's
* **No gradient accumulation.** `step += 1` per loader batch and one optimizer step per batch, so
  `pretrain.total_steps` counts optimizer updates here and loader batches there.
* `max_frames_per_batch = 20000`, looser than the transducer's 28,000 because there is no RNN-T
  lattice to bound: encoder + head only.
* `encoder_lr_scale` is **overridden to 1.0** (`model_copy(update=...)`) and `pretrain.lr_scale`
  (0.5) is applied uniformly to every group's peak instead. `BestRqModel.encoder` matches the
  `encoder.` prefix `build_optimizer` keys on, so the transducer's knob would otherwise run 53.8 M
  of the stage's 62.2 M trainable params at half LR while only the 8.4 M head ran at full. Held by
  `test_pretrain_does_not_apply_encoder_lr_scale`. **Any new stage that names a submodule `encoder`
  inherits this trap.**
* Same two-set gradient clip and same `project_constraints` call as the transducer, so
  `stack_in_proj_max_sigma`, `stack_bypass_min` and the `log_scale` window are all enforced here too.
* Same `chunk_sizes` pool `{0, 16, 32}`.
* **Dev probe**: `dev_batches = 16` fixed batches materialised once on the host, masked under a
  pinned seed (`_DEV_MASK_SEED = 1234`, RNG saved and restored around it) at full context. Without
  the pinned mask two probes differ by which frames they happened to hide as much as by anything the
  encoder learned. `pretrain/loss` is over targets drawn from the same utterances the encoder just
  saw; `dev/loss` and `dev/acc` are the only generalisation signal in the stage.
* Watch `pretrain/acc` and `dev/acc`, **not** the loss: random-projection targets have an
  irreducible entropy floor that makes the loss curve plateau long before the representation does.

### 7.5 The handoff, and this stage's observability gap
`pretrain.lr_scale = 0.5` restores the effective encoder LR of the pretrain that produced the only
model this project has shipped. Removing the `encoder_lr_scale` accident doubled it, and trunk gains
scale ≈ √LR per stack:

| in_proj gain `‖W‖_F/√n_out` | stack1 | stack2 | stack3 | stack4 | stack5 | product |
|---|---|---|---|---|---|---|
| old pretrain (0.5×, → 3.43 %) | 1.90 | 1.60 | 0.68 | 0.65 | 0.92 | 1.2 |
| new pretrain (1.0×, 2026-08-20) | 3.10 | 1.31 | 1.02 | 1.01 | 1.30 | 5.4 |

Realized trunk RMS at handoff went 2.2-3.6 → 5.7-17.0, against 1.5-2.7 for the *trained* model that
works (the transducer stage started ~5× louder than where a healthy run ends) and `bypass₂` went
0.42 → 0.20. Stack 2 is the one that collapsed in both 2026-08-2x runs. The pre-overhaul encoder
survives at `data/backup/bestrq_encoder.pt` (identical 592 keys, same step 180,000, drop-in
`--warm-start`).

**This stage logs neither `train/trunk_gain_max` nor `stack_mix/*`**, only `pretrain/{loss,acc,lr}`,
`train/{grad_norm,grad_norm_guarded,branch_gain_max}` and `dev/{loss,acc}`. The amplitude the
transducer inherits is therefore invisible while it is being built, and the table above had to be
reconstructed from checkpoints after the fact. That is the largest known observability gap in the
pipeline.

---

## 8. Optimization

### 8.1 Parameter partition (`Optimizer_Adapter.partition_params`)
2-D `nn.Linear.weight` that is **not** in a head pattern (`frontend, ctc_head, pred_head, interctc,
joiner.out, simple_am_proj, simple_lm_proj`) → **Muon**, everything else → **AdamW**.
Verified: **135 tensors / 46,481,408 params on Muon, 472 tensors / 8,785,698 on AdamW**.
The partition keys on the **declared module type**, not on what the forward pass does: 6.81 M of
that AdamW total is `ConvModule.pointwise1/2`, `nn.Conv1d` weights executed as `F.linear` (§3.6),
held off Muon by their `ndim == 3` and nothing else.
Tied weights (STREAM-LM ties its readout to the embedding) surface under two module names; first
placement wins, so the same tensor is never stepped twice.

Groups are bucketed by `(lr, weight_decay)`, because a param group carries only hyperparameters, so
bucketing is exactly equivalent to per-parameter groups and is what lets the fused/multi-tensor
kernels work: the transducer's 472 AdamW parameters collapse to 4 groups and the AdamW step drops
6.1 ms → 0.6 ms. AdamW runs `betas=(0.9, 0.98)`, fused on CUDA.

`weight_decay` applies only to `ndim ≥ 2`, so the 415 scalar/bias tensors (160,418 elements, 0.3 %
of the model) are exempt: **289** `bias`, **98** `BiasNorm.log_scale`, **16** `res_lambda`, **6**
`ZipformerStack.bypass`, **6** `SimpleDownsample.weights`. For each of those, 0 is a specific
degenerate setting (value residual disabled, stack skipped, uniform pooling), not a neutral shrink
target. `log_scale` is the exception that proves the rule (0 there *is* unit gain) and is exempted
for a different reason: at this model's LRs decay is far too weak to be the mechanism holding the
gain down (equilibrium `|log_scale| ~ 93` at the measured 0.93 sign-coherence), and leaving it off
keeps decay and the projection from being confused for each other.

### 8.2 `Muon`
```
buf   = momentum * buf + grad                       # momentum 0.95, one _foreach pass per group
U     = newton_schulz(buf, ns_steps=5)              # fp32, batched by shape, all σ → 1
U    *= sqrt(max(1, rows/cols))                     # Jordan's fan scaling, folded into the batch
p    *= (1 - lr * weight_decay)
p    -= lr * U
```
Quintic coefficients `(3.4445, -4.7750, 2.0315)`, iterating on the wide orientation after a
spectral-norm normalisation. Batched over same-shaped matrices keyed on `(shape, ns_steps)`: 22
shape classes, ~2700 launches → ~440. fp32 and not bf16, because bf16 moves the update direction ~7 %
against fp64 where fp32/TF32 moves it 0.09 %. Rejects any non-2-D parameter outright.

### 8.3 LR (`LrSchedule.lr_at`)
Linear warmup to `peak * stable_ratio`, then WSD: hold, then anneal over the last `decay_frac`
(0.25) of the budget with a `1 − √frac` profile down to `peak * min_ratio` (0.01). Cosine is the
other option. Under WSD, raising `total_steps` mid-run only moves the decay window; under cosine
`total` *is* the curve, so bumping it re-heats the LR at the resume step, which is how the
120k → 175k bump that produced the shipped checkpoint accidentally became a two-cycle schedule.

The function returns a **shape multiplier**; each trainer snapshots every group's calibrated peak at
build time and applies the shape per group per step, so Muon's much larger base LR is never
clobbered by a single absolute overwrite.

Peaks: `muon_lr = 2e-2`, `adamw_lr = 1.5e-3`, `encoder_lr_scale = 0.5` on `encoder.*` in the
transducer stage. BEST-RQ overrides `encoder_lr_scale` to 1.0 and applies its own uniform
`pretrain.lr_scale = 0.5` instead (§7.4).

**`total_steps` counts loader batches in the transducer stage, not optimizer updates** (`step += 1`
is outside the `grad_accum` window). Convert before reasoning:
`total_steps × max_frames_per_batch / 100 / 3600` hours. 600,000 × 28,000 = 46,667 h = **24.1
passes** over the 1,933 h `train_sp2` corpus, in 200,000 updates at `grad_accum = 3` (that is the
budget's ceiling: a batch that closes on the token or lattice cap carries fewer frames). In the BEST-RQ stage
there is no accumulation, so its `total_steps` *is* updates (§7.4).

### 8.4 Gradient clipping (`GradientClipping`)
Two disjoint sets, clipped differently:
* `guarded_parameters` (`ndim ≥ 2`): one group-norm clip; this is what `train/grad_norm_guarded`
  reports and the only set whose norm tracks model health.
* `unguarded_parameters` (`ndim < 2`): `clip_grads_per_tensor`, each bounded against **its own**
  norm in 3 `_foreach` launches, returning the group's pre-clip norm for `train/grad_norm`.

Order in the loop matters: the guard norm is read **before** its clip, so it sees the true gradient
rather than a rescaled one.

A single global clip is wrong here: ~99.9 % of the global norm is the scalar gates, so one gate
decided the factor every weight matrix's gradient was scaled by, swinging ~100× per step. Muon's
Newton-Schulz is scale-invariant within a step but `momentum_buffer` is an EMA across steps, so a
per-step-varying rescale reweights it. Measured, Muon's momentum norms collapsed 1.30 → 0.19
(`frontend.conv1` 0.98 → 0.12) while the raw matrix gradient never moved. **Splitting the clip in two
only relocated the coupling**: `encoder.stacks.3.bypass` then sat at exactly 4.9998 against a
`grad_clip` of 5.0 while the other gates were at 0.007-0.025, i.e. one gate ate the whole scalar
budget every step and rescaled all 414 other scalars. Hence per-tensor.

`grad_clip = 5.0` is a **diagnostic, not a safety bound**: Newton-Schulz renormalises Muon's update
regardless of gradient size, so clipping protects none of the encoder's capacity, and Adam is near
scale-invariant too. The LR is the only lever live mid-run, which makes *noticing* the failure the
whole job.

### 8.5 Projection (`ParameterProjection.project_constraints`)
Called after every optimizer step in **both** trainers; walks `model.modules()` and calls any
`project()` it finds, duck-typed so the shared kernel needs no slice imports (this is also what
reaches the stacks through `_Checkpointed` and the predictor's `BiasNorm`). Projected-gradient
descent, not clamping in `forward` alone: `clamp`'s gradient is exactly 0 past a bound, so a
parameter pushed outside would never receive gradient again, while one resting **on** the bound
still trains. MEASURED 2026-08-05: `stacks.5.bypass` sat at 1.0020-1.0049 across a 22k-step window
with AdamW `exp_avg` at −2.3e-22, frozen out of training.

It is deliberately **not** done inside `forward`: an in-place write to a leaf between two forwards
that share one backward invalidates the first forward's saved tensor, exactly what the CR-CTC
two-view path does.

* `BiasNorm.project`: `log_scale.clamp_(-2.0, 1.0)`
* `BiasNorm.project` on each stack's `trunk_norm`: the same clamp against the wider trunk window,
  `model.trunk_norm_log_scale_max = 1.2`. Reached by the same `project_constraints` walk, since a
  trunk norm is a `BiasNorm`; it carries its own `log_scale_max`, so one walk enforces two windows.
  `branch_gain_params` excludes them for the opposite reason: the trunk ceiling is above the
  branch ceiling `gains_at_ceiling` counts against, so including them would report five gains
  permanently pinned.
* `ZipformerStack.project`: `bypass.clamp_(0.1, 1.0)`; `in_proj` clipped to
  `σ₁ ≤ stack_in_proj_max_sigma` by deflating the top singular direction,
  `W −= (σ₁ − c)·u₁v₁ᵀ`, which is the minimal (Frobenius-nearest) projection onto that ball and
  costs one rank-1 update because the power iteration already holds a matched `u₁`/`v₁`. Over
  `σ₁ > c·1.05` (reachable only by loading a weight saved without this constraint, where the
  spectrum can be flat enough that deflating the top value merely promotes the second) it falls
  back to an exact `min(σᵢ, c)` clip through one SVD.
  **This was a uniform rescale `W *= c/σ₁` until 2026-08-24 and that is the fifth collapse**: a
  rescale trims every direction by the same factor while the gradient re-inflates only the top one,
  so it is harmless exactly as long as it fires rarely, and the 600k run made it fire on 99.2 % of
  steps from 71k on. See §5.5.
  σ₁ by 2 power iterations warm-started across steps (400 on the first call after a load, since
  `sigma_v` is `persistent=False`). Measured accuracy: ≥ 0.99999 of the true σ₁ warm, 1.00000 cold
  at 400 iterations on the flattest matrix in the model. `in_proj.bias` is left alone.

---

## 9. Training loop mechanics (`TransducerTrainer_Handler`)

### 9.1 One optimizer update
```
lr_shape = lr_at(step, ...)               → applied to every group's snapshotted peak
chunk    = random.choice({0, 16, 32})     → one chunk size per BATCH, not per utterance
autocast(bf16):  total, rnnt, ctc, ictc, cr, simple = model.joint_loss(batch, chunk, step)
loss = total / grad_accum ;  loss.backward()
accumulated += 1
if accumulated == grad_accum:
    last_guard_norm = grad_norm_of(guard_params)          # BEFORE the clip
    last_grad_norm  = clip_grads_per_tensor(gate_params, grad_clip)
    clip_grad_norm_(guard_params, grad_clip)
    for opt in optimizers: opt.step(); opt.zero_grad(set_to_none=True)
    project_constraints(model)
    accumulated = 0
step += 1                                  # NOTE: outside the accumulation window
```
`accumulated` is tracked explicitly rather than inferred from `step % grad_accum`, because an OOM
drops a partial window and the counter must reset with it.

### 9.2 OOM policy
`torch.OutOfMemoryError` around the forward/backward drops the batch: `oom_skips += 1`, the whole
accumulation window is discarded (a partial backward left gradients on only some params),
`zero_grad`, `empty_cache`, warn, continue. The RNN-T lattice is the run's largest allocation, so a
dense frame×token bucket is what fails first when another process takes part of the card. Occasional
skips are a transient; persistent skips mean the budgets in `config/training.yaml` are too high for
the free VRAM.

### 9.3 Validation and checkpoint selection
`_dev_metrics` every `val_every` (10,000) steps produces four numbers:
* `dev/ctc_wer`: greedy-CTC over the **whole** dev set (~54k words, σ ≈ 0.0011), full context.
* `dev/ctc_wer_stream`: the same, re-decoded at `decode.chunk_size` (the *deployed* chunk, not one
  of the training pool), and `dev/ctc_wer_stream_gap` = stream − full.
* `dev/transducer_wer`: greedy transducer over a `dev_wer_utts = 200` subsample (~1k words,
  σ ≈ 0.008). The subsample is strided evenly across the loader, because `FrameBucketSampler` emits
  duration-sorted batches and the leading 200 utterances are the shortest clips in dev.
* `dev/blank_frac`: fraction of CTC frames decoding to blank.

**`transducer_best.pt` is selected on `dev/ctc_wer`, not on the transducer probe.** Selecting on the
probe picks the luckiest noise trough: the 175k run picked step 152k, mid-anneal, whose full-dev
ctc-WER was 0.0680 against 0.0610 at 174k, and that checkpoint decoded 0.60 WER points worse on
test-clean than the tail average. Validation runs under
`torch.compiler.set_stance("force_eager")`, because it is eval/no-grad/fp32, a mode training never uses, and
letting dynamo trace it would build a second full set of graphs (§9.5).

### 9.4 Checkpoints, resume, and the guard
* Every write is `torch.save` to `.tmp` + `os.replace`, so an interrupt never truncates a checkpoint.
* Payload: `model`, per-optimizer `state_dict`s, `step`, `best_wer`, `resume_count`, `kind`, RNG
  (python + torch CPU + all CUDA devices) and an `extra` dict.
* `extra` carries the two things a guard must not relearn on resume: `guard_norm_floor` and the
  `GainCeilingWatch` baseline. The 600k run resumed 16 times; a guard relearning its reference from
  the state it resumes *into* would adopt a diverged regime as normal.
* Resume bumps `resume_count` and reseeds the sampler (`seed + resume_count`) for a fresh epoch
  order rather than replaying the pre-interrupt batch stream.
* Rolling snapshots: `transducer_step{N}.pt` every `ckpt_every = 5400`, `keep_last_n = 5`, i.e. a
  27k-step window. **A long anneal rolls the pre-anneal snapshot off the end**, so preserving one is a
  manual copy. Snapshots are byte copies of `transducer_last.pt`, so they carry optimizer moments
  and RNG and resume cleanly.
* `GradNormGuard` is fed `grad_norm_guarded` at `log_every` (where the value is already synced), and
  is **not armed until `warmup_steps`**: `guard_window * log_every` = 5,000 < warmup 7,500, so
  otherwise its running-minimum floor, the reference for the whole run, would be latched from the
  LR ramp against a freshly initialised joiner. It trips when the window median exceeds
  `guard_trend_factor` (4.0) × the run's quietest median for `guard_patience` (3) consecutive
  windows, and then **aborts without checkpointing**, because `transducer_last.pt` is at most
  `ckpt_every` steps old and healthier than the moment of the trip. It is one-sided: it cannot fire
  on a gradient *collapse*, which is the other failure signature the checkpoints show.
* `SignalGuard` wraps the loop so SIGINT/SIGTERM finishes the current step and saves cleanly.
* `_seed_all` seeds python/torch/CUDA; torch's seed also fixes the DataLoader workers'.
  `use_deterministic_algorithms` is deliberately not set, because cuDNN's CTC has no deterministic kernel.

### 9.5 Selective compilation
`compile_hot_modules` hands exactly four leaf modules to inductor: `BiasNorm` (−5.3 %),
`TransducerJoiner` (−3.7 %), `ConvModule` (−1.5 %), `SwiGluFfn` (−1.0 %), via `nn.Module.compile`
at `dynamic=True`. Measured together: 183.7 → 160.5 ms/step (−12.6 %), peak 6.96 → 5.98 GiB,
20,008 → 14,683 launches, ~8 s of one-off warmup.

The dividing line is `get_config()`, SDPA and data-dependent control flow. Compiling
`ZipformerBlock` instead measured the same −11.5 % but took 135 s of warmup and produced 49 graphs
against 16, because `RotaryAttention` reaches `rotary_tables`, which calls `get_config()`.
`torch.compile` on the whole model additionally hits this build's dynamic-shape tiling assert.

`nn.Module.compile` installs a compiled `_call_impl` on the instance rather than wrapping it, so
`state_dict` keys are unchanged (609, none polluted) and a checkpoint written with compilation on
loads into the eager decode/eval path, and the setting can be toggled across a resume.

**`_RECOMPILE_LIMIT = 32`.** Dynamo's recompile limit is per *code object*, and one code object
serves every instance, so `BiasNorm.forward` alone needs one graph per distinct width (192/256/384/
512) per (dtype, requires-grad) mode. That is exactly torch's default of 8, and exceeding it does
not raise. Dynamo logs one warning and silently demotes that function to eager **for the rest of the
process**, turning a −12.6 % step back into −7 % at the first validation, hours into a run.

---

## 10. Data path

### 10.1 Cache binding
`data/features/mel/{split}.{f16,index.npy,header.json}`: one flat fp16 memmap per split, indexed by
**manifest row order**. The header carries the five front-end params (`sample_rate, n_mels, n_fft,
win_length, hop_length`) and a `manifest_fingerprint`: a SHA-1 over every row's `(uttid, speed)`.
`LibriSpeechDataset` compares it at construction and raises. Without that check, pairing a clean-mel
cache with a speed-perturbed manifest reads the right row count and the right shapes and trains on
audio belonging to a different utterance, which nothing downstream would catch. **`--train-manifest`
and `--train-cache-split` always move together.**

The header is written only after the last utterance of a split lands, so it doubles as a completion
marker: an interrupted precompute leaves a short `.f16` under a stale or absent header, and a restart
redoes only that split. Delete the header to force a rebuild.

### 10.2 `FrameBucketSampler`: three budgets
Utterances are duration-sorted, then greedily filled into batches under three caps, any of which can
close a batch:
* `max_frames_per_batch` (28,000): a **sum** budget over frames.
* `max_tokens_per_batch` (4,400): a **sum** budget over transcript *characters* (a cheap upper
  bound on BPE tokens, so no tokenizer is needed in the sampler).
* `max_lattice_per_batch` (1.1e7): a **product** budget, `B × max(frames) × max(chars)`.

The two sum budgets bound the *average* batch and leave the tail free; the product budget bounds the
*worst* batch, which is the one peak VRAM tracks (the densest batch of an epoch ran 1.7× the p99.9
one, which is what produced periodic OOM drops). **All three move together, or the one left behind
silently pins B**: raising frames 32k → 45k → 58k with tokens held at 4,000 produced B = 18, 19, 19
and an identical 273 s of audio per step.

The batch list is built once and cached (a 562k-row pass per epoch, run in the main process with the
loader stalled behind it, becomes one at first use). `shuffle` permutes the *batch order* per epoch
from `seed + epoch`, preserving intra-batch length grouping; off, the stream is fully deterministic,
which is what dev evaluation needs.

`token_sort_window` (default 1 = off) re-sorts by transcript length inside a sliding window of the
duration order, so a batch is homogeneous in U as well as T. Measured on `train_sp2`: window 256
takes lattice waste 22.9 % → 5.6 % and epoch lattice work to 81.7 %. It is off because making every
batch homogeneous in speaking rate is a training-semantics change no run has isolated, and the
recipe that shipped 3.43 % did not have it.

### 10.3 Batch and collation
`FeatureCollator` pads to the batch max and yields a `FeatureBatch` of
`(features, feature_lengths, tokens, token_lengths)`, pinned; the four H2D copies in `joint_loss` are
`non_blocking=True` so they overlap the tail of the previous step. The pretrain stage uses
`MelOnlyDataset` + `collate_mels` (labels ignored).

Measured supply: the loader delivers 15,627× realtime against 1,426× needed, so **never chase the
loader**.

---

## 11. Observability: what each scalar can and cannot see

### 11.1 Transducer stage
All `train/*` are emitted every `log_every = 250` steps from a **single** device→host sync: 12
scalars plus 3×`n_stacks` mix values are concatenated and `.tolist()`ed once, because every `.item()`
drains the CUDA queue.

| Tag | Quantity | Blind to |
|---|---|---|
| `train/loss,rnnt,ctc,interctc` | loss terms, per token (`interctc` is the RAW mean over taps) | everything structural |
| `train/rnnt_simple` | simple-joiner term, only under `rnnt_loss: pruned` | n/a (a diverging simple loss silently corrupts the pruning band while the pruned term stays finite) |
| `train/cr_ctc` | consistency KL, only with CR-CTC on | n/a |
| `train/lr` | AdamW group 0's LR (`optimizers[-1]` is always AdamW) | Muon's LR, which is 13× larger |
| `train/grad_norm` | the **scalars'** group pre-clip norm (`ndim < 2`), not the old global | **it is ~95-99 % one scalar gate.** The old global is `√(1/share)` larger (+0.5 % at a 99 % share, +2.6 % at 95 %), so the two agree to 2-3 figures, not exactly. Two runs were aborted on this as false positives |
| `train/grad_norm_guarded` | norm over `ndim ≥ 2` only, read **pre-clip** | the gates, by design; this is the health signal |
| `train/guard_norm_median`, `train/guard_norm_floor` | `GradNormGuard`'s window median and its running minimum | emitted only once the window has filled; the floor is persisted through `extra` |
| `train/branch_gain_max` | `max exp(log_scale)` over all 98 | it is a LEVEL: a warm start ships 6 gains on the ceiling, so it reads pinned from step 0 |
| `train/gains_at_ceiling` | **count** at the ceiling (tolerance 1e-3, so a checkpoint written under a wider bound counts too) | n/a (this is the shape the branch failure had: 1 tensor → 3) |
| `train/trunk_gain_max` | `max σ₁(in_proj)` over the 5 `Linear` stacks | realized amplitude. This is a worst-case bound; the shipped model sits at 9.73 while its trunk is at RMS 1.5-2.7. Free: it reads the value `project` just computed |
| `stack_mix/i_residual` | `1 − b` | anything about `in_proj`; read *healthier* at the 2026-08-22 collapse than 20k steps before it |
| `stack_mix/i_processed` | `b · exp(log_scale)` of the last `norm_out` | the branch's realized RMS, which is this times `rms(x)/rms(x−bias)` of that norm. That ratio is now capped at `max_amplification`, so the metric under-reads by **at most 4×** (and can over-read when the ratio is < 1); the 27× gap it showed in 2026-08-21 (2.34 logged against 64 emitted) is a pre-cap measurement and is no longer reachable |
| **`stack_mix/i_trunk`** | **realized RMS of the trunk residual, POST `trunk_norm`** | nothing yet; stack 0's entry is the frontend's output, the one unbounded operator |
| **`train/trunk_stable_rank_min`** | `min ‖W‖_F²/σ₁²` over the 5 `Linear` stacks, i.e. how many directions the flattest trunk operator still carries | which stack it is, and the realized amplitude. Free for the same reason as `trunk_gain_max`. It exists because σ₁ pinned on its bound is uninformative by construction, which is the blind spot 2026-08-24 went through |
| `train/encoder_param_norm` | ‖encoder params‖, one fused `_foreach_norm` | direction, conditioning, per-tensor detail |
| `dev/ctc_wer` | full-dev greedy CTC, ~54k words, σ ≈ 0.0011 | the transducer head directly (it tracks it closely enough to rank checkpoints) |
| `dev/ctc_wer_stream`, `dev/ctc_wer_stream_gap` | the same at the deployed chunk, and the delta | it is a diagnostic, never the selection metric |
| `dev/transducer_wer` | greedy transducer on 200 utts | ~1k words, σ ≈ 0.008. **Never compare across runs with different sampling** |
| `dev/blank_frac` | CTC frames decoding to blank | n/a (a blank-collapse tell) |

`_train_utils.GainCeilingWatch` warns when the count at the ceiling reaches a new high-water mark,
not on a level, and once per new level rather than once per window (the gains train *on* the bound,
so the count rattles). `SignalGuard`/`GradNormGuard` reads `grad_norm_guarded`.

### 11.2 BEST-RQ stage
`pretrain/{loss,acc,lr}`, `train/{grad_norm,grad_norm_guarded,branch_gain_max}`, `dev/{loss,acc}`,
and nothing else. Notably absent: `trunk_gain_max`, `gains_at_ceiling` and all of `stack_mix/*`, i.e.
the amplitude the transducer inherits is not observed while it is built (§7.5). Read `dev/acc`, not
`pretrain/loss`.

---

## 12. Decode and evaluation

### 12.1 Search
`TransducerBeamSearch` is a Graves A/B time-synchronous search over encoder memory: blank scored
once per frame then the hypothesis retires to `t+1`, at most `max_symbols = 5` emissions per frame,
equal-prefix recombination (logadd merge) at every prune. Keeping blanked hyps in the live beam
instead would charge an idling hypothesis up to `max_symbols` blank log-probs at a single frame, a
systematic over-penalty that biases the search. The whole live beam is evaluated in one batched
predictor+joiner call per symbol step (batch dim = beam width), so a frame costs a few GPU launches
and one host sync rather than one per hypothesis. `greedy()` mirrors the trainer's dev probe exactly.

Predictor-state contract: `step(state, token) → (out, new_state)` where `new_state` is already the
context *after* consuming `token`. Both greedy and beam make exactly ONE `step` call per hypothesis
per emission attempt; re-deriving the state would duplicate the emitted token into its own context.

### 12.2 Rescoring
When `decode.lm_weight > 0`:
```
score = acoustic + α · lm_seq − β · ilm_seq + length_bonus · len
```
one batched STREAM-LM forward over the n-best, plus one batched internal-LM forward. **n-best
rescoring, not shallow fusion**, because per-emission fusion costs far more GPU time for a corpus decode.
All three weights default to 0.0, which reproduces the pure-acoustic decoder exactly (the regression
lock), and all three are swept together by `evaluate.py`'s dev sweep.

`InternalLmScorer` (Meng et al., ILME) evaluates the joiner with the encoder memory set to zero, so
only the joiner's bias and the predictor path survive, then renormalises over the **non-blank**
labels (blank is the last column, so slicing it off leaves column *i* = token id *i*; the constructor
asserts `blank_id == logits_width - 1`). With a stateless predictor the resulting prior conditions on
`predictor_context` tokens, which is the low-order regime LODR argues is the right thing to subtract.
The estimate has no EOS, so it covers the U emitted tokens only.

STREAM-LM is a 512×16 deep-narrow causal Transformer (~44 M) sharing this repo's `BiasNorm`,
`SwiGluFfn` and RoPE, plus GQA (8 query heads / 2 KV), QK-norm, tied embeddings and value residual
(`lambda = 1.0`), trained with Muon+AdamW and softmax z-loss on document-masked packed windows.
`lm.d_model` is load-bearing at inference: `lm_best.pt` must match it or `load_checkpoint` fails.

### 12.3 Modes and the evaluation harness
Streaming mode feeds feature-rate chunks (`2 * decode.chunk_size`) through `streaming_forward` with a
carried `StreamCache`; offline mode runs the whole utterance in one encoder call. Both funnel into
the same beam.

`evaluate.py` runs the ablation `greedy → beam → beam+LM` cumulatively in both modes.
`eval.workers = 4` decodes the six stage×mode passes **concurrently** in the QUALITY pass (a single
decode leaves the GPU ~70 % idle because the beam is Python-bound; 1 → 6 workers measured
96.0 s → 53.5 s, GPU 31 % → 85 %). Timing is a **separate serial pass** over `rtf_probe_utts = 200`
evenly-strided utterances, so the reported RTF / first-partial latency / finalize cost are
contention-free; audio IO sits outside the timer. Report goes to `runs/eval/report-{split}.json`,
`{split}` filled from `--clean`/`--other` so one cannot overwrite the other.

---

## 13. Invariants

1. `ZipformerEncoder.forward` signature `(features, lengths) → (encoded, out_lengths)` is frozen:
   the CTC heads, the joiner and streaming inference all depend on it. `return_intermediates`
   extends it to a 4-tuple; nothing else may.
2. Every stack is length-preserving at base rate; only `frontend` and `final_downsample` change the
   frame rate.
3. `streaming_forward` ≡ `forward(chunk_size=B)`, bit-for-bit, tested. Decode `chunk_size` must be a
   multiple of `chunk_lcm() = 8` base-rate frames.
4. Blank id is `vocab_size` (500), logits width is 501, and blank is the **last** column. CTC,
   InterCTC, the transducer and `InternalLmScorer`'s renormalisation all depend on it. The LM's
   `decoder_vocab_size` is 502 (id 500 is an unused hole so `eos_id = 501` is addressable).
5. A mel cache is bound to its manifest by fingerprint; `--train-manifest` and `--train-cache-split`
   move together.
6. `max_frames_per_batch`, `max_tokens_per_batch` and `max_lattice_per_batch` move together, or the
   one left behind silently pins the batch size.
7. Anything that sizes a VRAM budget must set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   **before the first CUDA allocation**, or it measures a different allocator than the run it is
   sizing. All three trainers set it.
8. Compile the four elementwise leaves (`BiasNorm`, `TransducerJoiner`, `ConvModule`, `SwiGluFfn`),
   never the model. The dividing line is `get_config()`, SDPA and data-dependent control flow.
9. `project_constraints` runs after **every** optimizer step in both acoustic trainers. A bound
   enforced only in `forward` leaves the parameter in `clamp`'s dead zone permanently.
10. `training.transducer.rnnt_loss` is an architecture key, not a loop setting: `pruned` adds
    `simple_am_proj`/`simple_lm_proj`, so checkpoints do not cross between the two values.
11. `total_steps` counts loader batches in the transducer stage and optimizer updates in the BEST-RQ
    stage. Convert to audio-hours before comparing any run to a published one.
12. Any stage that names a submodule `encoder` inherits `optim.encoder_lr_scale`. Override it
    deliberately or it silently halves that stage's LR (§7.4).
13. `ConvModule`'s two kernel-1 convs stay `nn.Conv1d`. They are *called* as `F.linear`, but their
    `ndim == 3` is what keeps 6.81 M params on AdamW; declaring them `nn.Linear` moves them to
    Muon without touching a checkpoint key (§3.6, §8.1).
14. A trainer refuses to start without `cmvn.pt` unless `cmvn_path=""` says so explicitly. The
    encoder's mean 0 / std 1 fallback exists for tests and for inference, where the checkpoint
    carries the real buffers (§2.2).
