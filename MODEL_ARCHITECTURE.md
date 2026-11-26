# MODEL_ARCHITECTURE.md: STREAM ASR v2.5

What the v2.5 acoustic model computes, operator by operator, why each choice was made, and how it is
trained, decoded and measured. `config/` holds the authoritative values; this document explains
them. Numbers marked **MEASURED** were run on one RTX 5070 (torch 2.11, CUDA 12.8).

Names used throughout: the encoder is the **S2 encoder**, its block the **FusedBranch block**, the
transducer loss the **banded RNN-T**.

| § | Contents |
|---|---|
| [1](#1-overview) | Overview and headline results |
| [2](#2-shapes-and-rates) | Shape and rate ledger, parameter counts |
| [3](#3-front-end) | Log-mel, CMVN and mean fill, the subsampling frontend |
| [4](#4-encoder) | U-Net layout, the FusedBranch block, gates, attention |
| [5](#5-amplitude-by-construction) | Why no activation can grow without bound |
| [6](#6-heads-and-loss) | CTC head, predictor, joiner, objective, banded RNN-T |
| [7](#7-streaming) | Streaming contract, bounded cache, tail chunk |
| [8](#8-decoding-and-evaluation) | Label-looping greedy, beam, LM rescoring, evaluation metrics, STREAM-LM |
| [9](#9-training) | Data, optimisation, schedule, step mechanics, validation |
| [10](#10-observability-and-debugging) | Logged scalars and a debug playbook |
| [11](#11-invariants) | Rules that keep the design correct |
| [12](#12-results) | Test-set results, timing, budget curve |
| [13](#13-design-notes) | How the recipe was chosen, open directions, references |

---

## 1. Overview

A from-scratch streaming transducer for English speech, trained on LibriSpeech 960 h plus Common
Voice 27.0 English (3,867 h after speed perturbation and trimming, §9.1):

* **S2 encoder**, 61.78 M parameters: a three-rate U-Net with blocks at 25 Hz and 12.5 Hz, output at
  12.5 Hz (80 ms per frame). Each block is gated attention followed by a conv-gated MLP.
* **Stateless predictor + additive joiner** over BPE-500, plus a CTC head. 63.08 M parameters total.
* **One set of weights decodes offline and streaming** (320 ms chunks, 5.12 s bounded left context).
* **Banded RNN-T**: the joiner runs on 17 lattice cells per frame around the CTC head's cumulative
  occupancy instead of the full lattice.
* **Amplitude bounded by construction**: every stack entry is a weightless RMSNorm and the only
  learnable residual-path scale is a floored gate. No parameter is ever projected or clamped.
* **STREAM-LM** (52.5 M, own 8k tokenizer, separate slice, §8.4) rescores the beam's n-best with
  internal-LM estimation.

Headline (test WER, beam 10 + LM + ILME, MEASURED, §12):

| | offline | streaming 320 ms |
|---|---|---|
| test-clean | **2.29 %** | **2.88 %** |
| test-other | **5.46 %** | **7.19 %** |
| cv-test | **13.42 %** | **16.80 %** |

Training: 1,200,000 loader batches (about 24 passes over the 3,867 h) in **35.7 h** on one RTX
5070, peak 5.07 GiB allocated.

---

## 2. Shapes and rates

One utterance of `S` samples at 16 kHz, batch `B`, `V = 501` (500 BPE pieces + blank 500).

| # | Stage | Op | Shape out | Rate |
|---|---|---|---|---|
| 0 | audio | `soundfile` load | `[S]` | 16 kHz |
| 1 | log-mel | `LogMel_Transform`, 25 ms window, 10 ms hop, 80 bins | `[T₀, 80]` | 100 fps |
| 2 | SpecAugment | train only, fill = per-bin CMVN mean in raw space (§9.1) | `[B, T₀, 80]` | 100 fps |
| 3 | pad + CMVN | pad `T₀` to a multiple of 16 with the CMVN mean, then `(x − mean)/std` | `[B, ⌈T₀/16⌉·16, 80]` | 100 fps |
| 4 | frontend | `DwSepSubsampling` ×4 time, ×4 freq, weightless RMSNorm | `[B, T₁, 320]`, `T₁ = ⌈T₀/4⌉` | 25 fps |
| 5 | S0 | 4 FusedBranch blocks | `[B, T₁, 320]` | 25 fps |
| 6 | S1 | pool ↓2, 8 blocks | `[B, T₂, 512]`, `T₂ = ⌈T₁/2⌉` | 12.5 fps |
| 7 | S2 | pool ↓2, linear path (no blocks) | `[B, T₃, 640]`, `T₃ = ⌈T₂/2⌉` | 6.25 fps |
| 8 | S3 | upsample ↑2 + U-Net skip from S1, 8 blocks | `[B, T₂, 512]` | 12.5 fps |
| 9 | memory | weightless RMSNorm | `[B, T₂, 512]` | 12.5 fps |
| 10a | CTC head | `Linear(512 → 501)` | `[B, T₂, 501]` | 12.5 fps |
| 10b | predictor | `StatelessPredictor` | `[B, U+1, 512]` | per label |
| 11 | joiner | additive, on the band only | `[B, T₂, 17, 501]` | |
| 12 | loss | banded RNN-T + 0.5 · CTC | scalar | |

Total reduction: 160 × 4 × 2 = **1,280 samples = 80 ms per memory frame**.

**Parameters (MEASURED).** `hf` is the conv-gated MLP width, `⌊8d/3⌋` rounded down to a multiple of 8.

| stack | rate | d | blocks | heads | hf | params |
|---|---|---|---|---|---|---|
| frontend | 50 → 25 Hz | 320 | | | | 0.84 M |
| S0 | 25 Hz | 320 | 4 | 5 | 848 | 5.36 M |
| S1 | 12.5 Hz | 512 | 8 | 8 | 1360 | 27.55 M |
| S2 | 6.25 Hz | 640 | 0 | | | 0.33 M (its projections) |
| S3 | 12.5 Hz | 512 | 8 | 8 | 1360 | 27.71 M |
| **encoder** | | | **20** | | | **61.78 M** |
| CTC head / predictor / joiner | | | | | | 0.26 / 0.26 / 0.78 M |
| **total** | | | | | | **63.08 M** |

Each block takes two residual steps with their own gates, so the trunk is 40 sequential residual
steps deep.

**Why 12.5 Hz output is safe for BPE-500 (MEASURED).** A transducer needs no more than one frame per
token per emission step, and CTC needs `T ≥ U + repeats`. On 20,000 training utterances:

| output rate | blank fraction | max U/T | p99.9 U/T | CTC-infeasible utterances |
|---|---|---|---|---|
| 25 Hz | 78.7 % | 0.38 | 0.34 | 0 % |
| **12.5 Hz** | **57.3 %** | 0.77 | 0.68 | **0 %** |
| 8.33 Hz | 35.9 % | 1.18 | 1.03 | 0.195 % |
| 6.25 Hz | 14.4 % | 1.57 | 1.37 | 17.4 % |

Over all 562,482 rows of `train_sp2` the 12.5 Hz maximum U/T is **0.943** and the smallest CTC slack
`T − (U + repeats)` is 2 frames: still feasible everywhere, with almost no margin. A lower output
rate, or a speed factor above 1.1, breaks feasibility. 6.25 Hz is not a legal CTC rate, which is why
no CTC tap sits on S2. BPE-500 carries 5.32 tokens per second of speech (p99 7.62).

---

## 3. Front end

### 3.1 Log-mel, CMVN and the mean fill

80-bin log-mel at 100 fps, computed once into an fp16 mmap cache (ExtractFeatures). CMVN is the
encoder's first op: `x ← (x − mean)/std` per bin, with the statistics from `data/features/cmvn.pt`
held as `state_dict` buffers, so a checkpoint carries its own normalisation.

**The only constant ever written into raw log-mel is the per-bin CMVN mean.** SpecAugment's masks,
the batch padding, the stride pad and the streaming tail all write `mean` in raw space, which is
exactly 0 after CMVN. A raw `0.0` would land at a median +1.5 σ after CMVN (+0.54 to +2.33 σ per
bin, MEASURED): a loud, spectrally shaped plateau where the model should see "nothing here". Writing
the mean before normalising keeps one encoder entry point that takes raw features in training,
offline decode and streaming alike.

**Pad to the total stride.** `forward` pads time to a multiple of 16 mel frames (×4 frontend, ×2,
×2 pooling) with the mean fill, so every pooling step averages frames computed from real or
mean-filled input, never an activation-space zero. Without it, an utterance's last 6.25 Hz frame
would depend on whether it is the longest in its batch, and differ again in streaming.

### 3.2 `DwSepSubsampling` (×4 time)

```
x:[B,T,80] → [B,1,T,80]                          # T a multiple of 16
conv1: Conv2d(1→128, k=3, stride=2, pad=(0,1)), left time-pad 1 → ReLU             50 Hz, 40 bins
dw2:   Conv2d(128→128, k=3, stride=2, pad=(0,1), groups=128), left time-pad 1      25 Hz, 20 bins
pw2:   Conv2d(128→128, k=1) → ReLU
reshape [B, T₁, 128·20 = 2560] → Linear(2560 → 320) → rms_norm (no weight)
```

Depthwise-separable subsampling (FastConformer's idea) costs 9.0 M MAC per second of audio, where a
dense 3×3 second conv would cost 147.5 M, and it delivers 25 Hz directly.

**Stride-aligned causality: left pad 1, not 2.** With a left pad of 2, output `i` of a stride-2,
k = 3 conv reads inputs `2i−2 … 2i`; stacked twice, 25 Hz frame `j` reads mel frames `4j−6 … 4j`, so
30 ms of each 40 ms frame's own audio is deferred to the next frame. With a left pad of 1 it reads
`4j−3 … 4j+3`: its whole group of four. The look-ahead stays inside a 4-frame group and every chunk
is a whole number of groups (§7), so this adds no streaming latency and keeps streaming ≡ batched.
The cache is two levels (`in_tail`, `mid_tail`) of 1 frame each.

---

## 4. Encoder

### 4.1 Layout: a three-rate U-Net

```
f   = frontend(x)                                  # 25 Hz, d 320, RMS 1
s0  = S0(f)                                        # 25 Hz, 4 blocks
s1  = S1(avgpool2(s0))                             # 12.5 Hz, 8 blocks
s2  = rms_norm(proj_640(avgpool2(s1)))             # 6.25 Hz, linear path, no blocks
s3  = S3(upsample2(s2), skip = s1)                 # 12.5 Hz, 8 blocks
mem = rms_norm(s3)                                 # encoder memory, RMS 1
```

Every stack begins the same way:

```
x = proj(x_in)                  # Linear(d_in → d), no bias; Identity when d_in == d
x = rms_norm(x + skip)          # skip only on S3; weightless: the stack entry has RMS exactly 1
```

`avgpool2` is a stride-2 mean over aligned pairs (the time axis is even at every rate because of the
pad to 16 mel frames). `upsample2` is `repeat_interleave(2)` trimmed to the skip's length. Both are
per-frame linear and chunk-aligned, hence transparent to streaming.

S2 holds no blocks but is not empty: S3's entry is `rms_norm(proj(upsample(s2)) + s1)`, a linear
6.25 Hz low-pass path beside the skip. The model was selected with it in place (§13).

### 4.2 `FusedBranchBlock`: attention, then a conv-gated MLP

```
q,k,v,g = rms_norm(x) @ W_in[q,k,v,g]                            # one GEMM, no bias
q, k    = rope(γq ⊙ rms_norm_head(q)), rope(γk ⊙ rms_norm_head(k)) # QK-norm per head
a       = SDPA(q, k, v, mask = chunk ∧ left window ∧ padding) ⊙ σ(g)
x       = x + β_attn ⊙ rms_norm(a @ W_out[a])                     # residual step 1
u,z     = rms_norm(x) @ W_in[u,z]                                 # one GEMM, no bias
c       = dwconv_causal_k15(u) ⊙ silu(z)                          # conv-gated MLP, width hf
x       = x + β ⊙ rms_norm(c @ W_out[c])                          # residual step 2
β       = 0.05 + 0.95 · σ(b)                                      # per channel, in (0.05, 1)
```

What each piece is and why:

* **Serial attention then MLP.** Two residual steps per block, each with one input norm, one input
  GEMM and one output GEMM. Serial order doubles the sequential depth over a parallel block for the
  same matrices.
* **Conv-gated MLP** (E-Branchformer's cgMLP): the depthwise causal conv runs on the MLP's hidden
  width, gated by SiLU. It replaces a separate convolution module and feed-forward pair.
* **QK-norm** with learnable per-dim gains `γq, γk` (OLMo 2, Gemma 3): attention logits are bounded
  by `‖γq‖∞ ‖γk‖∞ · √head_dim` whatever the residual's scale.
* **Gated attention**: an elementwise sigmoid gate on the SDPA output (Qiu et al., NeurIPS 2025),
  for stability at high LR and no attention sinks.
* **Peri-LN output norm** (Gemma 2/3, OLMo 2; Kim et al. 2025): each step's contribution has RMS
  exactly `β` per channel, whatever `W_out`'s scale is.
* **Per-slice parameters.** `W_in` and `W_out` are one `nn.Parameter` per logical slice (q, k, v, g,
  u, z; a, c) in a `ParameterDict`, concatenated in `forward` into the single GEMM, so Muon
  orthogonalises each slice on its own scale (§9.2).
* **No dropout, no macaron FFN, no value residual.** Dev WER was still falling at the end of the
  1.2M run (§12.3), with no over-fitting for dropout to fix.

Every config value is read in `__init__`, never in `forward`, so a whole block is one
`torch.compile` graph (§9.5).

### 4.3 Norms and gates

`rms_norm` is `F.rms_norm(x, (d,), eps=1e-6)` with **no weight**, everywhere except the QK-norm.
A learnable gain on a residual-path norm is a free amplitude, and Adam drives a sign-coherent scalar
into whatever bound is put on it. The only per-channel scales on the residual path are the `β`
gates, whose range the sigmoid fixes.

`b` starts at **−1.0** (`β₀ = 0.30`) and trains on AdamW with no weight decay. The floor of 0.05
exists because a gate that can reach 0 is an absorbing state: the block's weights stop receiving
gradient and never recover. With the floor, a block always sees at least 5 % of its gradient.
Saturation of `b` itself is soft but real: simulated AdamW at `lr 1.5e-3` stalls a sign-coherent
`b` at −15 to −20 after ~43k to 54k steps, so a gate on its floor rarely comes back. A rising
`train/res_gate_floor_count` therefore reads as "blocks switching off for good" (§10).

### 4.4 Attention

`head_dim = 64` everywhere; RoPE with base 10000 at each stack's own frame index. SDPA gets a boolean
mask (chunk-causal ∧ left-context window ∧ key padding), so it runs the memory-efficient kernel.

The left-context window is **chunk-granular**, because a ring buffer holds whole chunks: query `i` in
the chunk starting at `s(i)` sees key `j` iff `s(i) − L ≤ j < s(i) + C`, with `C` and `L` the chunk
and left context at that stack's rate. A per-frame sliding window would give each query in a chunk a
different window and break streaming ≡ batched. At full context (`chunk = 0`) the window is unlimited.

**A padded query attends to itself.** With a bounded window, a pad query far past a short row's
length can have no real key in view; SDPA returns NaN for an all-masked row, and `0 × NaN` would
carry it into every real query that reads the pad frame's value. Real queries always see
themselves, so only pad queries change.

**RoPE positions are window-local in streaming.** Absolute positions grow without limit in a live
session and fp32 `pos × inv_freq` loses relative-angle precision as they grow (MEASURED: 8e-4 rad
between adjacent frames after 1 h, 2.8e-2 rad after 24 h at 12.5 Hz). RoPE scores depend only on
`i − j`, so the ring caches **pre-RoPE** keys and rotates `[cache ∥ chunk]` with positions
`0 … L + C − 1` every chunk: exact, bounded, one rotation of `L` keys per chunk.

---

## 5. Amplitude by construction

Every operator that can change the activation scale and what bounds it. Every bound is a
consequence of the forward pass; none is enforced by a projection, a clamp or a watchdog.

| Operator | Normalised by | Bound |
|---|---|---|
| frontend output | weightless `rms_norm` | RMS = 1 per frame |
| stack entry (`proj` + skip) | weightless `rms_norm` | RMS = 1 per frame |
| one residual step `β ⊙ rms_norm(y)` | Peri-LN output norm | per-frame RMS ≤ `max β` < 1 |
| stack exit after `n` blocks (`2n` steps) | | RMS ≤ 1 + 2n coherent worst case, ≈ √(1 + Σβ²) if uncorrelated |
| attention logits | QK-norm | ≤ `‖γq‖∞ ‖γk‖∞ · √64` |
| encoder memory | weightless `rms_norm` | RMS = 1 per frame |
| joiner input to `tanh` | `enc_proj`, `pred_proj` | linear maps of RMS-1 inputs; `tanh` bounds the readout input |

Failure modes this rules out, by mechanism:

| Failure | Path it needs | Here |
|---|---|---|
| a learnable norm gain amplifies one stack until the next is arithmetically silenced | a gain on a residual-path norm | no gain on any residual-path norm |
| a norm that divides by `rms(x − bias)` but scales `x` | a free bias inside a norm | no bias in any norm |
| an inter-stack projection inflates in gain, anisotropy or rank | an unnormalised inter-stack Linear | every stack entry is renormalised, so `W`'s scale, spectrum and rank cannot reach the activation |
| a residual gate collapses to exactly 0 | a gate with a reachable 0 | `β ≥ 0.05` by parametrisation |
| one scalar gate dominates the global gradient clip | scalar gradients ∝ activation scale | activations are bounded, and scalars are clipped per tensor anyway (§9.2) |

What remains free, none of it able to move the trunk's amplitude:

* **Scale-free slices**: `q`, `k` feed a per-head norm; the frontend `Linear` and stack `proj` feed a
  weightless norm; `W_out` feeds the output norm. Their norms only set Muon's effective step size
  (`train/weight_norm_max`).
* **Scale-sensitive slices**: `g` feeds `σ`, `z` feeds SiLU, so their scale sets how saturated those
  gates are. A `σ(g)` stuck at 0 switches a head off (`train/gate_sat_frac`). At S3's entry,
  `‖proj‖` sets the deep-versus-skip mix (`stack_rms/3_skip_ratio`).
* **The QK gains** (attention sharpness) and **the joiner projections**.

---

## 6. Heads and loss

### 6.1 CTC head
`Linear(512 → 501)` on the memory at 12.5 Hz, blank 500 as the last column. It is an auxiliary loss,
the source of the band (§6.5), and the cheap full-dev greedy WER probe. There are no intermediate
CTC taps: S2 runs at 6.25 Hz, where 17.4 % of utterances are CTC-infeasible.

### 6.2 `StatelessPredictor`
Embedding 501 × 512, a depthwise causal Conv1d over the last 2 tokens, then a weightless
`rms_norm`. Streaming state is the last token id.

### 6.3 Joiner
Additive: `readout(tanh(enc_proj(mem)[:, :, None] + pred_proj(pred)))`, `joiner_dim 512`, readout
padded to 504 columns with a −inf bias on the pad (cuBLAS alignment). It is evaluated only on the
band's cells (§6.5).

### 6.4 Objective

```
N_tok = Σ token_lengths
total = rnnt / N_tok + 0.5 · ctc / N_tok
ctc   = F.ctc_loss(reduction = "sum")
rnnt  = Σ_b −log p(y_b | x_b) over the banded lattice
```

Both terms are divided by the batch's **token** count, so their ratio does not move with batch
composition or frame rate. The CTC weight of 0.5 was chosen by screening (§13).

### 6.5 Banded RNN-T

The RNN-T path at frame `t` sits near the number of tokens emitted so far, and the CTC head estimates
that number for free:

```
p_nb(t) = 1 − p_ctc(blank | t)                       # detached
c(t)    = Σ_{t' ≤ t} p_nb(t')                         # occupancy: expected tokens by frame t
c(t)   ← c(t) · U / c(T−1)                            # rescale so the path ends at (T−1, U)
W'      = min(W + 1, U + 1)                           # W = 16 → 17 cells, ±8 tokens
lo(t)   = clamp(⌊c(t)⌋ − W//2, 0, U + 1 − W')
lo(0)   = 0,  lo(T−1) = U + 1 − W'                    # both terminals inside the band
k       = W' − 1                                      # the largest legal rise of lo per frame
lo      ← cummin(lo − t·k) + t·k                      # (1) forward repair: rises ≤ k
lo(T−1) = U + 1 − W'
lo      ← max(lo, revcummax(lo − t·k) + t·k)          # (2) reverse repair: lift the tail
band(t) = { lo(t), …, lo(t) + W' − 1 }
```

`c` is a cumulative sum of non-negative numbers, so `lo` starts monotone. Pass (1) is the largest
`lo' ≤ lo` whose per-frame rise is at most `k`, which keeps the path connected. **Pass (1) alone is
not enough**: it only lowers `lo`, so a burst of non-blank mass late in the utterance drags
`(T−1, U)` out of the band. MEASURED on such a case (U = 90, T = 150, non-blank mass in the last 3
frames): `lo(T−1) = 48` against the 74 required, a **finite** cost of 2e30 that no NaN guard catches,
and an infinite arc posterior that poisons the batch's gradient. Pass (2) only raises `lo`, keeps
every rise ≤ `k`, and cannot lift `lo(0)` off 0 while `U ≤ T·k` (always: U/T ≤ 0.943 against
k = 16). The two passes held start, end, monotonicity and the rise bound on 20,000 adversarial
random occupancies. `train/band_infeasible` counts any band that still misses a terminal and must
read 0. A head with `c(T−1) < 0.5` (not yet emitting) falls back to the diagonal `c(t) = t·U/(T−1)`.

**The band is in tokens**, so its time coverage does not depend on the frame rate: ±8 tokens is
about ±1.5 s of speech at 5.3 tokens/s.

**Lossless in practice (MEASURED).** `dev/band_loss_gap`, the banded minus the full RNN-T cost per
token on a fixed dev batch, read **0.0000** at all 120 validations of the 1.2M run and
`train/band_infeasible` read 0 throughout. W = 16 keeps 31.5 % of the lattice cells on dev-other's
average utterance and less on longer training batches, because the band is a fixed 17 cells against
`U + 1`.

**Warmup.** The first `band_warmup_steps` = 10,000 loader batches use the full lattice, because a
fresh CTC head carries no alignment. The batch budgets are sized for that full lattice (§9.4).

**Implementation.** One autograd function serves full and banded lattices: the joiner evaluates
`[B, T, W', V]` at `u = lo + j` (predictor rows gathered after `pred_proj`), the log-softmax and arc
gather run on the band, the two arc grids are scattered into `[B, T, U+1]` grids filled with a large
negative constant, the anti-diagonal scan runs unchanged, and the backward gathers arc posteriors
back to the band. The full lattice is `lo = 0, W' = U + 1` through the same code, so "banded at full
width ≡ full" holds bit for bit, and both carry a float64 `gradcheck` (`test_rnnt_band`).

Prior art: CTC-guided lattice reduction (Wang et al. 2022) bands around a CTC Viterbi alignment and
alignment-restricted RNN-T (Mahadeokar et al. 2021) around an external one. The cumulative-occupancy
band needs no alignment pass: it is one `cumsum` over a tensor the step already computes.

---

## 7. Streaming

### 7.1 Contract

`S2Encoder.streaming_forward(chunk, cache, n_valid)` equals the batched `forward` with the same
`(chunk_size, left_context)` mask to fp32 round-off (gate `atol = 1e-4`, MEASURED 2.1e-6 to 2.7e-6),
including a partial final chunk and the kept tail. The time-coupling ops:

1. `DwSepSubsampling`: look-ahead confined to the 4-frame group, two 1-frame tails.
2. Every block's causal conv: the last 14 frames of `u`.
3. Every attention: a ring of the last `left_context` frames (at that stack's rate) of pre-RoPE `k`
   and `v`, plus their validity mask.
4. `avgpool2`: chunk boundaries align to pairs at every rate, so pooling never straddles one.

Units: `chunk_size` and `left_context` count **25 Hz frames** (40 ms). `chunk_lcm = 4` (rates ÷1,
÷2, ÷4), so a chunk is a multiple of 160 ms.

| | training pool (sampled per batch) | decode default |
|---|---|---|
| `chunk_size` | {0 = full context, 8 (320 ms), 16 (640 ms)} | 8 (320 ms) |
| `left_context` | {64 (2.56 s), 128 (5.12 s), unlimited} | 128 (5.12 s) |

### 7.2 Bounded state

At the default 5.12 s the whole streaming state is **1.73 M values, 6.9 MB in fp32** (MEASURED after
30 chunks), constant for any stream length: the k/v rings are 4 × 2 × 320 × 128 values at S0 and
16 × 2 × 512 × 64 at S1 and S3, plus the conv tails. Attention cost per chunk is constant too, so a
live session can run indefinitely.

### 7.3 Tail: finish the chunk, then decode one more

At end of stream the final partial chunk is completed with the CMVN mean and **one further whole
chunk** (`decode.tail_chunks: 1`, 320 ms) of mean fill follows. **Every output frame is kept**,
including those produced from the fill; the pad positions are key-masked, so real frames never attend
to fabricated ones. A transducer emits some final tokens a little late, and they need frames to land
on: dropping the tail's frames costs about 0.5 points of dev-other streaming WER, almost all of it in
the last two words. It costs one chunk of compute at finalize and no added latency on a live stream
that endpoints on trailing silence.

### 7.4 Optional: revise on endpoint

When a stream endpoints, the demo re-runs the offline mode over the buffered utterance and replaces
the streaming hypothesis. It costs one offline encoder pass at finalize. It is a product choice for
the demo, and **reported streaming WER never uses it**.

---

## 8. Decoding and evaluation

### 8.1 Greedy: label-looping

While a hypothesis emits blanks its predictor state does not change, so the joiner is evaluated on
the next 16 frames at once against that one state and the search jumps to the first frame whose
argmax is non-blank (Bataev et al. 2024). That is exactly frame-by-frame greedy, with one joiner
call per emitted token instead of one per frame (`test_label_looping_equals_framewise`). At 57 %
blank frames it removes most of the per-frame Python loop.

### 8.2 Beam and LM rescoring

A Graves-style time-synchronous beam (A/B sets, prefix recombination), `beam_size 10`,
`max_symbols 5` per frame (max U/T is 0.943 at 12.5 Hz). Each symbol step evaluates the whole live
beam in one batched predictor + joiner call.

With an LM attached the n-best is re-ranked by

```
score = acoustic + α · lm_seq − β · ilm_seq + λ · len
```

`lm_seq` is STREAM-LM's sequence log-probability of the hypothesis text (§8.4). `ilm_seq` is the internal-LM estimate (Meng et al.
2021): the joiner run with the encoder memory zeroed and renormalised over non-blank labels,
subtracted because the predictor already carries a language prior from the training transcripts.
`λ` offsets the transducer's deletion bias. All three weights default to 0 (the pure-acoustic
regression lock) and are tuned on dev per split and mode (§12).

### 8.3 Evaluation

`evaluate.py --clean | --other | --cv` scores the test set with weights tuned on the matching dev
set (`cv-dev2k` for cv). The (α, β, λ) sweep runs over a cached dev n-best, so its grid costs no
decoding; a weight that lands on the edge of its grid is a bound, not an optimum, and the grid is
widened rather than read as tuned.
Stages are cumulative (greedy → beam → beam + LM), each offline and streaming. Every row reports:

* WER and CER with raw error and reference counts, so two runs can be compared without re-decoding.
* **Edge error rates**: first-2 / middle / last-2 word error with the deletion share of each, to see
  where streaming errors sit.
* **Emission latency** (streaming): per correct word, the audio time its first token was shown (the
  end of its chunk) minus the word's end in a CTC forced alignment of the reference; median and p90.
  The alignment comes from the scored model's own CTC head, so this is the transducer's lag behind
  its own alignment, quantised to 80 ms frames. It tracks one model across decode settings; it is
  not comparable between two models.
* RTF, first-partial latency and finalize cost, from a separate serial timing pass on 200 strided
  utterances, because concurrent decoding makes timing meaningless.
* The n-best oracle WER on dev: the floor any rescoring could reach.
* Provenance: the transducer, both tokenizers and the LM by path and SHA-256, since every training
  run writes the same file names.

### 8.4 STREAM-LM

A 16-layer, 512-wide causal Transformer (52.5 M: 44.1 M in the blocks, 4.2 M each for the input
table and the readout) trained on LibriSpeech-LM + FineWeb-Edu + cv-train sentences, every line in
the acoustic output alphabet and every dev/test transcript held out.

* **Own tokenizer.** Rescoring reads hypothesis text, so the LM uses an 8,192-piece SentencePiece
  unigram trained on its corpus (`lm.tokenizer`) instead of the acoustic BPE-500: 1.23 tokens per
  word against 2.24 (measured), so a 256-token window holds twice the text and a step trains on
  twice the words. Each hypothesis is decoded to text and re-encoded, which also makes the score
  independent of how the acoustic model split a word into pieces.
* **Block.** Pre-norm with weightless RMSNorm (the acoustic stack's normaliser, no gain to grow):
  `x + Attn(rms_norm(x))`, then `x + SwiGLU(rms_norm(x))`, final `rms_norm` before the readout.
  Attention is GQA (8 query heads, 2 KV heads) with QK-norm and RoPE; K and V are separate matrices
  so Muon orthogonalises each. Layer 0's values are mixed into every deeper layer as
  `v + λ_l · v_0` with `λ_l` learned per layer (value residual, init 1.0). The SwiGLU FFN is
  bias-free.
* **Embeddings.** Untied, the readout zero-initialised, so step 0 starts at loss ln V. A readout
  through the N(0, 1) input table would give logits of std ≈ √d and a step-0 loss of hundreds of
  nats.
* **Sentence boundaries.** The packed stream is `line + EOS` per line, and EOS doubles as BOS:
  segment ids advance on each EOS, so the document mask trains every line as `[EOS] + line`, the
  exact row a hypothesis is scored as (inputs `[EOS] + ids`, targets `ids + [EOS]`).
* **Training.** Muon on the block matrices, AdamW on the tables and the `λ_l` scalars (no decay on
  vectors or scalars), matrices clipped as one group and vectors per tensor. WSD with a 1-sqrt
  anneal over the last 25 % of 140k steps of 64 × 256 tokens: 2.29e9 tokens, one pass over the
  packed corpus, read as non-overlapping windows shuffled without replacement. `lm_branch52500.pt`
  is the trunk at the decay start of a 70k budget, so one anneal from it measures what the second
  half of the corpus buys. bf16 autocast, each block compiled; z-loss 1e-4. Selection is perplexity
  over every window of `val.bin`, which is a proportional sample of all three sources.

---

## 9. Training

The recipe below produced every number in §12: LibriSpeech plus Common Voice, 1.2M batches.

### 9.1 Data and augmentation

* LibriSpeech `train-clean-100 + train-clean-360 + train-other-500`, each utterance paired with one
  copy at 0.9× or 1.1× speed (`train_sp2`: 562,482 rows, 1,933 h). The speed factor is resampled in
  the waveform at extraction and baked into the fp16 mel cache (104 GiB; `cv-train`'s is the same size), which is bound to its
  manifest by a fingerprint because row order is the cache index.
* Common Voice Scripted Speech 27.0 English (CC0), `cv-train`: `validated.tsv` minus every dev/test
  clip, speaker and sentence (1,335,344 clips, 2,031 h before trimming and 1,934 h after; the
  sentence filter alone removes 557k clips, since each sentence is read by many speakers). Transcripts go through
  `Transcript_Transform.normalise_transcript` (NFKD, upper-case `A-Z '`, a line with a digit or a
  spoken symbol is rejected). Leading/trailing silence is trimmed conservatively
  (`config/trim.yaml`: speech above floor + 25 % of the 10th-to-99th-percentile dB range, 0.5 s
  margin, untrimmed below a 15 dB range or a 1 s span), stored as `start_sample` / `num_samples`
  in the row and covered by the fingerprint. No speed perturbation.
* The two sources are read side by side (`train_sources`: one dataset per (manifest, cache) pair
  under a `ConcatDataset`, one sampler over their rows), so no cache is copied. The sampler buckets
  the pooled rows by length, so each source weighs in by its hours.
* Tokenizer (BPE-500 unigram) and CMVN are computed over both sources: the tokenizer over
  `train` + `cv-train` transcripts, CMVN over `train_sp2` + `cv-train` as the trainer reads them.
* SpecAugment as a GPU batch op on raw log-mel with the CMVN-mean fill: 2 frequency masks of up to
  27 bins; time masks by lhotse's rule at ×2.5 strength, up to 25 masks each ≤ 5 % of the
  utterance and ≤ 37.5 % together. No time warp.
* `FrameBucketSampler`: duration-sorted batches under three budgets (28,000 frames, 4,400
  transcript characters, and `B · max(frames) · max(chars)` ≤ 11 M for the full lattice of the
  warmup), shuffled per epoch.

### 9.2 Optimisation

| Tensor | Optimizer |
|---|---|
| `W_in` / `W_out` slices, stack `proj`, `frontend.linear`, joiner `enc_proj` / `pred_proj` | **Muon**, `lr 1e-2`, momentum 0.95, 5 Newton-Schulz steps, each slice orthogonalised on its own |
| depthwise convs, frontend convs, CTC head, joiner readout, predictor embedding | AdamW, `lr 7.5e-4` |
| gates `b`, QK gains, biases | AdamW, no weight decay |

Weight decay 1e-2. Orthogonalising per slice matters: Newton-Schulz on the fused concatenation would
couple the scales of unrelated projections.

**Clipping** is two-set (`GradientClipping`): the matrices are clipped as a group, scalars and
vectors per tensor. A vector parameter multiplies a whole `[B, T, C]` activation, so its gradient is
sized by the batch, not the model; in one shared clip a single gate decides every matrix's factor
and, through Muon's momentum EMA, reweights its steps.

**Runaway guard.** Newton-Schulz renormalises every Muon update, so clipping bounds no matrix step.
`GradNormGuard` watches the matrices' pre-clip norm and aborts, without checkpointing, when its
windowed median stays above 4× the run's quietest median for 3 windows.

### 9.3 Schedule and budget

WSD: 5,000 batches of linear warmup, a constant phase, and a `1 − √frac` anneal to 1 % over the
last 25 %. `grad_accum` 3. **`total_steps` counts loader batches, not optimizer updates.** The
budget is 1,200,000 batches, at most 93,333 h of encoder audio (the 28,000-frame budget is a
ceiling, not a fill), about 24 passes over `train_sp2` + `cv-train`.

**Branched cooldowns.** A WSD constant phase is uninformative on its own: most of the gain arrives
in the anneal. The trainer keeps the trunk checkpoint at each configured branch point,
and each shorter budget is one anneal resumed from its file. The 1.2M run keeps one,
`transducer_branch450000.pt` (the decay start of a 600k budget), so the run also yields an
equal-compute 600k model for 4.9 h more (§12.3).

### 9.4 Step mechanics

```
lr        = peak · wsd(step)
chunk, left ~ the §7.1 pool                          # one pair per batch
autocast(bf16): terms = model.joint_loss(batch, chunk, left, use_band = step ≥ 10k)
(terms.total / grad_accum).backward()
every grad_accum batches: guard reads matrix norm → two-set clip → opt.step() → zero_grad
```

MEASURED on the 1.2M run: 0.107 s per loader batch averaged over the whole run including
validation (single process, no resume), peak 5.07 GiB allocated with `expandable_segments`. The banded
lattice is small, so the encoder's activations set peak VRAM after the warmup. A batch that runs out
of memory is dropped with a warning and its accumulation window discarded. The three budgets move
together, or the one left behind silently pins `B`. `scripts/profile_transducer_step.py` measures
step time, VRAM and kernel counts on the real loader.

### 9.5 Compilation

Every `FusedBranchBlock` is compiled with `torch.compile(dynamic=True)`, SDPA included; the whole
model is not (this torch build hits a dynamic-shape tiling assert). The residual stream is fp32 from
every stack entry, so one block code object serves 5 graphs (one per width × mask kind) rather than
twice that. dynamo's recompile limit is per code object and exceeding it silently demotes the block
to eager, so the trainer raises it to 64. Validation runs eager.

### 9.6 Validation, checkpoints, averaging

Every 10,000 batches, over full dev-clean and dev-other and `cv-dev2k` (an evenly strided
2,000-clip slice of cv-dev, tag suffix `_cv`): greedy CTC and label-looping greedy
transducer WER, offline and in the deployed streaming condition (320 ms chunks, 5.12 s left context,
kept tail), plus `dev/band_loss_gap` and the attention-gate saturation on a fixed dev batch.
`transducer_best.pt` is selected on **dev-other streaming greedy-transducer WER**.

`transducer_last.pt` is written atomically every 5,400 batches with optimizer and RNG state; Ctrl-C
checkpoints and exits at the next step. The newest 5 rolling `transducer_step{N}.pt` snapshots are
averaged into `transducer_avg.pt`, the checkpoint every decode path loads.

---

## 10. Observability and debugging

| Tag | Quantity |
|---|---|
| `train/loss`, `rnnt`, `ctc` | loss terms per token |
| `train/lr` | AdamW group 0 (Muon follows the same shape) |
| `train/grad_norm_guarded` | pre-clip norm over the weight matrices: the health signal |
| `train/grad_norm` | pre-clip norm of the scalar/vector group |
| `train/res_gate_min`, `res_gate_floor_count` | minimum `β` over all 40 gates; count with `σ(b) < 0.01` |
| `stack_rms/{i}_exit` | realised RMS leaving each stack (entry is 1 by construction) |
| `stack_rms/3_skip_ratio` | RMS of `proj(s2)` over RMS of the S1 skip at S3's entry |
| `train/weight_norm_max` | max ‖W‖_F over Muon matrices (tracks Muon's effective step) |
| `train/qk_gain_max` | max `‖γq ⊙ γk‖∞` |
| `train/band_scale` | median `c(T−1)/U` before the rescale |
| `train/band_infeasible` | bands missing a terminal after the repair; **must be 0** |
| `train/gate_sat_frac` | fraction of attention-gate activations below 0.01, on the fixed dev batch |
| `dev/band_loss_gap` | (banded − full) RNN-T cost per token on the fixed dev batch |
| `dev/ctc_wer*`, `dev/transducer_wer*`, `dev/blank_frac` | full dev, greedy, offline and streaming, per split |

Everything is emitted from one host sync per `log_every`.

| Symptom | Read, in this order | Usually |
|---|---|---|
| dev WER turns upward mid-run | `stack_rms/*_exit` → `res_gate_floor_count` → `grad_norm_guarded` | Not amplitude (bounded by construction). A rising exit RMS means one block dominates its stack; gates on the floor mean blocks switching off |
| `res_gate_floor_count` rising | per-stack `res_gate_min` | Blocks deciding they are useless; a capacity hint, not a divergence |
| `dev/band_loss_gap` > 0.01 | `train/band_scale` → `dev/blank_frac` | The CTC head's token count drifting from the transducer's alignment; widen `band_width` or extend the warmup. The trainer warns |
| `train/rnnt` ~1e28 or an inf gradient | `train/band_infeasible` | A band missing `(T−1, U)`: the repair was bypassed |
| streaming last-word errors ≫ offline | `decode.tail_chunks` | The tail is being dropped (§7.3) |
| streaming WER ≫ offline | `dev/ctc_wer_stream_gap` → decode left context | The decode window is one the training pool never sampled |
| OOM mid-run | OOM-skip warnings → the three budgets | `max_frames_per_batch` too high for the densest batch |
| step time regresses hours in | the log, for a dynamo recompile-limit warning | A block silently demoted to eager |
| greedy disagrees with frame-wise greedy | `test_label_looping_equals_framewise` | A bug; label-looping is exact |

---

## 11. Invariants

1. **No learnable scale on the residual path except the floored gates `β`.** Every residual-path
   norm is weightless. A gain, a bias inside a norm or an unnormalised inter-stack operator re-opens
   the failure modes of §5. If an experiment needs one, bound it by parametrisation, never by
   projection.
2. Every stack entry, the frontend output and the encoder memory are `rms_norm`ed.
3. `S2Encoder.forward(features, lengths, chunk_size, left_context) → (memory, out_lengths)` takes
   raw log-mel on every path; `streaming_forward` ≡ `forward` with the same mask (`atol 1e-4`),
   tested. The left-context window is chunk-granular.
4. `chunk_size` and `left_context` count 25 Hz frames and are multiples of 4.
5. Blank is id 500, the last of 501 columns (CTC, transducer, internal-LM scorer).
6. The only constant written into raw log-mel is the per-bin CMVN mean.
7. Banded RNN-T at `W' = U + 1` is bit-identical to the full loss; both carry a float64 gradcheck.
   The band is built with both repair passes and `train/band_infeasible` reads 0.
8. The CTC posteriors that set the band are **detached** from it.
9. `rnnt` and `ctc` are both divided by the batch's token count.
10. `W_in` / `W_out` are orthogonalised per logical slice.
11. The mel cache is bound to its manifest by fingerprint; the three batch budgets move together;
    `expandable_segments` is set before the first CUDA allocation.
12. `total_steps` counts loader batches; convert to audio-hours before comparing runs.
13. The trunk checkpoint at each branch point is kept, never rolled off.
14. Streaming decode feeds `decode.tail_chunks` extra chunks of mean fill and keeps their frames.
15. The frontend pads time by 1 frame on the left per stride-2 stage.
16. Compile the blocks, never the whole model.

---

## 12. Results

`transducer_avg.pt` (mean of the snapshots at 1,177.2k to 1,198.8k of the 1.2M run), STREAM-LM
`lm_best.pt` (§8.4), beam 10, streaming at 320 ms chunks with 5.12 s left context and one kept tail
chunk. From `runs/eval/report-{clean,other,cv}.json`; rescoring weights tuned on the matching dev
set (cv: `cv-dev2k`), never on test. The one-setting row is from
`report-{clean,other,cv}-shipped.json`. MEASURED 2026-11-26.

### 12.1 Accuracy

| Decoding | clean offline | clean stream | other offline | other stream | cv offline | cv stream |
|---|---|---|---|---|---|---|
| greedy | 2.83 % | 3.80 % | 6.55 % | 9.07 % | 15.69 % | 20.07 % |
| beam 10 | 2.89 % | 3.76 % | 6.50 % | 8.75 % | 15.43 % | 19.31 % |
| beam 10 + LM + ILME | **2.29 %** | **2.88 %** | **5.46 %** | **7.19 %** | **13.42 %** | **16.80 %** |
| CER, beam + LM | 0.73 % | 0.99 % | 2.24 % | 3.13 % | 6.97 % | 8.99 % |
| beam + LM at (0.8, 0.2, 0.5) on every split | 2.31 % | 2.83 % | 5.46 % | 7.19 % | 13.40 % | 16.75 % |
| CER, same | 0.75 % | 0.96 % | 2.24 % | 3.13 % | 7.02 % | 8.98 % |

| tuned (α, β, λ) | offline | streaming |
|---|---|---|
| clean | (0.6, 0.1, 1.0) | (0.7, 0.0, 0.75) |
| other | (0.8, 0.2, 0.5) | (0.8, 0.2, 0.5) |
| cv | (0.9, 0.3, 1.0) | (1.0, 0.3, 0.5) |

Dev at the tuned weights: clean 2.13 / 2.60 %, other 5.32 / 7.16 %, cv-dev2k 10.27 / 13.20 %
(offline / streaming). n-best oracle on dev: clean 1.04 / 1.56 %, other 3.96 / 5.81 %, cv 7.48 /
10.15 %. Of the gap between the acoustic-only n-best top and its oracle, rescoring closes 28 %
(clean offline) to 56 % (other streaming). λ = 1.0 was the top of the
grid these reports were tuned on (clean and cv offline), so those two entries are bounds; the grid
now runs to 2.0.

* **The LM's share.** Beam → beam + LM is −21 / −23 % relative on clean, −16 / −18 % on other and
  −13 % on cv (offline / streaming). Without the LM the beam is level with greedy on LibriSpeech
  (±0.06 points): its job is the n-best it hands the LM, so greedy is the decoder when no LM runs.
* **One setting is enough.** The dev optima differ per domain (α 0.6 to 1.0, β 0.0 to 0.3), yet
  dev-other's (0.8, 0.2, 0.5) on every split is within 0.05 points of them in all six cells, and
  lower in three (clean streaming and both cv). Per-split tuning does not transfer a measurable gain
  to test, so (0.8, 0.2, 0.5) is the recommended deployment setting.

Where the errors sit (beam + LM, error % first-2 / middle / last-2 words, deletion share in
brackets):

| | offline | streaming |
|---|---|---|
| test-clean | 2.6 / 1.9 / 3.0 | 3.5 / 2.4 / 3.3 |
| test-other | 6.8 / 4.5 / 5.6 | 10.0 / 5.9 / 6.6 |
| cv-test | 12.1 / 11.2 / 14.9 (last 3.6 del) | 16.9 / 14.0 / 17.0 (last 4.9 del) |

Streaming costs 1.4 points mid-utterance on test-other and 3.2 on the first two words, which have
no left context to adapt from: the first-words penalty is the largest structural streaming loss
left. The kept tail chunk holds the last words near the middle on LibriSpeech. On cv the last two
words are the worst region even offline, and the LM trades tail substitutions for deletions (last-2
deletions 3.3 → 3.6 % offline, 4.2 → 4.9 % streaming, at a lower tail error overall). The training
trim keeps 0.5 s of margin, so it is not the cause; Common Voice recordings that stop on the last
word are the likely one, unchecked.

### 12.2 Speed and latency

Serial timing pass, 200 strided utterances, RTX 5070:

| | RTF offline | RTF streaming | finalize |
|---|---|---|---|
| greedy | 0.003 | 0.030 to 0.032 | 0.01 s |
| beam 10 | 0.016 to 0.018 | 0.045 to 0.048 | 0.09 to 0.12 s |
| beam 10 + LM | 0.017 to 0.018 | 0.046 to 0.048 | 0.10 to 0.12 s |

First partial 11 to 14 ms after a chunk arrives. Batch-1 streaming on this GPU is bound by kernel
launches, not arithmetic, which is why the encoder's width does not show in the RTF. Emission
latency against the model's own CTC alignment (§8.3): median 0 to 80 ms, p90 240 ms on LibriSpeech
and 320 ms on cv.

### 12.3 Budget curve (branched cooldowns)

Trainer validation, greedy transducer, final weights of each run (not averaged); dev-clean and
dev-other in full, cv on `cv-dev2k`:

| budget | GPU time of its own path | dev-clean off / stream | dev-other off / stream | cv off / stream |
|---|---|---|---|---|
| 600k (branch from 450k) | 13.4 h + 4.9 h | 2.94 / 3.92 | 7.39 / 10.03 | 13.15 / 17.57 |
| **1.2M** | **35.7 h** | **2.62 / 3.49** | **6.78 / 9.12** | **12.31 / 16.17** |

Doubling the budget fits WER ∝ steps^−0.14 on dev-other streaming, −0.17 on dev-clean and −0.12 on
cv, the flattest. At the 450k branch point, still on the constant learning rate, dev-other
streaming read 19.68 %; the 150k-batch anneal took it to 10.03 %. Two points fix an exponent and
cannot show curvature: whether 2.4M keeps paying is a projection (about −9 % for another 36 h).

---

## 13. Design notes

### 13.1 How the recipe was chosen

The design was validated on 60,000-batch proxy runs (full WSD, from scratch, scored on full dev)
and 15,000-batch screens, before any full run. The first design put five blocks at 6.25 Hz
(`[2, 4, 5, 4]`, conv-only S0, parallel attention ∥ MLP, CTC weight 0.1). Screens and a block
knockout found the allocation inverted:

| change (15k screens, dev-other streaming greedy) | effect |
|---|---|
| `[4, 8, 0, 8]` with attention in S0 | −4.3 points, the largest single gain |
| serial attention then MLP | −0.8 |
| CTC weight 0.1 → 0.3 | −0.6 (helps the CTC head most) |
| all three together | −6.4 (−19 % relative), transferred exactly to the 60k proxy |
| CTC weight 0.3 → 0.5 | −0.3, within screen noise, kept |
| 25 Hz output (an extra 25 Hz stack) | no gain, +30 % wall clock |
| 45 % wider stacks | −0.8 for +18 % wall clock |
| a bounded (non-normalised) block output | +2.2 worse |
| one more block per stack, or weight decay 0.1 | within noise |
| peak LR ×2 | ~1.5 points worse at 60k |

A block costs about the same compute at every rate here, but a 6.25 Hz block costs 4× the
parameters of a 25 Hz one, and the knockout showed S2's blocks were the least used: removing all
five cost +0.30 nats/token, removing one S0 block +0.47 to +0.68. Depth bought less than the same
wall clock spent on steps.

**Width, on the two-source data.** Two 60k proxies, identical but for `stack_dims`: the wide
`[320, 512, 640, 512]` read 17.36 % dev-other streaming greedy against 18.81 % for
`[256, 384, 512, 384]`, and 28.65 % against 30.23 % on `cv-dev2k`, for +17 % wall clock (1:49 vs
1:33). The gate was ≥ 0.5 points with no cv loss; wide was adopted. The 60k screen cannot say how
much of the margin survives 1.2M batches, and no full-scale base run exists to compare.

**Decided without a full-scale result:**

* **No consistency regularisation (CR-CTC).** Built and tested, never measured at a scale where it
  could pay: its published gain is compute-matched at ~150 data passes, this run is 24, and it
  halves the distinct audio per batch. Off.
* **No self-supervised pretraining.** On the proxy, a BEST-RQ warm start of the previous
  architecture was level with training from scratch at 30k batches, so the recipe trains from
  scratch and saves the pretrain's GPU hours.
* **Band width 16** from `dev/band_loss_gap` = 0 throughout, rather than a sweep.
* **5.12 s left context, no band offset, no dropout**: defaults, not swept.

### 13.2 Open directions

* **The first-words penalty** (§12.1: streaming adds 3.2 points on the first two words of
  test-other, 1.4 in the middle). Decode-only first: a 640 ms first chunk followed by 320 ms ones
  (both sizes are in the training pool, the mixed sequence is not), costing latency on the first
  words only. Then cross-utterance context, which seeds each cache from the previous segment of the
  same chapter; it changes the evaluation protocol, so it would be reported separately.
* **The streaming penalty overall** (1.7 points on test-other): a one-way full-context → streaming
  distillation between two views of the same batch (dual-mode training) targets it directly.
* **A longer budget.** 600k → 1.2M bought −9 % on dev-other streaming (steps^−0.14) and −8 % on cv;
  2.4M is a projection of about −9 % more for another 36 h.
* **STREAM-LM budget**: one anneal from `lm_branch52500.pt` measures what the second half of the
  corpus buys.
* **Consistency regularisation** at a long budget, judged against this recipe.
* **Removing the empty S2 stage**, which would drop `chunk_lcm` to 2 (a model change that needs its
  own run).
* **Left-context sweep** (2.56 s / 5.12 s / unlimited) is decode-only on the trained checkpoint.

### 13.3 References

Zipformer (Yao et al., ICLR 2024); CR-CTC (Yao et al., ICLR 2025, arXiv 2410.05101); FastConformer
(Rekesh et al. 2023); Branchformer / E-Branchformer; Peri-LN (Kim et al., ICML 2025); Gated
Attention (Qiu et al., NeurIPS 2025); Muon (Jordan et al. 2024); WSD schedules (Hu et al. 2024);
CTC-guided RNN-T lattice reduction (Wang et al. 2022, arXiv 2210.16481); Ar-RNN-T (Mahadeokar et
al. 2021); label-looping decoding (Bataev et al. 2024, arXiv 2406.06220); ILME (Meng et al. 2021,
arXiv 2011.01991); Dual-mode ASR (Yu et al., ICLR 2021); BEST-RQ (Chiu et al. 2022); SpecAugment
(Park et al. 2019); LibriSpeech (Panayotov et al., ICASSP 2015).
