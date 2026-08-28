import math
from typing import cast

import pytest
import torch

from src.shared_kernel.MaskUtils import make_pad_mask
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.BiasNorm import BiasNorm
from src.shared_kernel.SwiGluFfn import SwiGluFfn
from src.shared_kernel.RoPE_Transform import rotary_tables
from src.slices.TrainAcousticModel.RotaryAttention import RotaryAttention
from src.slices.TrainAcousticModel.ConvModule import ConvModule
from src.slices.TrainAcousticModel.Conv2dSubsampling import Conv2dSubsampling
from src.slices.TrainAcousticModel.Resample import SimpleDownsample, SimpleUpsample
from src.slices.TrainAcousticModel.ZipformerBlock import ZipformerBlock
from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder
from src.slices.TrainAcousticModel.ZipformerStack import ZipformerStack
from src.shared_kernel.ParameterProjection import project_constraints
from src.slices.TrainAcousticModel._train_utils import (
    _Checkpointed,
    branch_gain_params,
    stack_mix_params,
    trunk_gain_max,
    trunk_stable_rank_min,
)

_MODEL = get_config().model
_N_MELS = get_config().audio.n_mels


def _mask():
    return make_pad_mask(torch.tensor([7, 5]), max_len=7)  # [2, 7]


def test_biasnorm_preserves_shape_and_is_finite():
    x = torch.randn(2, 7, 16)
    out = BiasNorm(16)(x)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_biasnorm_matches_its_definition():
    # BiasNorm folds the reciprocal and exp(log_scale) into the [.., 1] statistic to save two
    # full-size passes over x. Pin it against the literal definition it is an optimisation of:
    # rms = sqrt(mean((x - bias)^2) + eps), out = x / rms * exp(log_scale).
    torch.manual_seed(0)
    norm = BiasNorm(24).double()
    with torch.no_grad():
        norm.bias.normal_()
        norm.log_scale.fill_(0.37)
    x = torch.randn(3, 9, 24, dtype=torch.float64) * 4
    rms = (x - norm.bias).pow(2).mean(dim=-1, keepdim=True).add(norm.eps).sqrt()
    assert torch.allclose(norm(x), x / rms * norm.log_scale.exp(), rtol=0, atol=1e-14)


def test_swiglu_preserves_shape():
    x = torch.randn(2, 7, 32)
    assert SwiGluFfn(32)(x).shape == x.shape


def test_rotary_attention_shape_and_backward():
    x = torch.randn(2, 7, 48, requires_grad=True)
    out, v = RotaryAttention(48, num_heads=4)(x, _mask())
    assert out.shape == x.shape
    assert v.shape == (2, 4, 7, 12)  # [B, heads, T, head_dim], the value-residual carrier
    out.sum().backward()
    assert x.grad is not None


def test_rotary_attention_value_residual_changes_output():
    # A non-zero lambda + injected layer-0 values must actually move the output (wiring guard),
    # and lambda 0 must be a no-op regardless of what is injected (vanilla-attention lock).
    torch.manual_seed(0)
    attn = RotaryAttention(48, num_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 7, 48)
    v0 = torch.randn(1, 4, 7, 12)
    with torch.no_grad():
        base, _ = attn(x, make_pad_mask(torch.tensor([7]), 7))
        attn.res_lambda.fill_(0.0)
        same, _ = attn(x, make_pad_mask(torch.tensor([7]), 7), value_residual=v0)
        attn.res_lambda.fill_(1.0)
        moved, _ = attn(x, make_pad_mask(torch.tensor([7]), 7), value_residual=v0)
    assert torch.allclose(base, same, atol=1e-6)
    assert not torch.allclose(base, moved, atol=1e-4)


def test_conv_module_shape_and_ignores_padding():
    x = torch.randn(2, 7, 32)
    out = ConvModule(32, kernel=15)(x, _mask())
    assert out.shape == x.shape


def test_conv_module_pointwise_matches_the_convolution_it_replaces():
    # The two kernel-1 convolutions are run as F.linear on the [B, T, C] layout the block already
    # carries, instead of transposing to [B, C, T] for cuDNN. The parameters stay Conv1d-shaped, so
    # this must be the *same function* -- not merely close -- or every checkpoint silently shifts.
    torch.manual_seed(0)
    conv = ConvModule(32, kernel=15).double().eval()
    x = torch.randn(2, 21, 32, dtype=torch.float64)
    mask = make_pad_mask(torch.tensor([21, 15]), 21)
    with torch.no_grad():
        expected = conv.pointwise1(x.transpose(1, 2)).transpose(1, 2)
    assert torch.equal(ConvModule._pointwise(conv.pointwise1, x), expected)
    with torch.no_grad():
        assert torch.isfinite(conv(x, mask)).all()


def test_conv_module_is_causal():
    # Output at frame t must not change when a strictly-future frame is perturbed.
    torch.manual_seed(0)
    conv = ConvModule(32, kernel=15).eval()
    x = torch.randn(1, 20, 32)
    no_pad = make_pad_mask(torch.tensor([20]), 20)
    with torch.no_grad():
        base = conv(x, no_pad)
        x2 = x.clone()
        x2[:, 15:] += 5.0  # perturb frames 15..19 only
        pert = conv(x2, no_pad)
    assert torch.allclose(base[:, :15], pert[:, :15], atol=1e-5)


def test_conv2d_subsampling_halves_time():
    x = torch.randn(2, 101, _N_MELS)
    lengths = torch.tensor([101, 60])
    y, out_len = Conv2dSubsampling()(x, lengths)
    assert y.shape[0] == 2 and y.shape[2] == _MODEL.encoder_dims[0]
    assert y.shape[1] == (101 - 1) // 2 + 1  # 51
    assert out_len.tolist() == [(101 - 1) // 2 + 1, (60 - 1) // 2 + 1]


def test_frontend_is_causal_in_time():
    # Base-rate output frame t depends only on input frames <= 2t (both convs causal in time).
    # Perturbing input frames >= 30 must leave outputs 0..14 unchanged (2*14=28 < 30) while changing
    # frame 15+ (2*15=30). A symmetric-in-time frontend leaks into frame 14, so checking through
    # frame 14 (:15) genuinely discriminates causal from non-causal; the second assert confirms the
    # perturbation actually propagates (guards against a degenerate input-ignoring implementation).
    torch.manual_seed(0)
    front = Conv2dSubsampling().eval()
    x = torch.randn(1, 60, _N_MELS)
    lengths = torch.tensor([60])
    with torch.no_grad():
        base, _ = front(x, lengths)
        x2 = x.clone()
        x2[:, 30:] += 5.0  # perturb input frames 30..59
        pert, _ = front(x2, lengths)
    assert torch.allclose(base[:, :15], pert[:, :15], atol=1e-5)
    assert not torch.allclose(base[:, 15:], pert[:, 15:], atol=1e-5)


def test_downsample_then_upsample_restores_length():
    x = torch.randn(2, 20, 8)
    lengths = torch.tensor([20, 13])
    down = SimpleDownsample(4)
    up = SimpleUpsample(4)
    y, dl = down(x, lengths)
    assert y.shape[1] == 5  # ceil(20/4)
    assert dl.tolist() == [5, 4]  # ceil(20/4), ceil(13/4)
    z = up(y, out_len=20)
    assert z.shape[1] == 20


def test_zipformer_block_shape_and_backward():
    x = torch.randn(2, 7, 64, requires_grad=True)
    out, v = ZipformerBlock(64, num_heads=4)(x, _mask())
    assert out.shape == x.shape
    assert v.shape == (2, 4, 7, 16)  # block exposes its attention values for the stack residual
    out.sum().backward()
    assert x.grad is not None


def test_encoder_value_residual_gates_init_zero():
    # Regression lock for the blank-collapse fix: under the shipped config every deeper
    # block's value-residual gate must start at 0, so a fresh encoder trains identically to the
    # proven no-value-residual baseline. A non-zero default here re-introduces the collapse.
    from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder

    enc = ZipformerEncoder(cmvn_path=None)
    for stack in enc.stacks:
        for block in stack.blocks:
            # Every gate must start at 0 (blank-collapse guard: fresh encoder == vanilla baseline).
            assert block.attn.res_lambda.item() == 0.0


def test_stack_changes_dim_preserves_time():
    x = torch.randn(2, 12, 32)
    lengths = torch.tensor([12, 9])
    base_mask = make_pad_mask(lengths, 12)
    stack = ZipformerStack(dim_in=32, dim=48, num_layers=2, downsample=4, num_heads=4)
    out = stack(x, lengths, base_mask)
    assert out.shape == (2, 12, 48)  # time preserved, channels -> 48
    out.sum().backward()


def _stack_with_bypass(value: float) -> ZipformerStack:
    stack = ZipformerStack(dim_in=32, dim=32, num_layers=1, downsample=1, num_heads=4)
    with torch.no_grad():
        stack.bypass.fill_(value)
    return stack


def _bypass_grad(stack: ZipformerStack) -> torch.Tensor | None:
    lengths = torch.tensor([12, 9])
    out = stack(torch.randn(2, 12, 32), lengths, make_pad_mask(lengths, 12))
    out.pow(2).sum().backward()
    return stack.bypass.grad


def test_bypass_past_the_bound_is_dead_and_the_projection_revives_it():
    # `clamp`'s gradient is exactly 0 past its bounds, so a gate the optimizer pushes above 1.0 is
    # frozen out of training for good. MEASURED 2026-08-05: encoder.stacks.5.bypass sat at
    # 1.0020-1.0049 across 22k steps with AdamW exp_avg at -2.3e-22, while stacks.3.bypass climbed
    # 0.9655 -> 0.9872 toward the same trap.
    dead = _stack_with_bypass(1.0035)
    assert float(_bypass_grad(dead)) == 0.0, "this is the bug: no gradient past the bound"

    dead.project()
    assert dead.bypass.detach().item() == 1.0
    dead.bypass.grad = None
    assert float(_bypass_grad(dead)) != 0.0, "on the bound, the gate must train again"


def test_project_leaves_an_in_range_gate_untouched():
    interior = _stack_with_bypass(0.6)
    expected = interior.bypass.detach().clone()
    interior.project()
    assert interior.bypass.detach().equal(expected)


def test_project_constraints_reaches_every_gate_through_the_checkpoint_wrapper():
    # The trainer replaces encoder.stacks with _Checkpointed wrappers when grad_checkpoint is on,
    # so the projection walks modules() rather than self.stacks. Both layouts must reach all six
    # gates, or the constraint silently stops being applied at the one setting that needs it.
    encoder = ZipformerEncoder(cmvn_path=None)
    with torch.no_grad():
        for stack in encoder.stacks:
            stack.bypass.fill_(1.5)
    encoder.stacks = torch.nn.ModuleList([_Checkpointed(s) for s in encoder.stacks])
    project_constraints(encoder)
    gates = [float(m.bypass.detach()) for m in encoder.modules() if isinstance(m, ZipformerStack)]
    assert len(gates) == len(get_config().model.encoder_dims)
    assert all(g == 1.0 for g in gates)


def test_project_constraints_reaches_a_biasnorm_outside_the_encoder():
    # It is called on the whole model, not on the encoder, precisely so it covers the predictor's
    # BiasNorm too -- an unbounded gain there feeds the joiner exactly as one in a stack feeds the
    # next stack.
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = BiasNorm(8)

    model = _Model()
    with torch.no_grad():
        model.norm.log_scale.fill_(9.0)
    project_constraints(model)
    assert float(model.norm.log_scale.detach()) == pytest.approx(
        get_config().model.biasnorm_log_scale_max
    )


def test_bypass_gradient_grows_with_the_branch_gain():
    # Why train/grad_norm explodes while the weight matrices stay healthy. A stack mixes as
    # `residual + bypass * (x - residual)` where `x`'s RMS is exp(log_scale) of the last block's
    # norm_out, so d(loss)/d(bypass) = <dL/dout, x - residual> carries that gain -- and it is a
    # reduction over a whole [B, T, C] activation, so it dwarfs every matrix's gradient. That is
    # the 2026-08-09 climb: train/grad_norm 1.95 -> 201 with train/grad_norm_guarded 0.82 -> 1.60.
    #
    # Measured on the upward branch only: |dL/db| has a minimum near gain 1, where the processed
    # branch and the residual have the same magnitude and their difference nearly cancels. Only
    # the growing side is the runaway.
    torch.manual_seed(3)
    grads = []
    for log_scale in (0.0, 0.5, 1.0):
        stack = _stack_with_bypass(0.6)
        with torch.no_grad():
            stack.blocks[-1].norm_out.log_scale.fill_(log_scale)
        grads.append(abs(float(_bypass_grad(stack))))
    assert grads[1] > 2 * grads[0] and grads[2] > 2 * grads[1]


def test_a_stack_emitting_a_large_amplitude_silences_the_next_one():
    # Why the gain is bounded at all, stated as behaviour rather than as a parameter value. Every
    # branch inside a block is PRE-normed, so it contributes an O(1) correction regardless of how
    # large the residual it is added to is: the work a block can do falls off as 1/RMS^2 of its
    # input. MEASURED on a fresh block -- 1 - cos(in, out) is 0.060 at RMS 1 and 0.0004 at RMS 12,
    # which is the arithmetic that silenced stacks 2..5 when stack 1 ran to exp(2.5) = 12.18.
    torch.manual_seed(0)
    block = ZipformerBlock(48, num_heads=4)
    pad = torch.zeros(2, 16, dtype=torch.bool)
    x = torch.randn(2, 16, 48)
    work = []
    for amplitude in (1.0, math.e, 12.18):
        with torch.no_grad():
            out, _ = block(x * amplitude, pad)
        cos = torch.nn.functional.cosine_similarity(x, out, dim=-1).mean()
        work.append(1.0 - float(cos))
    assert work[0] > work[1] > work[2]
    # The bound has to keep the middle case reachable and the last one out of reach: e is a
    # single-digit slowdown, 12.18 is two orders of magnitude.
    assert work[0] / work[1] < 20 and work[0] / work[2] > 50


def test_biasnorm_output_rms_is_the_gain_and_the_bound_caps_it_both_ways():
    # exp(log_scale) IS the output RMS, which is why an unbounded log_scale is an unbounded
    # activation. forward clamps the value it USES -- so a checkpoint written under a wider bound
    # already computes inside the current one -- and project() clamps the value it STORES, which is
    # what keeps the parameter off clamp's dead zone.
    lo = get_config().model.biasnorm_log_scale_min
    hi = get_config().model.biasnorm_log_scale_max
    norm = BiasNorm(64)
    x = torch.randn(4, 16, 64)

    with torch.no_grad():
        norm.log_scale.fill_(hi + 1.5)
    assert norm(x).pow(2).mean().sqrt().item() == pytest.approx(math.exp(hi), rel=0.05)
    norm.project()
    assert float(norm.log_scale) == pytest.approx(hi)

    with torch.no_grad():
        norm.log_scale.fill_(lo - 1.5)
    assert norm(x).pow(2).mean().sqrt().item() == pytest.approx(math.exp(lo), rel=0.05)
    norm.project()
    assert float(norm.log_scale) == pytest.approx(lo)

    interior = (lo + hi) / 2 + 0.1
    with torch.no_grad():
        norm.log_scale.fill_(interior)
    norm.project()
    assert float(norm.log_scale) == pytest.approx(interior), "inside the window must be untouched"
    assert norm(x).pow(2).mean().sqrt().item() == pytest.approx(math.exp(interior), rel=0.05)


def test_the_gain_bound_is_dead_only_outside_and_still_trains_on_it():
    # Same projected-gradient-descent argument as the bypass gate: resting ON the bound the gain
    # must still receive gradient, so it can come back down.
    hi = get_config().model.biasnorm_log_scale_max
    norm = BiasNorm(32)
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        norm.log_scale.fill_(hi + 0.5)
    norm(x).pow(2).sum().backward()
    assert float(norm.log_scale.grad) == 0.0, "past the bound the gain is frozen"

    norm.project()
    norm.log_scale.grad = None
    norm(x).pow(2).sum().backward()
    assert float(norm.log_scale.grad) != 0.0


def test_project_constraints_reaches_every_biasnorm_gain():
    # 98 of these live in the encoder and any one of them can be the runaway -- three were, on the
    # 600k run. The projection must reach all of them, including through the _Checkpointed wrapper
    # the trainer installs when grad_checkpoint is on.
    # Each against ITS OWN ceiling: the trunk normalisers are BiasNorms too and carry the wider
    # trunk window, because the whole healthy trunk ledger sits just under the branch ceiling.
    hi = get_config().model.biasnorm_log_scale_max
    encoder = ZipformerEncoder(cmvn_path=None)
    norms = [m for m in encoder.modules() if isinstance(m, BiasNorm)]
    assert len(norms) > 50
    with torch.no_grad():
        for norm in norms:
            norm.log_scale.fill_(hi + 5.0)
    encoder.stacks = torch.nn.ModuleList([_Checkpointed(s) for s in encoder.stacks])
    project_constraints(encoder)
    assert all(float(m.log_scale) == pytest.approx(m.log_scale_max) for m in norms)
    assert {m.log_scale_max for m in norms} == {hi, get_config().model.trunk_norm_log_scale_max}


def test_stack_mix_params_reads_the_two_coefficients_the_stack_actually_applies():
    # A stack returns (1 - b) * residual + (b * g) * x_hat. The pair is what the trainer logs, and
    # it has to be the LAST block's norm_out -- that is the one feeding the bypass mix. Reading any
    # other block's gain would report a healthy number through the exact divergence it exists for.
    encoder = ZipformerEncoder(cmvn_path=None)
    pairs = stack_mix_params(encoder)
    assert len(pairs) == len(get_config().model.encoder_dims)
    for stack, (bypass, log_scale) in zip(encoder.stacks, pairs):
        assert bypass is stack.bypass
        assert log_scale is stack.blocks[-1].norm_out.log_scale
        assert all(log_scale is not b.norm_out.log_scale for b in stack.blocks[:-1])


def test_rotary_pos_offset_matches_full_tail():
    cos_full, sin_full = rotary_tables(12, 12, torch.device("cpu"), torch.float32)
    cos_tail, sin_tail = rotary_tables(4, 12, torch.device("cpu"), torch.float32, pos_offset=8)
    assert torch.allclose(cos_full[8:], cos_tail, atol=1e-6)
    assert torch.allclose(sin_full[8:], sin_tail, atol=1e-6)


def _biasnorm_amplification(norm: BiasNorm, x: torch.Tensor) -> float:
    """What the module multiplies its own gain by: max over frames of RMS(x) / RMS(x - bias)."""
    with torch.no_grad():
        out = norm(x)
        per_frame = out.pow(2).mean(-1).sqrt() / float(norm.log_scale.clamp(-9, 9).exp())
    return float(per_frame.max())


def test_biasnorm_cannot_amplify_past_its_gain_by_more_than_the_cap():
    # THE BOUND ON log_scale IS NOT A BOUND ON THE OUTPUT. BiasNorm divides by RMS(x - bias) but
    # scales x, so its output RMS is exp(log_scale) * RMS(x)/RMS(x - bias) and the second factor is
    # free. MEASURED on the 2026-08-21 run: `stacks.1.blocks.1.norm_out` reached 68.7x per frame
    # with log_scale pinned at its 1.0 ceiling, emitting RMS 187 where the bound implies 2.72.
    # A frame sitting on the learned bias is all it takes.
    torch.manual_seed(0)
    norm = BiasNorm(64)
    with torch.no_grad():
        norm.bias.copy_(torch.randn(64) * 3.0)
        norm.log_scale.fill_(get_config().model.biasnorm_log_scale_max)
    on_the_bias = norm.bias.detach().expand(1, 8, 64) + 1e-3 * torch.randn(1, 8, 64)

    cap = get_config().model.biasnorm_max_amplification
    assert _biasnorm_amplification(norm, on_the_bias) <= cap * 1.01


def test_the_amplification_cap_leaves_the_healthy_regime_alone():
    # The cap is a bound, not a normalisation change. MEASURED over 158,081 frame-module pairs on
    # the healthy step43200 checkpoint: 99.99 % amplify by <2x and the single worst module reaches
    # 6.7x, so a cap of 4 moves the encoder output by 3.4e-4 relative. Anything the module does
    # away from its bias must be bit-identical.
    torch.manual_seed(1)
    norm = BiasNorm(64)
    with torch.no_grad():
        norm.bias.copy_(torch.randn(64) * 0.1)
    x = torch.randn(2, 16, 64) * 5.0  # far from the bias: amplification ~1
    inv_rms = (x - norm.bias).pow(2).mean(-1, keepdim=True).add(norm.eps).rsqrt()
    uncapped = x * (inv_rms * norm.log_scale.clamp(-9, 9).exp())
    assert torch.equal(norm(x), uncapped), "the cap must not bind in the regime the model trains in"


def test_the_cap_removes_the_gradient_that_drives_the_escape():
    # Why a cap and not just a warning: past it the denominator no longer depends on `bias`, so the
    # direction that was being ridden stops receiving gradient. The gain still trains, because the
    # capped output is still proportional to exp(log_scale) -- same projected-gradient argument as
    # the log_scale bound itself.
    torch.manual_seed(2)
    norm = BiasNorm(32)
    with torch.no_grad():
        norm.bias.copy_(torch.randn(32) * 3.0)
    x = norm.bias.detach().expand(1, 4, 32) + 1e-3 * torch.randn(1, 4, 32)
    norm(x).pow(2).sum().backward()
    assert float(norm.bias.grad.norm()) == 0.0, "past the cap the bias must stop being driven"
    assert float(norm.log_scale.grad) != 0.0, "the gain must keep training"


def _trunk_stack() -> ZipformerStack:
    # dim_in != dim, so in_proj is a real Linear rather than the Identity a same-width stack gets.
    return ZipformerStack(dim_in=32, dim=48, num_layers=1, downsample=1, num_heads=4)


def _block_grad_norm(stack: ZipformerStack) -> float:
    # `grad is None` counts as zero: at bypass 0 autograd does not merely deliver a zero gradient,
    # it never reaches the blocks at all.
    lengths = torch.tensor([12, 9])
    out = stack(torch.randn(2, 12, 32), lengths, make_pad_mask(lengths, 12))
    out.pow(2).sum().backward()
    grads = [p.grad for p in stack.blocks.parameters() if p.grad is not None]
    return math.sqrt(sum(float(g.pow(2).sum()) for g in grads))


def test_a_zero_bypass_freezes_the_whole_stack_and_the_floor_prevents_it():
    # THE 2026-08-22 FAILURE. At b = 0 the stack is `out = in_proj(input)` and every block is
    # multiplied by zero, so autograd delivers exactly no gradient to any of it -- measured on
    # transducer_step81000.pt, 0 of stack 2's 105 block parameters had a nonzero grad, freezing
    # 10.6 M of 53.8 M encoder parameters. It is an absorbing state, not merely a dead zone:
    # d(loss)/d(bypass) there was +0.91, i.e. descent pushing further into the floor.
    old = _trunk_stack()
    old.bypass_min = 0.0  # the behaviour this floor replaces
    with torch.no_grad():
        old.bypass.fill_(0.0)
    assert _block_grad_norm(old) == 0.0, "this is the bug: a stack that has removed itself"

    floored = _trunk_stack()
    with torch.no_grad():
        floored.bypass.fill_(0.0)
    assert floored.bypass_min > 0.0
    assert _block_grad_norm(floored) > 0.0, "the floor must keep every block trainable"


def test_bypass_floor_is_applied_to_the_value_used_and_the_value_stored():
    # Same two-sided contract as BiasNorm's gain: forward clamps what it USES, so a checkpoint
    # written before the floor existed (transducer_best.pt carries bypass = 0.0) computes inside
    # the bound on its first batch, and project clamps what it STORES, which keeps the parameter
    # resting ON the bound where gradient still flows.
    floor = get_config().model.stack_bypass_min
    stack = _trunk_stack()
    with torch.no_grad():
        stack.bypass.fill_(0.0)
    lengths = torch.tensor([12, 9])
    x = torch.randn(2, 12, 32)
    used = stack(x, lengths, make_pad_mask(lengths, 12))
    residual = stack.in_proj(x)
    assert not torch.allclose(used, residual), "forward must not run at the stored 0.0"

    stack.project()
    assert float(stack.bypass.detach()) == pytest.approx(floor)
    stack.bypass.grad = None
    assert float(_bypass_grad(stack)) != 0.0, "on the floor, the gate must still train"


def test_project_bounds_the_in_proj_spectral_norm():
    # in_proj is the encoder's trunk: the only operator between two stacks, and the only one no
    # BiasNorm sits on. Muon's Newton-Schulz gives a direction the loss cannot feel the same size
    # step as one it can, so it inflates without plateauing.
    #
    # This used to also assert "a projection rescales, it does not rotate", which was the bug: a
    # uniform rescale is not the projection onto {||W||_2 <= c} and preserving the matrix's
    # direction means trimming the directions that were never over the bound. See
    # test_the_trunk_projection_clips_the_top_singular_value_and_nothing_else.
    limit = get_config().model.stack_in_proj_max_sigma
    stack = _trunk_stack()
    weight = stack.in_proj.weight
    sigma = lambda: float(torch.linalg.matrix_norm(weight.detach(), ord=2))  # noqa: E731

    with torch.no_grad():
        weight.mul_(10.0 * limit / sigma())
    assert sigma() == pytest.approx(10.0 * limit)

    stack.project()
    assert sigma() == pytest.approx(limit, rel=1e-4)


def test_the_bound_is_spectral_because_the_isotropic_one_could_not_see_the_inflation():
    # THE reason this bound was rewritten. ||W||_F/sqrt(n_out) is the gain against ISOTROPIC input;
    # the inflation runs along a few directions instead, so an anisotropic matrix can multiply its
    # own top direction arbitrarily while the isotropic reading barely moves. MEASURED at step
    # 43,200 of the 600k run: isotropic 2.76 at stack 3 against a realized 6.5x on dev audio, and
    # a bound of 4.0 on that reading clipped nothing in 43k steps.
    stack = _trunk_stack()
    weight = stack.in_proj.weight
    n_out = weight.shape[0]
    with torch.no_grad():
        u, s_vals, vh = torch.linalg.svd(weight, full_matrices=False)
        s_vals[0] *= 6.0  # one direction only
        weight.copy_((u * s_vals) @ vh)

    isotropic = float(weight.detach().norm()) / math.sqrt(n_out)
    spectral = float(torch.linalg.matrix_norm(weight.detach(), ord=2))
    assert spectral > 4.0 * isotropic, "an inflated direction hides inside the Frobenius norm"

    x = torch.randn(2, 12, 32)
    aligned = x @ vh[0].outer(vh[0])  # input lying in the inflated direction
    realized = float((aligned @ weight.detach().t()).norm() / aligned.norm())
    assert realized > isotropic, "the isotropic reading is not an upper bound on realized gain"
    assert realized <= spectral + 1e-4, "the spectral one is"


def test_trunk_sigma_power_iteration_tracks_the_true_spectral_norm():
    # `project` runs every optimizer step, so it estimates sigma by two power iterations warm
    # started from the previous step rather than by an SVD. That is only sound if the estimate
    # stays tight while the weight drifts the way training drifts it.
    stack = _trunk_stack()
    weight = stack.in_proj.weight
    first = float(stack.trunk_sigma())
    true = float(torch.linalg.matrix_norm(weight.detach(), ord=2))
    assert first == pytest.approx(true, rel=1e-4), "the cold iteration must converge from random"
    assert first <= true, "power iteration approaches sigma from below, never above"

    torch.manual_seed(0)
    for _ in range(20):
        with torch.no_grad():
            # A Newton-Schulz step is spectrally flat, so perturb every direction equally at the
            # ~0.4 %/step the measured run moved its trunk by.
            weight.add_(
                torch.linalg.svd(torch.randn_like(weight), full_matrices=False)[0]
                @ torch.linalg.svd(torch.randn_like(weight), full_matrices=False)[2],
                alpha=0.004 * float(weight.norm()) / math.sqrt(min(weight.shape)),
            )
        estimate = float(stack.trunk_sigma())
        true = float(torch.linalg.matrix_norm(weight.detach(), ord=2))
        assert estimate == pytest.approx(true, rel=2e-3), "warm iteration must stay tight"


def test_forward_records_the_realized_trunk_amplitude():
    # The quantity every bound in ZipformerStack is a proxy for, and the one no metric read
    # through four collapses: at step 43,200 the trunk into stack 3 was at RMS 43.19 against 1.51
    # for the model that shipped 3.43 %, while trunk_gain_max read 3.24 out of a budget of 4.0.
    # POST trunk_norm, i.e. what the residual stream actually carries -- which is why a x7 on
    # in_proj no longer shows up here at all. That is the whole point of the normaliser: the
    # quantity that failed six times is now bounded rather than merely reported.
    stack = _trunk_stack()
    lengths = torch.tensor([12, 9])
    x = torch.randn(2, 12, 32)
    assert stack.trunk_rms is None, "nothing to report before a forward"
    stack(x, lengths, make_pad_mask(lengths, 12))
    expected = stack.trunk_norm(stack.in_proj(x)).detach().pow(2).mean().sqrt()
    assert float(stack.trunk_rms) == pytest.approx(float(expected), rel=1e-5)

    with torch.no_grad():
        stack.in_proj.weight.mul_(7.0)
        stack.in_proj.bias.mul_(7.0)
    stack(x, lengths, make_pad_mask(lengths, 12))
    assert float(stack.trunk_rms) == pytest.approx(float(expected), rel=1e-2), "held by the norm"


def test_project_leaves_an_in_range_in_proj_untouched():
    # The bound is calibrated to be inert on the model that works (release/stream-asr-v1.0 peaks
    # at sigma 9.73 against a bound of 10.0), so it must not touch anything inside it.
    stack = _trunk_stack()
    expected = stack.in_proj.weight.detach().clone()
    stack.project()
    assert stack.in_proj.weight.detach().equal(expected)


def test_project_constraints_bounds_every_trunk_gain_through_the_checkpoint_wrapper():
    # Same failure mode as the gate's: with grad_checkpoint on the trainer wraps each stack, so a
    # projection that walked self.stacks would silently stop applying at the one setting that
    # needs it. Stack 0 has dim_in == dim and therefore an Identity in_proj, which must be skipped
    # rather than crash.
    limit = get_config().model.stack_in_proj_max_sigma
    encoder = ZipformerEncoder(cmvn_path=None)
    linear = [s for s in encoder.stacks if isinstance(s.in_proj, torch.nn.Linear)]
    assert len(linear) == len(encoder.stacks) - 1, "stack 0 is same-width, so it has no in_proj"
    with torch.no_grad():
        for stack in linear:
            stack.in_proj.weight.mul_(50.0)
    encoder.stacks = torch.nn.ModuleList([_Checkpointed(s) for s in encoder.stacks])

    project_constraints(encoder)

    sigmas = [
        float(torch.linalg.matrix_norm(s.in_proj.weight.detach(), ord=2))
        for s in encoder.modules()
        if isinstance(s, ZipformerStack) and isinstance(s.in_proj, torch.nn.Linear)
    ]
    assert len(sigmas) == len(linear)
    assert all(sig == pytest.approx(limit, rel=1e-4) for sig in sigmas)
    assert float(trunk_gain_max(encoder)) == pytest.approx(limit, rel=1e-4)


def test_trunk_gain_max_sees_what_the_stack_mix_scalars_cannot():
    # The blind spot that hid the 2026-08-22 collapse. `stack_mix/*` reports (1 - b) and b * g,
    # both of which describe the PROCESSED half; the residual half is (1 - b) * in_proj(input) and
    # neither scalar depends on in_proj at all. So inflating the trunk must move trunk_gain_max and
    # leave the mix scalars bit-identical -- which is exactly how the run read healthy while its
    # trunk went to RMS 570.
    encoder = ZipformerEncoder(cmvn_path=None)
    before_mix = [(float(b), float(g)) for b, g in stack_mix_params(encoder)]
    before_trunk = float(trunk_gain_max(encoder))

    with torch.no_grad():
        encoder.stacks[2].in_proj.weight.mul_(20.0)

    assert [(float(b), float(g)) for b, g in stack_mix_params(encoder)] == before_mix
    assert float(trunk_gain_max(encoder)) > 10.0 * before_trunk


def _trained_shape_stack(sigma_top: float, decay: float = 0.78) -> ZipformerStack:
    # 384 -> 512 is stacks.3, the one the 2026-08-24 collapse ran through. Its spectrum is shaped
    # like a TRAINED in_proj rather than a fresh one: sigma_2/sigma_1 ran 0.72-0.79 across the run,
    # against 0.96-0.99 at init, and that gap is what makes the top value the only one over the
    # bound.
    stack = ZipformerStack(dim_in=384, dim=512, num_layers=1, downsample=1, num_heads=8)
    weight = cast(torch.nn.Linear, stack.in_proj).weight
    with torch.no_grad():
        u, s_vals, vh = torch.linalg.svd(weight, full_matrices=False)
        shaped = sigma_top * decay ** torch.arange(s_vals.numel(), dtype=s_vals.dtype)
        weight.copy_(u @ torch.diag(shaped) @ vh)
    return stack


def test_the_trunk_projection_clips_the_top_singular_value_and_nothing_else():
    # THE 2026-08-24 BUG. The projection used to be `W *= bound / sigma_1`, which trims every
    # direction by the same factor. Harmless while it fires rarely; the run made it fire on 99.2 %
    # of steps from 71k on, and stacks.3.in_proj's stable rank went 64.4 -> 21.7 with sigma_1
    # reading exactly 10.00 at every checkpoint. Only the singular values ABOVE the bound may move.
    bound = _MODEL.stack_in_proj_max_sigma
    stack = _trained_shape_stack(bound * 1.02)
    weight = cast(torch.nn.Linear, stack.in_proj).weight
    before = torch.linalg.svdvals(weight.detach().clone())
    assert before[1] < bound, "only the top value is over the bound, so deflation is exact here"

    stack.project()
    after = torch.linalg.svdvals(weight.detach())

    assert after[0] == pytest.approx(bound, rel=1e-4), "the top value must land on the bound"
    # Everything under the bound is untouched, which is what the uniform rescale destroyed: it
    # would have multiplied all of these by 1/1.02 on every one of 26,000 consecutive steps.
    assert torch.allclose(after[1:], before[1:], rtol=1e-4, atol=1e-5)


def test_the_trunk_projection_clips_a_flat_spectrum_exactly_too():
    # A weight arriving from outside the training loop can have many values over the bound at once,
    # and a fresh matrix is the flat case (sigma_1/sigma_2 = 1.007-1.042). Deflating the top one
    # there would just promote the second, so this routes to the exact clip.
    bound = _MODEL.stack_in_proj_max_sigma
    stack = ZipformerStack(dim_in=384, dim=512, num_layers=1, downsample=1, num_heads=8)
    weight = cast(torch.nn.Linear, stack.in_proj).weight
    with torch.no_grad():
        weight.mul_(50.0 * bound / torch.linalg.matrix_norm(weight, 2))
    before = torch.linalg.svdvals(weight.detach().clone())

    stack.project()
    after = torch.linalg.svdvals(weight.detach())

    assert after[0] == pytest.approx(bound, rel=1e-4)
    assert torch.all(after <= bound + 1e-3), "min(sigma_i, bound), not a rescale"
    kept = before <= bound
    assert torch.allclose(after[kept], before[kept], rtol=1e-4, atol=1e-4), "under the bound: kept"


def test_the_trunk_projection_does_not_collapse_the_spectrum_when_it_binds_every_step():
    # The regression proper: the failure was not one bad projection, it was 26,000 of them against
    # a gradient that re-inflates only the top direction. Under the rescale this loop drives stable
    # rank to ~1; the deflation has to leave it alone.
    bound = _MODEL.stack_in_proj_max_sigma
    stack = _trained_shape_stack(bound * 0.98)
    weight = cast(torch.nn.Linear, stack.in_proj).weight
    torch.manual_seed(0)
    u = torch.nn.functional.normalize(torch.randn(512), dim=0)
    v = torch.nn.functional.normalize(torch.randn(384), dim=0)
    start = stack.trunk_stable_rank().item()

    for _ in range(300):
        with torch.no_grad():
            weight.add_(torch.outer(u, v), alpha=0.05)  # Muon-like push along one direction
        stack.project()

    assert stack.trunk_sigma().item() == pytest.approx(bound, rel=1e-3), "bound still enforced"
    end = stack.trunk_stable_rank().item()
    assert end > 0.9 * start, f"spectrum collapsed under repeated projection: {start} -> {end}"


def test_trunk_stable_rank_min_reads_the_flattest_stack():
    # The metric that was missing: sigma_1 is constant by construction once the bound binds, so
    # nothing the trainer logged could see a matrix being squeezed toward rank 1 underneath it.
    encoder = ZipformerEncoder(cmvn_path=None)
    stacks = [s for s in encoder.modules() if isinstance(s, ZipformerStack)]
    flat = next(s for s in stacks if isinstance(s.in_proj, torch.nn.Linear))
    weight = cast(torch.nn.Linear, flat.in_proj).weight
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        s[1:] *= 0.01  # everything but the top direction all but removed
        weight.copy_(u @ torch.diag(s) @ vh)

    assert trunk_stable_rank_min(encoder).item() == pytest.approx(
        flat.trunk_stable_rank().item(), rel=1e-4
    )
    assert flat.trunk_stable_rank().item() < 1.1, "a rank-1 matrix has stable rank 1"


def test_trunk_sigma_returns_a_matched_pair_of_singular_vectors():
    # `project` deflates by `sigma * u v^T`, so a `u` carried out of the loop one half-step stale
    # would overshoot the bound by that staleness instead of landing on it.
    stack = _trained_shape_stack(6.0)
    weight = cast(torch.nn.Linear, stack.in_proj).weight
    sigma = stack.trunk_sigma()
    assert torch.allclose(weight @ stack.sigma_v, sigma * stack.sigma_u, rtol=1e-4, atol=1e-5)
    assert sigma.item() == pytest.approx(torch.linalg.matrix_norm(weight, 2).item(), rel=1e-4)


def test_trunk_norm_bounds_the_residual_stream_the_bounds_could_not():
    # THE point of the trunk norm. Six parameter bounds each moved the inflation into whatever
    # summary of `in_proj` was left free; this one bounds the ACTIVATION. Feed a stack an input
    # loud enough that any parameter bound would still let it through, and the residual the blocks
    # and the next stack see must come out inside the ceiling regardless.
    stack = ZipformerStack(dim_in=384, dim=512, num_layers=1, downsample=1, num_heads=8)
    assert isinstance(stack.trunk_norm, BiasNorm), "config has model.trunk_norm off"
    ceiling = math.exp(_MODEL.trunk_norm_log_scale_max) * _MODEL.biasnorm_max_amplification
    lengths = torch.tensor([12, 9])

    for scale in (1.0, 50.0, 5000.0):
        stack(torch.randn(2, 12, 384) * scale, lengths, make_pad_mask(lengths, 12))
        assert float(stack.trunk_rms) <= ceiling, f"trunk escaped at input scale {scale}"


def test_trunk_norm_is_skipped_where_in_proj_is_an_identity():
    # Stack 0 is same-width, so there is no inter-stack operator to normalise and no preceding
    # stack to compound from. `stack_mix/0_trunk` stays the frontend's output amplitude.
    same_width = ZipformerStack(dim_in=32, dim=32, num_layers=1, downsample=1, num_heads=4)
    assert isinstance(same_width.in_proj, torch.nn.Identity)
    assert isinstance(same_width.trunk_norm, torch.nn.Identity)


def test_trunk_norm_gets_its_own_window_and_is_not_counted_as_a_branch_gain():
    # The trunk ceiling sits ABOVE the branch ceiling (the whole healthy trunk ledger, 1.51-2.69,
    # sits just under exp(1.0)), so counting the trunk norms in the branch population would report
    # all five as permanently pinned and make `gains_at_ceiling` useless.
    encoder = ZipformerEncoder(cmvn_path=None)
    trunk = [
        m
        for n, m in encoder.named_modules()
        if n.endswith("trunk_norm") and isinstance(m, BiasNorm)
    ]
    assert len(trunk) == len(encoder.stacks) - 1, "one per stack with a Linear in_proj"
    assert all(t.log_scale_max == _MODEL.trunk_norm_log_scale_max for t in trunk)

    gains = branch_gain_params(encoder)
    assert len(gains) == sum(1 for m in encoder.modules() if isinstance(m, BiasNorm)) - len(trunk)
    assert all(all(g is not t.log_scale for t in trunk) for g in gains)


def test_project_constraints_holds_the_trunk_norms_in_their_own_window():
    # Same projected-gradient argument as every other bound here, against the OTHER ceiling.
    encoder = ZipformerEncoder(cmvn_path=None)
    trunk = [
        m
        for n, m in encoder.named_modules()
        if n.endswith("trunk_norm") and isinstance(m, BiasNorm)
    ]
    with torch.no_grad():
        for t in trunk:
            t.log_scale.fill_(9.0)

    project_constraints(encoder)

    ceiling = _MODEL.trunk_norm_log_scale_max
    assert all(float(t.log_scale) == pytest.approx(ceiling) for t in trunk)
    # and a branch gain is still held at the tighter branch ceiling by the same call
    branch = next(
        m for n, m in encoder.named_modules() if isinstance(m, BiasNorm) and "trunk_norm" not in n
    )
    with torch.no_grad():
        branch.log_scale.fill_(9.0)
    project_constraints(encoder)
    assert float(branch.log_scale) == pytest.approx(_MODEL.biasnorm_log_scale_max)
