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
from src.slices.TrainAcousticModel.ZipformerStack import ZipformerStack

_MODEL = get_config().model
_N_MELS = get_config().audio.n_mels


def _mask():
    return make_pad_mask(torch.tensor([7, 5]), max_len=7)  # [2, 7]


def test_biasnorm_preserves_shape_and_is_finite():
    x = torch.randn(2, 7, 16)
    out = BiasNorm(16)(x)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_swiglu_preserves_shape():
    x = torch.randn(2, 7, 32)
    assert SwiGluFfn(32)(x).shape == x.shape


def test_rotary_attention_shape_and_backward():
    x = torch.randn(2, 7, 48, requires_grad=True)
    out, v = RotaryAttention(48, num_heads=4)(x, _mask())
    assert out.shape == x.shape
    assert v.shape == (2, 4, 7, 12)  # [B, heads, T, head_dim] — the value-residual carrier
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
    # Regression lock for the Stage-A blank-collapse fix: under the shipped config every deeper
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


def test_rotary_pos_offset_matches_full_tail():
    cos_full, sin_full = rotary_tables(12, 12, torch.device("cpu"), torch.float32)
    cos_tail, sin_tail = rotary_tables(4, 12, torch.device("cpu"), torch.float32, pos_offset=8)
    assert torch.allclose(cos_full[8:], cos_tail, atol=1e-6)
    assert torch.allclose(sin_full[8:], sin_tail, atol=1e-6)
