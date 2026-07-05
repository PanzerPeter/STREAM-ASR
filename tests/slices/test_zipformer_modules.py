import torch
from src.shared_kernel.MaskUtils import make_pad_mask
from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.BiasNorm import BiasNorm
from src.slices.TrainAcousticModel.SwiGluFfn import SwiGluFfn
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
    out = RotaryAttention(48, num_heads=4)(x, _mask())
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_conv_module_shape_and_ignores_padding():
    x = torch.randn(2, 7, 32)
    out = ConvModule(32, kernel=15)(x, _mask())
    assert out.shape == x.shape


def test_conv2d_subsampling_halves_time():
    x = torch.randn(2, 101, _N_MELS)
    lengths = torch.tensor([101, 60])
    y, out_len = Conv2dSubsampling()(x, lengths)
    assert y.shape[0] == 2 and y.shape[2] == _MODEL.encoder_dims[0]
    assert y.shape[1] == (101 - 1) // 2 + 1  # 51
    assert out_len.tolist() == [(101 - 1) // 2 + 1, (60 - 1) // 2 + 1]


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
    out = ZipformerBlock(64, num_heads=4)(x, _mask())
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_stack_changes_dim_preserves_time():
    x = torch.randn(2, 12, 32)
    lengths = torch.tensor([12, 9])
    base_mask = make_pad_mask(lengths, 12)
    stack = ZipformerStack(dim_in=32, dim=48, num_layers=2, downsample=4, num_heads=4)
    out = stack(x, lengths, base_mask)
    assert out.shape == (2, 12, 48)  # time preserved, channels -> 48
    out.sum().backward()
