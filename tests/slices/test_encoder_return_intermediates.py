import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder


def _enc():
    torch.manual_seed(0)
    return ZipformerEncoder(cmvn_path=None).eval()


def test_default_forward_is_byte_identical():
    enc = _enc()
    n_mels = get_config().audio.n_mels
    x = torch.randn(2, 120, n_mels)
    lengths = torch.tensor([120, 96])
    with torch.no_grad():
        a = enc(x, lengths)
        b = enc(x, lengths, return_intermediates=None)
    assert len(a) == 2 and len(b) == 2
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_return_intermediates_shapes():
    enc = _enc()
    dims = get_config().model.encoder_dims
    n_mels = get_config().audio.n_mels
    x = torch.randn(2, 120, n_mels)
    lengths = torch.tensor([120, 96])
    with torch.no_grad():
        memory, out_len, inters, base_len = enc(x, lengths, return_intermediates=[3, 4])
    assert len(inters) == 2
    assert inters[0].shape[0] == 2 and inters[0].shape[2] == dims[3]
    assert inters[1].shape[2] == dims[4]
    # intermediates are at base rate: same time length as each other, >= out_len (pre-downsample)
    assert inters[0].shape[1] == inters[1].shape[1]
    assert int(base_len.max()) >= int(out_len.max())
    # memory unchanged vs default path
    assert torch.equal(memory, enc(x, lengths)[0])
