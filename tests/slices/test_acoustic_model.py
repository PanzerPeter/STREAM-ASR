import torch
from src.slices.TrainAcousticModel.AcousticModel import AcousticModel
from src.shared_kernel.Config_Adapter import get_config


def test_logits_shape_and_loss_backward():
    model = get_config().model
    n_mels = get_config().audio.n_mels
    acoustic = AcousticModel(cmvn_path=None)
    features = torch.randn(2, 200, n_mels)
    lengths = torch.tensor([200, 150])
    tokens = torch.randint(0, model.vocab_size, (2, 8))
    token_lengths = torch.tensor([8, 6])

    logits, out_lengths = acoustic(features, lengths)
    assert logits.shape[0] == 2 and logits.shape[2] == model.logits_width
    assert (out_lengths >= token_lengths).all(), "encoder must emit >= target length for CTC"

    loss = acoustic.ctc_loss(logits, out_lengths, tokens, token_lengths)
    assert torch.isfinite(loss)
    loss.backward()
