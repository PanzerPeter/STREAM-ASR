import torch

from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


def _batch():
    torch.manual_seed(0)
    n_mels = 80
    return collate_features(
        [(torch.randn(160, n_mels), [3, 4, 5, 6, 7]), (torch.randn(120, n_mels), [8, 9, 10])]
    )


def test_forward_shapes():
    model = TransducerModel(cmvn_path=None).eval()
    b = _batch()
    memory, out_len, ctc_logits, interctc, base_len = model(b.features, b.feature_lengths)
    assert ctc_logits.shape[0] == 2 and ctc_logits.shape[-1] == 501
    assert len(interctc) == len(model.interctc_heads)
    assert memory.shape[1] == ctc_logits.shape[1]


def test_joint_loss_finite_with_finite_grads():
    model = TransducerModel(cmvn_path=None).train()
    b = _batch()
    total, rnnt, ctc, ictc = model.joint_loss(b, chunk_size=0)
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_rnnt_loss_is_per_token_mean():
    # reduction="sum" divided by total tokens puts the RNN-T term on the same O(1) per-token scale
    # as F.ctc_loss("mean"), so the ctc/interctc aux weights are not silently ~1/avg_tokens weaker
    # than nominal. Locks the SP5 loss-normalisation fix (old code used reduction="mean").
    model = TransducerModel(cmvn_path=None).eval()
    assert model._rnnt.reduction == "sum"
    b = _batch()
    memory, out_len, *_ = model(b.features, b.feature_lengths)
    single = model.rnnt_loss(memory, out_len, b.tokens, b.token_lengths)
    dup = model.rnnt_loss(
        torch.cat([memory, memory]),
        torch.cat([out_len, out_len]),
        torch.cat([b.tokens, b.tokens]),
        torch.cat([b.token_lengths, b.token_lengths]),
    )
    assert torch.allclose(single, dup, atol=1e-4)  # a proper mean, not a batch-growing raw sum


def test_joint_loss_under_bf16_autocast():
    # CPU bf16 autocast reproduces the GPU training codepath that feeds the RNN-T loss: the
    # joiner's Linear layers emit bf16 logits, which torchaudio's RNNTLoss kernel rejects unless
    # TransducerModel.rnnt_loss casts them back to fp32 first. Before that cast this test raised
    # `RuntimeError: logits must be float32 or float16`.
    model = TransducerModel(cmvn_path=None).train()
    b = _batch()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        total, rnnt, ctc, ictc = model.joint_loss(b, chunk_size=0)
    assert torch.isfinite(total)
    total.backward()  # grads must flow through the fp32 cast
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
