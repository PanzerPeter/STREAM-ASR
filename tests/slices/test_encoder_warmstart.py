import pytest
import torch

from src.shared_kernel.BiasNorm import BiasNorm
from src.shared_kernel.Checkpoint_Adapter import save_checkpoint
from src.slices.PretrainEncoder.BestRqModel import BestRqModel
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import _warm_start_encoder


class _Log:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def test_warmstart_copies_encoder_weights(tmp_path):
    pre = BestRqModel(cmvn_path=None)
    path = str(tmp_path / "bestrq_encoder.pt")
    save_checkpoint(path, pre.encoder, [], step=0, kind="bestrq")

    model = TransducerModel(cmvn_path=None)
    before = model.encoder.frontend.linear.weight.clone()
    _warm_start_encoder(model, path, _Log())
    after = model.encoder.frontend.linear.weight
    assert torch.allclose(after, pre.encoder.frontend.linear.weight)
    assert not torch.allclose(after, before)  # weights actually changed


def test_warm_start_accepts_a_pretrain_that_predates_the_trunk_norms(tmp_path):
    # bestrq_encoder.pt was written before `model.trunk_norm` existed, so it cannot supply those
    # 10 parameters. That is the ONE absence a warm start tolerates -- and the norms must start at
    # gain 1.0 rather than inherit the pretrain's amplitude, which is the quantity CLAUDE.md
    # pitfall 15 measures at trunk RMS 5.7-17.0 against 1.5-2.7 for the model that works.
    model = TransducerModel(cmvn_path=None)
    sd = {k: v for k, v in model.encoder.state_dict().items() if ".trunk_norm." not in k}
    path = str(tmp_path / "pretrain.pt")
    torch.save({"model": sd}, path)

    _warm_start_encoder(model, path, _Log())

    live = [
        m
        for n, m in model.encoder.named_modules()
        if n.endswith("trunk_norm") and isinstance(m, BiasNorm)
    ]
    assert live, "config has model.trunk_norm off"
    assert all(float(t.log_scale) == 0.0 for t in live), "gain 1.0, not the pretrain's amplitude"


def test_warm_start_still_rejects_a_genuine_mismatch(tmp_path):
    # Anything OTHER than the trunk norms absent is a real disagreement between the checkpoint and
    # this encoder, which is what the strict load was there to catch.
    model = TransducerModel(cmvn_path=None)
    sd = dict(model.encoder.state_dict())
    del sd[next(k for k in sd if k.startswith("stacks.1.blocks."))]
    path = str(tmp_path / "wrong.pt")
    torch.save({"model": sd}, path)

    with pytest.raises(RuntimeError, match="warm start mismatch"):
        _warm_start_encoder(model, path, _Log())
