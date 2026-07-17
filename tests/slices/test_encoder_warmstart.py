import torch

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
