import torch

from src.shared_kernel.Checkpoint_Adapter import save_checkpoint
from src.slices.PretrainEncoder.BestRqModel import BestRqModel
from src.slices.TrainAcousticModel.AcousticModel import AcousticModel
from src.slices.TrainAcousticModel.StageATrainer_Handler import load_pretrained_encoder


def test_warmstart_copies_encoder_weights(tmp_path):
    pre = BestRqModel(cmvn_path=None)
    path = str(tmp_path / "bestrq_encoder.pt")
    save_checkpoint(path, pre.encoder, [], step=0, kind="bestrq")

    model = AcousticModel(cmvn_path=None)
    before = model.encoder.frontend.linear.weight.clone()
    load_pretrained_encoder(model, path)
    after = model.encoder.frontend.linear.weight
    assert torch.allclose(after, pre.encoder.frontend.linear.weight)
    assert not torch.allclose(after, before)  # weights actually changed
