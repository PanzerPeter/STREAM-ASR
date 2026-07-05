import torch
from src.slices.ComputeCmvn.ComputeCmvn_Command import ComputeCmvnCommand
from src.slices.ComputeCmvn.ComputeCmvn_Handler import compute_cmvn
from src.shared_kernel.Config_Adapter import get_config


def test_cmvn_shapes_and_persist(tmp_path):
    n_mels = get_config().audio.n_mels
    out = tmp_path / "cmvn.pt"
    stats = compute_cmvn(
        ComputeCmvnCommand(manifest="data/manifests/dev.jsonl", cmvn_out=str(out), max_utts=20)
    )

    assert stats["mean"].shape == (n_mels,)
    assert stats["std"].shape == (n_mels,)
    assert torch.isfinite(stats["mean"]).all()
    assert (stats["std"] > 0).all()

    reloaded = torch.load(out)
    assert torch.allclose(reloaded["mean"], stats["mean"])
