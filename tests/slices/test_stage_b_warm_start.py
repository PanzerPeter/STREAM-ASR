import torch

from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention
from src.slices.TrainAcousticModel.StageBTrainer_Handler import _warm_start


class _Log:
    # Minimal stand-in for the loguru logger _warm_start expects; records nothing.
    def info(self, *_a, **_k) -> None: ...

    def warning(self, *_a, **_k) -> None: ...


def _old_format_state_dict(model: HybridCtcAttention) -> dict:
    """Rewrite a current (causal) encoder state_dict into the pre-Plan-3b Stage-A layout: the
    frontend convs were an nn.Sequential (conv.0 / conv.2) and ConvModule's norm was a LayerNorm
    (per-channel `weight`) rather than BiasNorm (scalar `log_scale`). Values are irrelevant for the
    renamed keys -- they are dropped/re-init on load -- but shapes must match the old modules."""
    enc = model.encoder.state_dict()
    old: dict[str, torch.Tensor] = {}
    for k, v in enc.items():
        if k.startswith("frontend.conv1."):
            old["frontend.conv.0." + k[len("frontend.conv1.") :]] = v
        elif k.startswith("frontend.conv2."):
            old["frontend.conv.2." + k[len("frontend.conv2.") :]] = v
        elif k.endswith(".conv.norm.log_scale"):
            dim = enc[k[: -len("log_scale")] + "bias"].shape[0]  # sibling bias carries the channels
            old[k[: -len("log_scale")] + "weight"] = torch.ones(dim)
        else:
            old[k] = v
    sd = {"encoder." + k: v for k, v in old.items()}
    sd.update({"ctc_head." + k: v for k, v in model.ctc_head.state_dict().items()})
    return {"model": sd}


def test_warm_start_loads_pre_plan3b_checkpoint(tmp_path):
    # Regression: Plan-3b made the encoder causal, renaming the frontend convs and the ConvModule
    # norm. Warm-start must load the old Stage-A checkpoint by name -- transferring the matched
    # weights and re-initialising exactly the changed submodules -- not crash on strict=True.
    source = HybridCtcAttention(cmvn_path=None)
    ckpt = tmp_path / "stage_a_last.pt"
    torch.save(_old_format_state_dict(source), ckpt)

    target = HybridCtcAttention(cmvn_path=None)  # independent random init
    fresh_before = target.encoder.frontend.conv1.weight.clone()

    _warm_start(target, str(ckpt), _Log())

    # A matched key (frontend.linear survives the rework) must transfer from the checkpoint.
    assert torch.equal(target.encoder.frontend.linear.weight, source.encoder.frontend.linear.weight)
    # A Phase-0 key (frontend.conv1) is absent from the old checkpoint -> left at target's own init.
    assert torch.equal(target.encoder.frontend.conv1.weight, fresh_before)


def test_warm_start_rejects_mismatch_beyond_phase0_delta(tmp_path):
    # A key mismatch outside the known Phase-0 delta means the checkpoint no longer matches this
    # encoder -- a real regression that must fail loud, not silently train from scratch.
    source = HybridCtcAttention(cmvn_path=None)
    sd = _old_format_state_dict(source)
    corrupt = next(
        k for k in sd["model"] if k.startswith("encoder.stacks.") and "conv.norm" not in k
    )
    sd["model"]["encoder.bogus_extra_param"] = sd["model"].pop(corrupt)  # rename -> miss + extra

    ckpt = tmp_path / "stage_a_last.pt"
    torch.save(sd, ckpt)

    import pytest

    with pytest.raises(RuntimeError, match="beyond the Plan-3b Phase-0 delta"):
        _warm_start(HybridCtcAttention(cmvn_path=None), str(ckpt), _Log())
