import torch

from src.slices.TrainAcousticModel.FusedBranchBlock import FusedBranchBlock, rms_norm
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


def test_gates_are_floored_by_parametrisation():
    # beta >= 0.05 whatever b does: a gate that can reach 0 is an absorbing state.
    block = FusedBranchBlock(384)
    with torch.no_grad():
        block.gate.fill_(-100.0)
        block.gate_attn.fill_(-100.0)
    for attn in (False, True):
        assert torch.allclose(block.beta(attn), torch.full((384,), 0.05))
    assert abs(float(FusedBranchBlock(384).beta()[0]) - (0.05 + 0.95 * 0.2689)) < 1e-3


def _steps(block: FusedBranchBlock, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Runs the block and returns (x after the attention step, block output).
    seen: list[torch.Tensor] = []
    project = block._project

    def spy(h: torch.Tensor, names: str) -> list[torch.Tensor]:
        if names == "uz":
            seen.append(h)
        return project(h, names)

    block._project = spy  # type: ignore[method-assign]
    with torch.no_grad():
        out = block(x, None)
    return seen[0], out


def test_peri_ln_steps_have_rms_beta_whatever_w_out_is():
    # §5: each residual step adds exactly beta * rms_norm(y); W_out's scale cannot reach the trunk.
    torch.manual_seed(0)
    block = FusedBranchBlock(384).eval()
    x = rms_norm(torch.randn(2, 16, 384))
    with torch.no_grad():
        for w in block.w_out.values():
            w.mul_(1000.0)
    mid, out = _steps(block, x)
    for delta, beta in ((mid - x, block.beta(attn=True)), (out - mid, block.beta())):
        per_frame = (delta / beta).pow(2).mean(-1).sqrt()
        assert torch.allclose(per_frame, torch.ones_like(per_frame), atol=1e-3)


def test_every_stack_entry_and_the_memory_are_unit_rms():
    # Invariants 1-2 measured: a 1000x frontend weight changes nothing about the trunk's scale.
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    with torch.no_grad():
        model.encoder.frontend.linear.weight.mul_(1000.0)
        memory, _ = model.encoder(torch.randn(1, 160, 80), torch.tensor([160]))
    rms = memory.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
    assert all(float(r) < 1 + 5 for r in model.encoder.last_exit_rms)  # <= 1 + sum of step betas


def test_memory_does_not_depend_on_the_batch_it_is_in():
    # Batch padding is the CMVN mean and the input is padded to the total stride, so an
    # utterance's memory is the same alone and next to a longer one.
    torch.manual_seed(1)
    model = TransducerModel(cmvn_path=None).eval()
    short, long = torch.randn(1, 203, 80), torch.randn(1, 480, 80)
    batch = torch.cat([torch.nn.functional.pad(short, (0, 0, 0, 277)), long])
    with torch.no_grad():
        alone, n = model.encoder(short, torch.tensor([203]))
        together, _ = model.encoder(batch, torch.tensor([203, 480]))
    k = int(n[0])
    assert torch.allclose(alone[0, :k], together[0, :k], atol=1e-4)
