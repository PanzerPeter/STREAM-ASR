# The parameter- and activation-level health scalars of MODEL_ARCHITECTURE.md §10.
from collections.abc import Sequence

import torch

from src.slices.TrainAcousticModel.FusedBranchBlock import FusedBranchBlock
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel

# A gate pre-activation below this is sigmoid(b) < 0.01: beta within 1 % of its range above the
# floor, i.e. a block that has switched itself off (§4.3).
_GATE_OFF = -4.6


def health_scalars(model: TransducerModel, muon_params: Sequence[torch.Tensor]) -> dict:
    # Everything §10 reads off the parameters and the last forward, as 0-d device tensors so the
    # caller syncs the whole log line once.
    with torch.no_grad():
        blocks = [m for m in model.modules() if isinstance(m, FusedBranchBlock)]
        # Both residual steps' gates: the attention step's and the MLP step's (none under prenorm).
        gates: list[torch.Tensor] = [
            g for b in blocks if hasattr(b, "gate") for g in (b.gate_attn, b.gate)
        ]
        out = {
            "weight_norm_max": torch.stack(torch._foreach_norm(list(muon_params))).max(),
            "qk_gain_max": torch.stack([(b.q_gain * b.k_gain).abs().max() for b in blocks]).max(),
        }
        if gates:
            flat = torch.cat(gates)
            out["res_gate_min"] = (blocks[0].floor + (1 - blocks[0].floor) * flat.sigmoid()).min()
            out["res_gate_floor_count"] = (flat < _GATE_OFF).sum().float()
        enc = model.encoder
        for i, rms in enumerate(enc.last_exit_rms):
            out[f"stack_rms/{i}_exit"] = rms
        if enc.last_skip_ratio is not None:
            out["stack_rms/3_skip_ratio"] = enc.last_skip_ratio
    return out
