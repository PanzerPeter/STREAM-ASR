# Builds the optimizer stack: 2D hidden weight matrices -> Muon (spectrally normalized updates),
# everything else (embeddings, biases, gates, conv kernels, output heads) -> AdamW.
import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import OptimConfig
from src.shared_kernel.Muon_Optimizer import Muon

# Output heads / readouts that stay on AdamW, matched as substrings of the dotted module name:
# `ctc_head` and `joiner.out`, the transducer's two vocab-width readouts.
_DEFAULT_HEAD_PATTERNS = ("ctc_head", "joiner.out")


def partition_params(
    model: nn.Module, head_patterns: tuple[str, ...] = _DEFAULT_HEAD_PATTERNS
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    muon: list[nn.Parameter] = []
    adamw: list[nn.Parameter] = []
    # Tied weights (STREAM-LM ties its readout to the embedding table) surface under two module
    # names; taking both would step the same tensor twice per optimizer step. First placement wins,
    # and module order puts the embedding ahead of the readout.
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        excluded = isinstance(module, nn.Embedding) or any(
            pat in module_name for pat in head_patterns
        )
        for p in module.parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            # Any 2-D parameter, not only nn.Linear.weight: a FusedBranch block holds W_in/W_out as
            # one parameter per logical slice (q, k, v, g, u, z) so Newton-Schulz orthogonalises
            # each projection on its own scale instead of the fused concatenation.
            if p.ndim == 2 and not excluded:
                muon.append(p)
            else:
                adamw.append(p)
    return muon, adamw


def _lr_groups(params: list[nn.Parameter], lr: float, weight_decay: float) -> list[dict]:
    # Weight decay only on ndim >= 2. For every vector and scalar here (biases, the floored gate
    # pre-activations b, the QK-norm gains) 0 is not a neutral shrink target but a specific setting
    # (beta = 0.53, a QK gain that flattens attention), so decay would be a standing pull toward it.
    # Bucketing by decay keeps the fused multi-tensor AdamW kernel to two groups.
    decayed = [p for p in params if p.ndim >= 2]
    plain = [p for p in params if p.ndim < 2]
    groups = [{"params": decayed, "lr": lr, "weight_decay": weight_decay}]
    groups.append({"params": plain, "lr": lr, "weight_decay": 0.0})
    return [g for g in groups if g["params"]]


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> list[torch.optim.Optimizer]:
    fused = next(model.parameters()).is_cuda  # fused AdamW kernel on CUDA; False on CPU (tests)
    muon_p, adamw_p = partition_params(model)
    muon = Muon(
        _lr_groups(muon_p, cfg.muon_lr, cfg.weight_decay),
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        ns_steps=cfg.ns_steps,
        weight_decay=cfg.weight_decay,
    )
    adamw = torch.optim.AdamW(
        _lr_groups(adamw_p, cfg.adamw_lr, cfg.weight_decay),
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
        fused=fused,
    )
    return [muon, adamw]
