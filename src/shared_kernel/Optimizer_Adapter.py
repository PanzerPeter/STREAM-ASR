# src/shared_kernel/Optimizer_Adapter.py
# Builds the SP3 optimizer stack: 2D hidden weight matrices -> Muon (spectrally normalized updates),
# everything else (embeddings, biases, norms, input frontend, output heads) -> AdamW. When muP is
# on, per-parameter AdamW LRs are scaled by the _mup_lr_scale tags so a proxy-width LR transfers.
import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import OptimConfig
from src.shared_kernel.Muon_Optimizer import Muon
from src.shared_kernel.mup import mup_lr_scale

# Output heads / readouts that must stay on AdamW, matched as substrings of the dotted module name.
# `pred_head` is the BEST-RQ pretrain head (BestRqModel); `ctc_head`/`interctc` are the main and
# auxiliary CTC readouts; `joiner.out` is the transducer readout. The joiner's `enc_proj`/
# `pred_proj` are hidden 2D projections and are intentionally omitted here so they fall through to
# Muon.
_DEFAULT_HEAD_PATTERNS = (
    "frontend",
    "ctc_head",
    "pred_head",
    "interctc",
    "joiner.out",
)


def partition_params(
    model: nn.Module, head_patterns: tuple[str, ...] = _DEFAULT_HEAD_PATTERNS
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    muon: list[nn.Parameter] = []
    adamw: list[nn.Parameter] = []
    for module_name, module in model.named_modules():
        excluded = any(pat in module_name for pat in head_patterns)
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_hidden_matrix = isinstance(module, nn.Linear) and pname == "weight" and p.ndim == 2
            if is_hidden_matrix and not excluded:
                muon.append(p)
            else:
                adamw.append(p)
    return muon, adamw


def _lr_groups(
    params: list[nn.Parameter], base_lr: float, enc_ids: set[int], enc_scale: float
) -> list[dict]:
    # One group per param so each carries its own peak LR: encoder params get base_lr * enc_scale
    # (warm-started encoder fine-tuned gently while fresh heads adapt at full LR); muP tags (no-op
    # = 1.0 when muP is off) still multiply through. The trainer rescales every group's lr per step.
    groups: list[dict] = []
    for p in params:
        lr = base_lr * mup_lr_scale(p) * (enc_scale if id(p) in enc_ids else 1.0)
        groups.append({"params": [p], "lr": lr})
    return groups


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> list[torch.optim.Optimizer]:
    fused = next(model.parameters()).is_cuda  # fused AdamW kernel on CUDA; False on CPU (tests)
    enc_ids = {id(p) for name, p in model.named_parameters() if name.startswith("encoder.")}
    enc_scale = cfg.encoder_lr_scale
    if cfg.optimizer == "adamw":
        groups = _lr_groups(list(model.parameters()), cfg.adamw_lr, enc_ids, enc_scale)
        return [
            torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.98), fused=fused)
        ]
    muon_p, adamw_p = partition_params(model)
    muon = Muon(
        _lr_groups(muon_p, cfg.muon_lr, enc_ids, enc_scale),
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        ns_steps=cfg.ns_steps,
        weight_decay=cfg.weight_decay,
    )
    adamw = torch.optim.AdamW(
        _lr_groups(adamw_p, cfg.adamw_lr, enc_ids, enc_scale),
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
        fused=fused,
    )
    return [muon, adamw]
