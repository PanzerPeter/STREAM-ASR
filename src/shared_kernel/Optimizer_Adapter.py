# src/shared_kernel/Optimizer_Adapter.py
# Builds the SP3 optimizer stack: 2D hidden weight matrices -> Muon (spectrally normalized updates),
# everything else (embeddings, biases, norms, input frontend, output heads) -> AdamW. When muP is
# on, per-parameter AdamW LRs are scaled by the _mup_lr_scale tags so a proxy-width LR transfers.
import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import OptimConfig
from src.shared_kernel.Muon_Optimizer import Muon
from src.shared_kernel.mup import mup_lr_scale

# Output heads that must stay on AdamW, matched as substrings of the dotted module name. The decoder
# uses a single shared vocab head `decoder.out_proj`; it is spelled with the `decoder.` prefix on
# purpose so it does not over-match the cross-attention `...cross_attn.out_proj` projections, which
# are hidden 2D weights that belong on Muon.
_DEFAULT_HEAD_PATTERNS = ("frontend", "ctc_head", "pred_head", "decoder.out_proj")


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


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> list[torch.optim.Optimizer]:
    fused = next(model.parameters()).is_cuda  # fused AdamW kernel on CUDA; False on CPU (tests)
    if cfg.optimizer == "adamw":
        return [
            torch.optim.AdamW(
                model.parameters(),
                lr=cfg.adamw_lr,
                weight_decay=cfg.weight_decay,
                betas=(0.9, 0.98),
                fused=fused,
            )
        ]
    muon_p, adamw_p = partition_params(model)
    muon = Muon(
        muon_p,
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        ns_steps=cfg.ns_steps,
        weight_decay=cfg.weight_decay,
    )
    if cfg.mup_enabled:
        adamw_groups = [{"params": [p], "lr": cfg.adamw_lr * mup_lr_scale(p)} for p in adamw_p]
        adamw = torch.optim.AdamW(
            adamw_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.98), fused=fused
        )
    else:
        adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.adamw_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98), fused=fused
        )
    return [muon, adamw]
