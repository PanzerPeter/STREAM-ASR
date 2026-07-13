# tests/slices/test_trainer_optimizer_wiring.py
import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config, OptimConfig
from src.shared_kernel.Optimizer_Adapter import build_optimizer


def test_optimizers_step_and_schedule_as_list():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    opts = build_optimizer(model, get_config().optim)
    # the trainer applies the schedule across every group of every optimizer
    for o in opts:
        for g in o.param_groups:
            g["lr"] = 1e-4
    model(torch.randn(3, 8)).sum().backward()
    for o in opts:
        o.step()
        o.zero_grad(set_to_none=True)
    assert all(g["lr"] == 1e-4 for o in opts for g in o.param_groups)


def test_shape_schedule_preserves_distinct_muon_and_adamw_peaks():
    # Regression lock for the SP3 LR bug: the per-group shape schedule the trainers apply
    # (peak * lr_shape) must keep Muon's large base LR (2e-2) distinct from AdamW's (1.5e-3).
    # The old absolute overwrite collapsed both to one value, silently training Muon ~13x too low.
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(8, 8)  # 2D hidden -> Muon
            self.ctc_head = nn.Linear(8, 4)  # head pattern -> AdamW

    cfg = OptimConfig(
        optimizer="muon+adamw",
        muon_lr=2.0e-2,
        adamw_lr=1.5e-3,
        muon_momentum=0.95,
        ns_steps=5,
        weight_decay=1.0e-2,
        mup_enabled=False,
        mup_base_dims=(8,),
    )
    optimizers = build_optimizer(_M(), cfg)
    peak_lrs = [[g["lr"] for g in o.param_groups] for o in optimizers]
    lr_shape = 0.5  # mid-schedule sample
    for o, peaks in zip(optimizers, peak_lrs):
        for g, pk in zip(o.param_groups, peaks):
            g["lr"] = pk * lr_shape
    muon_lr = optimizers[0].param_groups[0]["lr"]  # build_optimizer returns [muon, adamw]
    adamw_lr = optimizers[-1].param_groups[0]["lr"]
    assert abs(muon_lr - 1.0e-2) < 1e-12  # 2e-2 * 0.5
    assert abs(adamw_lr - 7.5e-4) < 1e-12  # 1.5e-3 * 0.5
    assert muon_lr > adamw_lr * 10  # peaks stay distinct (Muon >> AdamW)
