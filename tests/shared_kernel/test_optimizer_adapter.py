import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Optimizer_Adapter import build_optimizer, partition_params
from src.shared_kernel.Muon_Optimizer import Muon


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Linear(4, 8)  # input layer -> AdamW
        self.hidden = nn.Linear(8, 8)  # 2D hidden -> Muon
        self.embed = nn.Embedding(5, 8)  # not Linear -> AdamW
        self.ctc_head = nn.Linear(8, 6)  # output head -> AdamW


def test_partition_routes_hidden_to_muon_only():
    net = _Net()
    muon_p, adamw_p = partition_params(net, head_patterns=("frontend", "ctc_head"))
    muon_ids = {id(p) for p in muon_p}
    assert id(net.hidden.weight) in muon_ids
    assert id(net.frontend.weight) not in muon_ids  # frontend excluded
    assert id(net.ctc_head.weight) not in muon_ids  # head excluded
    assert id(net.embed.weight) not in muon_ids  # embedding excluded
    assert id(net.hidden.bias) not in muon_ids  # bias excluded


def test_build_optimizer_muon_plus_adamw():
    net = _Net()
    cfg = get_config().optim
    opts = build_optimizer(net, cfg)
    assert isinstance(opts, list)
    if cfg.optimizer == "muon+adamw":
        assert any(isinstance(o, Muon) for o in opts)
        assert any(isinstance(o, torch.optim.AdamW) for o in opts)


def test_optim_config_loads():
    cfg = get_config().optim
    assert cfg.optimizer in ("adamw", "muon+adamw")
    assert cfg.adamw_lr > 0


def test_partition_excludes_real_decoder_and_ctc_heads():
    # Regression lock for the head-routing bug: on the REAL model both output heads must go to
    # AdamW, while hidden attention projections (incl. cross-attn out_proj) must go to Muon.
    from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention

    m = HybridCtcAttention(cmvn_path=None)
    muon, _ = partition_params(m)
    muon_ids = {id(p) for p in muon}
    assert id(m.ctc_head.weight) not in muon_ids
    assert id(m.decoder.out_proj.weight) not in muon_ids
    assert id(m.decoder.left.layers[0].cross_attn.out_proj.weight) in muon_ids
    assert id(m.encoder.stacks[0].blocks[0].attn.out.weight) in muon_ids


def test_mup_enabled_scales_adamw_group_lr():
    # Lock the mup_enabled per-param-LR branch (SP3's muP machinery): a tagged head weight routed
    # to AdamW gets lr = adamw_lr * its _mup_lr_scale.
    from src.shared_kernel.Config_Adapter import OptimConfig
    from src.shared_kernel.mup import mup_linear_

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(8, 8)  # -> Muon
            self.ctc_head = nn.Linear(8, 4)  # head pattern -> AdamW

    net = _M()
    mup_linear_(net.ctc_head, base_fan_in=4)  # ctc_head.weight._mup_lr_scale = 4/8 = 0.5
    cfg = OptimConfig(
        optimizer="muon+adamw",
        muon_lr=0.02,
        adamw_lr=0.01,
        muon_momentum=0.95,
        ns_steps=5,
        weight_decay=0.01,
        mup_enabled=True,
        mup_base_dims=(8,),
    )
    opts = build_optimizer(net, cfg)
    adamw = next(o for o in opts if isinstance(o, torch.optim.AdamW))
    lrs = {round(g["lr"], 6) for g in adamw.param_groups}
    assert 0.005 in lrs  # 0.01 * 0.5 for the muP-scaled ctc_head.weight
