import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Optimizer_Adapter import build_optimizer, partition_params
from src.shared_kernel.Muon_Optimizer import Muon


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Linear(4, 8)  # excluded by the explicit pattern below
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
    assert any(isinstance(o, Muon) for o in opts)
    assert any(isinstance(o, torch.optim.AdamW) for o in opts)


def test_optim_config_loads():
    cfg = get_config().optim
    assert cfg.adamw_lr > 0


def test_partition_routes_transducer_modules():
    # Readouts/embeddings -> AdamW; every 2-D hidden matrix -> Muon, including each logical slice of
    # a block's fused W_in/W_out (orthogonalised one by one, never as the concatenation).
    from src.slices.TrainAcousticModel.TransducerModel import TransducerModel

    m = TransducerModel(cmvn_path=None)
    muon, adamw = partition_params(m)
    muon_ids = {id(p) for p in muon}
    adamw_ids = {id(p) for p in adamw}
    for p in (m.joiner.out.weight, m.ctc_head.weight, m.predictor.embed.weight):
        assert id(p) in adamw_ids
    block = m.encoder.stacks[1].blocks[0]
    for p in (
        m.joiner.enc_proj.weight,
        m.joiner.pred_proj.weight,
        m.encoder.frontend.linear.weight,
        m.encoder.stacks[1].proj.weight,
        *block.w_in.values(),
        *block.w_out.values(),
    ):
        assert id(p) in muon_ids
    assert len(block.w_in) == 6 and len(block.w_out) == 2


def test_weight_decay_skips_gates_gains_and_biases():
    # Zero is a specific setting for every ndim<2 parameter here, not a neutral shrink target: a
    # gate pre-activation of 0 is beta = 0.53, a QK gain of 0 flattens attention.
    from src.slices.TrainAcousticModel.TransducerModel import TransducerModel

    m = TransducerModel(cmvn_path=None)
    cfg = get_config().optim
    decay_of = {
        id(p): g["weight_decay"]
        for o in build_optimizer(m, cfg)
        for g in o.param_groups
        for p in g["params"]
    }
    assert len(decay_of) == sum(1 for p in m.parameters() if p.requires_grad)
    for name, p in m.named_parameters():
        assert decay_of[id(p)] == (cfg.weight_decay if p.ndim >= 2 else 0.0), name
    block = m.encoder.stacks[1].blocks[0]
    for p in (block.gate, block.q_gain, block.k_gain, block.conv.bias):
        assert decay_of[id(p)] == 0.0


def test_muon_takes_only_matrices_and_decays_them():
    from src.slices.TrainAcousticModel.TransducerModel import TransducerModel

    cfg = get_config().optim
    muon = build_optimizer(TransducerModel(cmvn_path=None), cfg)[0]
    assert isinstance(muon, Muon)
    assert {g["weight_decay"] for g in muon.param_groups} == {cfg.weight_decay}
    assert all(p.ndim == 2 for g in muon.param_groups for p in g["params"])
