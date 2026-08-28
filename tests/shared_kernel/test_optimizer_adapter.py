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


def test_partition_routes_transducer_modules():
    # Regression lock: transducer readouts/embeddings -> AdamW, joiner's hidden 2D projections
    # -> Muon, matching the routing locked for the encoder.
    from src.slices.TrainAcousticModel.TransducerModel import TransducerModel

    m = TransducerModel(cmvn_path=None)
    muon, adamw = partition_params(m)
    muon_ids = {id(p) for p in muon}
    adamw_ids = {id(p) for p in adamw}
    # Joiner readout + InterCTC + CTC heads + predictor embedding -> AdamW.
    assert id(m.joiner.out.weight) in adamw_ids
    assert id(m.ctc_head.weight) in adamw_ids
    assert id(m.interctc_heads[0].weight) in adamw_ids
    assert id(m.predictor.embed.weight) in adamw_ids
    # Joiner hidden projections -> Muon (2D hidden Linears).
    assert id(m.joiner.enc_proj.weight) in muon_ids
    assert id(m.joiner.pred_proj.weight) in muon_ids
    # Encoder attention out projection stays on Muon (regression lock).
    assert id(m.encoder.stacks[0].blocks[0].attn.out.weight) in muon_ids


def test_encoder_lr_scale_downscales_encoder_groups():
    # Warm-start discriminative fine-tuning: params under `encoder.*` get base_lr * encoder_lr_scale
    # on BOTH Muon and AdamW, while fresh heads keep the full LR.
    from src.shared_kernel.Config_Adapter import OptimConfig

    class _Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(8, 8))  # encoder.* -> scaled
            self.ctc_head = nn.Linear(8, 6)  # fresh head -> full LR

    net = _Enc()
    cfg = OptimConfig(
        optimizer="muon+adamw",
        muon_lr=0.02,
        adamw_lr=0.01,
        muon_momentum=0.95,
        ns_steps=5,
        weight_decay=0.01,
        encoder_lr_scale=0.25,
    )
    muon, adamw = build_optimizer(net, cfg)
    enc_w = id(net.encoder[0].weight)  # 2D hidden -> Muon, encoder -> scaled
    head_w = id(net.ctc_head.weight)  # head -> AdamW, full LR
    muon_lr = {id(p): g["lr"] for g in muon.param_groups for p in g["params"]}
    adamw_lr = {id(p): g["lr"] for g in adamw.param_groups for p in g["params"]}
    assert muon_lr[enc_w] == 0.02 * 0.25  # encoder Muon group scaled
    assert adamw_lr[head_w] == 0.01  # fresh head at full AdamW LR


def test_weight_decay_skips_gates_norms_and_biases():
    # Zero is a degenerate setting for every ndim<2 parameter in this model, not a neutral shrink
    # target: bypass 0 skips a stack, res_lambda 0 disables the value residual, log_scale 0 pins
    # the norm scale to 1, and SimpleDownsample.weights 0 forces uniform pooling. Decaying them is
    # a standing pull away from whatever they learned.
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
        expected = cfg.weight_decay if p.ndim >= 2 else 0.0
        assert decay_of[id(p)] == expected, name
    # The gates this exists for, named explicitly so a routing change cannot silently re-decay them.
    for gate in (
        m.encoder.stacks[0].bypass,
        m.encoder.stacks[0].blocks[1].attn.res_lambda,
        m.encoder.stacks[1].downsample.weights,
        m.encoder.out_norm.log_scale,
    ):
        assert decay_of[id(gate)] == 0.0
    # Real weight matrices keep it, including the 4D frontend convs and the 2D readouts.
    for weight in (m.encoder.frontend.conv1.weight, m.joiner.out.weight, m.predictor.embed.weight):
        assert decay_of[id(weight)] == cfg.weight_decay


def test_muon_groups_all_still_decay():
    # Muon only ever receives 2D hidden matrices, so the split must leave it with one group per LR
    # exactly as before -- a zero-decay group means something non-matrix got routed there.
    from src.slices.TrainAcousticModel.TransducerModel import TransducerModel

    cfg = get_config().optim
    muon = build_optimizer(TransducerModel(cmvn_path=None), cfg)[0]
    assert isinstance(muon, Muon)
    assert {g["weight_decay"] for g in muon.param_groups} == {cfg.weight_decay}
    assert all(p.ndim == 2 for g in muon.param_groups for p in g["params"])


def test_simple_projections_go_to_adamw_not_muon():
    # They are vocab-width readouts like joiner.out and ctc_head, not hidden matrices. Muon's
    # spectral normalisation is wrong for a readout.
    import torch.nn as nn
    from src.shared_kernel.Optimizer_Adapter import partition_params

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(8, 8)
            self.simple_am_proj = nn.Linear(8, 4)
            self.simple_lm_proj = nn.Linear(8, 4)

    net = _M()
    muon, adamw = partition_params(net)
    adamw_ids = {id(p) for p in adamw}
    assert id(net.simple_am_proj.weight) in adamw_ids
    assert id(net.simple_lm_proj.weight) in adamw_ids
    assert id(net.hidden.weight) in {id(p) for p in muon}
