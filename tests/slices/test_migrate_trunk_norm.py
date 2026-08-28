# The trunk-norm migration re-keys AdamW's saved state, and the keys are POSITIONS. Guards the
# ordering that mapping has to use: `Optimizer.state_dict` numbers parameters by their place in
# the concatenated `param_groups`, which `_lr_groups` builds by bucketing on (lr, weight_decay) --
# not the order `partition_params` walks the model in.
import torch
import torch.nn as nn

from scripts.migrate_trunk_norm import adamw_param_names
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Optimizer_Adapter import build_optimizer, partition_params


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Two decayed tensors (the conv kernels, ndim >= 2) separated by undecayed ones, which is
        # what makes `_lr_groups`' bucketing reorder the list -- and is the real encoder's shape,
        # where `frontend.conv1.bias` sits between the two frontend convolutions.
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 6, 3), nn.LayerNorm(6), nn.Conv1d(6, 6, 3), nn.Linear(6, 6)
        )
        self.ctc_head = nn.Linear(6, 5)


def _adamw_of(net: nn.Module) -> torch.optim.AdamW:
    opts = build_optimizer(net, get_config().optim)
    adamw = [o for o in opts if isinstance(o, torch.optim.AdamW)]
    assert len(adamw) == 1
    return adamw[0]


def test_names_index_the_state_dict_keys() -> None:
    net = _Net()
    adamw = _adamw_of(net)
    for group in adamw.param_groups:
        for p in group["params"]:
            p.grad = torch.zeros_like(p)
    adamw.step()

    names = adamw_param_names(net, adamw)
    shapes = dict(net.named_parameters())
    state = adamw.state_dict()["state"]
    assert len(names) == sum(len(g["params"]) for g in adamw.param_groups)
    for index, entry in state.items():
        assert entry["exp_avg"].shape == shapes[names[index]].shape


def test_walk_order_is_not_the_state_order() -> None:
    # The bug this test exists for: mapping by `partition_params` order permuted 115 of the real
    # checkpoint's 466 AdamW entries onto the wrong parameters, and the resume died inside
    # `Adam.step` on a (576,) moment meeting a (192,) gradient.
    net = _Net()
    _, adamw_p = partition_params(net)
    by_id = {id(p): n for n, p in net.named_parameters()}
    assert [by_id[id(p)] for p in adamw_p] != adamw_param_names(net, _adamw_of(net))
