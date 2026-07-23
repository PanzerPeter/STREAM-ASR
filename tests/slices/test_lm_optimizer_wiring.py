import torch

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Muon_Optimizer import Muon
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.slices.TrainLanguageModel.TrainLm_Handler import TrainLm_Handler


def _built() -> tuple[StreamLmModel, list[torch.optim.Optimizer]]:
    torch.manual_seed(0)
    model = StreamLmModel()
    return model, TrainLm_Handler()._optimizers(model, get_config().lm)


def _owned(opts: list[torch.optim.Optimizer]) -> list[int]:
    return [id(p) for opt in opts for g in opt.param_groups for p in g["params"]]


def test_tied_embedding_is_owned_by_exactly_one_optimizer_group():
    # head.weight IS tok_emb.weight. Listing it twice would apply two updates (and two weight
    # decays) per step to the same tensor, which is silent and hard to spot in a loss curve.
    ids = _owned(_built()[1])
    assert len(ids) == len(set(ids))


def test_every_model_parameter_is_optimized_exactly_once():
    model, opts = _built()
    assert set(_owned(opts)) == {id(p) for p in model.parameters()}


def test_muon_takes_the_block_matrices_and_not_the_embedding():
    model, opts = _built()
    muon = [o for o in opts if isinstance(o, Muon)]
    assert len(muon) == 1, "muon+adamw config must build a Muon optimizer"
    muon_params = [p for g in muon[0].param_groups for p in g["params"]]
    # Muon's update is only defined for 2D hidden matrices; the tied table must stay on AdamW.
    assert muon_params and all(p.ndim == 2 for p in muon_params)
    assert all(id(p) != id(model.tok_emb.weight) for p in muon_params)
