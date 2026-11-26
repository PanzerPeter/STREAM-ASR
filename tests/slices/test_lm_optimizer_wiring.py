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


def test_every_model_parameter_is_optimized_exactly_once():
    model, opts = _built()
    ids = _owned(opts)
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters()}


def test_muon_takes_the_block_matrices_and_not_the_embedding_or_readout():
    model, opts = _built()
    muon = [o for o in opts if isinstance(o, Muon)]
    assert len(muon) == 1
    muon_ids = {id(p) for g in muon[0].param_groups for p in g["params"]}
    # Muon's update is only defined for 2D hidden matrices; the vocab-width tables stay on AdamW.
    assert muon_ids and all(p.ndim == 2 for p in model.parameters() if id(p) in muon_ids)
    assert id(model.tok_emb.weight) not in muon_ids
    assert id(model.head.weight) not in muon_ids


def test_no_weight_decay_on_vectors_or_scalars():
    # Decoupled decay pulls every parameter toward 0, a specific setting for a mixing weight, not a
    # neutral one; only matrices decay.
    _, opts = _built()
    adamw = [o for o in opts if isinstance(o, torch.optim.AdamW)][0]
    for g in adamw.param_groups:
        if any(p.ndim < 2 for p in g["params"]):
            assert g["weight_decay"] == 0.0
            assert all(p.ndim < 2 for p in g["params"])
