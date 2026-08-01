import pytest
import torch

from src.slices.Decode.TransducerBeamSearch import TransducerBeamSearch
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-graph capture is GPU-only"
)


def _model():
    torch.manual_seed(0)
    return TransducerModel(cmvn_path=None).cuda().eval()


def test_graphed_step_matches_eager_logits():
    # One captured predictor+joiner step must reproduce the eager step bit-for-bit over the valid
    # rows -- that equivalence is the whole safety argument for the opt-in fast path.
    from src.slices.Decode.CudaGraphedTransducerStep import CudaGraphedTransducerStep

    model = _model()
    beam = 8
    step = CudaGraphedTransducerStep(model, beam)
    n = 5  # fewer live hyps than beam -> exercises the padding/readback path
    states = model.predictor.init_state(n, torch.device("cuda"))
    lasts = torch.randint(0, model._blank + 1, (n,), device="cuda")
    enc_t = torch.randn(1, model.encoder.output_dim, device="cuda")
    with torch.no_grad():
        pred_out, exp_state = model.predictor.step(states, lasts)
        exp_logp = torch.log_softmax(model.joiner.step(enc_t, pred_out), dim=-1)
    logp, new_state = step.run(states, lasts, enc_t)
    assert torch.allclose(logp, exp_logp, atol=1e-4)
    assert torch.equal(new_state, exp_state)


def test_graphed_beam_matches_eager_beam():
    # End to end: the beam with cuda_graph on returns the same n-best ids as the eager beam.
    from src.slices.Decode.CudaGraphedTransducerStep import CudaGraphedTransducerStep

    model = _model()
    memory = torch.randn(1, 12, model.encoder.output_dim, device="cuda")
    eager = TransducerBeamSearch(model, beam_size=8, max_symbols=5)
    graphed = TransducerBeamSearch(
        model, beam_size=8, max_symbols=5, graph_step=CudaGraphedTransducerStep(model, 8)
    )
    assert [ids for ids, _ in graphed.search(memory)] == [ids for ids, _ in eager.search(memory)]
