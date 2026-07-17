# tests/slices/test_stateless_predictor.py
import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.StatelessPredictor import StatelessPredictor


def test_forward_shape():
    torch.manual_seed(0)
    p = StatelessPredictor().eval()
    labels = torch.randint(0, 500, (2, 7))
    out = p(labels)
    assert out.shape == (2, 7, get_config().transducer.predictor_dim)


def test_streaming_state_equals_batched_forward():
    # Step-by-step state advance must equal the batched forward on the same blank-prefixed sequence.
    torch.manual_seed(1)
    p = StatelessPredictor().eval()
    B, U = 2, 6
    labels = torch.randint(0, 500, (B, U))
    blank = get_config().model.blank_id
    prefixed = torch.cat([torch.full((B, 1), blank), labels], dim=1)  # [B, U+1]
    with torch.no_grad():
        batched = p(prefixed)  # [B, U+1, D]
        state = p.init_state(B, labels.device)
        outs = []
        # position 0 consumes the blank start symbol, positions 1..U consume labels[:, u-1]
        step_tokens = prefixed
        for u in range(U + 1):
            out, state = p.step(state, step_tokens[:, u])
            outs.append(out)
        stepped = torch.stack(outs, dim=1)
    assert torch.allclose(batched, stepped, atol=1e-5)
