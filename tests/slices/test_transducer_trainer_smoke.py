import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import greedy_transducer_decode


class _Tok:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_greedy_transducer_decode_runs():
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    b = collate_features([(torch.randn(160, 80), [3, 4, 5])])
    with torch.no_grad():
        memory, out_len, _, _, _ = model(b.features, b.feature_lengths)
        hyps = greedy_transducer_decode(model, memory, out_len, _Tok())
    assert isinstance(hyps, list) and len(hyps) == 1 and isinstance(hyps[0], str)


@torch.no_grad()
def _forward_reference_greedy(
    model: TransducerModel, memory: torch.Tensor, t_len: int
) -> list[int]:
    # Ground-truth greedy computed via `predictor.forward` (batched) instead of `.step` (stateful).
    # Task 3 proved `step(state, tok)` == `forward([..state, tok])[:, -1]`, so a history-based
    # forward call is an independently-correct reference: it never touches the `state` threading
    # that the bug corrupts, so it fails to reproduce the bug and instead pins down truth.
    blank = get_config().model.blank_id
    max_symbols = get_config().decode.max_symbols
    device = memory.device
    history = [blank]
    ids: list[int] = []
    for t in range(t_len):
        enc_t = memory[0, t].unsqueeze(0)  # [1, De]
        emitted = 0
        while emitted < max_symbols:
            hist_t = torch.tensor([history], dtype=torch.long, device=device)
            pred = model.predictor(hist_t)  # [1, len(history), D]
            pred_u = pred[:, -1]
            tok = int(model.joiner.step(enc_t, pred_u).argmax(dim=-1))
            if tok == blank:
                break
            ids.append(tok)
            history.append(tok)
            emitted += 1
    return ids


def test_greedy_matches_forward_reference():
    # Discriminates the state-threading bug: the buggy `greedy_transducer_decode` re-derives the
    # predictor state from the just-emitted token (context becomes [tok, tok] instead of
    # [prev, tok]) once >=2 non-blank tokens are emitted for the same frame, which the
    # forward-based reference (built straight from `history`, never touching `state`) cannot
    # reproduce. Equality here is exactly "state-threading matches history semantics".
    torch.manual_seed(3)
    model = TransducerModel(cmvn_path=None).eval()
    b = collate_features([(torch.randn(320, 80), [3, 4, 5, 6, 7])])
    memory, out_len, _, _, _ = model(b.features, b.feature_lengths)
    t_len = int(out_len[0])

    ref_ids = _forward_reference_greedy(model, memory, t_len)
    hyp_texts = greedy_transducer_decode(model, memory, out_len, _Tok())

    assert len(ref_ids) >= 2, "seed must exercise the 2nd-token predictor context to catch the bug"
    assert hyp_texts[0] == _Tok().decode(ref_ids)
