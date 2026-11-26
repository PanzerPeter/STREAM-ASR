import torch

from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


def _batch():
    torch.manual_seed(0)
    return collate_features(
        [(torch.randn(160, 80), [3, 4, 5, 6, 7]), (torch.randn(120, 80), [8, 9, 10])]
    )


def test_forward_shapes_and_unit_rms_memory():
    model = TransducerModel(cmvn_path=None).eval()
    b = _batch()
    memory, out_len, ctc_logits = model(b.features, b.feature_lengths)
    assert ctc_logits.shape[-1] == 501 and memory.shape[1] == ctc_logits.shape[1]
    assert out_len.tolist() == [20, 15]  # 12.5 Hz: one memory frame per 8 mel frames
    # Invariant 2: the encoder memory is rms_normed, RMS exactly 1 per frame.
    rms = memory.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_parameter_budget_matches_the_design():
    model = TransducerModel(cmvn_path=None)
    enc = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    assert abs(enc - 61.78) < 0.01  # §2: [4, 8, 0, 8] at [320, 512, 640, 512], attention then MLP


def test_joint_loss_finite_with_finite_grads_band_on_and_off():
    model = TransducerModel(cmvn_path=None).train()
    for use_band in (False, True):
        for chunk, left in ((0, -1), (8, 64)):
            model.zero_grad(set_to_none=True)
            terms = model.joint_loss(_batch(), chunk, left, use_band)
            assert torch.isfinite(terms.total) and float(terms.band_infeasible) == 0.0
            terms.total.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)


def test_rnnt_is_per_token():
    # Every term is divided by the batch's TOKEN count, so duplicating the batch leaves it fixed.
    model = TransducerModel(cmvn_path=None).eval()
    b = _batch()
    memory, out_len, _ = model(b.features, b.feature_lengths)
    single = model.transducer_cost(memory, out_len, b.tokens, b.token_lengths).sum()
    dup = model.transducer_cost(
        torch.cat([memory, memory]),
        torch.cat([out_len, out_len]),
        torch.cat([b.tokens, b.tokens]),
        torch.cat([b.token_lengths, b.token_lengths]),
    ).sum()
    n = b.token_lengths.sum()
    assert torch.allclose(single / n, dup / (2 * n), atol=1e-4)


def test_joint_loss_under_bf16_autocast():
    # CPU bf16 autocast reproduces the GPU path: bf16 logits reach RnntLoss as-is and gradients
    # flow back out of its fp32 accumulation.
    model = TransducerModel(cmvn_path=None).train()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        terms = model.joint_loss(_batch(), 8, 128, True)
    assert torch.isfinite(terms.total)
    terms.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def _framewise_greedy(model: TransducerModel, memory: torch.Tensor, max_symbols: int):
    # The reference: frame by frame, emit up to max_symbols non-blanks, then step time.
    blank = 500
    state = model.predictor.init_state(1, memory.device)
    prev = torch.full((1,), blank, dtype=torch.long)
    ids, frames = [], []
    for t in range(memory.shape[1]):
        for _ in range(max_symbols):
            out, new_state = model.predictor.step(state, prev)
            tok = int(model.joiner.step(memory[:, t], out).argmax(-1))
            if tok == blank:
                break
            ids.append(tok)
            frames.append(t)
            state, prev = new_state, torch.tensor([tok])
    return ids, frames


def test_label_looping_equals_framewise():
    # §8.1: label-looping is exact for greedy by construction; any disagreement is a bug.
    torch.manual_seed(3)
    model = TransducerModel(cmvn_path=None).eval()
    with torch.no_grad():
        # Bias the blank so the random-init model emits sparsely, like a trained one.
        model.joiner.out.bias[500] += 4.0
        memory = torch.randn(3, 41, model.encoder.output_dim)
        lengths = torch.tensor([41, 30, 7])
        for max_symbols in (1, 2, 5):
            ids, frames = model.greedy_decode(memory, lengths, max_symbols)
            for b in range(3):
                ref = _framewise_greedy(model, memory[b : b + 1, : int(lengths[b])], max_symbols)
                assert (ids[b], frames[b]) == ref
