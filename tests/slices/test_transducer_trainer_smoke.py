import os

import pytest
import torch
import torch._dynamo

from src.shared_kernel.Checkpoint_Adapter import load_checkpoint, save_checkpoint
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Optimizer_Adapter import build_optimizer
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import (
    _drop_snapshots_after,
    _require_cmvn,
    _write_rolling_snapshot,
    greedy_transducer_decode,
)
from src.slices.TrainAcousticModel._train_utils import _HOT_MODULES, compile_hot_modules


class _Tok:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class _StubLog:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def warning(self, message: str) -> None:
        self._sink.append(message)


def test_greedy_transducer_decode_runs():
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    b = collate_features([(torch.randn(160, 80), [3, 4, 5])])
    with torch.no_grad():
        memory, out_len, _, _, _ = model(b.features, b.feature_lengths)
        hyps = greedy_transducer_decode(model, memory, out_len, _Tok())
    assert isinstance(hyps, list) and len(hyps) == 1 and isinstance(hyps[0], str)


def test_greedy_decode_is_batch_invariant():
    # The batch walks the frame axis together, so correctness rests entirely on freezing: a lane
    # past its own out_length, or one that has already blanked for this frame, must keep the exact
    # predictor state the per-utterance loop would have left it with. Ragged lengths plus decoding
    # the same utterances one at a time is the independent check on that.
    torch.manual_seed(11)
    model = TransducerModel(cmvn_path=None).eval()
    lengths = (320, 96, 224, 160)
    b = collate_features([(torch.randn(n, 80), [3, 4, 5]) for n in lengths])
    with torch.no_grad():
        memory, out_len, _, _, _ = model(b.features, b.feature_lengths)
        together = greedy_transducer_decode(model, memory, out_len, _Tok())
        alone = [
            greedy_transducer_decode(
                model, memory[i : i + 1, : int(out_len[i])], out_len[i : i + 1], _Tok()
            )[0]
            for i in range(len(lengths))
        ]
    assert len(set(out_len.tolist())) > 1, "lengths must be ragged to exercise lane freezing"
    assert together == alone


@torch.no_grad()
def _forward_reference_greedy(
    model: TransducerModel, memory: torch.Tensor, t_len: int
) -> list[int]:
    # Ground-truth greedy computed via `predictor.forward` (batched) instead of `.step` (stateful).
    # `step(state, tok)` == `forward([..state, tok])[:, -1]`, so a history-based
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


def test_rolling_snapshot_is_a_loadable_copy_of_last(tmp_path):
    # The snapshot is copied from transducer_last.pt rather than re-serialised, so what needs
    # pinning is that the copy is byte-for-byte the same checkpoint AND still loads through the
    # normal path -- scripts/average_checkpoints.py reads these, and a truncated or half-written
    # one would only surface after training finished.
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None)
    optimizers = build_optimizer(model, get_config().optim)
    last = os.path.join(tmp_path, "transducer_last.pt")
    save_checkpoint(last, model, optimizers, 42, best_wer=0.25, resume_count=3, kind="transducer")

    _write_rolling_snapshot(str(tmp_path), last, 42, keep_last_n=2)
    snap = os.path.join(tmp_path, "transducer_step42.pt")
    assert open(snap, "rb").read() == open(last, "rb").read()
    assert not os.path.exists(snap + ".tmp")

    restored = load_checkpoint(snap, TransducerModel(cmvn_path=None))
    assert (restored["step"], restored["best_wer"], restored["resume_count"]) == (42, 0.25, 3)


def test_rolling_snapshot_prunes_to_keep_last_n(tmp_path):
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None)
    optimizers = build_optimizer(model, get_config().optim)
    last = os.path.join(tmp_path, "transducer_last.pt")
    save_checkpoint(last, model, optimizers, 0, kind="transducer")
    for step in (10, 5, 20, 15):  # out of order: pruning must sort numerically, not lexically
        _write_rolling_snapshot(str(tmp_path), last, step, keep_last_n=2)
    kept = sorted(f for f in os.listdir(tmp_path) if f.startswith("transducer_step"))
    assert kept == ["transducer_step15.pt", "transducer_step20.pt"]

    _write_rolling_snapshot(str(tmp_path), last, 30, keep_last_n=0)  # 0 disables
    assert not os.path.exists(os.path.join(tmp_path, "transducer_step30.pt"))


def test_snapshots_ahead_of_the_resume_point_are_dropped(tmp_path):
    # Rotation is by step number, so a snapshot left by a run that was later rolled back outranks
    # every snapshot the replacement run writes and survives forever. OBSERVED 2026-08-05: after
    # rolling back to step394200, a stale step415800 from the abandoned run sat in the directory,
    # and average_checkpoints.py --last-n 5 would have averaged that diverged model into the
    # decode checkpoint. Anything ahead of the resumed step describes a future this run did not
    # take.
    for step in (394200, 399600, 415800):
        open(os.path.join(tmp_path, f"transducer_step{step}.pt"), "w").close()
    logged: list[str] = []
    _drop_snapshots_after(str(tmp_path), 399600, _StubLog(logged))
    kept = sorted(f for f in os.listdir(tmp_path))
    assert kept == ["transducer_step394200.pt", "transducer_step399600.pt"]
    assert len(logged) == 1 and "step415800" in logged[0]

    _drop_snapshots_after(str(tmp_path), 399600, _StubLog(logged))
    assert len(logged) == 1, "a clean directory must not warn"


def test_compile_hot_modules_leaves_checkpoints_portable():
    # nn.Module.compile installs a compiled _call_impl on the instance; wrapping the module in an
    # OptimizedModule instead would rename every key to _orig_mod.*, and Decode/Evaluate load these
    # checkpoints into an EAGER model. That portability is the thing to pin, along with the count
    # (a silently-empty selection would make the whole speedup vanish without failing anything).
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None)
    before = list(model.state_dict())

    compiled = compile_hot_modules(model)

    assert compiled == sum(isinstance(m, _HOT_MODULES) for m in model.modules()) > 0
    assert list(model.state_dict()) == before
    TransducerModel(cmvn_path=None).load_state_dict(model.state_dict())


def test_recompile_limit_has_headroom_for_every_channel_width():
    # dynamo's recompile limit is per code object, and one BiasNorm.forward serves every instance,
    # so its graphs accumulate one per distinct channel width. Exceeding the limit does not raise --
    # dynamo demotes the function to eager for the rest of the process and only logs a warning, so
    # the speedup evaporates silently, hours in. torch's default of 8 was met exactly by four
    # widths x {train bf16, eval fp32}, which is what shipped broken.
    #
    # _dev_metrics now forces eager, removing the eval half, but the invariant worth pinning is the
    # one that survives someone widening the encoder: enough headroom for both modes at every width.
    compile_hot_modules(TransducerModel(cmvn_path=None))

    model_cfg = get_config().model
    widths = set(model_cfg.encoder_dims) | {get_config().transducer.predictor_dim}
    assert torch._dynamo.config.recompile_limit >= 2 * len(widths), (
        f"{len(widths)} distinct BiasNorm widths need up to {2 * len(widths)} graphs; "
        f"limit is {torch._dynamo.config.recompile_limit}"
    )


def test_missing_cmvn_is_an_error_but_empty_path_is_not(tmp_path):
    # The encoder's mean 0 / std 1 fallback is for tests and inference; reaching it from a trainer
    # means training on raw log-mel (mean -5.65, std 4.06) with no metric in the loop reporting it.
    with pytest.raises(FileNotFoundError, match="cmvn"):
        _require_cmvn(str(tmp_path / "absent.pt"))
    assert _require_cmvn("") is None
    stats = tmp_path / "cmvn.pt"
    torch.save({"mean": torch.zeros(80), "std": torch.ones(80)}, stats)
    assert _require_cmvn(str(stats)) == str(stats)
