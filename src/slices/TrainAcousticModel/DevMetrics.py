# Validation for the transducer trainer (MODEL_ARCHITECTURE.md §9.6, §10): full-dev greedy CTC and
# label-looping greedy transducer WER, offline and streaming, on every dev set, plus the band's
# measured loss gap and the attention-gate saturation on one fixed dev batch.
from dataclasses import dataclass

import jiwer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch
from src.slices.TrainAcousticModel.CtcGreedyDecoder import ctc_greedy_decode
from src.slices.TrainAcousticModel.FusedBranchBlock import FusedBranchBlock, rms_norm
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel

# sigmoid(g) below this counts as a saturated (closed) attention gate.
_GATE_SATURATED = 0.01


@dataclass(frozen=True)
class DevSet:
    name: str  # "clean" | "other": the suffix of every dev/* tag
    loader: DataLoader


def _decode_texts(ids: list[list[int]], tokenizer) -> list[str]:
    return [tokenizer.decode(x) for x in ids]


def _score_set(model: TransducerModel, ds: DevSet, tokenizer, device: str) -> dict[str, float]:
    d = get_config().decode
    blank = get_config().model.blank_id
    refs: list[str] = []
    hyps: dict[str, list[str]] = {k: [] for k in ("ctc", "ctc_s", "rnnt", "rnnt_s")}
    blank_frames = total_frames = 0
    for batch in ds.loader:
        feats = batch.features.to(device)
        lens = batch.feature_lengths.to(device)
        memory, out_len, logits = model(feats, lens)
        best = logits.argmax(dim=-1)
        valid = torch.arange(best.shape[1], device=device)[None] < out_len[:, None]
        blank_frames += int(((best == blank) & valid).sum())
        total_frames += int(out_len.sum())
        hyps["ctc"] += ctc_greedy_decode(best.cpu(), out_len.cpu(), tokenizer)
        ids, _ = model.greedy_decode(memory, out_len, d.max_symbols)
        hyps["rnnt"] += _decode_texts(ids, tokenizer)
        # The deployed streaming condition, batched: decode chunk + left context, the final chunk
        # completed and the tail chunk kept. Equal to streaming_forward by the equivalence test.
        mem_s, keep = model.decode_window(feats, lens, d.chunk_size, d.left_context, d.tail_chunks)
        stream_best = model.ctc_head(mem_s).argmax(dim=-1)
        hyps["ctc_s"] += ctc_greedy_decode(stream_best.cpu(), out_len.cpu(), tokenizer)
        ids_s, _ = model.greedy_decode(mem_s, keep, d.max_symbols)
        hyps["rnnt_s"] += _decode_texts(ids_s, tokenizer)
        refs += [
            tokenizer.decode(batch.tokens[i, : batch.token_lengths[i]].tolist())
            for i in range(batch.tokens.shape[0])
        ]
    out = {
        "ctc_wer": jiwer.wer(refs, hyps["ctc"]),
        "ctc_wer_stream": jiwer.wer(refs, hyps["ctc_s"]),
        "transducer_wer": jiwer.wer(refs, hyps["rnnt"]),
        "transducer_wer_stream": jiwer.wer(refs, hyps["rnnt_s"]),
        "blank_frac": blank_frames / max(total_frames, 1),
    }
    out["ctc_wer_stream_gap"] = out["ctc_wer_stream"] - out["ctc_wer"]
    return {f"{k}_{ds.name}": float(v) for k, v in out.items()}


def _probe_batch(model: TransducerModel, batch: FeatureBatch, device: str) -> dict[str, float]:
    feats = batch.features.to(device)
    lens = batch.feature_lengths.to(device)
    tokens = batch.tokens.to(device)
    tlens = batch.token_lengths.to(device)
    inputs: list[torch.Tensor] = []
    blocks = [b for b in model.modules() if isinstance(b, FusedBranchBlock)]
    hooks = [b.register_forward_pre_hook(lambda _m, a: inputs.append(a[0])) for b in blocks]
    try:
        memory, out_len, logits = model(feats, lens)
    finally:
        for h in hooks:
            h.remove()
    closed = sum(
        float((torch.sigmoid(F.linear(rms_norm(x), blk.w_in["g"])) < _GATE_SATURATED).sum())
        for blk, x in zip(blocks, inputs)
    )
    sat = closed / max(1, sum(x.shape[0] * x.shape[1] * blk.dim for blk, x in zip(blocks, inputs)))
    gap = float("nan")
    if get_config().transducer.band_width > 0:
        # (banded - full) per token, on the trained model: the band is a subset of the paths, so
        # this is >= 0, and above 0.01 nats/token it is dropping probability mass that matters.
        full = model.transducer_cost(memory, out_len, tokens, tlens).sum()
        lo, _, _ = model.band(F.log_softmax(logits.float(), dim=-1), out_len, tlens)
        banded = model.transducer_cost(memory, out_len, tokens, tlens, lo).sum()
        gap = float((banded - full) / tlens.sum().clamp(min=1))
    return {"band_loss_gap": gap, "gate_sat_frac": sat}


@torch.no_grad()
def dev_metrics(
    model: TransducerModel,
    dev_sets: list[DevSet],
    probe: FeatureBatch,
    tokenizer,
    device: str,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}
    # Eval/no-grad/fp32 is a mode training never runs; tracing it would build a second graph per
    # block width purely to serve validation, and push the blocks toward dynamo's recompile limit.
    with torch.compiler.set_stance("force_eager"):
        for ds in dev_sets:
            metrics.update(_score_set(model, ds, tokenizer, device))
        metrics.update(_probe_batch(model, probe, device))
    model.train()
    return metrics
