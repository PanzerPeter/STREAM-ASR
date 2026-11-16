# Where streaming errors and delays sit inside an utterance (MODEL_ARCHITECTURE.md §8.3): the
# first-2 / middle / last-2 word error rates with their deletion share, and per-word emission
# latency against a CTC forced alignment of the reference. Both read jiwer's word alignment.
from dataclasses import dataclass, field
from typing import Protocol

import jiwer
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from src.shared_kernel.LogMel_Transform import compute_log_mel

# Words at each end of an utterance counted as its "edge".
EDGE_WORDS = 2
_REGIONS = ("first", "middle", "last")


class _WordTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def is_word_start(self, token_id: int) -> bool: ...


@dataclass
class EdgeCounts:
    words: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_REGIONS, 0))
    errors: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_REGIONS, 0))
    deletions: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_REGIONS, 0))

    def add(self, ref: str, hyp: str) -> None:
        n = len(ref.split())
        if n == 0:
            return
        for chunk in jiwer.process_words(ref, hyp).alignments[0]:
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                region = _region(i, n)
                self.words[region] += 1
                if chunk.type != "equal":
                    self.errors[region] += 1
                if chunk.type == "delete":
                    self.deletions[region] += 1

    def rates(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in _REGIONS:
            out[f"{r}_err"] = self.errors[r] / max(1, self.words[r])
            out[f"{r}_del"] = self.deletions[r] / max(1, self.words[r])
        return out


def _region(i: int, n: int) -> str:
    if i < EDGE_WORDS:
        return "first"
    return "last" if i >= n - EDGE_WORDS else "middle"


def _word_groups(token_ids: list[int], tok: _WordTokenizer) -> list[list[int]]:
    # Token positions of each word: a word starts at every U+2581-prefixed piece.
    groups: list[list[int]] = []
    for pos, tid in enumerate(token_ids):
        if tok.is_word_start(tid) or not groups:
            groups.append([])
        groups[-1].append(pos)
    return groups


@torch.no_grad()
def reference_word_ends(
    model, tok: _WordTokenizer, wave: torch.Tensor, text: str, frame_s: float
) -> list[float] | None:
    """End time (s) of each reference word in a CTC forced alignment on the offline encoder, or
    None when the alignment is infeasible (more tokens than frames)."""
    device = model.ctc_head.weight.device
    feats = compute_log_mel(wave).unsqueeze(0).to(device)
    memory, out_len = model.encoder(feats, torch.tensor([feats.shape[1]], device=device))
    log_probs = F.log_softmax(model.ctc_head(memory[:, : int(out_len[0])]).float(), dim=-1)
    ids = tok.encode(text)
    if not ids or len(ids) > log_probs.shape[1]:
        return None
    blank = log_probs.shape[-1] - 1
    targets = torch.tensor([ids], dtype=torch.int32, device=device)
    try:
        labels, scores = AF.forced_align(log_probs, targets, blank=blank)
    except RuntimeError:
        return None
    spans = AF.merge_tokens(labels[0], scores[0].exp(), blank=blank)
    return [spans[g[-1]].end * frame_s for g in _word_groups(ids, tok)]


def emission_latencies(
    ref: str,
    hyp: str,
    ref_ends: list[float],
    hyp_ids: list[int],
    hyp_emit_s: list[float],
    tok: _WordTokenizer,
) -> list[float]:
    """Per correctly recognised word: when its first token was shown minus when the word ended."""
    groups = _word_groups(hyp_ids, tok)
    if len(groups) != len(hyp.split()) or len(ref_ends) != len(ref.split()):
        return []
    out: list[float] = []
    for chunk in jiwer.process_words(ref, hyp).alignments[0]:
        if chunk.type != "equal":
            continue
        for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
            first_token = groups[chunk.hyp_start_idx + k][0]
            out.append(hyp_emit_s[first_token] - ref_ends[chunk.ref_start_idx + k])
    return out
