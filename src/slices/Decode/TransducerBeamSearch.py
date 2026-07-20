# src/slices/Decode/TransducerBeamSearch.py
# Single-pass RNN-T decode over encoder memory, acoustic-only. Greedy for live partials; a small
# time-synchronous beam (per-frame non-blank expansion, capped at max_symbols) for the final. The
# LM is NOT consulted here -- shallow fusion was replaced by n-best rescoring in
# StreamingDecoder_Handler (score each final hypothesis once with the LM, re-rank by
# acoustic + alpha*lm), which is what keeps corpus decode inside its GPU budget. So this searcher is
# pure acoustic and lm_scorer=None-equivalent by construction.
#
# Predictor-state contract (StatelessPredictor.step): step(state, token) -> (out, new_state), where
# `new_state` is already the context AFTER consuming `token`. That is exactly what the NEXT call
# needs, so it must be reused directly -- never re-derived by calling step a second time with the
# just-emitted token (that would duplicate the emitted token into its own context window). Both
# greedy and search below make exactly ONE predictor.step call per hypothesis per emission attempt.
import math

import torch
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel

# A search hypothesis: (label ids, path log-prob, predictor state BEFORE `last`, last token id).
_Hyp = tuple[tuple[int, ...], float, torch.Tensor, int]


def _logadd(a: float, b: float) -> float:
    # log(exp(a)+exp(b)) in a stable form -- marginalises two alignments of the SAME label sequence.
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def _recombine(hyps: list[_Hyp]) -> list[_Hyp]:
    # RNN-T hypothesis recombination: two hyps with identical label ids are the same partial output
    # reached by different blank/emit alignments, so their probabilities must be *summed* (logadd),
    # not left to compete for separate beam slots (splitting mass, which can prune the true best).
    # For the stateless predictor, equal ids imply an identical (state, last), so the representative
    # kept here is exact. Returns best-first.
    merged: dict[tuple[int, ...], _Hyp] = {}
    for ids, score, state, last in hyps:
        prev = merged.get(ids)
        if prev is None:
            merged[ids] = (ids, score, state, last)
        else:
            merged[ids] = (ids, _logadd(prev[1], score), state, last)
    return sorted(merged.values(), key=lambda h: h[1], reverse=True)


class TransducerBeamSearch:
    def __init__(self, model: TransducerModel, beam_size: int, max_symbols: int) -> None:
        self.model = model
        self.beam_size = beam_size
        self.max_symbols = max_symbols
        self.blank = get_config().model.blank_id

    @torch.no_grad()
    def greedy(self, memory: torch.Tensor) -> list[int]:
        # memory [1, T, De] -> token ids. Mirrors greedy_transducer_decode (Task 8) exactly.
        device = memory.device
        state = self.model.predictor.init_state(1, device)
        prev = torch.full((1,), self.blank, dtype=torch.long, device=device)
        ids: list[int] = []
        for t in range(memory.shape[1]):
            enc_t = memory[:, t]  # [1, De]
            emitted = 0
            while emitted < self.max_symbols:
                # ONE step call; `new_state` is the context after `prev` -- reuse it verbatim.
                pred_out, new_state = self.model.predictor.step(state, prev)
                logits = self.model.joiner.step(enc_t, pred_out)  # [1, V]
                tok = int(logits.argmax(dim=-1))
                if tok == self.blank:
                    break
                ids.append(tok)
                state = new_state
                prev = torch.full((1,), tok, dtype=torch.long, device=device)
                emitted += 1
        return ids

    @torch.no_grad()
    def search(self, memory: torch.Tensor) -> list[tuple[list[int], float]]:
        # memory [1, T, De] -> n-best (ids, acoustic score) best-first, time-synchronous RNN-T beam.
        # Every live hypothesis' predictor+joiner is evaluated in ONE batched call per symbol step
        # (batch dim = live beam width), so a whole frame costs a handful of GPU launches + a single
        # host sync instead of one per hypothesis.
        #
        # Structure (Graves-style A/B time-synchronous search):
        #   `active`  -- hyps that may still EMIT another symbol at the current frame.
        #   `blanked` -- hyps that took the blank at this frame; blank is scored EXACTLY ONCE, then
        #                the hyp advances to t+1 and is NOT re-expanded here. (The previous
        #                implementation kept blanked hyps in the live beam, so a hyp that idled
        #                accrued up to max_symbols blank log-probs at one frame -- a systematic
        #                over-penalty biasing the search; this A/B split removes it.)
        # `_recombine` merges equal-prefix hyps by logadd at every prune so probability mass is
        # summed, not split across duplicate beam slots.
        #
        # Each hypothesis: (ids, score, predictor state BEFORE `last` [context-1], last token id).
        # `state` is the context that must precede `last` in the next predictor.step call.
        device = memory.device
        init_pred = self.model.predictor.init_state(1, device)[0]  # [context-1]
        active: list[_Hyp] = [((), 0.0, init_pred, self.blank)]
        for t in range(memory.shape[1]):
            enc_t = memory[:, t]  # [1, De] (broadcasts over the batched predictor outputs)
            blanked: list[_Hyp] = []
            for _ in range(self.max_symbols):
                if not active:
                    break
                states = torch.stack([h[2] for h in active])  # [n, context-1]
                lasts = torch.tensor([h[3] for h in active], dtype=torch.long, device=device)
                pred_out, new_states = self.model.predictor.step(
                    states, lasts
                )  # [n, D], [n, ctx-1]
                logp = F.log_softmax(self.model.joiner.step(enc_t, pred_out), dim=-1)  # [n, V]
                blank_lp = logp[:, self.blank].tolist()  # [n]
                topk = torch.topk(logp, min(self.beam_size, logp.shape[-1]), dim=-1)
                top_lp, top_tok = topk.values.tolist(), topk.indices.tolist()  # [n, k] each
                emitted: list[_Hyp] = []
                for i, (ids, score, state, last) in enumerate(active):
                    # Blank: score once, retire to `blanked` (predictor context unchanged -- blank
                    # is never fed to the predictor, so state/last carry over verbatim).
                    blanked.append((ids, score + blank_lp[i], state, last))
                    # Child state = new_states[i] (context AFTER `last`); NO second step call.
                    child_state = new_states[i]
                    for lp, tok in zip(top_lp[i], top_tok[i]):
                        if tok == self.blank:
                            continue
                        emitted.append((ids + (tok,), score + lp, child_state, tok))
                active = _recombine(emitted)[: self.beam_size]
            # Hyps that hit max_symbols without blanking still advance to t+1 (forced time step, no
            # extra blank cost); fold them in with the blanked set and prune for the next frame.
            active = _recombine(blanked + active)[: self.beam_size]
        return [(list(ids), score) for ids, score, _, _ in _recombine(active)]
