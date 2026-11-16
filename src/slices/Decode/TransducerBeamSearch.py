# Single-pass RNN-T beam search over encoder memory, acoustic-only. The LM is NOT consulted here: it
# attaches as n-best rescoring in StreamingDecoder_Handler, which keeps corpus decode inside its GPU
# budget. Greedy decoding is the model's own label-looping search (TransducerModel.greedy_decode).
#
# Predictor-state contract (StatelessPredictor.step): step(state, token) -> (out, new_state), where
# `new_state` is already the context AFTER consuming `token`, so it is reused directly and never
# re-derived by stepping the just-emitted token a second time.
import math

import torch
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel

# (label ids, path log-prob, predictor state BEFORE `last`, last token id, emission frame per id).
_Hyp = tuple[tuple[int, ...], float, torch.Tensor, int, tuple[int, ...]]


def _logadd(a: float, b: float) -> float:
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def _recombine(hyps: list[_Hyp]) -> list[_Hyp]:
    # Equal label ids are the same partial output reached by different alignments: their mass is
    # summed (logadd) rather than split across beam slots. With the stateless predictor equal ids
    # imply an identical (state, last); the emission frames kept are the likelier alignment's.
    merged: dict[tuple[int, ...], _Hyp] = {}
    for hyp in hyps:
        prev = merged.get(hyp[0])
        if prev is None:
            merged[hyp[0]] = hyp
            continue
        keep = prev if prev[1] >= hyp[1] else hyp
        merged[hyp[0]] = (keep[0], _logadd(prev[1], hyp[1]), keep[2], keep[3], keep[4])
    return sorted(merged.values(), key=lambda h: h[1], reverse=True)


class TransducerBeamSearch:
    def __init__(self, model: TransducerModel, beam_size: int, max_symbols: int) -> None:
        self.model = model
        self.beam_size = beam_size
        self.max_symbols = max_symbols
        self.blank = get_config().model.blank_id

    @torch.no_grad()
    def search(self, memory: torch.Tensor) -> list[tuple[list[int], float, list[int]]]:
        """memory [1, T, De] -> n-best (ids, acoustic score, emission frames), best first.

        Graves A/B time-synchronous search: `active` hyps may still emit at frame t; a hyp that
        takes the blank is scored for it exactly once and retires to t+1 (keeping it in the live
        beam would charge an idling hyp up to max_symbols blanks per frame). The whole live beam is
        one batched predictor+joiner call per symbol step, with one host sync.
        """
        device = memory.device
        init_pred = self.model.predictor.init_state(1, device)[0]  # [context-1]
        active: list[_Hyp] = [((), 0.0, init_pred, self.blank, ())]
        for t in range(memory.shape[1]):
            enc_t = memory[:, t]  # [1, De], broadcasts over the batched predictor outputs
            blanked: list[_Hyp] = []
            for _ in range(self.max_symbols):
                if not active:
                    break
                states = torch.stack([h[2] for h in active])
                lasts = torch.tensor([h[3] for h in active], dtype=torch.long, device=device)
                pred_out, new_states = self.model.predictor.step(states, lasts)
                logp = F.log_softmax(self.model.joiner.step(enc_t, pred_out), dim=-1)  # [n, V]
                k = min(self.beam_size, logp.shape[-1])
                top_lp, top_tok = torch.topk(logp, k, dim=-1)
                # The blank column, the top-k log-probs and the top-k ids leave in ONE copy; ids
                # through fp32 are exact for a 501-symbol vocabulary.
                packed = torch.cat(
                    [logp[:, self.blank : self.blank + 1].float(), top_lp.float(), top_tok.float()],
                    dim=1,
                ).tolist()
                emitted: list[_Hyp] = []
                for i, (ids, score, state, last, frames) in enumerate(active):
                    row = packed[i]
                    blanked.append((ids, score + row[0], state, last, frames))
                    for lp, tok_f in zip(row[1 : 1 + k], row[1 + k :]):
                        tok = int(tok_f)
                        if tok != self.blank:
                            emitted.append(
                                (ids + (tok,), score + lp, new_states[i], tok, frames + (t,))
                            )
                active = _recombine(emitted)[: self.beam_size]
            # Hyps that hit max_symbols still advance to t+1 (forced step, no blank cost).
            active = _recombine(blanked + active)[: self.beam_size]
        return [(list(h[0]), h[1], list(h[4])) for h in _recombine(active)]
