from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.RnntLoss import occupancy_band, rnnt_loss
from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch
from src.slices.ExtractFeatures.SpecAugmentBatch import apply_spec_augment_batch
from src.slices.TrainAcousticModel.DwSepSubsampling import FRONTEND_STRIDE
from src.slices.TrainAcousticModel.S2Encoder import S2Encoder
from src.slices.TrainAcousticModel.StatelessPredictor import StatelessPredictor
from src.slices.TrainAcousticModel.TransducerJoiner import TransducerJoiner

# Output frames the label-looping search scans per joiner call while a hypothesis emits nothing.
_LOOP_WINDOW = 16


@dataclass(frozen=True)
class LossTerms:
    """Every term is per TOKEN (MODEL_ARCHITECTURE.md §6.4); tensors stay on device."""

    total: torch.Tensor
    rnnt: torch.Tensor
    ctc: torch.Tensor
    band_scale: torch.Tensor  # median c(T-1)/U before the rescale; nan while the band is off
    band_infeasible: torch.Tensor  # utterances whose band misses a terminal; must stay 0


class TransducerModel(nn.Module):
    """S2 encoder + CTC head + stateless predictor + additive joiner, trained jointly with
    `rnnt + ctc_loss_scale * ctc` on a CTC-banded lattice."""

    def __init__(self, cmvn_path: str | None = "data/features/cmvn.pt") -> None:
        super().__init__()
        cfg = get_config()
        t = cfg.transducer
        self.encoder = S2Encoder(cmvn_path=cmvn_path)
        self.ctc_head = nn.Linear(self.encoder.output_dim, cfg.model.logits_width)
        self.predictor = StatelessPredictor()
        self.joiner = TransducerJoiner(self.encoder.output_dim)
        self._blank = cfg.model.blank_id
        self._ctc_scale = t.ctc_loss_scale
        self._band_width = t.band_width
        self._spec_augment = cfg.training.transducer.spec_augment

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        chunk_size: int = 0,
        left_context: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, out_lengths = self.encoder(features, lengths, chunk_size, left_context)
        return memory, out_lengths, self.ctc_head(memory)

    def transducer_cost(
        self,
        memory: torch.Tensor,
        out_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        band_lo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-utterance -log p(y|x) [B] on the full lattice, or on the band starting at band_lo."""
        blanks = tokens.new_full((tokens.shape[0], 1), self._blank)
        pred = self.predictor(torch.cat([blanks, tokens], dim=1))  # [B, U+1, Dp]
        cols = None
        if band_lo is not None:
            width = min(self._band_width + 1, tokens.shape[1] + 1)
            cols = band_lo.unsqueeze(2) + torch.arange(width, device=band_lo.device)
        # bf16 logits go to the loss as-is: it promotes per slab, so the bf16 lattice is the only
        # lattice-sized tensor alive until backward.
        logits = self.joiner(memory, pred, cols)
        return rnnt_loss(
            logits,
            tokens.int(),
            out_lengths.int(),
            token_lengths.int(),
            blank=self._blank,
            reduction="none",
            band_lo=band_lo,
        )

    def band(
        self, ctc_log_probs: torch.Tensor, out_lengths: torch.Tensor, token_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The CTC head that sets the band is DETACHED from it (MODEL_ARCHITECTURE.md §11).
        lo, _, scale, infeasible = occupancy_band(
            ctc_log_probs[..., self._blank].detach(), out_lengths, token_lengths, self._band_width
        )
        return lo, scale, infeasible

    def joint_loss(
        self, batch: FeatureBatch, chunk_size: int, left_context: int, use_band: bool
    ) -> LossTerms:
        device = self.ctc_head.weight.device
        # non_blocking pairs with the loader's pin_memory, so the copies overlap the previous step.
        feats = batch.features.to(device, non_blocking=True)
        flens = batch.feature_lengths.to(device, non_blocking=True)
        tokens = batch.tokens.to(device, non_blocking=True)
        tlens = batch.token_lengths.to(device, non_blocking=True)
        if self._spec_augment and self.training:
            feats = apply_spec_augment_batch(feats, flens, self.encoder.cmvn_mean)
        memory, out_len, logits = self(feats, flens, chunk_size, left_context)
        n_tok = tlens.sum().clamp(min=1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        ctc = (
            F.ctc_loss(
                log_probs.transpose(0, 1),
                tokens,
                out_len,
                tlens,
                blank=self._blank,
                reduction="sum",
                zero_infinity=True,
            )
            / n_tok
        )
        nan = log_probs.new_full((), float("nan"))
        band_lo, scale, infeasible = None, nan, log_probs.new_zeros(())
        if use_band and self._band_width > 0:
            band_lo, scales, flags = self.band(log_probs, out_len, tlens)
            scale, infeasible = scales.median(), flags.sum().float()
        rnnt = self.transducer_cost(memory, out_len, tokens, tlens, band_lo).sum() / n_tok
        total = rnnt + self._ctc_scale * ctc
        return LossTerms(total, rnnt, ctc, scale, infeasible)

    @torch.no_grad()
    def greedy_decode(
        self, memory: torch.Tensor, lengths: torch.Tensor, max_symbols: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Label-looping greedy (Bataev et al. 2024), batched: (token ids, emission frame) per row.

        While a hypothesis emits blanks its predictor state does not move, so the joiner is
        evaluated on the next `_LOOP_WINDOW` frames at once against that one state and the search
        jumps to the first frame whose argmax is non-blank. That is exactly frame-by-frame greedy
        (`test_label_looping_equals_framewise`) with one joiner call per emitted token instead of
        one per frame. A frame that has emitted `max_symbols` tokens is left, as frame-wise does.
        """
        batch, num_frames, _ = memory.shape
        device = memory.device
        lengths = lengths.to(device)
        enc = self.joiner.enc_proj(memory)  # [B, T, J]
        state = self.predictor.init_state(batch, device)
        prev = torch.full((batch,), self._blank, dtype=torch.long, device=device)
        out, next_state = self.predictor.step(state, prev)
        pred = self.joiner.pred_proj(out)
        t = torch.zeros(batch, dtype=torch.long, device=device)
        symbols = torch.zeros_like(t)
        offsets = torch.arange(_LOOP_WINDOW, device=device)
        ids: list[list[int]] = [[] for _ in range(batch)]
        frames: list[list[int]] = [[] for _ in range(batch)]
        while True:
            active = t < lengths
            idx = t.unsqueeze(1) + offsets  # [B, W]
            in_range = idx < lengths.unsqueeze(1)
            cells = enc.gather(
                1, idx.clamp(max=num_frames - 1).unsqueeze(2).expand(-1, -1, enc.shape[2])
            )
            best = self.joiner.out(torch.tanh(cells + pred.unsqueeze(1))).argmax(dim=-1)  # [B, W]
            non_blank = (best != self._blank) & in_range & active.unsqueeze(1)
            first = non_blank.int().argmax(dim=1)
            emit = non_blank.any(dim=1)
            at = t + first
            tok = best.gather(1, first.unsqueeze(1)).squeeze(1)
            packed = torch.stack([active.long(), emit.long(), tok, at]).tolist()  # one sync
            if not any(packed[0]):
                break
            for lane in range(batch):
                if packed[1][lane]:
                    ids[lane].append(packed[2][lane])
                    frames[lane].append(packed[3][lane])
            symbols = torch.where(emit, torch.where(at == t, symbols, 0) + 1, symbols)
            t = torch.where(emit, at, torch.where(active, t + _LOOP_WINDOW, t))
            leave = emit & (symbols >= max_symbols)
            t = torch.where(leave, t + 1, t)
            symbols = torch.where(leave, 0, symbols)
            state = torch.where(emit.unsqueeze(1), next_state, state)
            prev = torch.where(emit, tok, prev)
            out, stepped = self.predictor.step(state, prev)
            pred = torch.where(emit.unsqueeze(1), self.joiner.pred_proj(out), pred)
            next_state = torch.where(emit.unsqueeze(1), stepped, next_state)
        return ids, frames

    def decode_window(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        chunk_size: int,
        left_context: int,
        tail_chunks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched memory equal to what streaming decode produces: the final partial chunk is
        completed and `tail_chunks` whole chunks of mean fill follow, all of their frames kept.
        chunk_size 0 = offline (no tail). Returns (memory, kept lengths)."""
        if chunk_size <= 0:
            memory, out_len = self.encoder(features, lengths)
            return memory, out_len
        mel_chunk = chunk_size * FRONTEND_STRIDE
        extra = mel_chunk * (tail_chunks + 1)
        memory, _ = self.encoder(
            F.pad(features, (0, 0, 0, extra)), lengths, chunk_size, left_context
        )
        per_chunk = mel_chunk // self.encoder.mel_frames_per_output
        chunks = -(-lengths // mel_chunk) + tail_chunks
        return memory, (chunks * per_chunk).clamp(max=memory.shape[1])
