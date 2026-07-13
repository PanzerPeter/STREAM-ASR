# BEST-RQ pretraining head over the existing ZipformerEncoder (SP4). The encoder sees span-masked
# log-mel; the frozen quantizer labels the CLEAN log-mel (CMVN-normalized, then frame-stacked to the
# encoder's ~25 Hz output grid). Cross-entropy is computed only on masked, valid output positions.
# Warm-start later loads encoder.* into AcousticModel and discards pred_head.
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.RandomProjectionQuantizer import RandomProjectionQuantizer
from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder
from src.slices.PretrainEncoder.BestRqMask import apply_span_mask


def stack_frames(mel: torch.Tensor, stack: int) -> torch.Tensor:
    b, t, f = mel.shape
    t2 = t // stack
    return mel[:, : t2 * stack].reshape(b, t2, stack * f)


def _net_subsample() -> int:
    # frontend x2 * final_downsample = encoder net subsampling; quantizer target grid must match.
    return 2 * get_config().model.final_downsample


class BestRqModel(nn.Module):
    def __init__(self, cmvn_path: str | None = "data/features/cmvn.pt") -> None:
        super().__init__()
        p = get_config().pretrain
        self.encoder = ZipformerEncoder(cmvn_path=cmvn_path)
        self.stack = p.stack_frames or _net_subsample()
        n_mels = get_config().audio.n_mels
        self.quantizer = RandomProjectionQuantizer(
            in_dim=n_mels * self.stack,
            codebook_size=p.codebook_size,
            codebook_dim=p.codebook_dim,
            seed=p.seed,
        )
        self.pred_head = nn.Linear(self.encoder.output_dim, p.codebook_size)
        self._mask_prob = p.mask_prob
        self._mask_span = p.mask_span
        self._noise_std = p.noise_std

    def forward(self, mel: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        masked, mask = apply_span_mask(
            mel, lengths, self._mask_prob, self._mask_span, self._noise_std
        )
        enc, out_lengths = self.encoder(masked, lengths)  # [B, Tenc, D]
        logits = self.pred_head(enc)  # [B, Tenc, K]

        # Targets from CLEAN, CMVN-normalized mel on the encoder's output grid.
        mel_n = (mel - self.encoder.cmvn_mean) / self.encoder.cmvn_std
        targets = self.quantizer(stack_frames(mel_n, self.stack))  # [B, Tstack]

        # A stacked target position is "masked" if any of its stacked input frames were masked.
        t2 = targets.shape[1]
        tgt_mask = mask[:, : t2 * self.stack].reshape(mask.shape[0], t2, self.stack).any(dim=-1)

        # Align encoder-output and target lengths (they should match to within one frame).
        length = min(logits.shape[1], t2)
        logits = logits[:, :length]
        targets = targets[:, :length]
        tgt_mask = tgt_mask[:, :length].clone()
        valid = torch.arange(length, device=mel.device)[None, :] < out_lengths[:, None]
        select = tgt_mask & valid
        if not select.any():
            return logits.sum() * 0.0  # degenerate tiny batch: no masked positions
        return F.cross_entropy(logits[select], targets[select])
