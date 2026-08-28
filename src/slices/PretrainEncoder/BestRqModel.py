# BEST-RQ pretraining head over the ZipformerEncoder. The encoder sees span-masked log-mel; frozen
# quantizers label the CLEAN log-mel (CMVN-normalized, then frame-stacked to the encoder's ~25 Hz
# output grid). Cross-entropy is computed only on masked, valid output positions.
# Warm-start later loads encoder.* into TransducerModel and discards pred_head.
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
    """Encoder + N independent random-projection quantizers + one N-wide prediction head.

    N > 1 is USM's multi-softmax. Each quantizer is a different frozen random draw, each gets its
    own contiguous slice of the head's readout, and the loss is the mean of their cross-entropies
    -- which is the mean of N samples of a target distribution whose quality is otherwise decided
    by a single seed. MEASURED across 20 seeds on 300 dev utterances, target code entropy at
    K=8192, D=16 spans 7.64 to 8.90 bits; `config/pretrain.yaml`'s seed 42 draws 7.93.

    The head and the quantizers are both evaluated ONLY on the positions that enter the loss. Both
    are 8192-wide readouts and ~55 % of the grid is unmasked, so computing them on the full grid
    and then selecting -- what this did before -- threw away more than half of the two largest
    GEMMs in the stage. Selecting first is also what makes N=4 affordable.
    """

    def __init__(self, cmvn_path: str | None = "data/features/cmvn.pt") -> None:
        super().__init__()
        p = get_config().pretrain
        self.encoder = ZipformerEncoder(cmvn_path=cmvn_path)
        self.stack = p.stack_frames or _net_subsample()
        n_mels = get_config().audio.n_mels
        self.num_codebooks = p.num_codebooks
        self.codebook_size = p.codebook_size
        self.quantizers = nn.ModuleList(
            [
                RandomProjectionQuantizer(
                    in_dim=n_mels * self.stack,
                    codebook_size=p.codebook_size,
                    codebook_dim=p.codebook_dim,
                    seed=p.seed + i,
                )
                for i in range(p.num_codebooks)
            ]
        )
        self.pred_head = nn.Linear(self.encoder.output_dim, p.num_codebooks * p.codebook_size)
        self._mask_prob = p.mask_prob
        self._mask_span = p.mask_span
        self._noise_std = p.noise_std

    def forward(
        self, mel: torch.Tensor, lengths: torch.Tensor, chunk_size: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked, mask = apply_span_mask(
            mel,
            lengths,
            self._mask_prob,
            self._mask_span,
            self._noise_std,
            self.encoder.cmvn_mean,
            self.encoder.cmvn_std,
        )
        enc, out_lengths = self.encoder(masked, lengths, chunk_size)  # [B, Tenc, D]

        # Targets from CLEAN, CMVN-normalized mel on the encoder's output grid.
        mel_n = (mel - self.encoder.cmvn_mean) / self.encoder.cmvn_std
        stacked = stack_frames(mel_n, self.stack)  # [B, Tstack, stack*n_mels]

        # A stacked target position is "masked" if any of its stacked input frames were masked.
        t2 = stacked.shape[1]
        tgt_mask = mask[:, : t2 * self.stack].reshape(mask.shape[0], t2, self.stack).any(dim=-1)

        # Align encoder-output and target lengths (they should match to within one frame).
        length = min(enc.shape[1], t2)
        valid = torch.arange(length, device=mel.device)[None, :] < out_lengths[:, None]
        select = tgt_mask[:, :length] & valid

        # Boolean indexing syncs on the selected count either way, so this is the step's one stall
        # and the shape is on the host afterwards -- cheaper than the `select.any()` probe it
        # replaces, which paid the same stall for strictly less information.
        enc_sel = enc[:, :length][select]  # [S, D]
        if enc_sel.shape[0] == 0:  # degenerate tiny batch: no masked positions
            return enc.sum() * 0.0, enc.new_zeros(())
        feat_sel = stacked[:, :length][select]  # [S, stack*n_mels]

        # Labels in fp32 regardless of the caller's autocast: the quantizer is an argmax over 8192
        # near-ties, so a bf16 projection would resolve some of them differently from one step to
        # the next and feed the encoder label noise for free. It is 0.3 GFLOP, and frozen.
        with torch.autocast(device_type=feat_sel.device.type, enabled=False):
            targets = torch.stack([q(feat_sel.float()) for q in self.quantizers], dim=1)  # [S, N]

        logits = self.pred_head(enc_sel).view(-1, self.num_codebooks, self.codebook_size)
        accuracy = (logits.argmax(dim=-1) == targets).to(enc.dtype).mean()
        loss = F.cross_entropy(logits.reshape(-1, self.codebook_size), targets.reshape(-1))
        return loss, accuracy
