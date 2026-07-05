# src/slices/TrainAcousticModel/HybridModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.MaskUtils import make_pad_mask
from src.slices.ExtractFeatures.FeatureCollator import IGNORE_ID
from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch
from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder
from src.slices.TrainAcousticModel.AttentionDecoder import BiTransformerDecoder


class HybridCtcAttention(nn.Module):
    """M1 encoder + CTC head (streaming first pass) + bidirectional attention decoder (rescorer),
    trained with the U2++ joint loss. Encoder + CTC head are warm-started from Stage A."""

    def __init__(self, cmvn_path: str | None = "data/features/cmvn.pt") -> None:
        super().__init__()
        cfg = get_config().model
        self.encoder = ZipformerEncoder(cmvn_path=cmvn_path)
        self.ctc_head = nn.Linear(self.encoder.output_dim, cfg.logits_width)
        self.decoder = BiTransformerDecoder()

    def forward(self, features: torch.Tensor, lengths: torch.Tensor, chunk_size: int = 0):
        memory, out_lengths = self.encoder(features, lengths, chunk_size)
        ctc_logits = self.ctc_head(memory)
        return ctc_logits, memory, out_lengths

    def ctc_loss(self, ctc_logits, out_lengths, tokens, token_lengths) -> torch.Tensor:
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, V]
        targets = torch.cat([tokens[i, : token_lengths[i]] for i in range(tokens.shape[0])])
        return F.ctc_loss(
            log_probs,
            targets,
            out_lengths,
            token_lengths,
            blank=get_config().model.blank_id,
            zero_infinity=True,
        )

    def _ce(self, logits, target) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=IGNORE_ID,
            label_smoothing=get_config().training.stage_b.label_smoothing,
        )

    def attention_loss(self, memory, out_lengths, batch: FeatureBatch) -> torch.Tensor:
        rev_w = get_config().training.stage_b.reverse_weight
        mem_pad = make_pad_mask(out_lengths, memory.shape[1])
        ys_pad = make_pad_mask(batch.dec_lengths, batch.dec_in_l2r.shape[1])
        logits_l2r = self.decoder(memory, mem_pad, batch.dec_in_l2r, ys_pad, reverse=False)
        logits_r2l = self.decoder(memory, mem_pad, batch.dec_in_r2l, ys_pad, reverse=True)
        ce_l2r = self._ce(logits_l2r, batch.dec_out_l2r)
        ce_r2l = self._ce(logits_r2l, batch.dec_out_r2l)
        return (1 - rev_w) * ce_l2r + rev_w * ce_r2l

    def joint_loss(self, batch: FeatureBatch, chunk_size: int):
        sb = get_config().training.stage_b
        device = self.ctc_head.weight.device
        ctc_logits, memory, out_len = self.forward(
            batch.features.to(device), batch.feature_lengths.to(device), chunk_size
        )
        ctc = self.ctc_loss(
            ctc_logits, out_len, batch.tokens.to(device), batch.token_lengths.to(device)
        )
        attn = self.attention_loss(
            memory,
            out_len,
            FeatureBatch(
                batch.features,
                batch.feature_lengths,
                batch.tokens,
                batch.token_lengths,
                batch.dec_in_l2r.to(device),
                batch.dec_out_l2r.to(device),
                batch.dec_in_r2l.to(device),
                batch.dec_out_r2l.to(device),
                batch.dec_lengths.to(device),
            ),
        )
        total = sb.ctc_weight * ctc + (1 - sb.ctc_weight) * attn
        return total, ctc, attn
