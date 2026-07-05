# src/slices/TrainAcousticModel/AcousticModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder


class AcousticModel(nn.Module):
    def __init__(
        self, vocab_size: int | None = None, cmvn_path: str | None = "data/features/cmvn.pt"
    ) -> None:
        super().__init__()
        if vocab_size is None:
            vocab_size = get_config().model.vocab_size
        self.encoder = ZipformerEncoder(cmvn_path=cmvn_path)
        self.ctc_head = nn.Linear(self.encoder.output_dim, vocab_size + 1)  # +1 for CTC blank

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        encoded, out_lengths = self.encoder(features, lengths)
        logits = self.ctc_head(encoded)  # [B, T//4, vocab_size+1]
        return logits, out_lengths

    def ctc_loss(self, logits, out_lengths, tokens, token_lengths) -> torch.Tensor:
        # CTCLoss wants [T, B, V] log-probs and flattened, blank-free targets.
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, V]
        targets = torch.cat([tokens[i, : token_lengths[i]] for i in range(tokens.shape[0])])
        return F.ctc_loss(
            log_probs,
            targets,
            out_lengths,
            token_lengths,
            blank=get_config().model.blank_id,
            zero_infinity=True,
        )
