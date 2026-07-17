# src/slices/TrainAcousticModel/StatelessPredictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.BiasNorm import BiasNorm
from src.shared_kernel.Config_Adapter import get_config


class StatelessPredictor(nn.Module):
    """icefall-style stateless RNN-T predictor: embed the prediction-input token, then a depthwise
    Conv1d over a small left context. No recurrence -> streaming state is just the last
    `context-1` token ids, and the module is trivially causal. Output is BiasNorm-normalised to
    match the encoder's normalisation idiom before the joiner."""

    def __init__(self) -> None:
        super().__init__()
        t = get_config().transducer
        self.context = t.predictor_context
        self.output_dim = t.predictor_dim
        self.blank_id = get_config().model.blank_id
        # Embedding table covers the acoustic vocab plus the shared blank id (used as the sequence
        # start symbol), so it is vocab_size + 1 wide -> ids 0..blank_id are all valid.
        self.embed = nn.Embedding(get_config().model.logits_width, t.predictor_dim)
        # Depthwise causal conv over `context` frames. groups=channels keeps it cheap and per-dim.
        self.conv = nn.Conv1d(
            t.predictor_dim, t.predictor_dim, kernel_size=self.context, groups=t.predictor_dim
        )
        self.norm = BiasNorm(t.predictor_dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        # labels [B, U'] long (blank-prefixed prediction inputs) -> [B, U', D].
        # Pad in token-id space with blank_id (not zero-pad the embeddings) so the implicit
        # history for the first `context-1` outputs matches init_state's blank_id fill -- this
        # is what makes step-by-step streaming bit-equal to this batched forward.
        padded = F.pad(labels, (self.context - 1, 0), value=self.blank_id)
        emb = self.embed(padded).transpose(1, 2)  # [B, D, U'+context-1]
        out = self.conv(emb).transpose(1, 2)  # [B, U', D]
        return self.norm(out)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (batch_size, self.context - 1), self.blank_id, dtype=torch.long, device=device
        )

    def step(self, state: torch.Tensor, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # state [B, context-1] previous ids, token [B] newest id -> (out [B, D], new_state).
        window = torch.cat([state, token.unsqueeze(1)], dim=1)  # [B, context]
        emb = self.embed(window).transpose(1, 2)  # [B, D, context]
        out = self.conv(emb).transpose(1, 2)[:, -1]  # [B, D] (single valid position)
        new_state = window[:, 1:] if self.context > 1 else state
        return self.norm(out), new_state
