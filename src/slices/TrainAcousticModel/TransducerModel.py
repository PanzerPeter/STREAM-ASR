# src/slices/TrainAcousticModel/TransducerModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch
from src.slices.ExtractFeatures.SpecAugmentBatch import apply_spec_augment_batch
from src.slices.TrainAcousticModel.StatelessPredictor import StatelessPredictor
from src.slices.TrainAcousticModel.TransducerJoiner import TransducerJoiner
from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder


class TransducerModel(nn.Module):
    """Single-pass streaming RNN-T: unchanged Zipformer encoder + stateless predictor + additive
    joiner, trained with rnnt + ctc_aux_weight*ctc + sum(interctc_weights*interctc_k). The aux CTC
    head doubles as a cheap greedy dev-WER probe; InterCTC taps regularise intermediate stacks."""

    def __init__(self, cmvn_path: str | None = "data/features/cmvn.pt") -> None:
        super().__init__()
        model = get_config().model
        t = get_config().transducer
        self.encoder = ZipformerEncoder(cmvn_path=cmvn_path)
        self.ctc_head = nn.Linear(self.encoder.output_dim, model.logits_width)
        self.interctc_layers = list(t.interctc_layers)
        self.interctc_weights = list(t.interctc_weights)
        self.interctc_heads = nn.ModuleList(
            [nn.Linear(model.encoder_dims[i], model.logits_width) for i in self.interctc_layers]
        )
        self.predictor = StatelessPredictor()
        self.joiner = TransducerJoiner()
        self._blank = model.blank_id
        self._ctc_aux_weight = t.ctc_aux_weight
        self._spec_augment = get_config().training.transducer.spec_augment
        # reduction="sum" (not "mean") so we can normalise per-token below. torchaudio's "mean"
        # divides only by batch size, yielding a per-utterance sum (~O(#tokens) ≈ 30), whereas
        # F.ctc_loss("mean") is per-token (~O(1)). Mixing the two silently down-weights the CTC/
        # InterCTC aux terms by ~1/avg_tokens relative to their nominal weights -- so once the RNN-T
        # gradient matures it overpowers the aux heads (SP5: InterCTC diverged, dev-WER regressed).
        # Per-token normalisation puts all three losses on one O(1) scale so the weights mean what
        # they say. (Muon+AdamW are ~scale-invariant, so this reweights gradients, not step size.)
        self._rnnt = torchaudio.transforms.RNNTLoss(blank=self._blank, reduction="sum")

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor, chunk_size: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
        memory, out_lengths, inters, base_lengths = self.encoder(
            features, lengths, chunk_size, return_intermediates=self.interctc_layers
        )
        ctc_logits = self.ctc_head(memory)
        interctc_logits = [head(x) for head, x in zip(self.interctc_heads, inters)]
        return memory, out_lengths, ctc_logits, interctc_logits, base_lengths

    def _ctc(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, V]
        targets = torch.cat([tokens[i, : token_lengths[i]] for i in range(tokens.shape[0])])
        return F.ctc_loss(
            log_probs, targets, lengths, token_lengths, blank=self._blank, zero_infinity=True
        )

    def ctc_loss(
        self,
        ctc_logits: torch.Tensor,
        out_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self._ctc(ctc_logits, out_lengths, tokens, token_lengths)

    def interctc_terms(
        self,
        interctc_logits: list[torch.Tensor],
        base_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        # Raw (unweighted) CTC per tap. Intermediate taps are at base rate; their CTC input lengths
        # are base_lengths. CTC is rate-agnostic, so mixing these 50Hz aux heads with the 25Hz main
        # head is fine. Kept raw so training logs can show each stack's actual CTC-decodability
        # (the weighted sum hid whether a climb was the encoder eroding or just the weight).
        return [
            self._ctc(logits, base_lengths, tokens, token_lengths) for logits in interctc_logits
        ]

    def interctc_loss(
        self,
        interctc_logits: list[torch.Tensor],
        base_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        terms = self.interctc_terms(interctc_logits, base_lengths, tokens, token_lengths)
        total = terms[0].new_zeros(())
        for w, term in zip(self.interctc_weights, terms):
            total = total + w * term
        return total

    def rnnt_loss(
        self,
        memory: torch.Tensor,
        out_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # Blank-prefixed prediction inputs -> predictor -> full joiner lattice [B, T, U+1, V].
        batch_size = tokens.shape[0]
        blanks = torch.full((batch_size, 1), self._blank, dtype=torch.long, device=tokens.device)
        pred_in = torch.cat([blanks, tokens], dim=1)  # [B, U+1]
        pred = self.predictor(pred_in)  # [B, U+1, Dp]
        # Training runs this under bf16 autocast, so the joiner's Linear emits bf16 logits; cast
        # to fp32 (not fp16, for RNN-T dynamic range) before the loss, since torchaudio's RNNTLoss
        # kernel only accepts float32/float16 and raises on bf16.
        logits = self.joiner(memory, pred).float()  # [B, T, U+1, V], fp32 for the RNN-T kernel
        loss_sum: torch.Tensor = self._rnnt(
            logits,
            tokens.int(),
            out_lengths.int(),
            token_lengths.int(),
        )
        # Per-token mean -> same O(1) scale as the CTC aux (see RNNTLoss note in __init__).
        return loss_sum / token_lengths.sum().clamp(min=1)

    def joint_loss(
        self, batch: FeatureBatch, chunk_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.ctc_head.weight.device
        feats = batch.features.to(device)
        flens = batch.feature_lengths.to(device)
        tokens = batch.tokens.to(device)
        tlens = batch.token_lengths.to(device)
        # SpecAugment is a train-only input regulariser; joint_loss is only ever the training path,
        # but gate on self.training anyway so an eval-mode caller can never mask its own inputs.
        if self._spec_augment and self.training:
            feats = apply_spec_augment_batch(feats, flens)
        memory, out_len, ctc_logits, interctc_logits, base_len = self.forward(
            feats, flens, chunk_size
        )
        rnnt = self.rnnt_loss(memory, out_len, tokens, tlens)
        ctc = self.ctc_loss(ctc_logits, out_len, tokens, tlens)
        ictc_terms = self.interctc_terms(interctc_logits, base_len, tokens, tlens)
        ictc_weighted = ictc_terms[0].new_zeros(())
        for w, term in zip(self.interctc_weights, ictc_terms):
            ictc_weighted = ictc_weighted + w * term
        total = rnnt + self._ctc_aux_weight * ctc + ictc_weighted
        # Return the RAW mean interctc (not the weighted sum) for logging -- it tracks the actual
        # CTC-decodability of the tapped stacks, the signal that flagged the SP5 encoder erosion.
        ictc_raw = torch.stack(ictc_terms).mean()
        return total, rnnt, ctc, ictc_raw
