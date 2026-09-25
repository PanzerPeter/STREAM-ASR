import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.RnntLoss import rnnt_loss
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
        # reduction="sum" (not "mean") so we can normalise per-token below. A per-utterance "mean"
        # would divide only by batch size, yielding a per-utterance sum (~O(#tokens) ≈ 30), whereas
        # F.ctc_loss("mean") is per-token (~O(1)). Mixing the two silently down-weights the CTC/
        # InterCTC aux terms by ~1/avg_tokens relative to their nominal weights -- so once the RNN-T
        # gradient matures it overpowers the aux heads (InterCTC diverges, dev-WER regresses).
        # Per-token normalisation puts all three losses on one O(1) scale so the weights mean what
        # they say. (Muon+AdamW are ~scale-invariant, so this reweights gradients, not step size.)
        self._rnnt_reduction = "sum"

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor, chunk_size: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
        memory, out_lengths, inters, base_lengths = self.encoder(
            features, lengths, chunk_size, return_intermediates=self.interctc_layers
        )
        ctc_logits = self.ctc_head(memory)
        interctc_logits = [head(x) for head, x in zip(self.interctc_heads, inters)]
        return memory, out_lengths, ctc_logits, interctc_logits, base_lengths

    def ctc_loss(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # Pass the already-padded [B, S] targets straight to F.ctc_loss (its 2-D target form):
        # entries past token_lengths[i] are ignored, so no flatten is needed. Concatenating the
        # rows instead would slice each with a GPU scalar bound, costing B device->host syncs per
        # CTC call -- and this path is called 3x a step (main + 2 InterCTC taps).
        return F.ctc_loss(
            F.log_softmax(logits, dim=-1).transpose(0, 1),  # [T, B, V]
            tokens,
            lengths,
            token_lengths,
            blank=self._blank,
            zero_infinity=True,
        )

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
            self.ctc_loss(logits, base_lengths, tokens, token_lengths) for logits in interctc_logits
        ]

    def rnnt_loss(
        self,
        memory: torch.Tensor,
        out_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # Blank-prefixed prediction inputs -> predictor -> joiner lattice.
        blanks = torch.full(
            (tokens.shape[0], 1), self._blank, dtype=torch.long, device=tokens.device
        )
        pred = self.predictor(torch.cat([blanks, tokens], dim=1))  # [B, U+1, Dp]
        # Training runs this under bf16 autocast, so the joiner's Linear emits bf16 logits. They go
        # to the loss as-is: it promotes to fp32 inside its own log-softmax, so the bf16 lattice is
        # the only [B, T, U+1, V] tensor that has to survive until backward. Materialising an fp32
        # copy here instead would double that, on the step's largest allocation.
        logits = self.joiner(memory, pred)  # [B, T, U+1, V]
        cost = rnnt_loss(
            logits,
            tokens.int(),
            out_lengths.int(),
            token_lengths.int(),
            blank=self._blank,
            reduction=self._rnnt_reduction,
        )
        # Per-token mean -> same O(1) scale as the CTC aux (see the reduction note in __init__).
        return cost / token_lengths.sum().clamp(min=1)

    def joint_loss(
        self, batch: FeatureBatch, chunk_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(total, rnnt, ctc, interctc)``; ``interctc`` is the RAW mean over the taps."""
        device = self.ctc_head.weight.device
        # non_blocking pairs with the loader's pin_memory: the four H2D copies are issued async and
        # overlap with the tail of the previous step instead of blocking the launch thread on each.
        feats = batch.features.to(device, non_blocking=True)
        flens = batch.feature_lengths.to(device, non_blocking=True)
        tokens = batch.tokens.to(device, non_blocking=True)
        tlens = batch.token_lengths.to(device, non_blocking=True)
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
        ictc = sum(w * term for w, term in zip(self.interctc_weights, ictc_terms))
        total = rnnt + self._ctc_aux_weight * ctc + ictc
        # Log the RAW mean interctc (not the weighted sum) -- it tracks the actual
        # CTC-decodability of the tapped stacks, the signal that flags encoder erosion.
        return total, rnnt, ctc, torch.stack(ictc_terms).mean()
