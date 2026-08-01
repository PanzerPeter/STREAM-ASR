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
        # CR-CTC: two SpecAugment views + a consistency KL on the CTC head. Train-only regulariser.
        self._cr_ctc = t.cr_ctc
        self._cr_weight = t.cr_weight
        if self._cr_ctc and not self._spec_augment:
            # CR-CTC's signal *is* the disagreement between two independently masked views. With
            # masking off both views are the same tensor, the KL is identically zero, and the run
            # would silently pay for a second encoder forward that teaches nothing.
            raise ValueError(
                "transducer.cr_ctc requires training.transducer.spec_augment: the consistency "
                "term is defined between two SpecAugment views and degenerates to 0 without it"
            )
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

    def _ctc(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self._ctc_from_log_probs(
            F.log_softmax(logits, dim=-1), lengths, tokens, token_lengths
        )

    def _ctc_from_log_probs(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # Pass the already-padded [B, S] targets straight to F.ctc_loss (its 2-D target form):
        # entries past token_lengths[i] are ignored, so no flatten is needed. Concatenating the
        # rows instead would slice each with a GPU scalar bound, costing B device->host syncs per
        # CTC call -- and this path is called 3x a step (main + 2 InterCTC taps), 6x under CR-CTC.
        return F.ctc_loss(
            log_probs.transpose(0, 1),  # [T, B, V]
            tokens,
            lengths,
            token_lengths,
            blank=self._blank,
            zero_infinity=True,
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
        # Training runs this under bf16 autocast, so the joiner's Linear emits bf16 logits. They go
        # to the loss as-is: it promotes to fp32 inside its own log-softmax, so the bf16 lattice is
        # the only [B, T, U+1, V] tensor that has to survive until backward. Materialising an fp32
        # copy here instead would double that, on the step's largest allocation.
        logits = self.joiner(memory, pred)  # [B, T, U+1, V]
        loss_sum: torch.Tensor = rnnt_loss(
            logits,
            tokens.int(),
            out_lengths.int(),
            token_lengths.int(),
            blank=self._blank,
            reduction=self._rnnt_reduction,
        )
        # Per-token mean -> same O(1) scale as the CTC aux (see RNNTLoss note in __init__).
        return loss_sum / token_lengths.sum().clamp(min=1)

    def _interctc_weighted(self, ictc_terms: list[torch.Tensor]) -> torch.Tensor:
        total = ictc_terms[0].new_zeros(())
        for w, term in zip(self.interctc_weights, ictc_terms):
            total = total + w * term
        return total

    def _cr_consistency(
        self, log_pa: torch.Tensor, log_pb: torch.Tensor, out_lengths: torch.Tensor
    ) -> torch.Tensor:
        # Bidirectional KL between the two views' frame posteriors (R-Drop / CR-CTC), no stop-grad
        # so each view is pulled toward the other. Both views share the input length, hence the same
        # encoder output frames, so KL is frame-aligned. Masked to valid frames and mean-reduced,
        # giving the same O(1) per-frame scale as the CTC aux, so cr_weight means what it says.
        # Takes the two views' log-probs (already computed for the CTC terms) rather than raw
        # logits, so the caller's two log_softmax calls are reused instead of recomputed here.
        kl_ab = (log_pa.exp() * (log_pa - log_pb)).sum(-1)  # [B, T]
        kl_ba = (log_pb.exp() * (log_pb - log_pa)).sum(-1)
        per_frame = 0.5 * (kl_ab + kl_ba)
        idx = torch.arange(log_pa.shape[1], device=log_pa.device)
        mask = idx.unsqueeze(0) < out_lengths.to(idx.device).unsqueeze(1)
        return (per_frame * mask).sum() / mask.sum().clamp(min=1)

    def joint_loss(
        self, batch: FeatureBatch, chunk_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.ctc_head.weight.device
        # non_blocking pairs with the loader's pin_memory: the four H2D copies are issued async and
        # overlap with the tail of the previous step instead of blocking the launch thread on each.
        feats = batch.features.to(device, non_blocking=True)
        flens = batch.feature_lengths.to(device, non_blocking=True)
        tokens = batch.tokens.to(device, non_blocking=True)
        tlens = batch.token_lengths.to(device, non_blocking=True)
        if self._cr_ctc and self.training:
            return self._joint_loss_cr(feats, flens, tokens, tlens, chunk_size)
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
        total = rnnt + self._ctc_aux_weight * ctc + self._interctc_weighted(ictc_terms)
        # Return the RAW mean interctc (not the weighted sum) for logging -- it tracks the actual
        # CTC-decodability of the tapped stacks, the signal that flags encoder erosion.
        ictc_raw = torch.stack(ictc_terms).mean()
        return total, rnnt, ctc, ictc_raw, rnnt.new_zeros(())

    def _joint_loss_cr(
        self,
        feats: torch.Tensor,
        flens: torch.Tensor,
        tokens: torch.Tensor,
        tlens: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Two independently masked SpecAugment views. View 1 carries the whole objective; view 2 is
        # a second CTC-only forward whose head is tied to view 1's by the consistency KL. RNN-T runs
        # on view 1 ONLY -- its [B,T,U+1,V] lattice is the memory hog and CR-CTC's gain lives in the
        # CTC head, so a second lattice would only buy OOM on 12 GB.
        view1 = apply_spec_augment_batch(feats, flens)
        view2 = apply_spec_augment_batch(feats, flens)
        memory, out_len, ctc1, interctc_logits, base_len = self.forward(view1, flens, chunk_size)
        # View 2 feeds only the CTC head, so go through the encoder directly rather than forward():
        # forward() would also run both InterCTC head GEMMs over base-rate (2x) frames and allocate
        # their [B, T_base, V] logits, all of which this path discards.
        mem2, _ = self.encoder(view2, flens, chunk_size)
        ctc2 = self.ctc_head(mem2)
        rnnt = self.rnnt_loss(memory, out_len, tokens, tlens)
        # log_softmax each view once and reuse it for both the CTC alignment term and the KL;
        # passing raw logits to each would softmax every view twice.
        log_p1 = F.log_softmax(ctc1, dim=-1)
        log_p2 = F.log_softmax(ctc2, dim=-1)
        # Average the two views' CTC so both get a direct alignment gradient, not only the KL pull.
        ctc = 0.5 * (
            self._ctc_from_log_probs(log_p1, out_len, tokens, tlens)
            + self._ctc_from_log_probs(log_p2, out_len, tokens, tlens)
        )
        ictc_terms = self.interctc_terms(interctc_logits, base_len, tokens, tlens)
        cr = self._cr_consistency(log_p1, log_p2, out_len)
        total = (
            rnnt
            + self._ctc_aux_weight * ctc
            + self._interctc_weighted(ictc_terms)
            + self._cr_weight * cr
        )
        ictc_raw = torch.stack(ictc_terms).mean()
        return total, rnnt, ctc, ictc_raw, cr
