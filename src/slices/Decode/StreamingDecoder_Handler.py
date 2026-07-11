# The Decode slice's orchestration: audio -> streaming encoder -> CTC prefix-beam (live partials)
# -> attention rescore at the endpoint. Offline mode swaps the streaming encoder for one
# full-context forward but reuses both passes verbatim.
import time
from typing import Protocol

import torch
import torch.nn.functional as F

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Checkpoint_Adapter import load_checkpoint
from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention
from src.slices.TrainAcousticModel.StreamCache import StreamCache
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.slices.Decode.CtcPrefixBeam import CtcPrefixBeam
from src.slices.Decode.AttentionRescorer import AttentionRescorer
from src.slices.Decode.LmScorer import LmScorer
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command
from src.slices.Decode.StreamingDecode_Response import StreamingDecode_Response, SegmentResult


class _Tokenizer(Protocol):
    # Structural type: the handler only needs `.decode(ids) -> str`; it must not import a
    # concrete tokenizer implementation (SentencePiece vs. a test stub) to type-check against.
    def decode(self, ids: list[int]) -> str: ...


class StreamingDecoder_Handler:
    def __init__(
        self,
        model: HybridCtcAttention,
        tokenizer: _Tokenizer,
        beam_size: int | None = None,
        use_rescore: bool = True,
        fuse_lm_beam: bool = True,
        fuse_lm_rescore: bool = True,
        lm_weight: float | None = None,
    ) -> None:
        # The four ablation gates default to the full two-pass decoder; the Evaluate slice flips
        # them per stage (greedy CTC -> prefix beam -> +rescore -> +LM rescore -> +LM fusion)
        # without mutating global config. lm_weight == 0 forces the LM off regardless of the gates.
        self.model = model
        self.tok = tokenizer
        self.cfg = get_config()
        self.beam_size = beam_size if beam_size is not None else self.cfg.decode.beam_size
        self.use_rescore = use_rescore
        # lm_weight override lets Evaluate sweep alpha on dev without mutating the authoritative
        # decode.yaml (whose lm_weight=0.0 is the alpha=0 regression lock); None = configured value.
        self.lm_weight = lm_weight if lm_weight is not None else self.cfg.decode.lm_weight
        # Load the LM only when a gate actually consumes it AND lm_weight > 0; lm_weight == 0 (or a
        # stage that fuses in neither pass) keeps it None, so no checkpoint is read and both passes
        # reproduce the pre-LM decoder exactly.
        needs_lm = (fuse_lm_beam or fuse_lm_rescore) and self.lm_weight > 0
        self.lm_scorer = self._load_lm() if needs_lm else None
        self.beam_lm = self.lm_scorer if fuse_lm_beam else None
        self.rescorer = AttentionRescorer(
            model.decoder, self.lm_scorer if fuse_lm_rescore else None
        )

    def _load_lm(self) -> LmScorer:
        device = self.model.ctc_head.weight.device
        lm = StreamLmModel()
        load_checkpoint(self.cfg.decode.lm_checkpoint, lm)
        lm.to(device).eval()
        return LmScorer(lm, self.lm_weight)

    def decode(self, cmd: StreamingDecode_Command) -> StreamingDecode_Response:
        return self.decode_waveform(load_audio(cmd.audio_path), cmd.streaming)

    def decode_waveform(self, wave: torch.Tensor, streaming: bool) -> StreamingDecode_Response:
        # Waveform-in entry (the demo server decodes uploaded bytes / a live mic buffer that never
        # touch disk). decode() is just load_audio + this; both share the exact two-pass path.
        # Front end is CPU-only (soundfile + torchaudio mel); move to the model's device so decode
        # runs on the GPU the CLI placed the model on. Everything downstream inherits feats.device.
        device = self.model.ctc_head.weight.device
        feats = compute_log_mel(wave).unsqueeze(0).to(device)  # [1, T, n_mels]
        audio_seconds = wave.shape[0] / self.cfg.audio.sample_rate
        start = time.perf_counter()
        if streaming:
            memory, beam, first_latency = self._stream_encode(feats, start)
        else:
            memory, beam, first_latency = self._offline_encode(feats)
        mem_pad = torch.zeros(1, memory.shape[1], dtype=torch.bool, device=memory.device)
        if self.use_rescore:
            # Acoustic-only CTC scores: the rescorer applies the LM once, so first-pass shallow
            # fusion (which shaped this n-best) is not double-counted in the final score.
            nbest = beam.nbest_acoustic()[: self.beam_size]
            rescored = self.rescorer.rescore(
                memory, mem_pad, [h for h, _ in nbest], [s for _, s in nbest]
            )
        else:  # first-pass-only stages (ctc_greedy / prefix_beam) keep the beam's own ranking
            rescored = [(list(h), s) for h, s in beam.nbest()[: self.beam_size]]
        best_ids = rescored[0][0] if rescored else []
        text = self.tok.decode(best_ids)
        seg = SegmentResult(text=text, nbest=[(self.tok.decode(h), sc) for h, sc in rescored])
        rtf = (time.perf_counter() - start) / max(audio_seconds, 1e-6)
        return StreamingDecode_Response(
            text=text, segments=[seg], rtf=rtf, first_partial_latency_s=first_latency
        )

    def _valid_out_frames(self, n_feats: int) -> int:
        # Output frames the encoder yields for n_feats real input frames: ×2 conv frontend then the
        # ×2 final downsample (ceil, matching SimpleDownsample). Used to discard the frames produced
        # by the silence padded onto the tail, so streaming stays consistent with offline.
        base = (n_feats - 1) // 2 + 1
        f = self.cfg.model.final_downsample
        return (base + f - 1) // f

    def _stream_encode(
        self, feats: torch.Tensor, start: float
    ) -> tuple[torch.Tensor, CtcPrefixBeam, float]:
        enc = self.model.encoder
        base = self.cfg.decode.chunk_size
        feat_chunk = 2 * base  # feature-rate chunk -> base base-rate frames
        n_valid = self._valid_out_frames(feats.shape[1])  # real output frames, before padding
        pad = (-feats.shape[1]) % feat_chunk
        if pad:  # trailing partial chunk: pad with edge silence so every step is aligned
            feats = F.pad(feats, (0, 0, 0, pad))
        beam = CtcPrefixBeam(self.cfg.model.blank_id, self.beam_size, self.beam_lm)
        beam.reset()
        cache = StreamCache.init(enc, batch_size=1, device=feats.device)
        mems: list[torch.Tensor] = []
        first_latency = 0.0
        emitted = 0
        for i, s in enumerate(range(0, feats.shape[1], feat_chunk)):
            mem, cache = enc.streaming_forward(feats[:, s : s + feat_chunk], cache)
            take = min(mem.shape[1], n_valid - emitted)  # drop frames that come from tail padding
            if take > 0:
                mem = mem[:, :take]
                beam.advance(F.log_softmax(self.model.ctc_head(mem)[0], dim=-1))
                mems.append(mem)
                emitted += take
            if i == 0:
                first_latency = time.perf_counter() - start
        memory = (
            torch.cat(mems, dim=1)
            if mems
            else torch.zeros(1, 0, enc.output_dim, device=feats.device)
        )
        return memory, beam, first_latency

    def _offline_encode(self, feats: torch.Tensor) -> tuple[torch.Tensor, CtcPrefixBeam, float]:
        lengths = torch.tensor([feats.shape[1]], device=feats.device)
        logits, memory, out_len = self.model(feats, lengths, chunk_size=0)
        beam = CtcPrefixBeam(self.cfg.model.blank_id, self.beam_size, self.beam_lm)
        beam.reset()
        beam.advance(F.log_softmax(logits[0, : int(out_len[0])], dim=-1))
        return memory, beam, 0.0
