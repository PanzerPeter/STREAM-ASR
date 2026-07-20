# The Decode slice's orchestration: audio -> (streaming | offline) encoder memory -> single-pass
# RNN-T beam search. Streaming feeds the encoder chunk-by-chunk via StreamCache; offline runs one
# full-context encoder forward. Both funnel into the same TransducerBeamSearch so the only
# difference between the two modes is how `memory` gets built.
import time
from typing import Protocol

import torch
import torch.nn.functional as F

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Checkpoint_Adapter import load_checkpoint
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.StreamCache import StreamCache
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.slices.Decode.TransducerBeamSearch import TransducerBeamSearch
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
        model: TransducerModel,
        tokenizer: _Tokenizer,
        beam_size: int | None = None,
        fuse_lm: bool = True,
        lm_weight: float | None = None,
    ) -> None:
        # Single ablation gate (fuse_lm) replaces the old two-pass beam/rescore gate pair -- there
        # is only one pass now, so there is only one place the LM can attach. lm_weight == 0 forces
        # the LM off regardless of fuse_lm.
        self.model = model
        self.tok = tokenizer
        self.cfg = get_config()
        self.beam_size = beam_size if beam_size is not None else self.cfg.decode.beam_size
        # lm_weight override lets Evaluate sweep alpha on dev without mutating the authoritative
        # decode.yaml (whose lm_weight=0.0 is the alpha=0 regression lock); None = configured value.
        self.lm_weight = lm_weight if lm_weight is not None else self.cfg.decode.lm_weight
        # Per-token bonus applied at n-best re-ranking to offset RNN-T's deletion bias.
        self.length_bonus = self.cfg.decode.length_bonus
        # Load the LM only when fuse_lm actually consumes it AND lm_weight > 0; lm_weight == 0 (or
        # fuse_lm=False) keeps it None, so no checkpoint is read and search() reproduces the
        # pre-LM decoder exactly (the alpha=0 regression lock).
        needs_lm = fuse_lm and self.lm_weight > 0
        self.lm_scorer = self._load_lm() if needs_lm else None
        # The searcher is pure acoustic; the LM (when present) re-ranks its n-best in
        # _search_rescore below -- there is no per-step fusion path any more.
        self.searcher = TransducerBeamSearch(model, self.beam_size, self.cfg.decode.max_symbols)

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
        # touch disk). decode() is just load_audio + this; both share the exact single-pass path.
        audio_seconds = wave.shape[0] / self.cfg.audio.sample_rate
        start = time.perf_counter()
        memory, first_latency = self._encode(wave, streaming, start)
        nbest = self._search_rescore(memory)
        best_ids = nbest[0][0] if nbest else []
        text = self.tok.decode(best_ids)
        seg = SegmentResult(text=text, nbest=[(self.tok.decode(h), sc) for h, sc in nbest])
        rtf = (time.perf_counter() - start) / max(audio_seconds, 1e-6)
        return StreamingDecode_Response(
            text=text, segments=[seg], rtf=rtf, first_partial_latency_s=first_latency
        )

    def _search_rescore(self, memory: torch.Tensor) -> list[tuple[list[int], float]]:
        # Acoustic beam, then re-rank the n-best by acoustic + alpha*lm_seq + length_bonus*len:
        # the LM term is one full-sequence forward per hyp (replacing per-step shallow fusion),
        # the length term offsets RNN-T's deletion bias. With no LM AND length_bonus==0 the acoustic
        # order is returned untouched (the alpha=0 / no-LM regression lock).
        nbest = self.searcher.search(memory)
        lb = self.length_bonus
        if self.lm_scorer is None and lb == 0.0:
            return nbest
        scored = [
            (
                ids,
                ac
                + (self.lm_scorer.sequence_score(ids) if self.lm_scorer is not None else 0.0)
                + lb * len(ids),
            )
            for ids, ac in nbest
        ]
        scored.sort(key=lambda c: c[1], reverse=True)
        return scored

    def nbest_for_rescore(
        self, cmd: StreamingDecode_Command
    ) -> list[tuple[list[int], float, float]]:
        # Alpha-tuning support: return the acoustic n-best as (ids, acoustic_logp, lm_seq_logp) with
        # the *unweighted* LM sequence logprob attached. The acoustic beam is alpha-independent, so
        # a whole alpha sweep reuses one decode: the caller ranks by acoustic + alpha*lm at any
        # alpha without re-decoding. Requires an LM.
        if self.lm_scorer is None:
            raise ValueError("nbest_for_rescore requires an LM: fuse_lm=True with lm_weight>0")
        memory, _ = self._encode(load_audio(cmd.audio_path), cmd.streaming, time.perf_counter())
        return [
            (ids, ac, self.lm_scorer.raw_sequence_logprob(ids))
            for ids, ac in self.searcher.search(memory)
        ]

    def _encode(
        self, wave: torch.Tensor, streaming: bool, start: float
    ) -> tuple[torch.Tensor, float]:
        # Waveform -> encoder memory (streaming | offline), shared by decode_waveform and
        # nbest_for_rescore. Front end is CPU-only (soundfile + torchaudio mel); move to the model's
        # device so decode runs on the GPU the CLI placed the model on -- everything downstream
        # inherits feats.device.
        device = self.model.ctc_head.weight.device
        feats = compute_log_mel(wave).unsqueeze(0).to(device)  # [1, T, n_mels]
        if streaming:
            return self._stream_encode(feats, start)
        return self._offline_encode(feats)

    def _valid_out_frames(self, n_feats: int) -> int:
        # Output frames the encoder yields for n_feats real input frames: ×2 conv frontend then the
        # ×2 final downsample (ceil, matching SimpleDownsample). Used to discard the frames produced
        # by the silence padded onto the tail, so streaming stays consistent with offline.
        base = (n_feats - 1) // 2 + 1
        f = self.cfg.model.final_downsample
        return (base + f - 1) // f

    def _stream_encode(self, feats: torch.Tensor, start: float) -> tuple[torch.Tensor, float]:
        enc = self.model.encoder
        base = self.cfg.decode.chunk_size
        feat_chunk = 2 * base  # feature-rate chunk -> base base-rate frames
        n_valid = self._valid_out_frames(feats.shape[1])  # real output frames, before padding
        pad = (-feats.shape[1]) % feat_chunk
        if pad:  # trailing partial chunk: pad with edge silence so every step is aligned
            feats = F.pad(feats, (0, 0, 0, pad))
        cache = StreamCache.init(enc, batch_size=1, device=feats.device)
        mems: list[torch.Tensor] = []
        first_latency = 0.0
        emitted = 0
        for i, s in enumerate(range(0, feats.shape[1], feat_chunk)):
            mem, cache = enc.streaming_forward(feats[:, s : s + feat_chunk], cache)
            take = min(mem.shape[1], n_valid - emitted)  # drop frames that come from tail padding
            if take > 0:
                mem = mem[:, :take]
                mems.append(mem)
                emitted += take
            if i == 0:
                first_latency = time.perf_counter() - start
        memory = (
            torch.cat(mems, dim=1)
            if mems
            else torch.zeros(1, 0, enc.output_dim, device=feats.device)
        )
        return memory, first_latency

    def _offline_encode(self, feats: torch.Tensor) -> tuple[torch.Tensor, float]:
        lengths = torch.tensor([feats.shape[1]], device=feats.device)
        memory, out_len = self.model.encoder(feats, lengths, chunk_size=0)
        return memory[:, : int(out_len[0])], 0.0
