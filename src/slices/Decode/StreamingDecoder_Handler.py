# The Decode slice's orchestration: audio -> (streaming | offline) encoder memory -> single-pass
# RNN-T search. Streaming feeds the encoder chunk-by-chunk with a bounded StreamCache; offline runs
# one full-context forward. Both funnel into the same search, so the only difference between the
# two modes is how `memory` gets built.
import time
from typing import Protocol

import torch

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.Checkpoint_Adapter import load_checkpoint
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.Decode.InternalLmScorer import InternalLmScorer
from src.slices.Decode.LmScorer import LmScorer
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command
from src.slices.Decode.StreamingDecode_Response import (
    NbestEntry,
    SegmentResult,
    StreamingDecode_Response,
)
from src.slices.Decode.TransducerBeamSearch import TransducerBeamSearch
from src.slices.TrainAcousticModel.DwSepSubsampling import FRONTEND_STRIDE
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel

# (ids, score, emission frames), best first.
_Nbest = list[tuple[list[int], float, list[int]]]


class _Tokenizer(Protocol):
    # The handler only needs `.decode(ids) -> str`, so a test stub type-checks like SentencePiece.
    def decode(self, ids: list[int]) -> str: ...


class StreamingDecoder_Handler:
    def __init__(
        self,
        model: TransducerModel,
        tokenizer: _Tokenizer,
        beam_size: int | None = None,
        fuse_lm: bool = True,
        lm_weight: float | None = None,
        ilm_weight: float | None = None,
        length_bonus: float | None = None,
        lm_checkpoint: str | None = None,
        lm_tokenizer: str | None = None,
    ) -> None:
        # The overrides let Evaluate sweep weights on dev without mutating decode.yaml, whose zeros
        # are the pure-acoustic regression lock; None = the configured value.
        self.model = model
        self.tok = tokenizer
        self.cfg = get_config()
        d = self.cfg.decode
        self.beam_size = beam_size if beam_size is not None else d.beam_size
        self.lm_weight = lm_weight if lm_weight is not None else d.lm_weight
        self.length_bonus = length_bonus if length_bonus is not None else d.length_bonus
        # ILME only means something next to an external LM; with fuse_lm off there is no double
        # count to remove, so the stage stays identical to pure acoustic decoding.
        beta = ilm_weight if ilm_weight is not None else d.ilm_weight
        self.ilm_weight = beta if fuse_lm else 0.0
        self.ilm_scorer = InternalLmScorer(model)
        # A released LM is a .safetensors file beside its own tokenizer, not the paths in the yaml.
        self._lm_checkpoint = lm_checkpoint or d.lm_checkpoint
        self._lm_tokenizer = lm_tokenizer or self.cfg.lm.tokenizer
        self.lm_scorer = self._load_lm() if fuse_lm and self.lm_weight > 0 else None
        self.searcher = TransducerBeamSearch(model, self.beam_size, d.max_symbols)
        self.model.encoder.check_window(d.chunk_size, d.left_context)

    def _load_lm(self) -> LmScorer:
        device = self.model.ctc_head.weight.device
        lm_tok = SentencePieceTokenizer(self._lm_tokenizer)
        if lm_tok.vocab_size != self.cfg.lm.vocab_size:
            raise ValueError(
                f"{self._lm_tokenizer} has {lm_tok.vocab_size} pieces, "
                f"lm.vocab_size is {self.cfg.lm.vocab_size}"
            )
        lm = StreamLmModel()
        load_checkpoint(self._lm_checkpoint, lm)
        lm.to(device).eval()
        return LmScorer(lm, lm_tok, self.lm_weight)

    @property
    def frame_s(self) -> float:
        """Audio seconds per encoder memory frame (80 ms at 12.5 Hz)."""
        a = self.cfg.audio
        return self.model.encoder.mel_frames_per_output * a.hop_length / a.sample_rate

    def decode(self, cmd: StreamingDecode_Command) -> StreamingDecode_Response:
        return self.decode_waveform(load_audio(cmd.audio_path), cmd.streaming)

    def decode_waveform(self, wave: torch.Tensor, streaming: bool) -> StreamingDecode_Response:
        # Audio IO sits outside the timer on purpose: rtf/decode_s measure the model.
        audio_seconds = wave.shape[0] / self.cfg.audio.sample_rate
        start = time.perf_counter()
        memory, first_latency = self._encode(wave, streaming, start)
        encoded = time.perf_counter()
        nbest = self._search_rescore(memory)
        best_ids, _, best_frames = nbest[0] if nbest else ([], 0.0, [])
        text = self.tok.decode(best_ids)
        seg = SegmentResult(text=text, nbest=[(self.tok.decode(h), sc) for h, sc, _ in nbest])
        end = time.perf_counter()
        return StreamingDecode_Response(
            text=text,
            segments=[seg],
            rtf=(end - start) / max(audio_seconds, 1e-6),
            decode_s=end - start,
            audio_s=audio_seconds,
            finalize_s=end - encoded,
            first_partial_latency_s=first_latency,
            token_ids=best_ids,
            token_emit_s=self._emit_times(best_frames, streaming),
        )

    def _emit_times(self, frames: list[int], streaming: bool) -> list[float]:
        # A streaming frame is only visible once its whole chunk has arrived.
        per_chunk = self.cfg.decode.chunk_size // self.model.encoder.out_div if streaming else 1
        return [(f // per_chunk + 1) * per_chunk * self.frame_s for f in frames]

    def _search(self, memory: torch.Tensor) -> _Nbest:
        if self.beam_size > 1:
            return self.searcher.search(memory)
        ids, frames = self.model.greedy_decode(
            memory, torch.tensor([memory.shape[1]]), self.cfg.decode.max_symbols
        )
        return [(ids[0], 0.0, frames[0])]

    def _search_rescore(self, memory: torch.Tensor) -> _Nbest:
        # Acoustic search, then re-rank the n-best by
        #     acoustic + alpha*lm_seq - beta*ilm_seq + length_bonus*len
        # with each LM term ONE batched forward over the whole n-best. With no LM, beta == 0 and no
        # length bonus, the acoustic order is returned untouched (the regression lock).
        nbest = self._search(memory)
        lb = self.length_bonus
        if self.lm_scorer is None and self.ilm_weight == 0.0 and lb == 0.0:
            return nbest
        ids_only = [ids for ids, _, _ in nbest]
        lm = (
            self.lm_scorer.sequence_scores([self.tok.decode(ids) for ids in ids_only])
            if self.lm_scorer is not None
            else None
        )
        ilm = self.ilm_scorer.sequence_logprob_batch(ids_only) if self.ilm_weight else None
        scored = [
            (
                ids,
                ac
                + (lm[i] if lm is not None else 0.0)
                - self.ilm_weight * (ilm[i] if ilm is not None else 0.0)
                + lb * len(ids),
                frames,
            )
            for i, (ids, ac, frames) in enumerate(nbest)
        ]
        scored.sort(key=lambda c: c[1], reverse=True)
        return scored

    def nbest_for_rescore(self, cmd: StreamingDecode_Command) -> list[NbestEntry]:
        # Weight tuning: the acoustic n-best with the UNWEIGHTED LM and internal-LM sequence
        # logprobs attached. Neither weight moves the acoustic beam, so one decode serves the grid.
        if self.lm_scorer is None:
            raise ValueError("nbest_for_rescore requires an LM: fuse_lm=True with lm_weight>0")
        memory, _ = self._encode(load_audio(cmd.audio_path), cmd.streaming, time.perf_counter())
        nbest = self._search(memory)
        ids_only = [ids for ids, _, _ in nbest]
        lm = self.lm_scorer.raw_sequence_logprobs([self.tok.decode(ids) for ids in ids_only])
        ilm = self.ilm_scorer.sequence_logprob_batch(ids_only)
        return [
            NbestEntry(ids=ids, acoustic=ac, lm=lm[i], ilm=ilm[i])
            for i, (ids, ac, _) in enumerate(nbest)
        ]

    def _encode(
        self, wave: torch.Tensor, streaming: bool, start: float
    ) -> tuple[torch.Tensor, float | None]:
        # The front end is CPU (soundfile + torchaudio mel); everything after runs on the model's
        # device.
        device = self.model.ctc_head.weight.device
        feats = compute_log_mel(wave).unsqueeze(0).to(device)  # [1, T, n_mels]
        if streaming:
            return self._stream_encode(feats, start)
        lengths = torch.tensor([feats.shape[1]], device=device)
        memory, out_len = self.model.encoder(feats, lengths)
        # No partials exist in a one-shot encoder call, so the latency is None (not 0.0).
        return memory[:, : int(out_len[0])], None

    def _stream_encode(
        self, feats: torch.Tensor, start: float
    ) -> tuple[torch.Tensor, float | None]:
        # The final partial chunk is completed and `tail_chunks` more chunks follow, every frame of
        # them KEPT: tokens the model emits a little late need frames to land on (dropping them
        # costs ~0.5 dev-other WER, MODEL_ARCHITECTURE.md §7.3). The encoder mean-fills and
        # key-masks from n_valid on.
        d = self.cfg.decode
        enc = self.model.encoder
        mel_chunk = d.chunk_size * FRONTEND_STRIDE
        n_real = feats.shape[1]
        n_chunks = -(-n_real // mel_chunk) + d.tail_chunks
        padded = torch.zeros(1, n_chunks * mel_chunk, feats.shape[2], device=feats.device)
        padded[:, :n_real] = feats
        cache = enc.init_stream(1, d.left_context, feats.device)
        mems: list[torch.Tensor] = []
        first_latency: float | None = None
        for i in range(n_chunks):
            s = i * mel_chunk
            n_valid = max(0, min(mel_chunk, n_real - s))
            mem, cache = enc.streaming_forward(padded[:, s : s + mel_chunk], cache, n_valid)
            mems.append(mem)
            if i == 0:
                first_latency = time.perf_counter() - start
        return torch.cat(mems, dim=1), first_latency
