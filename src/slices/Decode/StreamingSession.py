# src/slices/Decode/StreamingSession.py — a live microphone session for the Demo slice.
# Audio arrives incrementally (PCM chunks over a WebSocket); this drives the causal encoder
# chunk-by-chunk and returns a running greedy-transducer hypothesis as a *partial*. When the
# speaker stops, finalize() re-decodes the fully buffered waveform through the handler's verified
# offline path (full context, single-pass RNN-T beam search) — the authoritative result.
#
# Why two paths: the log-mel front end is center-padded, so a partial recomputed on a growing
# buffer is only exact away from its trailing edge — perfect for a live caption, wrong for a final
# number. Offline decode over the whole utterance both removes that edge effect and yields the
# best WER, so the endpointed result replaces the partials verbatim.
import torch

from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.Decode.StreamingDecode_Response import StreamingDecode_Response
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.TrainAcousticModel.StreamCache import StreamCache


class StreamingSession:
    def __init__(self, handler: StreamingDecoder_Handler) -> None:
        # Reuses the handler's model/tokenizer/searcher (incl. any LM fusion) so a live session and
        # a file decode share one configuration; finalize() delegates back to it. The searcher is
        # NOT rebuilt here -- self.h.searcher is the single TransducerBeamSearch Task 10 built.
        self.h = handler
        self.cfg = handler.cfg
        self.device = handler.model.ctc_head.weight.device
        self.feat_chunk = 2 * self.cfg.decode.chunk_size  # feature-rate frames per encoder step
        self.reset()

    def reset(self) -> None:
        self.buffer = torch.zeros(0, dtype=torch.float32)  # accumulated 16 kHz mono waveform
        self.cache = StreamCache.init(self.h.model.encoder, batch_size=1, device=self.device)
        self._mems: list[torch.Tensor] = []  # encoder memory chunks accumulated so far
        self.fed = 0  # feature frames already handed to the encoder

    @torch.no_grad()
    def accept_audio(self, pcm: torch.Tensor) -> str:
        # Append new mono PCM, (re)compute features on the whole buffer, feed every *complete* new
        # encoder chunk, and return a running greedy-transducer partial over all memory accumulated
        # so far. A trailing partial chunk waits for more audio; the center-padded tail is resolved
        # exactly by finalize().
        self.buffer = torch.cat([self.buffer, pcm.reshape(-1).to(torch.float32)])
        if self.buffer.numel() == 0:
            return ""
        feats = compute_log_mel(self.buffer).unsqueeze(0).to(self.device)  # [1, T, n_mels]
        enc = self.h.model.encoder
        while self.fed + self.feat_chunk <= feats.shape[1]:
            chunk = feats[:, self.fed : self.fed + self.feat_chunk]
            mem, self.cache = enc.streaming_forward(chunk, self.cache)
            self._mems.append(mem)
            self.fed += self.feat_chunk
        if not self._mems:
            return ""
        # Greedy over the accumulated memory is cheap and monotonic-enough for live captions; the
        # exact final comes from finalize()'s beam search over the offline-encoded memory.
        ids = self.h.searcher.greedy(torch.cat(self._mems, dim=1))
        return self.h.tok.decode(ids)

    @torch.no_grad()
    def finalize(self) -> StreamingDecode_Response:
        # Endpoint: authoritative offline decode (full context + beam search) over the full buffer.
        return self.h.decode_waveform(self.buffer, streaming=False)
