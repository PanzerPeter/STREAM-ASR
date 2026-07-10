# src/slices/Decode/StreamingSession.py — a live microphone session for the Demo slice.
# Audio arrives incrementally (PCM chunks over a WebSocket); this drives the causal encoder
# chunk-by-chunk and returns the CTC prefix-beam's running best hypothesis as a *partial*. When
# the speaker stops, finalize() re-decodes the fully buffered waveform through the handler's
# verified offline two-pass path (full context + attention rescore) — the authoritative result.
#
# Why two paths: the log-mel front end is center-padded, so a partial recomputed on a growing
# buffer is only exact away from its trailing edge — perfect for a live caption, wrong for a final
# number. Offline decode over the whole utterance both removes that edge effect and yields the
# best WER (the 9 % test-clean figure), so the endpointed result replaces the partials verbatim.
import torch
import torch.nn.functional as F

from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.TrainAcousticModel.StreamCache import StreamCache
from src.slices.Decode.CtcPrefixBeam import CtcPrefixBeam
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Decode.StreamingDecode_Response import StreamingDecode_Response


class StreamingSession:
    def __init__(self, handler: StreamingDecoder_Handler) -> None:
        # Reuses the handler's model/tokenizer/beam config (incl. shallow-fusion LM) so a live
        # session and a file decode share one configuration; finalize() delegates back to it.
        self.h = handler
        self.cfg = handler.cfg
        self.device = handler.model.ctc_head.weight.device
        self.feat_chunk = 2 * self.cfg.decode.chunk_size  # feature-rate frames per encoder step
        self.reset()

    def reset(self) -> None:
        self.buffer = torch.zeros(0, dtype=torch.float32)  # accumulated 16 kHz mono waveform
        self.cache = StreamCache.init(self.h.model.encoder, batch_size=1, device=self.device)
        self.beam = CtcPrefixBeam(self.cfg.model.blank_id, self.h.beam_size, self.h.beam_lm)
        self.beam.reset()
        self.fed = 0  # feature frames already handed to the encoder

    @torch.no_grad()
    def accept_audio(self, pcm: torch.Tensor) -> str:
        # Append new mono PCM, (re)compute features on the whole buffer, feed every *complete* new
        # encoder chunk, and return the running partial transcript. A trailing partial chunk waits
        # for more audio; the center-padded tail is resolved exactly by finalize().
        self.buffer = torch.cat([self.buffer, pcm.reshape(-1).to(torch.float32)])
        if self.buffer.numel() == 0:
            return ""
        feats = compute_log_mel(self.buffer).unsqueeze(0).to(self.device)  # [1, T, n_mels]
        enc = self.h.model.encoder
        while self.fed + self.feat_chunk <= feats.shape[1]:
            chunk = feats[:, self.fed : self.fed + self.feat_chunk]
            mem, self.cache = enc.streaming_forward(chunk, self.cache)
            # Mid-stream chunks are all real audio (no tail padding yet) -> every output frame is
            # valid, matching StreamingDecoder_Handler._stream_encode's non-tail steps.
            self.beam.advance(F.log_softmax(self.h.model.ctc_head(mem)[0], dim=-1))
            self.fed += self.feat_chunk
        return self.h.tok.decode(self.beam.partial())

    @torch.no_grad()
    def finalize(self) -> StreamingDecode_Response:
        # Endpoint: authoritative offline two-pass decode over the full utterance.
        return self.h.decode_waveform(self.buffer, streaming=False)
